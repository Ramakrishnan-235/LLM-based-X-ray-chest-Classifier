import io
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from services.inference.classification.densenet_predictor import CXRClassifier


class DenseNetGradCAM:
    """
    Selective Grad-CAM generator for TorchXRayVision DenseNet-121.
    Computes spatial attribution overlays strictly for flagged or uncertain findings.
    """

    def __init__(self, classifier: CXRClassifier, alpha: float = 0.35):
        self.classifier = classifier
        self.model = classifier.model
        self.device = classifier.device
        self.alpha = alpha

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Target the final spatial conv layer (7x7x32 in DenseBlock4 Layer 16)
        target_layer = self.model.features.denseblock4.denselayer16.conv2
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input_tensor, output_tensor):
        self._activations = output_tensor

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def _render_overlay_png(
        self,
        cam_2d: np.ndarray,
        base_png_bytes: bytes,
    ) -> bytes:
        """Upsamples 7x7 Grad-CAM to native resolution and blends with the grayscale X-ray."""
        base_img = Image.open(io.BytesIO(base_png_bytes)).convert("RGB")
        base_np = np.array(base_img, dtype=np.uint8)
        orig_h, orig_w = base_np.shape[:2]

        cam_resized = cv2.resize(cam_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cam_uint8 = np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8)

        # Apply JET colormap (convert BGR from OpenCV to RGB)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Suppress low-activation background (< 15% peak) so normal lung fields stay clear
        mask = (cam_resized > 0.15)[..., None].astype(np.float32)
        blended = (
            base_np.astype(np.float32) * (1.0 - self.alpha * mask)
            + heatmap_rgb.astype(np.float32) * (self.alpha * mask)
        ).astype(np.uint8)

        out_buffer = io.BytesIO()
        Image.fromarray(blended).save(out_buffer, format="PNG")
        return out_buffer.getvalue()

    def predict_and_explain(
        self,
        tensor_input: np.ndarray,
        normalized_png_bytes: bytes,
        explain_states: Tuple[str, ...] = ("flagged_for_review", "uncertain"),
    ) -> Dict[str, object]:
        """
        Runs a single forward pass, formats calibrated predictions, and generates
        Grad-CAM PNG overlays only for findings whose state is in explain_states.
        """
        self.model.zero_grad(set_to_none=True)
        x = torch.from_numpy(tensor_input).to(self.device, dtype=torch.float32)
        x.requires_grad_(True)

        logits = self.model(x)[0]
        findings: List[dict] = self.classifier._format_predictions(logits.detach())
        overlays: Dict[str, bytes] = {}

        for item in findings:
            if item["result"] not in explain_states:
                item["grad_cam_generated"] = False
                continue

            finding_key = item["finding_key"]
            class_idx = self.classifier.target_indices[finding_key]

            self.model.zero_grad(set_to_none=True)
            logits[class_idx].backward(retain_graph=True)

            if self._activations is None or self._gradients is None:
                item["grad_cam_generated"] = False
                continue

            # Global average pooling of gradients across spatial dimensions (H, W)
            weights = torch.mean(self._gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self._activations, dim=1, keepdim=True)
            if cam.detach().max().item() > 0:
                cam = F.relu(cam)
            else:
                cam = cam - cam.detach().min()

            cam_np = cam.squeeze().detach().cpu().numpy()
            max_val = float(np.max(cam_np))
            if max_val < 1e-6:
                # Near-zero gradient norm; suppress noisy heatmap
                item["grad_cam_generated"] = False
                continue

            cam_norm = (cam_np - float(np.min(cam_np))) / (max_val + 1e-8)
            overlay_bytes = self._render_overlay_png(cam_norm, normalized_png_bytes)

            slug = item["label"].lower().replace(" ", "_")
            overlays[slug] = overlay_bytes
            item["grad_cam_generated"] = True

        # Free hook tensors immediately to keep VRAM footprint flat
        self._activations = None
        self._gradients = None
        self.model.zero_grad(set_to_none=True)

        return {
            "model_version": f"xrv-{self.classifier.backbone_name}-cal-{self.classifier.configuration_hash}",
            "configuration_hash": self.classifier.configuration_hash,
            "localization_metadata": {
                "method": "grad_cam",
                "target_layer": "features.denseblock4.denselayer16.conv2",
                "clinical_location_validated": False,
            },
            "findings": findings,
            "overlays": overlays,
        }

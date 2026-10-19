import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torchxrayvision as xrv
import yaml


DEFAULT_CONFIG_PATH = Path("ml/configs/findings_scope.yaml")


class CXRClassifier:
    """
    Production wrapper around TorchXRayVision DenseNet-121 with per-class
    Temperature Scaling and 3-zone clinical thresholding.
    """

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)

        self.findings_cfg: Dict[str, dict] = self.raw_config["findings"]
        self.backbone_name: str = self.raw_config.get(
            "model_backbone", "densenet121-res224-all"
        )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pretrained TorchXRayVision DenseNet-121
        self.model = xrv.models.DenseNet(weights=self.backbone_name).to(self.device)
        self.model.eval()

        # Map our 5 locked finding keys to TorchXRayVision output indices
        self.target_indices: Dict[str, int] = {}
        for finding_key in self.findings_cfg.keys():
            if finding_key not in self.model.pathologies:
                raise ValueError(
                    f"Finding '{finding_key}' not supported by {self.backbone_name}."
                )
            self.target_indices[finding_key] = self.model.pathologies.index(finding_key)

        self.configuration_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        payload = json.dumps(
            {"backbone": self.backbone_name, "findings": self.findings_cfg},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _format_predictions(self, logits_1d: torch.Tensor) -> List[dict]:
        results: List[dict] = []
        for finding_key, idx in self.target_indices.items():
            cfg = self.findings_cfg[finding_key]
            raw_logit = float(logits_1d[idx].item())
            temp = float(cfg["temperature"])
            low_t = float(cfg["low_threshold"])
            high_t = float(cfg["high_threshold"])

            cal_prob = float(torch.sigmoid(torch.tensor(raw_logit / temp)).item())

            if cal_prob >= high_t:
                state = "flagged_for_review"
            elif cal_prob >= low_t:
                state = "uncertain"
            else:
                state = "not_flagged"

            results.append(
                {
                    "finding_key": finding_key,
                    "label": cfg["ui_label"],
                    "raw_logit": round(raw_logit, 4),
                    "score": round(cal_prob, 4),
                    "uncertainty_floor": low_t,
                    "threshold": high_t,
                    "temperature": temp,
                    "result": state,
                }
            )
        return results

    @torch.inference_mode()
    def predict(self, tensor_input: np.ndarray) -> Dict[str, object]:
        """Fast inference path when Grad-CAM heatmaps are not requested."""
        x = torch.from_numpy(tensor_input).to(self.device, dtype=torch.float32)
        logits = self.model(x)[0]
        findings = self._format_predictions(logits)
        return {
            "model_version": f"xrv-{self.backbone_name}-cal-{self.configuration_hash}",
            "configuration_hash": self.configuration_hash,
            "findings": findings,
        }

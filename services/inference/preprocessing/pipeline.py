import io
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset
try:
    from pydicom.pixels import apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import torchxrayvision as xrv


SAFE_DICOM_TAGS = {
    "Modality",
    "ViewPosition",
    "PatientPosition",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "PixelSpacing",
    "ImagerPixelSpacing",
}


@dataclass
class PreprocessedStudyResult:
    tensor_input: Optional[np.ndarray]          # Shape (1, 1, 224, 224) float32 in [-1024, 1024]
    normalized_png_bytes: Optional[bytes]       # 8-bit grayscale PNG for viewer & MedGemma
    qc_status: str                              # accepted | flagged_unsupported_view | rejected_low_quality
    view_position: str                          # PA | AP | LATERAL | UNKNOWN
    photometric_inverted: bool
    border_phi_masked: bool
    original_shape: Tuple[int, int]
    safe_metadata: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None


class CXRPreprocessor:
    SUPPORTED_VIEWS = {"PA", "AP"}

    def __init__(
        self,
        target_size: int = 224,
        min_dimension: int = 224,
        border_mask_ratio: float = 0.08,
    ):
        self.target_size = target_size
        self.min_dimension = min_dimension
        self.border_mask_ratio = border_mask_ratio

    def process_bytes(self, file_bytes: bytes, filename: str = "") -> PreprocessedStudyResult:
        """Entry point supporting both raw DICOM (.dcm) and standard radiographs (.png/.jpg)."""
        is_dicom = filename.lower().endswith(".dcm") or self._has_dicom_magic(file_bytes)
        if is_dicom:
            return self._process_dicom(file_bytes)
        return self._process_standard_image(file_bytes)

    @staticmethod
    def _has_dicom_magic(data: bytes) -> bool:
        return len(data) > 132 and data[128:132] == b"DICM"

    def _extract_safe_metadata(self, ds: Dataset) -> Dict[str, Any]:
        """Whitelist technical imaging attributes; strip all Patient/Institution PHI tags."""
        meta: Dict[str, Any] = {}
        for tag_name in SAFE_DICOM_TAGS:
            if hasattr(ds, tag_name):
                val = getattr(ds, tag_name)
                meta[tag_name] = str(val) if not isinstance(val, (int, float, str)) else val
        return meta

    def _mask_peripheral_burned_in_text(self, img_uint8: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detects high-contrast small character blocks in the top/bottom border strips
        and masks them out while preserving central thoracic anatomy.
        """
        h, w = img_uint8.shape
        strip_h = max(1, int(h * self.border_mask_ratio))
        cleaned = img_uint8.copy()
        masked_any = False

        for y_start, y_end in [(0, strip_h), (h - strip_h, h)]:
            strip = cleaned[y_start:y_end, :]
            # Threshold extreme bright burned-in characters (> 240 intensity)
            _, thresh = cv2.threshold(strip, 240, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Filter for multi-character text strings (wider than single L/R marker, small height)
                if cw > int(w * 0.06) and ch < strip_h:
                    cv2.rectangle(strip, (x, y), (x + cw, y + ch), 0, thickness=-1)
                    masked_any = True

        return cleaned, masked_any

    def _finalize_pipeline(
        self,
        raw_float: np.ndarray,
        view_pos: str,
        inverted: bool,
        safe_meta: Dict[str, Any],
    ) -> PreprocessedStudyResult:
        h, w = raw_float.shape[:2]

        # 1. Quality Gate: Dimensions & Blank/Corrupted Check
        if h < self.min_dimension or w < self.min_dimension:
            return PreprocessedStudyResult(
                tensor_input=None,
                normalized_png_bytes=None,
                qc_status="rejected_low_quality",
                view_position=view_pos,
                photometric_inverted=inverted,
                border_phi_masked=False,
                original_shape=(h, w),
                safe_metadata=safe_meta,
                rejection_reason=f"Image dimensions ({w}x{h}) below minimum {self.min_dimension}px.",
            )

        std_val = float(np.std(raw_float))
        if std_val < 1e-3 or np.isnan(std_val):
            return PreprocessedStudyResult(
                tensor_input=None,
                normalized_png_bytes=None,
                qc_status="rejected_low_quality",
                view_position=view_pos,
                photometric_inverted=inverted,
                border_phi_masked=False,
                original_shape=(h, w),
                safe_metadata=safe_meta,
                rejection_reason="Near-zero pixel variance (blank or corrupted frame).",
            )

        # 2. Rescale dynamically to [0, 255] uint8 for peripheral PHI masking & viewer export
        min_v, max_v = float(np.min(raw_float)), float(np.max(raw_float))
        scaled_uint8 = ((raw_float - min_v) / (max_v - min_v + 1e-8) * 255.0).astype(np.uint8)

        # 3. Mask peripheral burned-in PHI text
        cleaned_uint8, phi_masked = self._mask_peripheral_burned_in_text(scaled_uint8)

        # 4. Encode lossless 8-bit PNG for Cornerstone3D viewer and MedGemma VLM
        pil_img = Image.fromarray(cleaned_uint8, mode="L")
        png_buffer = io.BytesIO()
        pil_img.save(png_buffer, format="PNG")
        png_bytes = png_buffer.getvalue()

        # 5. Normalize to [-1024, 1024] for TorchXRayVision DenseNet-121
        xrv_norm = xrv.datasets.normalize(cleaned_uint8.astype(np.float32), maxval=255.0)
        resized = cv2.resize(
            xrv_norm,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        tensor_input = resized[None, None, ...].astype(np.float32)  # (1, 1, 224, 224)

        # 6. Projection QC Gate
        qc_status = "accepted"
        if view_pos not in self.SUPPORTED_VIEWS and view_pos != "UNKNOWN":
            qc_status = "flagged_unsupported_view"

        return PreprocessedStudyResult(
            tensor_input=tensor_input,
            normalized_png_bytes=png_bytes,
            qc_status=qc_status,
            view_position=view_pos,
            photometric_inverted=inverted,
            border_phi_masked=phi_masked,
            original_shape=(h, w),
            safe_metadata=safe_meta,
        )

    def _process_dicom(self, dicom_bytes: bytes) -> PreprocessedStudyResult:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
        safe_meta = self._extract_safe_metadata(ds)

        Modality = str(getattr(ds, "Modality", "CR")).upper()
        if Modality not in {"CR", "DX", "OT", ""}:
            return PreprocessedStudyResult(
                tensor_input=None,
                normalized_png_bytes=None,
                qc_status="rejected_low_quality",
                view_position="UNKNOWN",
                photometric_inverted=False,
                border_phi_masked=False,
                original_shape=(0, 0),
                safe_metadata=safe_meta,
                rejection_reason=f"Unsupported DICOM Modality: {Modality} (expected CR or DX).",
            )

        view_pos = str(getattr(ds, "ViewPosition", "UNKNOWN")).strip().upper() or "UNKNOWN"
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()

        # Apply VOI LUT windowing if present
        pixel_array = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        if pixel_array.ndim == 3:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)

        inverted = False
        if photometric == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
            inverted = True

        return self._finalize_pipeline(pixel_array, view_pos, inverted, safe_meta)

    def _process_standard_image(self, image_bytes: bytes) -> PreprocessedStudyResult:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
        pixel_array = np.array(pil_img, dtype=np.float32)
        return self._finalize_pipeline(
            raw_float=pixel_array,
            view_pos="UNKNOWN",
            inverted=False,
            safe_meta={"source_format": "raster_image"},
        )

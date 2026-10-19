import torch
from services.inference.preprocessing.pipeline import CXRPreprocessor
from services.inference.classification.densenet_predictor import CXRClassifier
from services.inference.explainability.gradcam import DenseNetGradCAM
from tests.unit.test_preprocessing import create_synthetic_dicom


def test_classifier_and_selective_gradcam_vram_budget():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1. Preprocess synthetic PA DICOM from Sprint 1
    preprocessor = CXRPreprocessor()
    dcm_bytes = create_synthetic_dicom(photometric="MONOCHROME2", view_position="PA")
    prep_res = preprocessor.process_bytes(dcm_bytes, filename="sprint2_test.dcm")
    assert prep_res.qc_status == "accepted"
    assert prep_res.tensor_input is not None
    assert prep_res.normalized_png_bytes is not None

    # 2. Initialize Classifier & Grad-CAM wrapper
    classifier = CXRClassifier()
    gradcam = DenseNetGradCAM(classifier)

    # Force low thresholds temporarily on one finding to guarantee Grad-CAM triggers in test
    classifier.findings_cfg["Effusion"]["low_threshold"] = 0.001

    output = gradcam.predict_and_explain(
        tensor_input=prep_res.tensor_input,
        normalized_png_bytes=prep_res.normalized_png_bytes,
    )

    assert "model_version" in output
    assert len(output["findings"]) == 5
    assert output["localization_metadata"]["clinical_location_validated"] is False

    # Verify Pleural Effusion triggered a valid PNG overlay
    assert "pleural_effusion" in output["overlays"]
    overlay_bytes = output["overlays"]["pleural_effusion"]
    assert overlay_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # 3. Verify GPU VRAM ceiling (< 1.1 GB on RTX 2050)
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n[VRAM Check] Peak allocated GPU memory: {peak_vram_mb:.1f} MB")
        assert peak_vram_mb < 1100.0, f"VRAM exceeded 1.1 GB budget: {peak_vram_mb:.1f} MB"

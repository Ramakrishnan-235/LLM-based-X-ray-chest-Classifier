import httpx
from fastapi.testclient import TestClient
from apps.api.main import app
from services.worker.celery_app import celery_app
from tests.unit.test_preprocessing import create_synthetic_dicom


def test_end_to_end_study_upload_idempotency_and_inference():
    # Execute Celery tasks synchronously in-process for deterministic integration testing
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    with TestClient(app) as client:
        # 1. Create Study
        res_study = client.post("/api/v1/studies", json={"modality": "DX"})
        assert res_study.status_code == 201
        study_id = res_study.json()["study_id"]

        # 2. Request Pre-Signed MinIO Upload URL
        res_url = client.post(
            f"/api/v1/studies/{study_id}/upload-url",
            json={"filename": "chest_pa.dcm", "content_type": "application/dicom"},
        )
        assert res_url.status_code == 200
        upload_url = res_url.json()["upload_url"]

        # 3. Upload Synthetic DICOM directly to MinIO via Pre-Signed PUT URL
        dcm_bytes = create_synthetic_dicom(photometric="MONOCHROME2", view_position="PA")
        put_resp = httpx.put(
            upload_url,
            content=dcm_bytes,
            headers={"Content-Type": "application/dicom"},
        )
        assert put_resp.status_code in (200, 204)

        # 4. Trigger Async Analysis Pipeline
        res_analyze_1 = client.post(f"/api/v1/studies/{study_id}/analyze")
        assert res_analyze_1.status_code == 202
        job_id_1 = res_analyze_1.json()["job_id"]

        # 5. Verify Idempotency: Duplicate /analyze call returns the exact same job_id
        res_analyze_2 = client.post(f"/api/v1/studies/{study_id}/analyze")
        assert res_analyze_2.status_code == 202
        assert res_analyze_2.json()["job_id"] == job_id_1

        # 6. Verify Predictions Endpoint Contract
        res_preds = client.get(f"/api/v1/studies/{study_id}/predictions")
        assert res_preds.status_code == 200
        preds_data = res_preds.json()
        assert preds_data["status"] == "review_required"
        assert preds_data["image_quality"]["status"] == "accepted"
        assert preds_data["image_quality"]["view_position"] == "PA"
        assert preds_data["localization"]["clinical_location_validated"] is False
        assert len(preds_data["findings"]) == 5

        # 7. Verify Draft Report Endpoint Contract
        res_report = client.get(f"/api/v1/studies/{study_id}/report")
        assert res_report.status_code == 200
        report_data = res_report.json()
        assert report_data["status"] == "draft"
        assert report_data["reconciliation"]["status"] == "consistent"

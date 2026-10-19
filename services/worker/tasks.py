import json
import uuid
from typing import Any, Dict, Optional
import redis
from services.worker.celery_app import celery_app
from apps.api.config import settings
from apps.api.database import SessionLocal
from apps.api.models import (
    AnalysisJob,
    AuditEvent,
    ImageRecord,
    Prediction,
    Report,
    Study,
    Visualization,
)
from apps.api.storage import storage_service
from services.inference.preprocessing.pipeline import CXRPreprocessor
from services.inference.classification.densenet_predictor import CXRClassifier
from services.inference.explainability.gradcam import DenseNetGradCAM


redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

# Lazy worker singletons so GPU models initialize only once per worker process
_preprocessor: Optional[CXRPreprocessor] = None
_classifier: Optional[CXRClassifier] = None
_gradcam: Optional[DenseNetGradCAM] = None


def get_cv_pipeline():
    global _preprocessor, _classifier, _gradcam
    if _preprocessor is None:
        _preprocessor = CXRPreprocessor()
    if _classifier is None:
        _classifier = CXRClassifier()
    if _gradcam is None:
        _gradcam = DenseNetGradCAM(_classifier)
    return _preprocessor, _classifier, _gradcam


def publish_job_event(
    db,
    job: AnalysisJob,
    state: str,
    stage: str,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
):
    job.state = state
    job.current_stage = stage
    if error_code:
        job.error_code = error_code
        job.error_message = error_message
    db.commit()

    payload = json.dumps(
        {
            "job_id": str(job.id),
            "study_id": str(job.study_id),
            "state": state,
            "stage": stage,
            "error_code": error_code,
            "error_message": error_message,
        }
    )
    redis_client.set(f"job_latest:{job.id}", payload, ex=3600)
    redis_client.publish(f"job_progress:{job.id}", payload)


@celery_app.task(
    name="services.worker.tasks.run_cv_pipeline",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def run_cv_pipeline(self, study_id: str, job_id: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        study_uuid = uuid.UUID(study_id)
        job_uuid = uuid.UUID(job_id)

        job = db.query(AnalysisJob).filter_by(id=job_uuid).one()
        study = db.query(Study).filter_by(id=study_uuid).one()
        image_rec = db.query(ImageRecord).filter_by(study_id=study_uuid).one()

        job.attempts += 1
        study.status = "processing"
        publish_job_event(db, job, state="running", stage="preprocessing")

        # 1. Fetch raw DICOM/PNG from MinIO and run Sprint 1 Preprocessor
        raw_bytes = storage_service.get_object_bytes(image_rec.storage_key)
        preprocessor, _, gradcam = get_cv_pipeline()
        prep_res = preprocessor.process_bytes(raw_bytes, filename=image_rec.storage_key)

        study.qc_status = prep_res.qc_status
        study.view_position = prep_res.view_position
        study.photometric_inverted = prep_res.photometric_inverted
        study.border_phi_masked = prep_res.border_phi_masked
        image_rec.projection = prep_res.view_position
        db.commit()

        if prep_res.qc_status == "rejected_low_quality" or prep_res.tensor_input is None:
            study.status = "failed"
            publish_job_event(
                db,
                job,
                state="failed",
                stage="rejected_low_quality",
                error_code="QC_REJECTED",
                error_message=prep_res.rejection_reason or "Image failed quality check.",
            )
            return {"status": "aborted_qc", "study_id": study_id, "job_id": job_id}

        # 2. Save normalized 8-bit PNG to MinIO for viewer & VLM
        processed_key = f"processed/{study_id}/image.png"
        storage_service.put_object_bytes(
            processed_key, prep_res.normalized_png_bytes, content_type="image/png"
        )
        image_rec.processed_storage_key = processed_key
        db.commit()

        # 3. Run Sprint 2 Calibrated DenseNet-121 + Selective Grad-CAM
        publish_job_event(db, job, state="running", stage="classifying_and_explaining")
        cv_output = gradcam.predict_and_explain(
            tensor_input=prep_res.tensor_input,
            normalized_png_bytes=prep_res.normalized_png_bytes,
        )

        # Clear any partial predictions from prior retry attempts for idempotency
        db.query(Prediction).filter_by(job_id=job_uuid).delete()
        db.commit()

        serialized_findings = []
        for item in cv_output["findings"]:
            pred_row = Prediction(
                job_id=job_uuid,
                finding=item["label"],
                raw_logit=item["raw_logit"],
                score=item["score"],
                uncertainty_floor=item["uncertainty_floor"],
                threshold=item["threshold"],
                result=item["result"],
            )
            db.add(pred_row)
            db.flush()

            slug = item["label"].lower().replace(" ", "_")
            grad_cam_key = None
            if slug in cv_output["overlays"]:
                grad_cam_key = f"visualizations/{study_id}/{slug}_gradcam.png"
                storage_service.put_object_bytes(
                    grad_cam_key,
                    cv_output["overlays"][slug],
                    content_type="image/png",
                )
                viz_row = Visualization(
                    prediction_id=pred_row.id,
                    storage_key=grad_cam_key,
                    method="grad_cam",
                    clinical_location_validated=False,
                )
                db.add(viz_row)

            serialized_findings.append(
                {
                    "label": item["label"],
                    "score": item["score"],
                    "uncertainty_floor": item["uncertainty_floor"],
                    "threshold": item["threshold"],
                    "result": item["result"],
                    "grad_cam_key": grad_cam_key,
                }
            )

        db.add(
            AuditEvent(
                study_id=study_uuid,
                action="cv_inference_completed",
                details={
                    "job_id": job_id,
                    "model_version": cv_output["model_version"],
                    "qc_status": prep_res.qc_status,
                },
            )
        )
        db.commit()

        return {
            "status": "cv_completed",
            "study_id": study_id,
            "job_id": job_id,
            "processed_png_key": processed_key,
            "findings": serialized_findings,
        }
    except Exception as exc:
        db.rollback()
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    name="services.worker.tasks.run_vlm_report_pipeline",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def run_vlm_report_pipeline(self, cv_result: Dict[str, Any]) -> Dict[str, Any]:
    if cv_result.get("status") == "aborted_qc":
        return cv_result

    db = SessionLocal()
    try:
        study_uuid = uuid.UUID(cv_result["study_id"])
        job_uuid = uuid.UUID(cv_result["job_id"])

        job = db.query(AnalysisJob).filter_by(id=job_uuid).one()
        study = db.query(Study).filter_by(id=study_uuid).one()

        publish_job_event(db, job, state="running", stage="drafting_report")

        findings = cv_result.get("findings", [])
        flagged = [f["label"] for f in findings if f["result"] == "flagged_for_review"]
        uncertain = [f["label"] for f in findings if f["result"] == "uncertain"]

        # Deterministic structured draft (Upgraded to MedGemma 1.5 4B + Reconciliation in Sprint 4)
        if flagged:
            findings_narrative = (
                f"Automated analysis flagged {', '.join(flagged)} above the review threshold. "
            )
        else:
            findings_narrative = "No supported findings exceeded the high-confidence review threshold. "

        if uncertain:
            findings_narrative += (
                f"Equivocal signal detected in the uncertainty zone for {', '.join(uncertain)}."
            )

        impression_narrative = (
            "Preliminary AI-assisted triage report. Mandatory validation by a qualified "
            "radiologist is required prior to clinical interpretation."
        )

        # Idempotent report persistence
        db.query(Report).filter_by(job_id=job_uuid).delete()
        report = Report(
            study_id=study_uuid,
            job_id=job_uuid,
            draft_findings=findings_narrative.strip(),
            draft_impression=impression_narrative,
            reconciliation_status="consistent",
            conflicts_json=[],
            requires_mandatory_override=False,
            review_status="draft",
        )
        db.add(report)

        study.status = "review_required"
        db.add(
            AuditEvent(
                study_id=study_uuid,
                action="draft_report_generated",
                details={"job_id": str(job_uuid), "flagged_count": len(flagged)},
            )
        )
        publish_job_event(db, job, state="completed", stage="review_required")

        return {
            "status": "review_required",
            "study_id": str(study_uuid),
            "job_id": str(job_uuid),
            "report_id": str(report.id),
        }
    finally:
        db.close()

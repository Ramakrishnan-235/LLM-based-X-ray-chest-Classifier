import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from celery import chain
import redis.asyncio as aioredis
from sqlalchemy.orm import Session

from apps.api.config import settings
from apps.api.database import Base, SessionLocal, engine, get_db
from apps.api.models import (
    AnalysisJob,
    AuditEvent,
    ImageRecord,
    ModelVersion,
    Prediction,
    Report,
    Study,
    seed_defaults,
)
from apps.api.schemas import (
    AnalyzeResponse,
    CreateStudyRequest,
    CreateStudyResponse,
    FindingItem,
    StudyPredictionsResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from apps.api.storage import storage_service
from services.worker.tasks import run_cv_pipeline, run_vlm_report_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_defaults(db)
    yield


app = FastAPI(
    title="CXR Insight AI — Radiology Assistance API",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "cxr-insight-api"}


@app.get("/api/v1/models")
def list_approved_models(db: Session = Depends(get_db)):
    models = db.query(ModelVersion).filter_by(is_approved=True).all()
    return [
        {
            "id": str(m.id),
            "name": m.name,
            "version": m.version,
            "role": m.role,
            "configuration_hash": m.configuration_hash,
        }
        for m in models
    ]


@app.post("/api/v1/studies", response_model=CreateStudyResponse, status_code=status.HTTP_201_CREATED)
def create_study(payload: CreateStudyRequest, db: Session = Depends(get_db)):
    user, _ = seed_defaults(db)
    study_uid = payload.study_uid or f"cxr_{uuid.uuid4().hex[:12]}"

    study = Study(
        user_id=user.id,
        organization_id=user.organization_id,
        study_uid=study_uid,
        status="awaiting_upload",
    )
    db.add(study)
    db.flush()

    db.add(
        AuditEvent(
            actor_id=user.id,
            study_id=study.id,
            action="study_created",
            details={"study_uid": study_uid},
        )
    )
    db.commit()
    db.refresh(study)
    return CreateStudyResponse(
        study_id=study.id, study_uid=study.study_uid, status=study.status
    )


@app.post("/api/v1/studies/{study_id}/upload-url", response_model=UploadUrlResponse)
def create_upload_url(
    study_id: uuid.UUID, payload: UploadUrlRequest, db: Session = Depends(get_db)
):
    study = db.query(Study).filter_by(id=study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")

    ext = ".dcm" if payload.filename.lower().endswith(".dcm") else ".png"
    storage_key = f"originals/{study.id}/image{ext}"

    img_rec = db.query(ImageRecord).filter_by(study_id=study.id).first()
    if not img_rec:
        img_rec = ImageRecord(study_id=study.id, storage_key=storage_key)
        db.add(img_rec)
    else:
        img_rec.storage_key = storage_key
    db.commit()

    upload_url = storage_service.generate_presigned_upload_url(
        object_key=storage_key, content_type=payload.content_type
    )
    return UploadUrlResponse(
        study_id=study.id,
        storage_key=storage_key,
        upload_url=upload_url,
    )


@app.post(
    "/api/v1/studies/{study_id}/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_analysis_job(study_id: uuid.UUID, db: Session = Depends(get_db)):
    study = db.query(Study).filter_by(id=study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")

    img_rec = db.query(ImageRecord).filter_by(study_id=study.id).first()
    if not img_rec or not storage_service.object_exists(img_rec.storage_key):
        raise HTTPException(
            status_code=400,
            detail="Source radiograph not found in private storage. Upload file first.",
        )

    _, classifier_version = seed_defaults(db)

    # Idempotency check: return existing job if already created for (study_id, model_version_id)
    existing_job = (
        db.query(AnalysisJob)
        .filter_by(study_id=study.id, model_version_id=classifier_version.id)
        .first()
    )
    if existing_job and existing_job.state in {"queued", "running", "completed"}:
        return AnalyzeResponse(
            job_id=existing_job.id,
            study_id=study.id,
            state=existing_job.state,
            current_stage=existing_job.current_stage,
        )

    job = AnalysisJob(
        study_id=study.id,
        model_version_id=classifier_version.id,
        state="queued",
        current_stage="queued",
    )
    study.status = "queued"
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch Dual-Queue Celery Workflow Chain
    workflow = chain(
        run_cv_pipeline.s(str(study.id), str(job.id)).set(queue="cxr_cv_local"),
        run_vlm_report_pipeline.s().set(queue="cxr_vlm_remote"),
    )
    workflow.apply_async()

    return AnalyzeResponse(
        job_id=job.id,
        study_id=study.id,
        state=job.state,
        current_stage=job.current_stage,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=AnalyzeResponse)
def get_job_status(job_id: uuid.UUID, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return AnalyzeResponse(
        job_id=job.id,
        study_id=job.study_id,
        state=job.state,
        current_stage=job.current_stage,
    )


@app.get("/api/v1/jobs/{job_id}/stream")
async def stream_job_progress(job_id: uuid.UUID):
    """Server-Sent Events (SSE) endpoint streaming real-time worker stage updates."""
    async def event_generator():
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        pubsub = r.pubsub()
        channel = f"job_progress:{job_id}"
        await pubsub.subscribe(channel)
        try:
            # Immediately yield latest cached state if available
            latest = await r.get(f"job_latest:{job_id}")
            if latest:
                yield f"data: {latest}\n\n"
                parsed = json.loads(latest)
                if parsed.get("state") in {"completed", "failed"}:
                    return

            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("data"):
                    data_str = message["data"]
                    yield f"data: {data_str}\n\n"
                    parsed = json.loads(data_str)
                    if parsed.get("state") in {"completed", "failed"}:
                        break
                await asyncio.sleep(0.2)
        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/studies/{study_id}/predictions", response_model=StudyPredictionsResponse)
def get_study_predictions(study_id: uuid.UUID, db: Session = Depends(get_db)):
    study = db.query(Study).filter_by(id=study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")

    job = (
        db.query(AnalysisJob)
        .filter_by(study_id=study.id)
        .order_by(AnalysisJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="No analysis job found for study.")

    model_ver = db.query(ModelVersion).filter_by(id=job.model_version_id).first()
    predictions = db.query(Prediction).filter_by(job_id=job.id).all()

    finding_items = []
    for p in predictions:
        viz = p.visualizations[0] if p.visualizations else None
        grad_key = viz.storage_key if viz else None
        grad_url = (
            storage_service.generate_presigned_download_url(grad_key)
            if grad_key
            else None
        )
        finding_items.append(
            FindingItem(
                label=p.finding,
                score=p.score,
                uncertainty_floor=p.uncertainty_floor,
                threshold=p.threshold,
                result=p.result,
                grad_cam_key=grad_key,
                grad_cam_url=grad_url,
            )
        )

    return StudyPredictionsResponse(
        study_id=study.id,
        status=study.status,
        model_version=model_ver.version if model_ver else "unknown",
        image_quality={
            "status": study.qc_status,
            "view_position": study.view_position,
            "photometric_inverted": study.photometric_inverted,
            "border_phi_masked": study.border_phi_masked,
        },
        localization={
            "method": "grad_cam",
            "clinical_location_validated": False,
        },
        findings=finding_items,
    )


@app.get("/api/v1/studies/{study_id}/report")
def get_study_report(study_id: uuid.UUID, db: Session = Depends(get_db)):
    report = (
        db.query(Report)
        .filter_by(study_id=study_id)
        .order_by(Report.created_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet generated.")

    return {
        "report_id": str(report.id),
        "study_id": str(report.study_id),
        "status": report.review_status,
        "findings": report.draft_findings,
        "impression": report.draft_impression,
        "reconciliation": {
            "status": report.reconciliation_status,
            "conflicts": report.conflicts_json,
            "requires_mandatory_override": report.requires_mandatory_override,
        },
    }

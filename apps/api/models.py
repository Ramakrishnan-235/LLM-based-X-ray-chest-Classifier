import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
from apps.api.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(50), default="reviewer")  # clinician | reviewer | admin
    organization_id: Mapped[str] = mapped_column(String(100), index=True, default="org_demo_hospital")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True, default="org_demo_hospital")
    study_uid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(50), index=True, default="awaiting_upload")
    qc_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    view_position: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    photometric_inverted: Mapped[bool] = mapped_column(Boolean, default=False)
    border_phi_masked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    images: Mapped[List["ImageRecord"]] = relationship(back_populates="study", cascade="all, delete-orphan")
    jobs: Mapped[List["AnalysisJob"]] = relationship(back_populates="study", cascade="all, delete-orphan")
    reports: Mapped[List["Report"]] = relationship(back_populates="study", cascade="all, delete-orphan")


class ImageRecord(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("studies.id"), index=True)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    processed_storage_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    modality: Mapped[str] = mapped_column(String(32), default="DX")
    projection: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    study: Mapped["Study"] = relationship(back_populates="images")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(32))  # classifier | vlm
    configuration_hash: Mapped[str] = mapped_column(String(64))
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        UniqueConstraint("study_id", "model_version_id", name="uq_study_model_version_job"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("studies.id"), index=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id"), index=True)
    state: Mapped[str] = mapped_column(String(50), index=True, default="queued")
    current_stage: Mapped[str] = mapped_column(String(64), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    study: Mapped["Study"] = relationship(back_populates="jobs")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_jobs.id"), index=True)
    finding: Mapped[str] = mapped_column(String(100), index=True)
    raw_logit: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    uncertainty_floor: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    result: Mapped[str] = mapped_column(String(50))  # flagged_for_review | uncertain | not_flagged
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped["AnalysisJob"] = relationship(back_populates="predictions")
    visualizations: Mapped[List["Visualization"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class Visualization(Base):
    __tablename__ = "visualizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("predictions.id"), index=True)
    storage_key: Mapped[str] = mapped_column(String(512))
    method: Mapped[str] = mapped_column(String(50), default="grad_cam")
    clinical_location_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="visualizations")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("studies.id"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_jobs.id"), index=True)
    draft_findings: Mapped[str] = mapped_column(Text)
    draft_impression: Mapped[str] = mapped_column(Text)
    reconciliation_status: Mapped[str] = mapped_column(String(50), default="consistent")
    conflicts_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    requires_mandatory_override: Mapped[bool] = mapped_column(Boolean, default=False)
    review_status: Mapped[str] = mapped_column(String(50), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    study: Mapped["Study"] = relationship(back_populates="reports")
    reviews: Mapped[List["Review"]] = relationship(back_populates="report", cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id"), index=True)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    decision: Mapped[str] = mapped_column(String(50))  # approved | approved_with_edits | rejected
    edited_findings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    edited_impression: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diff_patch: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    report: Mapped["Report"] = relationship(back_populates="reviews")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    study_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def seed_defaults(db: Session) -> tuple[User, ModelVersion]:
    """Ensures a default demo reviewer and approved DenseNet-121 version exist."""
    user = db.query(User).filter_by(email="radiologist@cxr-insight.local").first()
    if not user:
        user = User(
            email="radiologist@cxr-insight.local",
            role="reviewer",
            organization_id="org_demo_hospital",
        )
        db.add(user)

    model_ver = db.query(ModelVersion).filter_by(name="densenet121-res224-all").first()
    if not model_ver:
        model_ver = ModelVersion(
            name="densenet121-res224-all",
            version="xrv-densenet121-res224-all-cal-v1",
            role="classifier",
            configuration_hash="sprint2-cal",
            is_approved=True,
        )
        db.add(model_ver)

    db.commit()
    db.refresh(user)
    db.refresh(model_ver)
    return user, model_ver

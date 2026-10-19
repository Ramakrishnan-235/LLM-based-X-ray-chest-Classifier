import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class CreateStudyRequest(BaseModel):
    study_uid: Optional[str] = None
    modality: str = "DX"


class CreateStudyResponse(BaseModel):
    study_id: uuid.UUID
    study_uid: str
    status: str


class UploadUrlRequest(BaseModel):
    filename: str = "image.dcm"
    content_type: str = "application/dicom"


class UploadUrlResponse(BaseModel):
    study_id: uuid.UUID
    storage_key: str
    upload_url: str
    expires_in_seconds: int = 900


class AnalyzeResponse(BaseModel):
    job_id: uuid.UUID
    study_id: uuid.UUID
    state: str
    current_stage: str


class FindingItem(BaseModel):
    label: str
    score: float
    uncertainty_floor: float
    threshold: float
    result: str
    grad_cam_key: Optional[str] = None
    grad_cam_url: Optional[str] = None


class StudyPredictionsResponse(BaseModel):
    study_id: uuid.UUID
    status: str
    model_version: str
    image_quality: Dict[str, Any]
    localization: Dict[str, Any]
    findings: List[FindingItem]

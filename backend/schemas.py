from typing import List, Optional

from pydantic import BaseModel


class CaseAccepted(BaseModel):
    case_id: str
    job_id: str
    status: str
    created_at: str


class CaseSummary(BaseModel):
    case_id: str
    filename: str
    sha256: str
    size_bytes: int
    created_at: str
    job_id: str
    status: str
    stage: str
    progress: float


class CaseList(BaseModel):
    items: List[CaseSummary]
    limit: int
    offset: int


class CaseDetail(BaseModel):
    case_id: str
    filename: str
    sha256: str
    size_bytes: int
    created_at: str
    job_id: str
    status: str
    stage: str
    progress: float
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    case_id: str
    status: str
    stage: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ArtifactItem(BaseModel):
    key: str
    size_bytes: int
    url: str


class ArtifactList(BaseModel):
    items: List[ArtifactItem]



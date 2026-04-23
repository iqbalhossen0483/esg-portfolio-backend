from pydantic import BaseModel


class UploadAck(BaseModel):
    job_id: int
    file_name: str
    file_size: int
    file_sha256: str
    status: str = "queued"


class IngestionStatus(BaseModel):
    job_id: int
    file_name: str
    status: str
    total_chunks: int | None = None
    chunks_processed: int = 0
    records_stored: dict | None = None
    quality_report: dict | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class IngestionJob(BaseModel):
    job_id: int
    file_name: str
    file_size: int | None = None
    status: str
    total_chunks: int | None = None
    chunks_processed: int = 0
    records_stored: dict | None = None
    quality_report: dict | None = None
    started_at: str | None = None
    completed_at: str | None = None

    class Config:
        from_attributes = True

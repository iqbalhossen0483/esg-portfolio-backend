from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    file_name: str
    file_size: int | None = None
    status: str = "processing"
    message: str = "File uploaded. Ingestion pipeline started."


class IngestionStatus(BaseModel):
    job_id: str
    file_name: str
    status: str
    total_chunks: int | None = None
    chunks_processed: int = 0
    records_stored: dict | None = None
    quality_report: dict | None = None


class IngestionJob(BaseModel):
    job_id: str
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

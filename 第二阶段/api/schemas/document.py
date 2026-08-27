"""临时上传文件 API schema。"""

from pydantic import BaseModel, Field

from 第二阶段.services.session_service import SessionDocumentSummary


class DocumentResponse(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    chunk_count: int = Field(ge=0)
    status: str

    @classmethod
    def from_summary(
        cls, summary: SessionDocumentSummary
    ) -> "DocumentResponse":
        return cls(
            document_id=summary.document_id,
            file_name=summary.file_name,
            file_type=summary.file_type,
            chunk_count=summary.chunk_count,
            status=summary.status,
        )


class DocumentListResponse(BaseModel):
    session_id: str
    documents: list[DocumentResponse] = Field(default_factory=list)


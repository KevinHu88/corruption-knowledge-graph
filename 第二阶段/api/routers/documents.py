"""临时文档上传与列表 Endpoint。"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from 第二阶段.api.dependencies import get_qa_service
from 第二阶段.api.schemas.document import DocumentListResponse, DocumentResponse
from 第二阶段.services.qa_service import QAService

router = APIRouter(prefix="/sessions/{session_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    session_id: str,
    file: Annotated[UploadFile, File(...)],
    service: Annotated[QAService, Depends(get_qa_service)],
) -> DocumentResponse:
    summary = service.add_document(
        session_id,
        file_name=file.filename or "",
        content_type=file.content_type,
        file_object=file.file,
    )
    return DocumentResponse.from_summary(summary)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    session_id: str,
    service: Annotated[QAService, Depends(get_qa_service)],
) -> DocumentListResponse:
    documents = [
        DocumentResponse.from_summary(item)
        for item in service.list_documents(session_id)
    ]
    return DocumentListResponse(session_id=session_id, documents=documents)


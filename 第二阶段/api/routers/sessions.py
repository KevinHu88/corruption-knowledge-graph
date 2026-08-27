"""Session 创建与删除 Endpoint。"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from 第二阶段.api.dependencies import get_qa_service
from 第二阶段.api.schemas.session import DeleteSessionResponse, SessionResponse
from 第二阶段.services.qa_service import QAService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    service: Annotated[QAService, Depends(get_qa_service)],
) -> SessionResponse:
    return SessionResponse.from_state(service.create_session())


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
def delete_session(
    session_id: str,
    service: Annotated[QAService, Depends(get_qa_service)],
) -> DeleteSessionResponse:
    service.delete_session(session_id)
    return DeleteSessionResponse(session_id=session_id)


"""知识问答 Endpoint；业务逻辑全部委托 QAService。"""

from typing import Annotated

from fastapi import APIRouter, Depends

from 第二阶段.api.dependencies import get_qa_service
from 第二阶段.api.schemas.question import QuestionRequest, QuestionResponse
from 第二阶段.services.qa_service import QAService

router = APIRouter(prefix="/sessions/{session_id}/questions", tags=["questions"])


@router.post("", response_model=QuestionResponse)
def answer_question(
    session_id: str,
    request: QuestionRequest,
    service: Annotated[QAService, Depends(get_qa_service)],
) -> QuestionResponse:
    result = service.answer_question(
        session_id,
        request.question,
        case_id=request.case_id,
        search_scope=request.search_scope,
        selected_case_ids=request.selected_case_ids,
    )
    return QuestionResponse.from_result(result)

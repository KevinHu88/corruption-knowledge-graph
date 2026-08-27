"""将内部异常统一转换为不含堆栈信息的 HTTP 错误。"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from 第二阶段.exceptions import (
    AmbiguousEntityError,
    DocumentParsingError,
    FileTooLargeError,
    GraphRetrievalError,
    InvalidQuestionError,
    LLMGenerationError,
    SessionNotFoundError,
)
from 第二阶段.parsing.base import UnsupportedFileTypeError
from 第二阶段.pipeline.qa_pipeline import PipelineConfigurationError

logger = logging.getLogger(__name__)


def _handler(status_code: int):
    async def handle(_: Request, exc: Exception) -> JSONResponse:
        logger.warning("API request failed: %s", exc)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handle


async def _ambiguous_entity_handler(
    _: Request, exc: AmbiguousEntityError
) -> JSONResponse:
    logger.warning("API request requires entity disambiguation: %s", exc)
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "entity_name": exc.entity_name,
            "candidate_case_ids": exc.candidate_case_ids,
        },
    )


async def _unexpected_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unexpected API error: %s",
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(SessionNotFoundError, _handler(404))
    app.add_exception_handler(FileTooLargeError, _handler(413))
    app.add_exception_handler(UnsupportedFileTypeError, _handler(415))
    app.add_exception_handler(DocumentParsingError, _handler(400))
    app.add_exception_handler(InvalidQuestionError, _handler(400))
    app.add_exception_handler(AmbiguousEntityError, _ambiguous_entity_handler)
    app.add_exception_handler(GraphRetrievalError, _handler(503))
    app.add_exception_handler(LLMGenerationError, _handler(503))
    app.add_exception_handler(PipelineConfigurationError, _handler(503))
    app.add_exception_handler(Exception, _unexpected_handler)

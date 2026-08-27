"""连接 HTTP 用例与现有 KnowledgeQAPipeline。"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from 第二阶段.config import QAConfig
from 第二阶段.exceptions import (
    DocumentParsingError,
    FileTooLargeError,
    InvalidQuestionError,
)
from 第二阶段.parsing.base import ParserError, UnsupportedFileTypeError
from 第二阶段.pipeline.qa_pipeline import KnowledgeQAPipeline
from 第二阶段.schemas.models import AnswerResult, UploadedDocument
from 第二阶段.schemas.models import PathSearchScope
from 第二阶段.services.session_service import (
    SessionDocumentSummary,
    SessionService,
    SessionState,
)
from 第二阶段.storage.session_document_store import SessionDocumentStore

PipelineFactory = Callable[[SessionDocumentStore], KnowledgeQAPipeline]


class QAService:
    """只负责编排 Session、上传文件与现有 Pipeline。"""

    def __init__(
        self,
        session_service: SessionService,
        pipeline_factory: PipelineFactory,
        config: QAConfig,
    ) -> None:
        self.session_service = session_service
        self.pipeline_factory = pipeline_factory
        self.config = config

    def create_session(self) -> SessionState:
        return self.session_service.create_session()

    def delete_session(self, session_id: str) -> SessionState:
        return self.session_service.delete_session(session_id)

    def list_documents(
        self, session_id: str
    ) -> list[SessionDocumentSummary]:
        return self.session_service.list_documents(session_id)

    def add_document(
        self,
        session_id: str,
        *,
        file_name: str,
        content_type: str | None,
        file_object: BinaryIO,
    ) -> SessionDocumentSummary:
        state = self.session_service.get_session(session_id)
        safe_name = Path((file_name or "").replace("\\", "/")).name
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in self.config.allowed_file_types:
            raise UnsupportedFileTypeError(
                f"Unsupported file type. Allowed: {', '.join(self.config.allowed_file_types)}"
            )
        content = file_object.read(self.config.max_upload_size + 1)
        if len(content) > self.config.max_upload_size:
            raise FileTooLargeError(
                f"File exceeds {self.config.max_upload_size} byte limit."
            )
        document_id = str(uuid4())
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="qa-upload-", suffix=suffix, delete=False
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            uploaded = UploadedDocument.from_path(
                temporary_path,
                document_id=document_id,
                mime_type=content_type,
                metadata={"original_file_name": safe_name},
            )
            uploaded.file_name = safe_name
            pipeline = self.pipeline_factory(state.document_store)
            pipeline.ingest_files([uploaded])
        except UnsupportedFileTypeError:
            raise
        except ParserError as exc:
            raise DocumentParsingError("Document could not be parsed.") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self._document_summary(session_id, document_id)

    def answer_question(
        self,
        session_id: str,
        question: str,
        *,
        case_id: str | None = None,
        search_scope: PathSearchScope = "same_case",
        selected_case_ids: list[str] | None = None,
    ) -> AnswerResult:
        normalized = question.strip()
        if not normalized or len(normalized) > self.config.question_max_chars:
            raise InvalidQuestionError(
                f"Question must contain 1..{self.config.question_max_chars} characters."
            )
        state = self.session_service.get_session(session_id)
        normalized_case_id = case_id.strip() if case_id else None
        return self.pipeline_factory(state.document_store).answer(
            normalized,
            case_id=normalized_case_id,
            search_scope=search_scope,
            selected_case_ids=selected_case_ids,
        )

    def _document_summary(
        self, session_id: str, document_id: str
    ) -> SessionDocumentSummary:
        return next(
            item
            for item in self.session_service.list_documents(session_id)
            if item.document_id == document_id
        )

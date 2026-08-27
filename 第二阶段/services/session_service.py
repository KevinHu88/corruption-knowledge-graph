"""内存 Session 生命周期与文档隔离管理。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from 第二阶段.exceptions import SessionNotFoundError
from 第二阶段.schemas.models import UploadedDocument
from 第二阶段.storage.session_document_store import SessionDocumentStore


@dataclass(slots=True)
class SessionState:
    session_id: str
    created_at: datetime
    document_store: SessionDocumentStore


@dataclass(frozen=True, slots=True)
class SessionDocumentSummary:
    document_id: str
    file_name: str
    file_type: str
    chunk_count: int
    status: str = "ready"


class SessionService:
    """为每个 Session 创建独立 SessionDocumentStore。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = RLock()

    def create_session(self) -> SessionState:
        with self._lock:
            session_id = str(uuid4())
            state = SessionState(
                session_id=session_id,
                created_at=datetime.now(timezone.utc),
                document_store=SessionDocumentStore(session_id),
            )
            self._sessions[session_id] = state
            return state

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError("Session not found.")
        return state

    def delete_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            raise SessionNotFoundError("Session not found.")
        state.document_store.clear()
        return state

    def add_document(
        self, session_id: str, document: UploadedDocument
    ) -> None:
        self.get_session(session_id).document_store.add_document(document)

    def list_documents(self, session_id: str) -> list[SessionDocumentSummary]:
        store = self.get_session(session_id).document_store
        summaries = []
        for document in store.get_documents():
            file_type = str(
                document.metadata.get("file_type")
                or document.file_name.rsplit(".", 1)[-1].lower()
            )
            summaries.append(
                SessionDocumentSummary(
                    document_id=document.document_id,
                    file_name=document.file_name,
                    file_type=file_type,
                    chunk_count=len(store.get_chunks(document.document_id)),
                )
            )
        return sorted(summaries, key=lambda item: item.document_id)


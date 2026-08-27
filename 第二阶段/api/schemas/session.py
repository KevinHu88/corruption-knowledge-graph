"""Session API schema。"""

from datetime import datetime

from pydantic import BaseModel

from 第二阶段.services.session_service import SessionState


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime

    @classmethod
    def from_state(cls, state: SessionState) -> "SessionResponse":
        return cls(session_id=state.session_id, created_at=state.created_at)


class DeleteSessionResponse(BaseModel):
    deleted: bool = True
    session_id: str


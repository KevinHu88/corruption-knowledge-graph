"""多个 Endpoint 共用的 API schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from 第二阶段.schemas.models import Evidence


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    detail: str
    entity_name: str | None = None
    candidate_case_ids: list[str] | None = None


class EvidenceResponse(BaseModel):
    id: str
    source_type: Literal["document", "graph"]
    content: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "EvidenceResponse":
        metadata = dict(evidence.metadata)
        if evidence.source_type == "document":
            metadata.pop("file_path", None)
        return cls(
            id=evidence.id,
            source_type=evidence.source_type,
            content=evidence.content,
            score=evidence.score,
            source=evidence.source,
            metadata=metadata,
        )


class SourceResponse(BaseModel):
    source_type: Literal["document", "graph"]
    name: str

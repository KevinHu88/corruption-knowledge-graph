"""知识问答请求与响应 schema。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from 第二阶段.api.schemas.common import EvidenceResponse, SourceResponse
from 第二阶段.schemas.models import AnswerResult


class QuestionRequest(BaseModel):
    question: str
    case_id: str | None = None
    search_scope: Literal["same_case", "selected_cases", "all_cases"] = (
        "same_case"
    )
    selected_case_ids: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("case_id")
    @classmethod
    def normalize_case_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("selected_case_ids")
    @classmethod
    def normalize_selected_case_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_search_scope(self) -> "QuestionRequest":
        if self.search_scope != "same_case" and not self.case_id:
            raise ValueError("跨案件相似检索必须提供锚点 case_id")
        if self.search_scope == "selected_cases" and not self.selected_case_ids:
            raise ValueError("selected_cases 必须提供 selected_case_ids")
        if self.search_scope != "selected_cases" and self.selected_case_ids:
            raise ValueError("selected_case_ids 仅适用于 selected_cases")
        return self


class QuestionResponse(BaseModel):
    answer: str
    route: Literal["DOCUMENT", "GRAPH", "HYBRID"]
    evidence: list[EvidenceResponse]
    sources: list[SourceResponse]

    @classmethod
    def from_result(cls, result: AnswerResult) -> "QuestionResponse":
        evidence = [
            EvidenceResponse.from_evidence(item)
            for item in result.retrieval.evidence
        ]
        sources: list[SourceResponse] = []
        seen: set[tuple[str, str]] = set()
        for item in result.retrieval.evidence:
            name = item.source or (
                "Neo4j" if item.source_type == "graph" else "uploaded document"
            )
            key = (item.source_type, name)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceResponse(source_type=item.source_type, name=name)
                )
        return cls(
            answer=result.answer,
            route=result.query_plan.route,
            evidence=evidence,
            sources=sources,
        )

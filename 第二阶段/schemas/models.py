"""知识问答各层共享的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Route = Literal["DOCUMENT", "GRAPH", "HYBRID"]
SourceType = Literal["document", "graph"]
PathSearchScope = Literal["same_case", "selected_cases", "all_cases"]


@dataclass(slots=True)
class UploadedDocument:
    """当前会话中用户上传的原始文件。"""

    document_id: str
    file_path: str
    file_name: str
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        file_path: str | Path,
        *,
        document_id: str,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UploadedDocument":
        path = Path(file_path)
        return cls(
            document_id=document_id,
            file_path=str(path),
            file_name=path.name,
            mime_type=mime_type,
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class ParsedDocument:
    """不同文件解析器的统一输出。"""

    document_id: str
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """可检索的文档片段。"""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryPlan:
    """问题的检索路由决策。"""

    route: Route
    question: str
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Evidence:
    """文档与图谱检索共用的证据模型。"""

    id: str
    source_type: SourceType
    content: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """一次问答检索的结构化结果。"""

    query: str
    route: Route
    evidence: list[Evidence] = field(default_factory=list)
    document_evidence: list[Evidence] = field(default_factory=list)
    graph_evidence: list[Evidence] = field(default_factory=list)


@dataclass(slots=True)
class AnswerResult:
    """知识问答流水线的最终输出。"""

    question: str
    answer: str
    query_plan: QueryPlan
    retrieval: RetrievalResult
    context: str
    prompt: str
    uploaded_document_ids: list[str] = field(default_factory=list)

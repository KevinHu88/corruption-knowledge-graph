"""第二阶段的轻量配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FIRST_STAGE_DIR = PROJECT_ROOT / "第一阶段"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须为 true 或 false")


@dataclass(frozen=True, slots=True)
class QAConfig:
    """可由环境变量覆盖的问答参数。"""

    chunk_size: int = 800
    chunk_overlap: int = 100
    document_top_k: int = 5
    retrieval_mode: str = "hybrid"
    embedding_provider: str = "hashing"
    embedding_dimensions: int = 384
    vector_candidate_multiplier: int = 4
    vector_min_score: float = 0.10
    rerank_bm25_weight: float = 0.45
    rerank_vector_weight: float = 0.45
    rerank_coverage_weight: float = 0.10
    vector_failure_fallback: bool = True
    graph_top_k: int = 10
    graph_path_max_hops: int = 3
    graph_path_candidate_limit: int = 100
    graph_path_similarity_threshold: float = 0.55
    fusion_limit: int = 12
    max_context_chars: int = 12000
    app_name: str = "Knowledge QA API"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    api_mode: str = "mock"
    max_upload_size: int = 20 * 1024 * 1024
    allowed_file_types: tuple[str, ...] = (".txt", ".pdf", ".docx")
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    )
    question_max_chars: int = 4000

    @classmethod
    def from_env(cls) -> "QAConfig":
        return cls(
            chunk_size=int(os.getenv("QA_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("QA_CHUNK_OVERLAP", "100")),
            document_top_k=int(os.getenv("QA_DOCUMENT_TOP_K", "5")),
            retrieval_mode=os.getenv("QA_RETRIEVAL_MODE", "hybrid").strip().lower(),
            embedding_provider=os.getenv(
                "QA_EMBEDDING_PROVIDER", "hashing"
            ).strip().lower(),
            embedding_dimensions=int(os.getenv("QA_EMBEDDING_DIMENSIONS", "384")),
            vector_candidate_multiplier=int(
                os.getenv("QA_VECTOR_CANDIDATE_MULTIPLIER", "4")
            ),
            vector_min_score=float(os.getenv("QA_VECTOR_MIN_SCORE", "0.10")),
            rerank_bm25_weight=float(os.getenv("QA_RERANK_BM25_WEIGHT", "0.45")),
            rerank_vector_weight=float(
                os.getenv("QA_RERANK_VECTOR_WEIGHT", "0.45")
            ),
            rerank_coverage_weight=float(
                os.getenv("QA_RERANK_COVERAGE_WEIGHT", "0.10")
            ),
            vector_failure_fallback=_env_bool("QA_VECTOR_FAILURE_FALLBACK", True),
            graph_top_k=int(os.getenv("QA_GRAPH_TOP_K", "10")),
            graph_path_max_hops=int(
                os.getenv("QA_GRAPH_PATH_MAX_HOPS", "3")
            ),
            graph_path_candidate_limit=int(
                os.getenv("QA_GRAPH_PATH_CANDIDATE_LIMIT", "100")
            ),
            graph_path_similarity_threshold=float(
                os.getenv("QA_GRAPH_PATH_SIMILARITY_THRESHOLD", "0.55")
            ),
            fusion_limit=int(os.getenv("QA_FUSION_LIMIT", "12")),
            max_context_chars=int(os.getenv("QA_MAX_CONTEXT_CHARS", "12000")),
            app_name=os.getenv("QA_APP_NAME", "Knowledge QA API").strip(),
            app_env=os.getenv("QA_APP_ENV", "development").strip(),
            host=os.getenv("QA_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("QA_PORT", "8000")),
            api_mode=os.getenv("QA_API_MODE", "mock").strip().lower(),
            max_upload_size=int(
                os.getenv("QA_MAX_UPLOAD_SIZE", str(20 * 1024 * 1024))
            ),
            allowed_file_types=tuple(
                item.strip().lower()
                for item in os.getenv(
                    "QA_ALLOWED_FILE_TYPES", ".txt,.pdf,.docx"
                ).split(",")
                if item.strip()
            ),
            cors_origins=tuple(
                item.strip()
                for item in os.getenv(
                    "QA_CORS_ORIGINS",
                    (
                        "http://localhost:3000,http://localhost:5173,"
                        "http://127.0.0.1:3000,http://127.0.0.1:5173"
                    ),
                ).split(",")
                if item.strip()
            ),
            question_max_chars=int(
                os.getenv("QA_QUESTION_MAX_CHARS", "4000")
            ),
        )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须位于 0..chunk_size-1")
        for name in ("document_top_k", "graph_top_k", "fusion_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if not 1 <= self.graph_path_max_hops <= 5:
            raise ValueError("graph_path_max_hops 必须位于 1..5")
        if self.graph_path_candidate_limit <= 0:
            raise ValueError("graph_path_candidate_limit 必须大于 0")
        if not 0.0 <= self.graph_path_similarity_threshold <= 1.0:
            raise ValueError("graph_path_similarity_threshold 必须位于 0..1")
        if self.retrieval_mode not in {"bm25", "hybrid"}:
            raise ValueError("retrieval_mode 必须为 bm25 或 hybrid")
        if self.embedding_provider not in {"hashing", "first_stage"}:
            raise ValueError("embedding_provider 必须为 hashing 或 first_stage")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions 必须大于 0")
        if self.vector_candidate_multiplier < 1:
            raise ValueError("vector_candidate_multiplier 必须大于等于 1")
        if not -1.0 <= self.vector_min_score <= 1.0:
            raise ValueError("vector_min_score 必须位于 -1..1")
        rerank_weights = (
            self.rerank_bm25_weight,
            self.rerank_vector_weight,
            self.rerank_coverage_weight,
        )
        if any(weight < 0 for weight in rerank_weights) or sum(rerank_weights) <= 0:
            raise ValueError("重排权重必须非负且总和大于 0")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars 必须大于 0")
        if not self.app_name or not self.host:
            raise ValueError("app_name 和 host 不能为空")
        if not 1 <= self.port <= 65535:
            raise ValueError("port 必须位于 1..65535")
        if self.api_mode not in {"mock", "production"}:
            raise ValueError("api_mode 必须为 mock 或 production")
        if self.max_upload_size <= 0:
            raise ValueError("max_upload_size 必须大于 0")
        supported = {".txt", ".pdf", ".docx"}
        if not self.allowed_file_types or not set(self.allowed_file_types) <= supported:
            raise ValueError("allowed_file_types 只能包含 .txt/.pdf/.docx")
        if self.question_max_chars <= 0:
            raise ValueError("question_max_chars 必须大于 0")
        if "*" in self.cors_origins:
            raise ValueError("cors_origins 不允许使用通配符 *")

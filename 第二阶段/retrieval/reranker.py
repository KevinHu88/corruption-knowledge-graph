"""文档候选的 BM25、向量与查询覆盖率混合重排。"""

from __future__ import annotations

from dataclasses import dataclass

from 第二阶段.retrieval.text_features import tokenize
from 第二阶段.schemas.models import Chunk


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    chunk: Chunk
    score: float
    bm25_score: float
    vector_score: float
    coverage_score: float


class HybridReranker:
    def __init__(
        self,
        *,
        bm25_weight: float = 0.45,
        vector_weight: float = 0.45,
        coverage_weight: float = 0.10,
    ) -> None:
        weights = (bm25_weight, vector_weight, coverage_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("重排权重必须非负且总和大于 0")
        total = sum(weights)
        self.bm25_weight = bm25_weight / total
        self.vector_weight = vector_weight / total
        self.coverage_weight = coverage_weight / total

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        bm25_scores: dict[str, float],
        vector_scores: dict[str, float],
        top_k: int,
    ) -> list[RerankedChunk]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        maximum_bm25 = max(bm25_scores.values(), default=0.0)
        maximum_vector = max(
            (max(0.0, value) for value in vector_scores.values()),
            default=0.0,
        )
        query_terms = set(tokenize(query))
        results: list[RerankedChunk] = []
        for chunk in chunks:
            raw_bm25 = bm25_scores.get(chunk.chunk_id, 0.0)
            raw_vector = max(0.0, vector_scores.get(chunk.chunk_id, 0.0))
            normalized_bm25 = raw_bm25 / maximum_bm25 if maximum_bm25 else 0.0
            normalized_vector = (
                raw_vector / maximum_vector if maximum_vector else 0.0
            )
            chunk_terms = set(tokenize(chunk.content))
            coverage = (
                len(query_terms & chunk_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            score = (
                self.bm25_weight * normalized_bm25
                + self.vector_weight * normalized_vector
                + self.coverage_weight * coverage
            )
            results.append(
                RerankedChunk(
                    chunk=chunk,
                    score=score,
                    bm25_score=raw_bm25,
                    vector_score=raw_vector,
                    coverage_score=coverage,
                )
            )
        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:top_k]

"""会话文档的 BM25、向量召回与混合重排。"""

from __future__ import annotations

import logging
import math
from collections import Counter

from 第二阶段.retrieval.embedding import EmbeddingClient, cosine_similarity
from 第二阶段.retrieval.reranker import HybridReranker, RerankedChunk
from 第二阶段.retrieval.text_features import tokenize
from 第二阶段.schemas.models import Chunk, Evidence
from 第二阶段.storage.session_document_store import SessionDocumentStore

logger = logging.getLogger(__name__)

# 保留旧名称，避免内部或下游代码因轻量分词函数重命名而中断。
_terms = tokenize


class DocumentRetriever:
    """BM25 与可选向量召回取并集，再执行混合重排。"""

    def __init__(
        self,
        store: SessionDocumentStore,
        *,
        embedding_client: EmbeddingClient | None = None,
        reranker: HybridReranker | None = None,
        candidate_multiplier: int = 4,
        vector_min_score: float = 0.10,
        vector_failure_fallback: bool = True,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier 必须大于等于 1")
        if not -1.0 <= vector_min_score <= 1.0:
            raise ValueError("vector_min_score 必须位于 -1..1")
        self.store = store
        self.embedding_client = embedding_client
        self.reranker = reranker or HybridReranker()
        self.candidate_multiplier = candidate_multiplier
        self.vector_min_score = vector_min_score
        self.vector_failure_fallback = vector_failure_fallback

    def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        chunks = self.store.get_chunks()
        query_terms = tokenize(query)
        if not chunks or not query_terms:
            return []

        bm25_scores = self._bm25_scores(query_terms, chunks)
        vector_scores: dict[str, float] = {}
        if self.embedding_client is not None:
            try:
                vector_scores = self._vector_scores(query, chunks)
            except Exception:
                if not self.vector_failure_fallback:
                    raise
                logger.warning(
                    "Vector retrieval failed; falling back to BM25",
                    exc_info=True,
                )

        candidate_limit = max(top_k, top_k * self.candidate_multiplier)
        candidate_ids = {
            chunk_id
            for chunk_id, _ in sorted(
                bm25_scores.items(), key=lambda item: (-item[1], item[0])
            )[:candidate_limit]
        }
        candidate_ids.update(
            chunk_id
            for chunk_id, score in sorted(
                vector_scores.items(), key=lambda item: (-item[1], item[0])
            )[:candidate_limit]
            if score >= self.vector_min_score
        )
        if not candidate_ids:
            return []

        candidates = [chunk for chunk in chunks if chunk.chunk_id in candidate_ids]
        reranked = self.reranker.rerank(
            query,
            candidates,
            bm25_scores=bm25_scores,
            vector_scores=vector_scores,
            top_k=top_k,
        )
        mode = "hybrid" if vector_scores else "bm25"
        return [self._to_evidence(item, mode=mode) for item in reranked]

    @staticmethod
    def _bm25_scores(
        query_terms: list[str], chunks: list[Chunk]
    ) -> dict[str, float]:
        tokenized = [tokenize(chunk.content) for chunk in chunks]
        document_frequency = Counter(
            term for terms in tokenized for term in set(terms)
        )
        average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
        scores: dict[str, float] = {}
        for chunk, terms in zip(chunks, tokenized, strict=True):
            frequencies = Counter(terms)
            score = 0.0
            for term in set(query_terms):
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(terms) / max(1.0, average_length)
                )
                score += idf * frequency * 2.5 / denominator
            if score > 0:
                scores[chunk.chunk_id] = score
        return scores

    def _vector_scores(
        self, query: str, chunks: list[Chunk]
    ) -> dict[str, float]:
        if self.embedding_client is None:
            return {}
        namespace = self.embedding_client.cache_namespace
        vectors: dict[str, list[float]] = {}
        missing: list[Chunk] = []
        for chunk in chunks:
            cached = self.store.get_chunk_vector(namespace, chunk)
            if cached is None:
                missing.append(chunk)
            else:
                vectors[chunk.chunk_id] = cached
        if missing:
            embedded = self.embedding_client.embed([chunk.content for chunk in missing])
            if len(embedded) != len(missing):
                raise RuntimeError("Embedding 客户端返回的向量数量不匹配")
            for chunk, vector in zip(missing, embedded, strict=True):
                vectors[chunk.chunk_id] = vector
                self.store.set_chunk_vector(namespace, chunk, vector)
        query_vectors = self.embedding_client.embed([query])
        if len(query_vectors) != 1:
            raise RuntimeError("Embedding 客户端未返回查询向量")
        query_vector = query_vectors[0]
        return {
            chunk.chunk_id: cosine_similarity(query_vector, vectors[chunk.chunk_id])
            for chunk in chunks
        }

    @staticmethod
    def _to_evidence(item: RerankedChunk, *, mode: str) -> Evidence:
        chunk = item.chunk
        source = str(chunk.metadata.get("file_name") or chunk.document_id)
        return Evidence(
            id=chunk.chunk_id,
            source_type="document",
            content=chunk.content,
            score=item.score,
            source=source,
            metadata={
                "document_id": chunk.document_id,
                **chunk.metadata,
                "retrieval": {
                    "mode": mode,
                    "bm25_score": item.bm25_score,
                    "vector_score": item.vector_score,
                    "coverage_score": item.coverage_score,
                    "rerank_score": item.score,
                },
            },
        )

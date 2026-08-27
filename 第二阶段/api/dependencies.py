"""应用级依赖装配；Router 不自行构造 Pipeline。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from 第二阶段.chunking.chunker import Chunker
from 第二阶段.config import QAConfig
from 第二阶段.context.context_builder import ContextBuilder
from 第二阶段.generation.llm_client import (
    FirstStageLLMClient,
    LLMClient,
    MockLLMClient,
)
from 第二阶段.generation.prompt_builder import PromptBuilder
from 第二阶段.graph.first_stage_adapter import FirstStageGraphAdapter
from 第二阶段.graph.graph_repository import GraphRepository
from 第二阶段.parsing.router import ParserRouter
from 第二阶段.pipeline.qa_pipeline import KnowledgeQAPipeline
from 第二阶段.retrieval.document_retriever import DocumentRetriever
from 第二阶段.retrieval.embedding import (
    EmbeddingClient,
    FirstStageEmbeddingClient,
    HashingEmbeddingClient,
)
from 第二阶段.retrieval.fusion import EvidenceFusion
from 第二阶段.retrieval.graph_retriever import GraphRetriever
from 第二阶段.retrieval.reranker import HybridReranker
from 第二阶段.routing.query_router import QueryRouter
from 第二阶段.services.qa_service import QAService
from 第二阶段.services.session_service import SessionService
from 第二阶段.storage.session_document_store import SessionDocumentStore


class MockGraphRepository:
    """默认 Mock API 使用的无外部依赖图谱。"""

    entity = {
        "entity_uid": "mock-person-zhang",
        "name": "张某",
        "entity_type": "PER",
        "case_id": "mock-case",
    }
    path_entities = [
        {
            "entity_uid": "mock-person-xie",
            "name": "谢晚林",
            "entity_type": "PER",
            "case_id": "mock-case",
        },
        {
            "entity_uid": "mock-money-19900",
            "name": "1.99万元",
            "entity_type": "MONEY",
            "case_id": "mock-case",
        },
        {
            "entity_uid": "mock-person-liu",
            "name": "刘某",
            "entity_type": "PER",
            "case_id": "mock-case",
        },
    ]
    path_claims = [
        {
            "claim_id": "mock-claim-receive-money",
            "relation_type": "收受金额",
            "status": "HUMAN_VERIFIED",
            "case_id": "mock-case",
            "doc_id": "mock-path-doc",
            "evidence_text": "谢晚林收受刘某支付的1.99万元。",
        },
        {
            "claim_id": "mock-claim-pay-money",
            "relation_type": "支付金额",
            "status": "HUMAN_VERIFIED",
            "case_id": "mock-case",
            "doc_id": "mock-path-doc",
            "evidence_text": "刘某向谢晚林支付1.99万元。",
        },
    ]

    @classmethod
    def _path_record(
        cls,
        entities: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        directions: list[str],
    ) -> dict[str, Any]:
        return {
            "path_entities": entities,
            "path_claims": claims,
            "directions": directions,
            "hop_count": len(claims),
        }

    def find_entities_in_text(
        self, text: str, limit: int = 10, *, case_id: str | None = None
    ):
        entities = [self.entity, *self.path_entities]
        return [
            {"entity": entity}
            for entity in entities
            if entity["name"] in text
            and (case_id is None or case_id == entity["case_id"])
        ][:limit]

    def find_entity_by_name(
        self, name: str, limit: int = 10, *, case_id: str | None = None
    ):
        entities = [self.entity, *self.path_entities]
        return [
            {"entity": entity}
            for entity in entities
            if entity["name"] == name
            and (case_id is None or case_id == entity["case_id"])
        ][:limit]

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        del limit
        if entity_uid != self.entity["entity_uid"] or (
            case_id is not None and case_id != self.entity["case_id"]
        ):
            return []
        return [
            {
                "claim": {
                    "claim_id": "mock-claim",
                    "relation_type": "请托",
                    "status": "HUMAN_VERIFIED",
                    "case_id": "mock-case",
                    "doc_id": "mock-doc",
                    "text_id": "mock-text",
                    "evidence_text": "张某请托李某协助办理项目审批。",
                },
                "head": self.entity,
                "tail": {
                    "entity_uid": "mock-person-li",
                    "name": "李某",
                    "entity_type": "PER",
                    "case_id": "mock-case",
                },
                "evidence": [
                    {
                        "text": "张某请托李某协助办理项目审批。",
                        "text_id": "mock-text",
                    }
                ],
                "document": {"title": "Mock 案件材料", "doc_id": "mock-doc"},
                "case": {"case_id": "mock-case"},
            }
        ]

    def find_simple_paths(
        self,
        start_uid: str,
        end_uid: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
        max_hops: int = 3,
    ):
        del max_hops
        endpoints = {self.path_entities[0]["entity_uid"], self.path_entities[-1]["entity_uid"]}
        if (
            {start_uid, end_uid} != endpoints
            or (case_id is not None and case_id != "mock-case")
        ):
            return []
        if start_uid == self.path_entities[0]["entity_uid"]:
            entities = self.path_entities
            claims = self.path_claims
            directions = ["forward", "reverse"]
        else:
            entities = list(reversed(self.path_entities))
            claims = list(reversed(self.path_claims))
            directions = ["forward", "reverse"]
        return [self._path_record(entities, claims, directions)][:limit]

    def find_path_candidates(
        self,
        relation_types: list[str],
        *,
        start_entity_type: str | None = None,
        end_entity_type: str | None = None,
        exclude_claim_ids: list[str] | None = None,
        case_id: str | None = None,
        case_ids: list[str] | None = None,
        max_hops: int = 3,
        limit: int = 100,
    ):
        del start_entity_type, end_entity_type, case_id, max_hops
        if not {"收受金额", "支付金额"}.intersection(relation_types):
            return []
        excluded = set(exclude_claim_ids or [])
        candidates = [
            self._similar_candidate("mock-case", "same", "周某", "3.2万元", "孙某"),
            self._similar_candidate("mock-case-2", "cross", "赵某", "2万元", "钱某"),
        ]
        allowed_cases = set(case_ids or [])
        return [
            candidate
            for candidate in candidates
            if not allowed_cases
            or candidate["path_claims"][0]["case_id"] in allowed_cases
            if not excluded.intersection(
                claim["claim_id"] for claim in candidate["path_claims"]
            )
        ][:limit]

    @classmethod
    def _similar_candidate(
        cls,
        case_id: str,
        prefix: str,
        recipient: str,
        money: str,
        payer: str,
    ) -> dict[str, Any]:
        entities = [
            {
                "entity_uid": f"mock-{prefix}-recipient",
                "name": recipient,
                "entity_type": "PER",
                "case_id": case_id,
            },
            {
                "entity_uid": f"mock-{prefix}-money",
                "name": money,
                "entity_type": "MONEY",
                "case_id": case_id,
            },
            {
                "entity_uid": f"mock-{prefix}-payer",
                "name": payer,
                "entity_type": "PER",
                "case_id": case_id,
            },
        ]
        claims = [
            {
                "claim_id": f"mock-{prefix}-receive",
                "relation_type": "收受金额",
                "status": "MODEL_PREDICTED",
                "case_id": case_id,
                "doc_id": f"mock-{prefix}-doc",
                "evidence_text": f"{recipient}收受{payer}支付的{money}。",
            },
            {
                "claim_id": f"mock-{prefix}-pay",
                "relation_type": "支付金额",
                "status": "MODEL_PREDICTED",
                "case_id": case_id,
                "doc_id": f"mock-{prefix}-doc",
                "evidence_text": f"{payer}向{recipient}支付{money}。",
            },
        ]
        return cls._path_record(entities, claims, ["forward", "reverse"])


@dataclass(slots=True)
class ApplicationContainer:
    config: QAConfig
    session_service: SessionService
    qa_service: QAService
    closeables: list[Any] = field(default_factory=list)

    def close(self) -> None:
        for resource in reversed(self.closeables):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def build_container(
    config: QAConfig | None = None,
    *,
    graph_retriever: Any | None = None,
    llm_client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> ApplicationContainer:
    settings = config or QAConfig.from_env()
    closeables: list[Any] = []
    if graph_retriever is None:
        if settings.api_mode == "production":
            adapter = FirstStageGraphAdapter()
            graph_retriever = GraphRetriever(
                GraphRepository(adapter),
                max_path_hops=settings.graph_path_max_hops,
                path_candidate_limit=settings.graph_path_candidate_limit,
                path_similarity_threshold=(
                    settings.graph_path_similarity_threshold
                ),
            )
            closeables.append(adapter)
        else:
            graph_retriever = GraphRetriever(
                MockGraphRepository(),
                max_path_hops=settings.graph_path_max_hops,
                path_candidate_limit=settings.graph_path_candidate_limit,
                path_similarity_threshold=(
                    settings.graph_path_similarity_threshold
                ),
            )
    if llm_client is None:
        if settings.api_mode == "production":
            llm_client = FirstStageLLMClient()
            closeables.append(llm_client)
        else:
            llm_client = MockLLMClient(
                "Mock answer based on document and graph evidence."
            )

    if settings.retrieval_mode == "hybrid" and embedding_client is None:
        if settings.embedding_provider == "first_stage":
            embedding_client = FirstStageEmbeddingClient()
            closeables.append(embedding_client)
        else:
            embedding_client = HashingEmbeddingClient(
                settings.embedding_dimensions
            )

    parser_router = ParserRouter()
    chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
    query_router = QueryRouter()
    fusion = EvidenceFusion()
    context_builder = ContextBuilder(settings.max_context_chars)
    prompt_builder = PromptBuilder()
    reranker = HybridReranker(
        bm25_weight=settings.rerank_bm25_weight,
        vector_weight=settings.rerank_vector_weight,
        coverage_weight=settings.rerank_coverage_weight,
    )

    def pipeline_factory(store: SessionDocumentStore) -> KnowledgeQAPipeline:
        return KnowledgeQAPipeline(
            parser_router=parser_router,
            chunker=chunker,
            document_store=store,
            query_router=query_router,
            document_retriever=DocumentRetriever(
                store,
                embedding_client=(
                    embedding_client
                    if settings.retrieval_mode == "hybrid"
                    else None
                ),
                reranker=reranker,
                candidate_multiplier=settings.vector_candidate_multiplier,
                vector_min_score=settings.vector_min_score,
                vector_failure_fallback=settings.vector_failure_fallback,
            ),
            graph_retriever=graph_retriever,
            fusion=fusion,
            context_builder=context_builder,
            prompt_builder=prompt_builder,
            llm_client=llm_client,
            document_top_k=settings.document_top_k,
            graph_top_k=settings.graph_top_k,
            fusion_limit=settings.fusion_limit,
        )

    session_service = SessionService()
    qa_service = QAService(session_service, pipeline_factory, settings)
    return ApplicationContainer(
        config=settings,
        session_service=session_service,
        qa_service=qa_service,
        closeables=closeables,
    )


def get_qa_service(request: Request) -> QAService:
    return request.app.state.container.qa_service

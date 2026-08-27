"""编排上传、解析、检索、上下文与生成，不承载各层具体算法。"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from collections.abc import Sequence

from 第二阶段.chunking.chunker import Chunker
from 第二阶段.context.context_builder import ContextBuilder
from 第二阶段.exceptions import (
    AmbiguousEntityError,
    GraphRetrievalError,
    LLMGenerationError,
)
from 第二阶段.generation.llm_client import LLMClient
from 第二阶段.generation.prompt_builder import PromptBuilder
from 第二阶段.parsing.router import ParserRouter
from 第二阶段.retrieval.document_retriever import DocumentRetriever
from 第二阶段.retrieval.fusion import EvidenceFusion
from 第二阶段.retrieval.graph_retriever import GraphRetriever
from 第二阶段.routing.query_router import QueryRouter
from 第二阶段.schemas.models import (
    AnswerResult,
    PathSearchScope,
    RetrievalResult,
    UploadedDocument,
)
from 第二阶段.storage.session_document_store import SessionDocumentStore


class PipelineConfigurationError(RuntimeError):
    """路由请求了尚未配置的检索来源。"""


class KnowledgeQAPipeline:
    def __init__(
        self,
        *,
        parser_router: ParserRouter,
        chunker: Chunker,
        document_store: SessionDocumentStore,
        query_router: QueryRouter,
        document_retriever: DocumentRetriever,
        graph_retriever: GraphRetriever | None,
        fusion: EvidenceFusion,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        document_top_k: int = 5,
        graph_top_k: int = 10,
        fusion_limit: int = 12,
    ) -> None:
        self.parser_router = parser_router
        self.chunker = chunker
        self.document_store = document_store
        self.query_router = query_router
        self.document_retriever = document_retriever
        self.graph_retriever = graph_retriever
        self.fusion = fusion
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.document_top_k = document_top_k
        self.graph_top_k = graph_top_k
        self.fusion_limit = fusion_limit

    def answer(
        self,
        question: str,
        uploaded_files: Sequence[str | Path | UploadedDocument] | None = None,
        *,
        case_id: str | None = None,
        search_scope: PathSearchScope = "same_case",
        selected_case_ids: list[str] | None = None,
    ) -> AnswerResult:
        uploaded_ids = self.ingest_files(uploaded_files or [])
        has_documents = bool(self.document_store.get_chunks())
        plan = self.query_router.route(question, has_documents)

        document_evidence = []
        graph_evidence = []
        if plan.route in {"DOCUMENT", "HYBRID"}:
            document_evidence = self.document_retriever.retrieve(
                question, top_k=self.document_top_k
            )
        if plan.route in {"GRAPH", "HYBRID"}:
            if self.graph_retriever is None:
                raise PipelineConfigurationError(
                    f"路由结果为 {plan.route}，但未配置 GraphRetriever"
                )
            try:
                graph_evidence = self.graph_retriever.retrieve(
                    question,
                    top_k=self.graph_top_k,
                    case_id=case_id,
                    search_scope=search_scope,
                    selected_case_ids=selected_case_ids,
                )
            except (AmbiguousEntityError, GraphRetrievalError):
                raise
            except Exception as exc:
                raise GraphRetrievalError(
                    "Knowledge graph service is unavailable."
                ) from exc

        merged = self.fusion.fuse(
            document_evidence,
            graph_evidence,
            limit=self.fusion_limit,
        )
        context = self.context_builder.build(merged)
        prompt = self.prompt_builder.build(question, context)
        try:
            answer = self.llm_client.generate(prompt)
        except LLMGenerationError:
            raise
        except Exception as exc:
            raise LLMGenerationError(
                "LLM generation service is unavailable."
            ) from exc
        retrieval = RetrievalResult(
            query=question,
            route=plan.route,
            evidence=merged,
            document_evidence=document_evidence,
            graph_evidence=graph_evidence,
        )
        return AnswerResult(
            question=question,
            answer=answer,
            query_plan=plan,
            retrieval=retrieval,
            context=context,
            prompt=prompt,
            uploaded_document_ids=uploaded_ids,
        )

    def ingest_files(
        self, uploaded_files: Sequence[str | Path | UploadedDocument]
    ) -> list[str]:
        uploaded_ids: list[str] = []
        for item in uploaded_files:
            if isinstance(item, UploadedDocument):
                uploaded = item
                parsed = self.parser_router.parse(item.file_path, item.mime_type)
                parsed.document_id = item.document_id
                parsed.file_name = item.file_name
            else:
                path = Path(item)
                mime_type = mimetypes.guess_type(path.name)[0]
                parsed = self.parser_router.parse(path, mime_type)
                uploaded = UploadedDocument.from_path(
                    path,
                    document_id=parsed.document_id,
                    mime_type=mime_type,
                )
            chunks = self.chunker.chunk(parsed)
            uploaded.metadata["file_type"] = parsed.file_type
            uploaded.metadata["chunk_count"] = len(chunks)
            self.document_store.add_document(uploaded)
            self.document_store.add_parsed_document(parsed)
            self.document_store.add_chunks(chunks)
            uploaded_ids.append(uploaded.document_id)
        return uploaded_ids

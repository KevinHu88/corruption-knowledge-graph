"""第二阶段 Mock 模式最小可运行示例。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from 第二阶段.chunking.chunker import Chunker
from 第二阶段.config import QAConfig
from 第二阶段.context.context_builder import ContextBuilder
from 第二阶段.generation.llm_client import MockLLMClient
from 第二阶段.generation.prompt_builder import PromptBuilder
from 第二阶段.parsing.router import ParserRouter
from 第二阶段.pipeline.qa_pipeline import KnowledgeQAPipeline
from 第二阶段.retrieval.document_retriever import DocumentRetriever
from 第二阶段.retrieval.fusion import EvidenceFusion
from 第二阶段.retrieval.graph_retriever import GraphRetriever
from 第二阶段.routing.query_router import QueryRouter
from 第二阶段.schemas.models import Chunk
from 第二阶段.storage.session_document_store import SessionDocumentStore


class DemoGraphRepository:
    """仅用于本地演示，不连接真实 Neo4j。"""

    entity = {
        "entity_uid": "person-zhang",
        "name": "张某",
        "normalized_name": "张某",
        "entity_type": "PER",
        "case_id": "case-demo",
    }

    def find_entities_in_text(
        self, text: str, limit: int = 10, *, case_id: str | None = None
    ):
        del limit
        matches_case = case_id is None or case_id == self.entity["case_id"]
        return [{"entity": self.entity}] if "张某" in text and matches_case else []

    def find_entity_by_name(
        self, name: str, limit: int = 10, *, case_id: str | None = None
    ):
        del limit
        matches_case = case_id is None or case_id == self.entity["case_id"]
        return [{"entity": self.entity}] if name == "张某" and matches_case else []

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        del limit
        if entity_uid != "person-zhang" or (
            case_id is not None and case_id != self.entity["case_id"]
        ):
            return []
        return [
            {
                "claim": {
                    "claim_id": "claim-demo",
                    "relation_type": "请托",
                    "status": "HUMAN_VERIFIED",
                    "case_id": "case-demo",
                    "doc_id": "doc-demo",
                    "evidence_text": "张某请托李某协助办理项目审批。",
                },
                "head": self.entity,
                "tail": {
                    "entity_uid": "person-li",
                    "name": "李某",
                    "entity_type": "PER",
                    "case_id": "case-demo",
                },
                "evidence": [{"text": "张某请托李某协助办理项目审批。"}],
                "document": {"title": "演示案件材料", "doc_id": "doc-demo"},
                "case": {"case_id": "case-demo"},
            }
        ]


def build_mock_pipeline() -> KnowledgeQAPipeline:
    config = QAConfig.from_env()
    store = SessionDocumentStore(session_id="demo")
    store.add_chunks(
        [
            Chunk(
                chunk_id="document-demo-1",
                document_id="uploaded-demo",
                content="上传材料记载：张某请托李某协助办理项目审批。",
                metadata={"file_name": "演示材料.txt", "chunk_index": 0},
            )
        ]
    )
    return KnowledgeQAPipeline(
        parser_router=ParserRouter(),
        chunker=Chunker(config.chunk_size, config.chunk_overlap),
        document_store=store,
        query_router=QueryRouter(),
        document_retriever=DocumentRetriever(store),
        graph_retriever=GraphRetriever(DemoGraphRepository()),
        fusion=EvidenceFusion(),
        context_builder=ContextBuilder(config.max_context_chars),
        prompt_builder=PromptBuilder(),
        llm_client=MockLLMClient("Mock 回答：张某请托李某协助办理项目审批。"),
        document_top_k=config.document_top_k,
        graph_top_k=config.graph_top_k,
        fusion_limit=config.fusion_limit,
    )


def main() -> None:
    pipeline = build_mock_pipeline()
    result = pipeline.answer("上传材料中的证据显示张某与李某存在什么关系？")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from 第二阶段.chunking.chunker import Chunker
from 第二阶段.context.context_builder import ContextBuilder
from 第二阶段.generation.llm_client import MockLLMClient
from 第二阶段.generation.prompt_builder import PromptBuilder
from 第二阶段.parsing.router import ParserRouter
from 第二阶段.pipeline.qa_pipeline import KnowledgeQAPipeline
from 第二阶段.retrieval.document_retriever import DocumentRetriever
from 第二阶段.retrieval.fusion import EvidenceFusion
from 第二阶段.retrieval.graph_retriever import GraphRetriever
from 第二阶段.routing.query_router import QueryRouter
from 第二阶段.storage.session_document_store import SessionDocumentStore

from 第二阶段.tests.test_graph_retriever import MockGraphRepository


def build_pipeline(store, mock_llm):
    return KnowledgeQAPipeline(
        parser_router=ParserRouter(),
        chunker=Chunker(chunk_size=30, chunk_overlap=5),
        document_store=store,
        query_router=QueryRouter(),
        document_retriever=DocumentRetriever(store),
        graph_retriever=GraphRetriever(MockGraphRepository()),
        fusion=EvidenceFusion(),
        context_builder=ContextBuilder(max_chars=2000),
        prompt_builder=PromptBuilder(),
        llm_client=mock_llm,
    )


def test_complete_hybrid_pipeline_with_mock_llm(tmp_path) -> None:
    upload = tmp_path / "material.txt"
    upload.write_text("原始材料显示张某请托李某办理项目审批。", encoding="utf-8")
    store = SessionDocumentStore("test-session")
    mock_llm = MockLLMClient("证据显示二人存在请托关系。")
    pipeline = build_pipeline(store, mock_llm)
    result = pipeline.answer(
        "张某与李某有什么关系，原文证据是什么？",
        uploaded_files=[upload],
    )
    assert result.query_plan.route == "HYBRID"
    assert result.retrieval.document_evidence
    assert result.retrieval.graph_evidence
    assert {item.source_type for item in result.retrieval.evidence} == {"document", "graph"}
    assert "Evidence ID" in result.context
    assert "不得虚构" in result.prompt
    assert result.answer == "证据显示二人存在请托关系。"
    assert len(mock_llm.prompts) == 1


def test_complete_document_only_pipeline(tmp_path) -> None:
    upload = tmp_path / "report.txt"
    upload.write_text("报告说明该项目由李某负责审批。", encoding="utf-8")
    store = SessionDocumentStore("document-session")
    pipeline = build_pipeline(store, MockLLMClient("项目由李某负责审批。"))
    result = pipeline.answer("上传报告中如何描述该项目？", [upload])
    assert result.query_plan.route == "DOCUMENT"
    assert result.retrieval.document_evidence
    assert result.retrieval.graph_evidence == []
    assert result.answer == "项目由李某负责审批。"


def test_complete_graph_only_pipeline() -> None:
    store = SessionDocumentStore("graph-session")
    pipeline = build_pipeline(store, MockLLMClient("张某与李某存在请托关系。"))
    result = pipeline.answer("张某与哪些人存在关系？")
    assert result.query_plan.route == "GRAPH"
    assert result.retrieval.document_evidence == []
    assert result.retrieval.graph_evidence
    assert result.answer == "张某与李某存在请托关系。"


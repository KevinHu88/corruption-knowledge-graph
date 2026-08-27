from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

from flows.annotation_flow import AnnotationFlowResult
from flows.retrieval_flow import RetrievalFlowResult, retrieval_flow
from models import RawDocument
from src.services.retrieval_service import RetrievalBatch, RetrievalService
from src.services.tavily_service import TavilyRequestError


class FakeTavily:
    def search_source(self, source, *, today=None):
        return [
            {
                "title": "Case notice",
                "url": "https://example.test/case/1",
                "canonical_url": "https://example.test/case/1",
                "score": 0.9,
                "published_at": "2026-07-28T00:00:00Z",
            }
        ]

    def extract(self, urls):
        return [
            {
                "url": urls[0],
                "canonical_url": urls[0],
                "raw_content": "full case content",
            }
        ]


def test_retrieval_service_maps_external_results_to_stable_documents() -> None:
    result = RetrievalService(FakeTavily()).retrieve_source(
        {"source_id": "court", "keywords": ["case"]}
    )

    assert result.searched_count == 1
    assert result.extracted_count == 1
    assert result.documents[0].doc_id.startswith("retrieval-")
    assert result.documents[0].raw_text == "full case content"
    assert result.documents[0].content_sha256
    assert result.documents[0].metadata["retrieval_backend"] == "tavily"


def test_retrieval_service_records_allowed_extract_failure() -> None:
    service = FakeTavily()
    service.extract = lambda urls: (_ for _ in ()).throw(
        TavilyRequestError("temporary")
    )

    result = RetrievalService(service).retrieve_source(
        {"source_id": "court", "keywords": ["case"]},
        continue_on_extract_error=True,
    )

    assert result.documents == []
    assert result.skipped_count == 1
    assert "extract failed" in result.errors[0]


def test_retrieval_flow_aggregates_domain_batches(monkeypatch) -> None:
    retrieval_module = importlib.import_module("flows.retrieval_flow")
    document = RawDocument(
        doc_id="doc-1", source_id="court", raw_text="content"
    )

    def fake_task(source, **kwargs):
        return RetrievalBatch(
            source_id=source["source_id"],
            documents=[document],
            searched_count=2,
        )

    monkeypatch.setattr(retrieval_module, "retrieve_source_task", fake_task)
    result = retrieval_flow.fn(
        sources=[{"source_id": "court", "enabled": True}]
    )

    assert result.raw_documents == [document]
    assert result.source_ids == ["court"]
    assert result.searched_count == 2


def test_ingestion_passes_retrieved_documents_to_annotation(monkeypatch) -> None:
    ingestion_module = importlib.import_module("flows.ingestion_flow")
    document = RawDocument(
        doc_id="doc-1", source_id="court", raw_text="content"
    )
    captured = {}

    def fake_retrieval_flow(**kwargs):
        return RetrievalFlowResult(
            raw_documents=[document], source_ids=["court"]
        )

    async def fake_annotation_flow(**kwargs):
        captured.update(kwargs)
        return AnnotationFlowResult()

    monkeypatch.setattr(ingestion_module, "retrieval_flow", fake_retrieval_flow)
    monkeypatch.setattr(ingestion_module, "annotation_flow", fake_annotation_flow)
    monkeypatch.setattr(
        ingestion_module, "flow_run", SimpleNamespace(id="flow-1")
    )

    result = asyncio.run(
        ingestion_module.ingestion_flow.fn(
            run_retrieval=True,
            run_annotation=True,
            run_dataset_build=False,
        )
    )

    assert captured["raw_documents"] == [document]
    assert result.retrieval_result["source_ids"] == ["court"]

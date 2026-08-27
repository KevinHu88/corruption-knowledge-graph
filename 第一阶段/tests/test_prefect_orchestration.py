from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

import task.annotation_tasks as annotation_module
import task.graph_tasks as graph_module
from flows import (
    annotation_flow,
    dataset_build_flow,
    graph_ingestion_flow,
    ingestion_flow,
    review_sync_flow,
    retrieval_flow,
    training_flow,
)
from models import ModelExtractionResult
from src.services.neo4j_service import (
    Neo4jConnectionError,
    Neo4jValidationError,
)
from src.services.llm_service import (
    LLMConnectionError,
    LLMResponseError,
)
from task import (
    annotation_task,
    dataset_build_task,
    graph_ingestion_task,
    llm_preannotation_task,
    review_sync_task,
    retrieve_source_task,
    training_task,
)

ingestion_module = importlib.import_module("flows.ingestion_flow")


class StubLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def test_prefect_names_and_retry_policies_are_stage_specific() -> None:
    assert annotation_flow.name == "annotation-flow"
    assert review_sync_flow.name == "review-sync-flow"
    assert retrieval_flow.name == "retrieval-flow"
    assert dataset_build_flow.name == "dataset-build-flow"
    assert training_flow.name == "training-flow"
    assert graph_ingestion_flow.name == "graph-ingestion-flow"
    assert ingestion_flow.name == "ingestion-flow"

    assert annotation_task.retries == 0
    assert llm_preannotation_task.retries == 2
    assert llm_preannotation_task.retry_condition_fn is not None
    assert review_sync_task.retries == 2
    assert retrieve_source_task.retries == 2
    assert retrieve_source_task.retry_condition_fn is not None
    assert dataset_build_task.retries == 0
    assert training_task.retries == 0
    assert graph_ingestion_task.retries == 2
    assert graph_ingestion_task.retry_condition_fn is not None


def test_llm_retry_condition_only_accepts_transient_errors() -> None:
    class FailedState:
        def __init__(self, error: Exception) -> None:
            self.error = error

        def result(self):
            raise self.error

    assert annotation_module._retry_transient_llm_error(
        None,
        None,
        FailedState(LLMConnectionError("temporary upstream 500")),
    )
    assert not annotation_module._retry_transient_llm_error(
        None,
        None,
        FailedState(LLMResponseError("invalid structured response")),
    )


def test_graph_retry_condition_only_accepts_transient_errors() -> None:
    class FailedState:
        def __init__(self, error: Exception) -> None:
            self.error = error

        def result(self):
            raise self.error

    assert graph_module._retry_transient_neo4j_error(
        None,
        None,
        FailedState(Neo4jConnectionError("temporary")),
    )
    assert not graph_module._retry_transient_neo4j_error(
        None,
        None,
        FailedState(Neo4jValidationError("invalid")),
    )


def test_annotation_task_delegates_exact_service_input(monkeypatch) -> None:
    extraction = ModelExtractionResult(
        text="张三",
        ner_model_version="ner-v1",
        relation_model_version="re-v1",
        inference_seconds=0,
    )
    expected = SimpleNamespace(entities=[], relations=[])
    captured: dict[str, object] = {}

    class Service:
        def to_canonical(self, value, **kwargs):
            captured["value"] = value
            captured.update(kwargs)
            return expected

    monkeypatch.setattr(annotation_module, "AnnotationService", Service)
    monkeypatch.setattr(
        annotation_module, "get_run_logger", lambda: StubLogger()
    )

    result = annotation_task.fn(
        extraction,
        annotation_id="ann-1",
        case_id="case-1",
        doc_id="doc-1",
        text_id="text-1",
    )

    assert result is expected
    assert captured == {
        "value": extraction,
        "annotation_id": "ann-1",
        "case_id": "case-1",
        "doc_id": "doc-1",
        "text_id": "text-1",
    }


def test_annotation_task_propagates_service_error(monkeypatch) -> None:
    extraction = ModelExtractionResult(
        text="张三",
        ner_model_version="ner-v1",
        relation_model_version="re-v1",
        inference_seconds=0,
    )

    class Service:
        def to_canonical(self, *args, **kwargs):
            raise RuntimeError("annotation failed")

    monkeypatch.setattr(annotation_module, "AnnotationService", Service)
    monkeypatch.setattr(
        annotation_module, "get_run_logger", lambda: StubLogger()
    )

    with pytest.raises(RuntimeError, match="annotation failed"):
        annotation_task.fn(
            extraction,
            annotation_id="ann-1",
            case_id="case-1",
            doc_id="doc-1",
            text_id="text-1",
        )


def test_graph_task_closes_task_local_service_on_error(monkeypatch) -> None:
    service = SimpleNamespace(closed=False)

    def ingest(*args, **kwargs):
        raise RuntimeError("neo4j unavailable")

    def close():
        service.closed = True

    service.ingest_annotations_batch = ingest
    service.close = close
    monkeypatch.setattr(graph_module, "Neo4jService", lambda: service)
    monkeypatch.setattr(
        graph_module, "get_run_logger", lambda: StubLogger()
    )

    with pytest.raises(RuntimeError, match="neo4j unavailable"):
        graph_ingestion_task.fn([])

    assert service.closed is True


def test_top_level_dry_run_is_serializable(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion_module,
        "flow_run",
        SimpleNamespace(id="flow-run-1"),
    )

    result = asyncio.run(
        ingestion_flow.fn(batch_id="batch-1", dry_run=True)
    )

    assert result.flow_run_id == "flow-run-1"
    assert result.batch_id == "batch-1"
    assert result.status == "dry_run"
    assert result.workflow_phase == "automated"
    assert result.errors == []
    assert result.model_dump(mode="json")["finished_at"]


def test_dry_run_reports_human_review_phase(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion_module,
        "flow_run",
        SimpleNamespace(id="flow-run-2"),
    )

    submission = asyncio.run(
        ingestion_flow.fn(publish_for_review=True, dry_run=True)
    )
    consumption = asyncio.run(
        ingestion_flow.fn(
            run_annotation=False,
            run_review_sync=True,
            dry_run=True,
        )
    )

    assert submission.workflow_phase == "human_review_submission"
    assert consumption.workflow_phase == "human_review_consumption"

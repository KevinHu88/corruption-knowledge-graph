from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

import main as main_module
from flows.annotation_flow import AnnotationFlowResult, AnnotationJob
from flows.training_flow import TrainingFlowResult
from models import IngestionFlowResult
from flows.retrieval_flow import RetrievalFlowResult


def test_execute_validates_and_preserves_annotation_models(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_annotation_flow(**kwargs):
        captured.update(kwargs)
        return AnnotationFlowResult()

    monkeypatch.setattr(
        main_module, "annotation_flow", fake_annotation_flow
    )
    monkeypatch.setattr(
        main_module.StageRoutingService,
        "validate_model_artifacts",
        lambda self: None,
    )
    result = asyncio.run(
        main_module.execute(
            "annotation",
            {
                "jobs": [
                    {
                        "annotation_id": "ann-1",
                        "case_id": "case-1",
                        "doc_id": "doc-1",
                        "text_id": "text-1",
                        "text": "张三",
                    }
                ]
            },
        )
    )

    assert isinstance(result, AnnotationFlowResult)
    assert isinstance(captured["jobs"][0], AnnotationJob)


def test_execute_dispatches_synchronous_training_flow(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_training_flow(**kwargs):
        captured.update(kwargs)
        return TrainingFlowResult(
            dataset_version=kwargs["dataset_version"]
        )

    monkeypatch.setattr(main_module, "training_flow", fake_training_flow)
    monkeypatch.setattr(
        main_module.StageRoutingService,
        "validate_dataset_artifact",
        lambda self, dataset_version: None,
    )
    result = asyncio.run(
        main_module.execute(
            "training",
            {"dataset_version": "dataset-v1", "task_types": ["ner"]},
        )
    )

    assert result.dataset_version == "dataset-v1"
    assert captured["task_types"] == ["ner"]


def test_execute_dispatches_retrieval_flow(monkeypatch) -> None:
    captured: dict[str, object] = {}
    preflight: list[str] = []

    def fake_retrieval_flow(**kwargs):
        captured.update(kwargs)
        return RetrievalFlowResult(source_ids=["court"])

    monkeypatch.setattr(main_module, "retrieval_flow", fake_retrieval_flow)
    monkeypatch.setattr(
        main_module,
        "require_preflight",
        lambda features: preflight.extend(features),
    )
    result = asyncio.run(
        main_module.execute("retrieval", {"source_ids": ["court"]})
    )

    assert result.source_ids == ["court"]
    assert captured["source_ids"] == ["court"]
    assert preflight == ["tavily"]


def test_execute_does_not_call_flow_when_precondition_fails(
    monkeypatch,
) -> None:
    called = False

    async def fake_annotation_flow(**kwargs):
        nonlocal called
        called = True
        return AnnotationFlowResult()

    monkeypatch.setattr(main_module, "annotation_flow", fake_annotation_flow)

    with pytest.raises(main_module.StagePreconditionError, match="missing_input"):
        asyncio.run(main_module.execute("annotation", {}))

    assert called is False


def test_ingestion_does_not_build_dataset_by_default() -> None:
    assert main_module.IngestionRequest().run_dataset_build is False


def test_training_request_rejects_unsafe_dataset_version() -> None:
    with pytest.raises(main_module.ValidationError):
        main_module.TrainingRequest(dataset_version="../dataset")


def test_cli_prints_json_result(monkeypatch, capsys) -> None:
    async def fake_execute(command, payload):
        assert command == "ingestion"
        assert isinstance(payload, main_module.IngestionRequest)
        assert payload.dry_run is True
        now = datetime.now()
        return IngestionFlowResult(
            flow_run_id="flow-1",
            status="dry_run",
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(main_module, "execute", fake_execute)
    exit_code = main_module.main(
        ["ingestion", "--json", '{"dry_run": true}', "--pretty"]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["flow_run_id"] == "flow-1"
    assert output["status"] == "dry_run"


def test_cli_reports_validation_error() -> None:
    assert main_module.main(["training", "--json", "{}"]) == 2


def test_cli_reports_flow_value_error_as_execution_failure(
    monkeypatch,
) -> None:
    async def failing_execute(command, payload):
        raise ValueError("service rejected input")

    monkeypatch.setattr(main_module, "execute", failing_execute)

    assert (
        main_module.main(
            ["ingestion", "--json", '{"dry_run": true}']
        )
        == 1
    )


def test_load_payload_reads_utf8_file(tmp_path) -> None:
    path = tmp_path / "request.json"
    path.write_text('{"dataset_version": "数据集-v1"}', encoding="utf-8")

    result = main_module._load_payload(
        input_path=str(path),
        json_payload=None,
    )

    assert result == {"dataset_version": "数据集-v1"}

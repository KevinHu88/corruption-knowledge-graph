from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from main import (
    AnnotationRequest,
    DatasetBuildRequest,
    IngestionRequest,
    TrainingRequest,
)
from models import AnnotationStatus, CanonicalAnnotation
from src.services.dataset_service import DatasetManifest, DatasetStatistics
from src.services.stage_routing_service import (
    StagePreconditionError,
    StageRoutingService,
)


def annotation(
    status: AnnotationStatus = AnnotationStatus.APPROVED,
) -> CanonicalAnnotation:
    return CanonicalAnnotation(
        annotation_id="ann-1",
        case_id="case-1",
        doc_id="doc-1",
        text_id="text-1",
        text="valid text",
        annotation_source="HUMAN",
        schema_version="relation_v2.0",
        status=status,
    )


def statistics() -> DatasetStatistics:
    return DatasetStatistics(
        annotation_count=1,
        case_count=1,
        text_count=1,
        total_characters=10,
        entity_count=0,
        relation_positive_count=0,
        relation_negative_count=0,
        entity_distribution={},
        relation_distribution={},
        split_entity_counts={},
        split_relation_counts={},
        split_annotation_counts={"train": 1},
        split_case_counts={"train": 1},
        text_length={},
        entity_length={},
        average_entities_per_text=0,
        average_relations_per_text=0,
        no_entity_text_count=1,
        no_relation_text_count=1,
        positive_negative_ratio=None,
        rare_relations=[],
        missing_train_relations=[],
        validation_or_test_only_relations=[],
        case_sets_disjoint=True,
        leakage_detected=False,
    )


def write_dataset(tmp_path, *, content: bytes = b"sample") -> None:
    root = tmp_path / "dataset-v1"
    artifact = root / "ner" / "train.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(content)
    manifest = DatasetManifest(
        dataset_version="dataset-v1",
        dataset_fingerprint="fingerprint",
        schema_version="relation_v2.0",
        created_at=datetime(2026, 8, 1),
        random_seed=42,
        split_ratios={"train": 1.0},
        frozen_test_case_ids=[],
        train_case_ids=["case-1"],
        validation_case_ids=[],
        test_case_ids=[],
        source_annotation_ids=["ann-1"],
        label2id={},
        relation2id={},
        negative_sampling={},
        files={"ner_train": "ner/train.jsonl"},
        configuration={},
        file_checksums={
            "ner/train.jsonl": hashlib.sha256(content).hexdigest()
        },
        statistics=statistics(),
        python_version="3.13",
    )
    (root / "manifest.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )


def test_ingestion_plan_preserves_stage_order_and_features(monkeypatch) -> None:
    service = StageRoutingService()
    monkeypatch.setattr(service, "validate_model_artifacts", lambda: None)
    request = IngestionRequest(
        run_retrieval=True,
        run_annotation=True,
        publish_for_review=True,
    )

    plan = service.prepare("ingestion", request)

    assert plan.stages == [
        "retrieval",
        "annotation",
    ]
    assert plan.required_features == ["tavily", "label_studio"]


def test_review_consumption_plan_can_continue_downstream() -> None:
    request = IngestionRequest(
        run_annotation=False,
        run_review_sync=True,
        run_dataset_build=True,
        run_training=True,
        run_graph_ingestion=True,
    )

    plan = StageRoutingService().prepare("ingestion", request)

    assert plan.stages == [
        "review-sync",
        "dataset-build",
        "training",
        "graph-ingestion",
    ]
    assert plan.required_features == ["label_studio", "neo4j"]


def test_review_submission_and_consumption_cannot_share_flow() -> None:
    request = IngestionRequest(
        annotation_jobs=[
            {
                "annotation_id": "ann-1",
                "case_id": "case-1",
                "doc_id": "doc-1",
                "text_id": "text-1",
                "text": "valid text",
            }
        ],
        run_annotation=True,
        publish_for_review=True,
        run_review_sync=True,
    )

    with pytest.raises(StagePreconditionError, match="human_review_phase_conflict"):
        StageRoutingService().prepare("ingestion", request)


def test_review_submission_must_end_before_downstream_stages() -> None:
    request = IngestionRequest(
        annotation_jobs=[
            {
                "annotation_id": "ann-1",
                "case_id": "case-1",
                "doc_id": "doc-1",
                "text_id": "text-1",
                "text": "valid text",
            }
        ],
        run_annotation=True,
        publish_for_review=True,
        run_dataset_build=True,
    )

    with pytest.raises(StagePreconditionError, match="review_submission_must_stop"):
        StageRoutingService().prepare("ingestion", request)


def test_annotation_requires_actual_input() -> None:
    with pytest.raises(StagePreconditionError, match="missing_input"):
        StageRoutingService().prepare("annotation", AnnotationRequest())


def test_annotation_checks_model_artifacts_before_flow() -> None:
    request = AnnotationRequest(
        jobs=[
            {
                "annotation_id": "ann-1",
                "case_id": "case-1",
                "doc_id": "doc-1",
                "text_id": "text-1",
                "text": "valid text",
            }
        ]
    )

    service = StageRoutingService()
    service.config.training["modeling"]["ner"]["checkpoint_path"] = None
    service.config.training["modeling"]["relation"]["checkpoint_path"] = None

    with pytest.raises(StagePreconditionError, match="checkpoint_not_configured"):
        service.prepare("annotation", request)


def test_dataset_build_rejects_non_approved_annotations() -> None:
    request = DatasetBuildRequest(
        annotations=[annotation(AnnotationStatus.PENDING_REVIEW)]
    )

    with pytest.raises(StagePreconditionError, match="annotations_not_approved"):
        StageRoutingService().prepare("dataset-build", request)


def test_ingestion_blocks_model_output_from_direct_dataset_build() -> None:
    request = IngestionRequest(
        annotation_jobs=[
            {
                "annotation_id": "ann-1",
                "case_id": "case-1",
                "doc_id": "doc-1",
                "text_id": "text-1",
                "text": "valid text",
            }
        ],
        run_annotation=True,
        run_dataset_build=True,
    )

    with pytest.raises(StagePreconditionError, match="review_required"):
        StageRoutingService().prepare("ingestion", request)


def test_training_validates_manifest_schema_and_checksums(tmp_path) -> None:
    write_dataset(tmp_path)
    service = StageRoutingService(datasets_root=tmp_path)

    plan = service.prepare(
        "training", TrainingRequest(dataset_version="dataset-v1")
    )

    assert plan.stages == ["training"]
    assert service.validate_dataset_artifact("dataset-v1").dataset_version == (
        "dataset-v1"
    )


def test_training_rejects_changed_dataset_artifact(tmp_path) -> None:
    write_dataset(tmp_path)
    (tmp_path / "dataset-v1" / "ner" / "train.jsonl").write_bytes(
        b"changed"
    )
    service = StageRoutingService(datasets_root=tmp_path)

    with pytest.raises(StagePreconditionError, match="dataset_checksum_mismatch"):
        service.prepare(
            "training", TrainingRequest(dataset_version="dataset-v1")
        )

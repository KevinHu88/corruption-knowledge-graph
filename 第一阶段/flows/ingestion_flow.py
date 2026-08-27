"""Top-level Prefect orchestration for the available service pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal

from prefect import flow
from prefect.runtime import flow_run
from pydantic import BaseModel

from flows.annotation_flow import AnnotationJob, annotation_flow
from flows.dataset_build_flow import dataset_build_flow
from flows.graph_ingestion_flow import graph_ingestion_flow
from flows.review_sync_flow import review_sync_flow
from flows.retrieval_flow import retrieval_flow
from flows.training_flow import training_flow
from models import (
    CanonicalAnnotation,
    CaseDocument,
    IngestionFlowResult,
    RawDocument,
    SourceDocument,
)
from src.services.dataset_service import DatasetManifest
from src.services.stage_routing_service import (
    StagePreconditionError,
    StageRoutingService,
)


# 中文注释：把子 Flow 的 Pydantic 结果转换为总流程可以汇总的 JSON 字典。
def _json_result(value: BaseModel | None) -> dict[str, Any] | None:
    return value.model_dump(mode="json") if value is not None else None


# 中文注释：项目总编排入口，按照阶段开关串联标注、人审、数据集、训练和图谱写入。
@flow(name="ingestion-flow")
async def ingestion_flow(
    *,
    batch_id: str | None = None,
    case_id: str | None = None,
    annotation_jobs: Sequence[AnnotationJob] | None = None,
    raw_documents: Sequence[RawDocument] | None = None,
    annotations: Sequence[CanonicalAnnotation] | None = None,
    run_retrieval: bool = False,
    retrieval_source_ids: Sequence[str] | None = None,
    retrieval_date: date | None = None,
    extract_missing_content: bool = True,
    continue_on_retrieval_error: bool = False,
    run_annotation: bool = True,
    publish_for_review: bool = False,
    run_review_sync: bool = False,
    run_dataset_build: bool = False,
    run_training: bool = False,
    run_graph_ingestion: bool = False,
    project_id: int | None = None,
    review_task_ids: Sequence[int] | None = None,
    review_max_items: int = 1000,
    dataset_version: str | None = None,
    previous_manifest: DatasetManifest | Mapping[str, Any] | None = None,
    rebuild_test_set: bool = False,
    overwrite_dataset: bool = False,
    training_task_types: Sequence[
        Literal["ner", "relation"]
    ] = ("ner", "relation"),
    source_documents: Mapping[
        str, SourceDocument | Mapping[str, Any]
    ] | None = None,
    case_documents: Mapping[
        str, CaseDocument | Mapping[str, Any]
    ] | None = None,
    entity_uid_maps: Mapping[str, Mapping[str, str]] | None = None,
    continue_on_graph_error: bool = False,
    dry_run: bool = False,
) -> IngestionFlowResult:
    """Run only requested stages and establish dependencies through results."""

    started_at = datetime.now()
    run_id = str(flow_run.id)
    workflow_phase = (
        "human_review_submission"
        if publish_for_review
        else (
            "human_review_consumption"
            if run_review_sync
            else "automated"
        )
    )
    if dry_run:
        return IngestionFlowResult(
            flow_run_id=run_id,
            batch_id=batch_id,
            case_id=case_id,
            workflow_phase=workflow_phase,
            status="dry_run",
            started_at=started_at,
            finished_at=datetime.now(),
        )

    current_documents = list(raw_documents or ())
    retrieval_result = None
    if run_retrieval:
        retrieval_result = retrieval_flow(
            source_ids=retrieval_source_ids,
            today=retrieval_date,
            extract_missing_content=extract_missing_content,
            continue_on_error=continue_on_retrieval_error,
        )
        by_id = {item.doc_id: item for item in current_documents}
        for item in retrieval_result.raw_documents:
            by_id.setdefault(item.doc_id, item)
        current_documents = list(by_id.values())

    current_annotations = list(annotations or ())
    annotation_result = None
    if run_annotation:
        if not annotation_jobs and not current_documents:
            raise ValueError(
                "run_annotation=True requires annotation_jobs or raw_documents"
            )
        # Retrieval can produce more than Prefect's 512 KiB flow-parameter
        # limit once full document bodies are included. Execute the decorated
        # flow's implementation in the current flow process so documents are
        # not serialized again as child-flow parameters; its tasks still keep
        # their normal Prefect tracking and retry behavior.
        annotation_runner = getattr(annotation_flow, "fn", annotation_flow)
        annotation_result = await annotation_runner(
            jobs=annotation_jobs,
            raw_documents=current_documents,
            publish_for_review=publish_for_review,
            project_id=project_id,
        )
        current_annotations = annotation_result.annotations

    review_result = None
    if run_review_sync:
        review_result = review_sync_flow(
            task_ids=review_task_ids,
            project_id=project_id,
            max_items=review_max_items,
        )
        current_annotations = review_result.annotations

    dataset_result = None
    if run_dataset_build:
        StageRoutingService.require_approved_annotations(
            current_annotations, stage="dataset-build"
        )
        dataset_result = dataset_build_flow(
            current_annotations,
            dataset_version=dataset_version,
            previous_manifest=previous_manifest,
            rebuild_test_set=rebuild_test_set,
            overwrite=overwrite_dataset,
        )

    training_result = None
    if run_training:
        selected_dataset_version = (
            dataset_result.dataset.dataset_version
            if dataset_result is not None
            else dataset_version
        )
        if not selected_dataset_version:
            raise ValueError(
                "run_training=True requires a built dataset or dataset_version"
            )
        if dataset_result is not None and dataset_result.dataset.status != "READY":
            raise StagePreconditionError(
                "training",
                "dataset_not_ready",
                f"新构建数据集状态不是 READY：{dataset_result.dataset.status}",
            )
        training_result = training_flow(
            selected_dataset_version,
            task_types=training_task_types,
        )

    graph_result = None
    if run_graph_ingestion:
        StageRoutingService.require_approved_annotations(
            current_annotations, stage="graph-ingestion"
        )
        graph_result = graph_ingestion_flow(
            current_annotations,
            source_documents=source_documents,
            case_documents=case_documents,
            entity_uid_maps=entity_uid_maps,
            continue_on_error=continue_on_graph_error,
        )

    retrieval_summary = None
    workflow_errors: list[str] = []
    if retrieval_result is not None:
        retrieval_summary = retrieval_result.model_dump(mode="json")
        workflow_errors.extend(retrieval_result.errors)
    annotation_summary = None
    if annotation_result is not None:
        annotation_summary = {
            "annotation_ids": [
                item.annotation_id for item in annotation_result.annotations
            ],
            "annotation_count": len(annotation_result.annotations),
            "processed_document_count": (
                annotation_result.processed_document_count
            ),
            "skipped_document_count": (
                annotation_result.skipped_document_count
            ),
            "failed_document_count": annotation_result.failed_document_count,
            "review_import": _json_result(annotation_result.review_import),
        }
        if annotation_result.failed_document_count:
            workflow_errors.append(
                "text processing failed for "
                f"{annotation_result.failed_document_count} document(s)"
            )
        if (
            annotation_result.review_import is not None
            and annotation_result.review_import.failed_count
        ):
            workflow_errors.append(
                "Label Studio import failed for "
                f"{annotation_result.review_import.failed_count} annotation(s)"
            )
    review_summary = None
    if review_result is not None:
        review_summary = {
            "annotation_ids": [
                item.annotation_id for item in review_result.annotations
            ],
            "synced_count": len(review_result.annotations),
            "task_ids": review_result.task_ids,
            "failed_task_ids": review_result.failed_task_ids,
            "errors": review_result.errors,
            "latency_seconds": review_result.latency_seconds,
        }
        workflow_errors.extend(review_result.errors)
    if dataset_result is not None and dataset_result.validation.invalid_count:
        workflow_errors.append(
            "dataset validation rejected "
            f"{dataset_result.validation.invalid_count} annotation(s)"
        )
    if graph_result is not None and graph_result.failed_batches:
        workflow_errors.extend(
            item.error or f"graph batch {item.batch_index} failed"
            for item in graph_result.batches
            if not item.success
        )

    return IngestionFlowResult(
        flow_run_id=run_id,
        batch_id=batch_id,
        case_id=case_id,
        workflow_phase=workflow_phase,
        retrieval_result=retrieval_summary,
        annotation_result=annotation_summary,
        review_sync_result=review_summary,
        dataset_result=_json_result(dataset_result),
        training_result=_json_result(training_result),
        graph_ingestion_result=_json_result(graph_result),
        status=(
            "completed_with_errors" if workflow_errors else "completed"
        ),
        started_at=started_at,
        finished_at=datetime.now(),
        errors=workflow_errors,
    )


__all__ = ["ingestion_flow"]

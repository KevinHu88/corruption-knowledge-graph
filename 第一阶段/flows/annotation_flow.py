"""Prefect flow for text processing, inference, and canonical annotation."""

from __future__ import annotations

from collections.abc import Sequence

from prefect import flow
from pydantic import BaseModel, Field

from config import load_project_config
from models import CanonicalAnnotation, RawDocument
from src.services.label_studio_service import LabelStudioImportResult
from task.annotation_tasks import (
    annotation_task,
    llm_preannotation_task,
    publish_annotations_task,
)
from task.ingestion_tasks import inference_task
from task.parsing_tasks import process_documents_task


# 中文注释：已经完成模型推理的单条标注任务；Flow 会把它转换为规范标注。
class AnnotationJob(BaseModel):
    """Serializable identifiers and text needed by the two annotation services."""

    annotation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


# 中文注释：标注 Flow 的汇总结果，区分文本处理结果、规范标注和 Label Studio 发布结果。
class AnnotationFlowResult(BaseModel):
    """Canonical results plus the optional Label Studio import summary."""

    annotations: list[CanonicalAnnotation] = Field(default_factory=list)
    processed_document_count: int = 0
    skipped_document_count: int = 0
    failed_document_count: int = 0
    review_import: LabelStudioImportResult | None = None


# 中文注释：核心标注编排；根据输入选择“原始文档处理+推理”或“直接规范化”，并可选发布到 Label Studio。
@flow(name="annotation-flow")
async def annotation_flow(
    *,
    jobs: Sequence[AnnotationJob] | None = None,
    raw_documents: Sequence[RawDocument] | None = None,
    publish_for_review: bool = False,
    project_id: int | None = None,
    model_version: str | None = None,
) -> AnnotationFlowResult:
    """Create canonical annotations from explicit jobs and/or raw documents."""

    pending = list(jobs or ())
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    if raw_documents:
        processed = await process_documents_task(list(raw_documents))
        processed_count = len(processed)
        skipped_count = sum(
            item.processing_status == "irrelevant" for item in processed
        )
        failed_count = sum(
            item.processing_status == "failed" for item in processed
        )
        for document in processed:
            if document.processing_status == "failed":
                continue
            pending.extend(
                AnnotationJob(
                    annotation_id=chunk.chunk_id,
                    case_id=document.case_id,
                    doc_id=document.doc_id,
                    text_id=chunk.text_id,
                    text=chunk.text,
                )
                for chunk in document.model_input_chunks
                if chunk.model_ready
            )

    annotations: list[CanonicalAnnotation] = []
    if pending:
        annotator = str(
            load_project_config().workflow.get("annotation", {}).get(
                "default_annotator", "deep_model"
            )
        ).lower()
        if annotator == "llm":
            annotations = [
                llm_preannotation_task(
                    job.text,
                    annotation_id=job.annotation_id,
                    case_id=job.case_id,
                    doc_id=job.doc_id,
                    text_id=job.text_id,
                )
                for job in pending
            ]
        elif annotator == "deep_model":
            extractions = inference_task([item.text for item in pending])
            annotations = [
                annotation_task(
                    extraction,
                    annotation_id=job.annotation_id,
                    case_id=job.case_id,
                    doc_id=job.doc_id,
                    text_id=job.text_id,
                )
                for job, extraction in zip(pending, extractions, strict=True)
            ]
        else:
            raise ValueError(f"不支持的 default_annotator：{annotator}")

    review_import = None
    if publish_for_review:
        review_import = publish_annotations_task(
            annotations,
            project_id=project_id,
            model_version=model_version,
        )
    return AnnotationFlowResult(
        annotations=annotations,
        processed_document_count=processed_count,
        skipped_document_count=skipped_count,
        failed_document_count=failed_count,
        review_import=review_import,
    )


__all__ = ["AnnotationFlowResult", "AnnotationJob", "annotation_flow"]

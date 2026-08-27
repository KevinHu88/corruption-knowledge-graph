"""Thin Prefect wrappers around annotation-related services."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prefect import task
from prefect.logging import get_run_logger

from models import CanonicalAnnotation, ModelExtractionResult
from src.services.annotation_service import AnnotationService
from src.services.label_studio_service import (
    LabelStudioConnectionError,
    LabelStudioImportResult,
    LabelStudioService,
)
from src.services.llm_service import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMService,
)


# 中文注释：只允许 Label Studio 连接类故障触发 Prefect 重试，业务校验错误不会重复执行。
def _retry_label_studio_connection(task, task_run, state) -> bool:
    """Retry transport failures, but not auth, schema, or conversion errors."""

    del task, task_run
    try:
        state.result()
    except LabelStudioConnectionError:
        return True
    except Exception:
        return False
    return False


# 中文注释：仅重试大模型的瞬时连接、超时、5xx 和限流错误；配置、解析及业务校验错误直接暴露。
def _retry_transient_llm_error(task, task_run, state) -> bool:
    """Retry recoverable provider failures without hiding bad responses."""

    del task, task_run
    try:
        state.result()
    except (LLMConnectionError, LLMRateLimitError):
        return True
    except Exception:
        return False
    return False


# 中文注释：将单条深度模型抽取结果规范化为跨系统统一的 CanonicalAnnotation。
@task(name="annotation", timeout_seconds=600)
def annotation_task(
    extraction: ModelExtractionResult,
    *,
    annotation_id: str,
    case_id: str,
    doc_id: str,
    text_id: str,
) -> CanonicalAnnotation:
    """Validate and convert one model extraction through AnnotationService."""

    logger = get_run_logger()
    started = time.perf_counter()
    logger.info(
        "step=annotation annotation_id=%s case_id=%s text_id=%s samples=1",
        annotation_id,
        case_id,
        text_id,
    )
    try:
        result = AnnotationService().to_canonical(
            extraction,
            annotation_id=annotation_id,
            case_id=case_id,
            doc_id=doc_id,
            text_id=text_id,
        )
    except Exception:
        logger.exception(
            "step=annotation status=failed annotation_id=%s case_id=%s",
            annotation_id,
            case_id,
        )
        raise
    logger.info(
        "step=annotation success=1 failed=0 skipped=0 entities=%d "
        "relations=%d output=memory elapsed=%.3fs",
        len(result.entities),
        len(result.relations),
        time.perf_counter() - started,
    )
    return result


# 中文注释：直接使用大模型对单个文本块生成实体和关系预标注。
@task(
    name="llm-preannotation",
    retries=2,
    retry_delay_seconds=[5, 15],
    retry_condition_fn=_retry_transient_llm_error,
    timeout_seconds=300,
)
def llm_preannotation_task(
    text: str,
    *,
    annotation_id: str,
    case_id: str,
    doc_id: str,
    text_id: str,
) -> CanonicalAnnotation:
    """Generate one reviewable annotation through Structured Outputs."""

    logger = get_run_logger()
    started = time.perf_counter()
    logger.info(
        "step=llm-preannotation annotation_id=%s chars=%d",
        annotation_id,
        len(text),
    )
    try:
        llm = LLMService()
        try:
            result = AnnotationService(
                llm_service=llm
            ).preannotate_with_llm(
                text,
                annotation_id=annotation_id,
                case_id=case_id,
                doc_id=doc_id,
                text_id=text_id,
            )
        finally:
            llm.close()
    except Exception:
        logger.exception(
            "step=llm-preannotation status=failed annotation_id=%s",
            annotation_id,
        )
        raise
    logger.info(
        "step=llm-preannotation status=completed annotation_id=%s "
        "entities=%d relations=%d elapsed=%.3fs",
        annotation_id,
        len(result.entities),
        len(result.relations),
        time.perf_counter() - started,
    )
    return result


# 中文注释：把规范标注批量发布为 Label Studio 任务，并在连接故障时由 Prefect 重试。
@task(
    name="publish-annotations-for-review",
    retries=2,
    retry_delay_seconds=10,
    retry_condition_fn=_retry_label_studio_connection,
    timeout_seconds=600,
)
def publish_annotations_task(
    annotations: Sequence[CanonicalAnnotation],
    *,
    project_id: int | None = None,
    model_version: str | None = None,
) -> LabelStudioImportResult:
    """Publish canonical annotations through LabelStudioService."""

    logger = get_run_logger()
    started = time.perf_counter()
    items = list(annotations)
    logger.info(
        "step=publish-annotations project_id=%s input_samples=%d",
        project_id,
        len(items),
    )
    try:
        result = LabelStudioService().import_annotations(
            items,
            project_id=project_id,
            model_version=model_version,
        )
    except Exception:
        logger.exception(
            "step=publish-annotations status=failed project_id=%s "
            "input_samples=%d",
            project_id,
            len(items),
        )
        raise
    logger.info(
        "step=publish-annotations success=%d failed=%d skipped=%d "
        "output=label-studio elapsed=%.3fs",
        result.imported_count,
        result.failed_count,
        result.queued_count,
        time.perf_counter() - started,
    )
    return result


__all__ = [
    "annotation_task",
    "llm_preannotation_task",
    "publish_annotations_task",
]

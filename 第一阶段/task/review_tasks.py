"""Thin Prefect wrapper around Label Studio review synchronization."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prefect import task
from prefect.logging import get_run_logger

from src.services.label_studio_service import (
    LabelStudioConnectionError,
    LabelStudioService,
    LabelStudioSyncResult,
)


# 中文注释：人审同步只在 Label Studio 连接异常时重试，格式或业务错误直接暴露。
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


# 中文注释：从 Label Studio 读取人工审核结果，并转换为项目统一的已批准标注。
@task(
    name="review-sync",
    retries=2,
    retry_delay_seconds=10,
    retry_condition_fn=_retry_label_studio_connection,
    timeout_seconds=600,
)
def review_sync_task(
    *,
    task_ids: Sequence[int] | None = None,
    project_id: int | None = None,
    max_items: int = 1000,
) -> LabelStudioSyncResult:
    """Fetch and convert reviewed tasks; the service remains the data authority."""

    logger = get_run_logger()
    started = time.perf_counter()
    logger.info(
        "step=review-sync project_id=%s requested_task_ids=%d max_items=%d",
        project_id,
        len(task_ids or ()),
        max_items,
    )
    try:
        result = LabelStudioService().sync_reviewed_annotations(
            task_ids=task_ids,
            project_id=project_id,
            max_items=max_items,
        )
    except Exception:
        logger.exception(
            "step=review-sync status=failed project_id=%s", project_id
        )
        raise
    logger.info(
        "step=review-sync success=%d failed=%d skipped=0 output=memory "
        "elapsed=%.3fs",
        len(result.annotations),
        len(result.failed_task_ids),
        time.perf_counter() - started,
    )
    return result


__all__ = ["review_sync_task"]

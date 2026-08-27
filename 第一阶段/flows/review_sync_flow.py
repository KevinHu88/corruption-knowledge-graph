"""Prefect flow for human-review synchronization."""

from __future__ import annotations

from collections.abc import Sequence

from prefect import flow

from src.services.label_studio_service import LabelStudioSyncResult
from task.review_tasks import review_sync_task


# 中文注释：人审同步编排，从 Label Studio 拉取最新有效审核结果并转换为 APPROVED 标注。
@flow(name="review-sync-flow")
def review_sync_flow(
    *,
    task_ids: Sequence[int] | None = None,
    project_id: int | None = None,
    max_items: int = 1000,
) -> LabelStudioSyncResult:
    """Return the service's structured synchronization result unchanged."""

    return review_sync_task(
        task_ids=task_ids,
        project_id=project_id,
        max_items=max_items,
    )


__all__ = ["review_sync_flow"]

"""Thin Prefect wrapper around DatasetService."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from prefect import task
from prefect.logging import get_run_logger

from models import CanonicalAnnotation
from src.services.dataset_service import (
    DatasetBuildResult,
    DatasetManifest,
    DatasetService,
)


# 中文注释：数据集阶段的调度边界，负责调用 DatasetService 并返回版本化产物信息。
@task(name="dataset-build", timeout_seconds=3600)
def dataset_build_task(
    annotations: Sequence[CanonicalAnnotation],
    *,
    dataset_version: str | None = None,
    previous_manifest: DatasetManifest | Mapping[str, Any] | None = None,
    rebuild_test_set: bool = False,
    strict: bool | None = None,
    overwrite: bool = False,
) -> DatasetBuildResult:
    """Create one immutable dataset version through DatasetService."""

    logger = get_run_logger()
    started = time.perf_counter()
    items = list(annotations)
    logger.info(
        "step=dataset-build dataset_version=%s input_annotations=%d "
        "overwrite=%s rebuild_test_set=%s",
        dataset_version,
        len(items),
        overwrite,
        rebuild_test_set,
    )
    try:
        result = DatasetService().create_dataset_version(
            items,
            dataset_version=dataset_version,
            previous_manifest=previous_manifest,
            rebuild_test_set=rebuild_test_set,
            strict=strict,
            overwrite=overwrite,
        )
    except Exception:
        logger.exception(
            "step=dataset-build status=failed dataset_version=%s "
            "input_annotations=%d",
            dataset_version,
            len(items),
        )
        raise
    dataset = result.dataset
    logger.info(
        "step=dataset-build success=%d failed=%d skipped=%d "
        "dataset_version=%s train=%d validation=%d test=%d "
        "output=%s elapsed=%.3fs",
        result.validation.valid_count,
        result.validation.invalid_count,
        result.deduplication.duplicate_count,
        dataset.dataset_version,
        dataset.train_size,
        dataset.validation_size,
        dataset.test_size,
        result.output_dir,
        time.perf_counter() - started,
    )
    return result


__all__ = ["dataset_build_task"]

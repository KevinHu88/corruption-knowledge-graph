"""Prefect flow for versioned dataset construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prefect import flow

from models import CanonicalAnnotation
from src.services.dataset_service import DatasetBuildResult, DatasetManifest
from task.dataset_tasks import dataset_build_task


# 中文注释：数据集构建编排，把已审核规范标注交给 Dataset Task 生成版本化训练数据。
@flow(name="dataset-build-flow")
def dataset_build_flow(
    annotations: Sequence[CanonicalAnnotation],
    *,
    dataset_version: str | None = None,
    previous_manifest: DatasetManifest | Mapping[str, Any] | None = None,
    rebuild_test_set: bool = False,
    strict: bool | None = None,
    overwrite: bool = False,
) -> DatasetBuildResult:
    """Build a dataset with an explicit dependency on annotation values."""

    return dataset_build_task(
        annotations,
        dataset_version=dataset_version,
        previous_manifest=previous_manifest,
        rebuild_test_set=rebuild_test_set,
        strict=strict,
        overwrite=overwrite,
    )


__all__ = ["dataset_build_flow"]

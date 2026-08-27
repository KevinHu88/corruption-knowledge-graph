"""Thin Prefect wrappers around TrainingService."""

from __future__ import annotations

import time
from typing import Literal

from prefect import task
from prefect.logging import get_run_logger

from models import TrainingResult
from src.services.training_service import TrainingService


# 中文注释：长耗时模型训练任务，根据 task_type 选择 BERT-CRF 或 BERTEntity Trainer。
@task(name="model-training", timeout_seconds=86400)
def training_task(
    dataset_version: str,
    *,
    task_type: Literal["ner", "relation"],
) -> TrainingResult:
    """Train exactly one model through the service; expensive work is not retried."""

    logger = get_run_logger()
    started = time.perf_counter()
    logger.info(
        "step=model-training task_type=%s dataset_version=%s",
        task_type,
        dataset_version,
    )
    try:
        service = TrainingService()
        if task_type == "ner":
            result = service.train_ner(dataset_version)
        else:
            result = service.train_relation_model(dataset_version)
    except Exception:
        logger.exception(
            "step=model-training status=failed task_type=%s "
            "dataset_version=%s",
            task_type,
            dataset_version,
        )
        raise
    logger.info(
        "step=model-training success=1 failed=0 skipped=0 task_type=%s "
        "dataset_version=%s model_version=%s checkpoint=%s "
        "metrics=%s elapsed=%.3fs",
        task_type,
        dataset_version,
        result.model_version,
        result.checkpoint_path,
        sorted(result.metrics),
        time.perf_counter() - started,
    )
    return result


__all__ = ["training_task"]

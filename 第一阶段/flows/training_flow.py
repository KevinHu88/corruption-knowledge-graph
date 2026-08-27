"""Prefect flow for NER and relation-model training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from prefect import flow
from pydantic import BaseModel, Field

from models import TrainingResult
from task.training_tasks import training_task


# 中文注释：训练 Flow 的统一输出，分别保存 NER 和关系模型的训练结果。
class TrainingFlowResult(BaseModel):
    """Serializable collection of service-level training results."""

    dataset_version: str
    results: list[TrainingResult] = Field(default_factory=list)


# 中文注释：训练编排；根据 model_type 顺序调用 NER 和/或关系模型训练 Task。
@flow(name="training-flow")
def training_flow(
    dataset_version: str,
    *,
    task_types: Sequence[Literal["ner", "relation"]] = ("ner", "relation"),
) -> TrainingFlowResult:
    """Train only the explicitly requested model families."""

    requested = list(dict.fromkeys(task_types))
    results = [
        training_task(dataset_version, task_type=task_type)
        for task_type in requested
    ]
    return TrainingFlowResult(
        dataset_version=dataset_version,
        results=results,
    )


__all__ = ["TrainingFlowResult", "training_flow"]

"""TrainingService 委托和清单路径测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models import TrainingResult
from src.services.training_service import (
    TrainingService,
    TrainingServiceError,
)


def test_training_service_delegates_to_trainer(tmp_path):
    checkpoint = tmp_path / "ner-v1"
    checkpoint.mkdir()
    expected = TrainingResult(
        task_type="ner",
        model_version="ner-v1",
        dataset_version="data-v1",
        schema_version="relation_v2.0",
        random_seed=42,
        hyperparameters={"epochs": 1},
        metrics={"f1": 0.8},
        checkpoint_path=str(checkpoint),
    )
    trainer = MagicMock()
    trainer.train.return_value = expected
    service = TrainingService(ner_trainer=trainer)

    assert service.train_ner("data-v1") == expected
    trainer.train.assert_called_once_with("data-v1")
    assert (checkpoint / "model_manifest.json").is_file()


def test_missing_model_directory_is_clear(tmp_path):
    service = TrainingService()
    service.config.training["modeling"]["ner"]["checkpoint_dir"] = str(
        tmp_path / "missing"
    )

    with pytest.raises(TrainingServiceError, match="模型目录不存在"):
        service.get_champion_model("ner")


@pytest.mark.parametrize(
    ("method_name", "task_name"),
    [
        ("train_ner", "NER"),
        ("train_relation_model", "关系"),
    ],
)
def test_missing_training_dataset_is_reported_before_model_loading(
    tmp_path, method_name, task_name
):
    service = TrainingService()
    service.config.training["modeling"]["datasets_root"] = str(tmp_path)

    with pytest.raises(FileNotFoundError, match=task_name):
        getattr(service, method_name)("missing-version")

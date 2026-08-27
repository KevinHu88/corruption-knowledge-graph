"""统一训练、评估与模型清单查询服务。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from config import ProjectConfig, load_project_config
from models import EvaluationResult, TrainingResult
from src.modeling.bert_crf.trainer import BertCrfTrainer
from src.modeling.bert_entity.trainer import BertEntityTrainer
from src.modeling.common.model_manifest import ModelManifest


class TrainingServiceError(RuntimeError):
    """训练配置、训练执行或清单查询失败。"""


# 中文注释：训练协调层；把 NER/关系训练委托给各自 Trainer，并统一记录模型 manifest。
class TrainingService:
    """调用两个 trainer；不在 Service 内实现训练循环。"""

    def __init__(
        self,
        ner_trainer: BertCrfTrainer | None = None,
        relation_trainer: BertEntityTrainer | None = None,
        *,
        project_config: ProjectConfig | None = None,
        end_to_end_evaluator: Callable[
            [str, str], EvaluationResult
        ] | None = None,
    ) -> None:
        self.config = project_config or load_project_config()
        self.ner_trainer = ner_trainer or BertCrfTrainer(
            project_config=self.config
        )
        self.relation_trainer = relation_trainer or BertEntityTrainer(
            project_config=self.config
        )
        self.end_to_end_evaluator = end_to_end_evaluator

    # 中文注释：使用指定数据集版本训练 BERT-CRF，并在 checkpoint 存在后记录产物清单。
    def train_ner(self, dataset_version: str) -> TrainingResult:
        """训练 NER 模型。"""

        result = self.ner_trainer.train(dataset_version)
        self._record_manifest(result)
        return result

    # 中文注释：使用指定数据集版本训练 BERTEntity 关系模型，并记录可追溯 manifest。
    def train_relation_model(self, dataset_version: str) -> TrainingResult:
        """训练关系分类模型。"""

        result = self.relation_trainer.train(dataset_version)
        self._record_manifest(result)
        return result

    def evaluate_ner(self, model_version: str) -> EvaluationResult:
        """评估 NER 模型。"""

        return self.ner_trainer.evaluate(model_version)

    def evaluate_relation_model(
        self, model_version: str
    ) -> EvaluationResult:
        """评估关系分类模型。"""

        return self.relation_trainer.evaluate(model_version)

    def run_end_to_end_evaluation(
        self,
        ner_model_version: str,
        relation_model_version: str,
    ) -> EvaluationResult:
        """调用已配置的端到端标注数据评估器。"""

        if self.end_to_end_evaluator is None:
            raise TrainingServiceError(
                "尚未配置 end_to_end_evaluator 或端到端评估数据"
            )
        return self.end_to_end_evaluator(
            ner_model_version, relation_model_version
        )

    # 中文注释：扫描模型产物目录，返回最新的 Champion manifest，而不是重新加载模型。
    def get_champion_model(
        self,
        task_type: Literal["ner", "relation"],
    ) -> ModelManifest:
        """从 artifacts 下的 model_manifest.json 查找 Champion。"""

        modeling = self.config.training["modeling"]
        root = Path(modeling[task_type]["checkpoint_dir"])
        if not root.is_dir():
            raise TrainingServiceError(f"模型目录不存在：{root}")
        champions = [
            manifest
            for path in root.glob("*/model_manifest.json")
            for manifest in [ModelManifest.load(path)]
            if manifest.role == "CHAMPION"
        ]
        if not champions:
            raise TrainingServiceError(
                f"未找到 task_type={task_type} 的 Champion 模型"
            )
        return max(champions, key=lambda item: item.created_at)

    # 中文注释：核验真实 checkpoint 后写入统一 ModelManifest，供推理和模型选择使用。
    def _record_manifest(self, result: TrainingResult) -> Path:
        """仅在真实 checkpoint 已存在时写入实验清单。"""

        checkpoint = Path(result.checkpoint_path)
        if not checkpoint.exists():
            raise TrainingServiceError(
                f"trainer 返回的 checkpoint 不存在：{checkpoint}"
            )
        task_config = self.config.training["modeling"][result.task_type]
        artifact_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
        expected_checkpoint = (
            "model.safetensors"
            if result.task_type == "ner"
            else "model.pth.tar"
        )
        checkpoint_file = (
            checkpoint.name
            if checkpoint.is_file()
            else expected_checkpoint
        )
        mapping_name = (
            "label_map.json"
            if result.task_type == "ner"
            else "relation_map.json"
        )
        mapping_path = artifact_dir / mapping_name
        configured_mapping = (
            task_config.get("label_map_path")
            or task_config.get("relation_map_path")
            or ""
        )
        manifest = ModelManifest(
            task_type=result.task_type,
            model_version=result.model_version,
            dataset_version=result.dataset_version,
            schema_version=result.schema_version,
            architecture=str(task_config["architecture"]),
            pretrained_model=str(task_config["pretrained_model"]),
            checkpoint_file=checkpoint_file,
            tokenizer_dir=".",
            mapping_file=(
                mapping_name if mapping_path.is_file()
                else str(configured_mapping)
            ),
            random_seed=result.random_seed,
            hyperparameters=result.hyperparameters,
            metrics=result.metrics,
            created_at=result.created_at,
        )
        manifest_path = artifact_dir / "model_manifest.json"
        return manifest.save(manifest_path)

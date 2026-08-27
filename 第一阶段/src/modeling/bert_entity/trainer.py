"""BERTEntity 的配置驱动训练、评估与 artifact 保存。"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ProjectConfig, load_project_config
from models import EvaluationResult, TrainingResult
from src.modeling.common.device import move_model_to_device
from src.modeling.common.label_mapping import load_relation_mapping
from .dataset import (
    BertEntityDataset,
    build_dataloader,
    build_relation_features,
    collate_relation_features,
)
from .metrics import micro_prf
from .model import BertEntityForRelation

logger = logging.getLogger(__name__)


class BertEntityTrainingError(RuntimeError):
    """BERTEntity 训练配置、数据或 checkpoint 不可用。"""


# 中文注释：BERTEntity 关系分类训练器，负责数据准备、交叉熵训练、评估和产物保存。
class BertEntityTrainer:
    """执行真实 BERTEntity 训练；支持 runner 注入以保持原接口兼容。"""

    def __init__(
        self,
        train_runner: Callable[[str], TrainingResult] | None = None,
        evaluate_runner: Callable[[str], EvaluationResult] | None = None,
        *,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.train_runner = train_runner
        self.evaluate_runner = evaluate_runner
        self.config = project_config or load_project_config()

    # 中文注释：读取指定版本关系数据集，训练分类模型并保存 checkpoint、tokenizer 和关系映射。
    def train(self, dataset_version: str) -> TrainingResult:
        """从 OpenNRE JSONL 训练正式 schema 对应的关系分类器。"""

        if self.train_runner is not None:
            return self.train_runner(dataset_version)
        torch, tokenizer, model, train_loader, eval_loader, context = (
            self._prepare_training(dataset_version)
        )
        training = self.config.training["relation"]["bert_entity"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training.get("learning_rate", 2e-5)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        criterion = torch.nn.CrossEntropyLoss()
        epochs = int(training.get("epochs", 5))
        total_loss = 0.0
        steps = 0
        model.train()
        for _ in range(epochs):
            for batch in train_loader:
                batch = {
                    key: value.to(context["device"])
                    for key, value in batch.items()
                }
                optimizer.zero_grad()
                logits = model(
                    batch["token"],
                    batch["att_mask"],
                    batch["pos1"],
                    batch["pos2"],
                )
                loss = criterion(logits, batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                steps += 1

        metrics = self._evaluate_loader(
            model,
            eval_loader,
            context["device"],
            context["id_to_relation"],
            context["negative_label"],
        )
        metrics["train_loss"] = total_loss / steps if steps else 0.0
        artifact_dir = self._artifact_dir(dataset_version)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = artifact_dir / "model.pth.tar"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "relation_mapping": context["relation_mapping"],
                "pretrained_model": context["pretrained_model"],
            },
            checkpoint,
        )
        tokenizer.save_pretrained(artifact_dir)
        (artifact_dir / "relation_map.json").write_text(
            json.dumps(
                context["relation_mapping"], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        logger.info("BERTEntity 训练完成：%s", artifact_dir)
        return TrainingResult(
            task_type="relation",
            model_version=artifact_dir.name,
            dataset_version=dataset_version,
            schema_version=self._schema_version,
            random_seed=context["seed"],
            hyperparameters={
                "pretrained_model": context["pretrained_model"],
                "epochs": epochs,
                "batch_size": context["batch_size"],
                "max_length": context["max_length"],
                "learning_rate": float(training.get("learning_rate", 2e-5)),
                "mask_entity": context["mask_entity"],
            },
            metrics=metrics,
            checkpoint_path=str(artifact_dir),
        )

    # 中文注释：加载指定模型版本，在对应验证/测试集上计算关系分类指标。
    def evaluate(self, model_version: str) -> EvaluationResult:
        """加载版本化权重，在 test/validation JSONL 上执行关系评估。"""

        if self.evaluate_runner is not None:
            return self.evaluate_runner(model_version)
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise BertEntityTrainingError(
                "评估 BERTEntity 需要安装 torch 和 transformers"
            ) from exc
        modeling = self.config.training["modeling"]
        relation_config = modeling["relation"]
        artifact_dir = Path(relation_config["checkpoint_dir"]) / model_version
        manifest = self._load_manifest(artifact_dir)
        dataset_version = str(manifest["dataset_version"])
        mapping = load_relation_mapping(
            schema=self.config.schema_config,
            relation_map_path=artifact_dir / "relation_map.json",
        )
        checkpoint = self._load_checkpoint(torch, artifact_dir / "model.pth.tar")
        pretrained = str(
            checkpoint.get("pretrained_model")
            or manifest["pretrained_model"]
        )
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir, use_fast=False)
        model = BertEntityForRelation(pretrained, len(mapping))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model, device = move_model_to_device(model, modeling["device"])
        loader = self._loader(
            self._split_path(dataset_version, evaluate=True),
            tokenizer,
            mapping,
            int(relation_config["max_length"]),
            bool(relation_config.get("mask_entity", False)),
            int(relation_config["batch_size"]),
            shuffle=False,
        )
        id_to_relation = {value: key for key, value in mapping.items()}
        negative = str(
            relation_config.get(
                "negative_label",
                self.config.schema_config["negative_relation"],
            )
        )
        metrics = self._evaluate_loader(
            model, loader, device, id_to_relation, negative
        )
        return EvaluationResult(
            task_type="relation",
            model_version=model_version,
            dataset_version=dataset_version,
            metrics=metrics,
            metadata={
                "dataset_path": str(
                    self._split_path(dataset_version, evaluate=True)
                )
            },
        )

    def _prepare_training(
        self, dataset_version: str
    ) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise BertEntityTrainingError(
                "训练 BERTEntity 需要安装 torch 和 transformers"
            ) from exc
        modeling = self.config.training["modeling"]
        model_config = modeling["relation"]
        training = self.config.training["relation"]["bert_entity"]
        train_path = self._split_path(dataset_version, evaluate=False)
        eval_path = self._split_path(dataset_version, evaluate=True)
        seed = int(training.get("random_seed", 42))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        pretrained = str(
            model_config.get("pretrained_model")
            or training["pretrained_model"]
        )
        mapping = load_relation_mapping(
            schema=self.config.schema_config,
            relation_map_path=model_config.get("relation_map_path"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained, use_fast=False
        )
        model = BertEntityForRelation(pretrained, len(mapping))
        model, device = move_model_to_device(model, modeling["device"])
        max_length = int(
            model_config.get("max_length", training.get("max_length", 180))
        )
        batch_size = int(
            model_config.get("batch_size", training.get("batch_size", 8))
        )
        mask_entity = bool(model_config.get("mask_entity", False))
        train_loader = self._loader(
            train_path,
            tokenizer,
            mapping,
            max_length,
            mask_entity,
            batch_size,
            shuffle=True,
        )
        eval_loader = self._loader(
            eval_path,
            tokenizer,
            mapping,
            max_length,
            mask_entity,
            batch_size,
            shuffle=False,
        )
        return (
            torch,
            tokenizer,
            model,
            train_loader,
            eval_loader,
            {
                "device": device,
                "seed": seed,
                "pretrained_model": pretrained,
                "relation_mapping": mapping,
                "id_to_relation": {
                    value: key for key, value in mapping.items()
                },
                "negative_label": str(
                    self.config.schema_config["negative_relation"]
                ),
                "batch_size": batch_size,
                "max_length": max_length,
                "mask_entity": mask_entity,
            },
        )

    @staticmethod
    def _loader(
        path: Path,
        tokenizer: Any,
        mapping: dict[str, int],
        max_length: int,
        mask_entity: bool,
        batch_size: int,
        *,
        shuffle: bool,
    ) -> Any:
        features = build_relation_features(
            BertEntityDataset(path),
            tokenizer,
            mapping,
            max_length=max_length,
            mask_entity=mask_entity,
        )
        if not features:
            raise BertEntityTrainingError(f"关系数据集为空：{path}")
        return build_dataloader(
            features,
            batch_size,
            shuffle=shuffle,
            collate_fn=collate_relation_features,
        )

    @staticmethod
    def _evaluate_loader(
        model: Any,
        loader: Any,
        device: Any,
        id_to_relation: dict[int, str],
        negative_label: str,
    ) -> dict[str, float]:
        import torch

        model.eval()
        predicted: list[str] = []
        gold: list[str] = []
        losses: list[float] = []
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(
                    batch["token"],
                    batch["att_mask"],
                    batch["pos1"],
                    batch["pos2"],
                )
                losses.append(
                    float(
                        criterion(logits, batch["label"]).detach().cpu().item()
                    )
                )
                ids = logits.argmax(-1).detach().cpu().tolist()
                labels = batch["label"].detach().cpu().tolist()
                predicted.extend(id_to_relation[int(value)] for value in ids)
                gold.extend(id_to_relation[int(value)] for value in labels)
        metrics = micro_prf(predicted, gold, negative_label)
        metrics["accuracy"] = (
            sum(p == g for p, g in zip(predicted, gold)) / len(gold)
            if gold
            else 0.0
        )
        metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
        model.train()
        return metrics

    def _split_path(self, dataset_version: str, *, evaluate: bool) -> Path:
        root = Path(self.config.training["modeling"]["datasets_root"])
        directory = root / dataset_version / "relation"
        if not evaluate:
            path = directory / "train.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"关系训练数据不存在：{path}")
            return path
        for name in ("test.jsonl", "validation.jsonl", "dev.jsonl"):
            path = directory / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"关系评估数据不存在：{directory}/"
            "test.jsonl|validation.jsonl|dev.jsonl"
        )

    def _artifact_dir(self, dataset_version: str) -> Path:
        root = Path(
            self.config.training["modeling"]["relation"]["checkpoint_dir"]
        )
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "-", dataset_version).strip("-")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return root / f"relation-{safe or 'dataset'}-{timestamp}"

    @property
    def _schema_version(self) -> str:
        return str(self.config.schema_config.get("schema_version", "unknown"))

    @staticmethod
    def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
        path = artifact_dir / "model_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"模型 manifest 不存在：{path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"关系模型权重不存在：{path}")
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise BertEntityTrainingError(f"关系 checkpoint 格式错误：{path}")
        return checkpoint

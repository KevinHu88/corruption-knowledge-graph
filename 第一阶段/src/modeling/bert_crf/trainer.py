"""BERT-CRF 的配置驱动训练、评估与 artifact 保存。"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ProjectConfig, load_project_config
from models import EvaluationResult, TrainingResult
from src.modeling.common.device import move_model_to_device
from src.modeling.common.label_mapping import load_ner_label_mapping
from .dataset import (
    BertCrfDataset,
    build_dataloader,
    build_ner_features,
    load_ner_records,
)
from .model import BertCrfForNer

logger = logging.getLogger(__name__)


class BertCrfTrainingError(RuntimeError):
    """BERT-CRF 训练配置、数据或 checkpoint 不可用。"""


# 中文注释：BERT-CRF 训练器，负责准备数据、真实训练循环、评估和本地 artifact 保存。
class BertCrfTrainer:
    """执行真实 BERT-CRF 训练；仍支持注入 runner 以便测试和扩展。"""

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

    # 中文注释：读取指定版本 BIO 数据集，完成多 epoch 训练、梯度裁剪、评估并保存权重。
    def train(self, dataset_version: str) -> TrainingResult:
        """从版本化 JSONL 数据集训练模型并保存真实权重。"""

        if self.train_runner is not None:
            return self.train_runner(dataset_version)
        torch, tokenizer, model, train_loader, eval_loader, context = (
            self._prepare_training(dataset_version)
        )
        training = self.config.training["ner"]
        optimizer = self._optimizer(torch, model, training)
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
                loss = model(**batch)[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                steps += 1

        metrics = self._evaluate_loader(model, eval_loader, context["device"])
        metrics["train_loss"] = total_loss / steps if steps else 0.0
        artifact_dir = self._artifact_dir(dataset_version, "ner")
        artifact_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(artifact_dir)
        tokenizer.save_pretrained(artifact_dir)
        (artifact_dir / "label_map.json").write_text(
            json.dumps(
                context["label_mapping"], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        logger.info("BERT-CRF 训练完成：%s", artifact_dir)
        return TrainingResult(
            task_type="ner",
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
            },
            metrics=metrics,
            checkpoint_path=str(artifact_dir),
        )

    # 中文注释：根据模型 manifest 找回对应数据集，在 validation/test 集上计算 NER 指标。
    def evaluate(self, model_version: str) -> EvaluationResult:
        """加载版本目录及其 manifest，在 test/validation JSONL 上评估。"""

        if self.evaluate_runner is not None:
            return self.evaluate_runner(model_version)
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise BertCrfTrainingError(
                "评估 BERT-CRF 需要安装 torch 和 transformers"
            ) from exc
        modeling = self.config.training["modeling"]
        artifact_dir = Path(modeling["ner"]["checkpoint_dir"]) / model_version
        manifest = self._load_manifest(artifact_dir)
        dataset_version = str(manifest["dataset_version"])
        dataset_path = self._split_path(dataset_version, "ner", evaluate=True)
        mapping = load_ner_label_mapping(
            label_map_path=artifact_dir / "label_map.json"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir, use_fast=True
        )
        model = BertCrfForNer.from_pretrained(artifact_dir)
        model, device = move_model_to_device(model, modeling["device"])
        features = build_ner_features(
            load_ner_records(dataset_path),
            tokenizer,
            mapping,
            max_length=int(modeling["ner"]["max_length"]),
            stride=int(modeling["ner"].get("stride", 0)),
        )
        loader = build_dataloader(
            BertCrfDataset(features),
            int(self.config.training["ner"].get("eval_batch_size", 4)),
            shuffle=False,
        )
        metrics = self._evaluate_loader(model, loader, device)
        return EvaluationResult(
            task_type="ner",
            model_version=model_version,
            dataset_version=dataset_version,
            metrics=metrics,
            metadata={"dataset_path": str(dataset_path)},
        )

    def _prepare_training(
        self, dataset_version: str
    ) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise BertCrfTrainingError(
                "训练 BERT-CRF 需要安装 torch 和 transformers"
            ) from exc
        modeling = self.config.training["modeling"]
        model_config = modeling["ner"]
        training = self.config.training["ner"]
        train_path = self._split_path(
            dataset_version, "ner", evaluate=False
        )
        eval_path = self._split_path(
            dataset_version, "ner", evaluate=True
        )
        seed = int(training.get("random_seed", 42))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        pretrained = str(
            model_config.get("pretrained_model")
            or training["pretrained_model"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained, use_fast=True
        )
        mapping = load_ner_label_mapping(
            label_order=model_config.get("label_order"),
            label_map_path=model_config.get("label_map_path"),
        )
        model = BertCrfForNer.from_pretrained(
            pretrained, num_labels=len(mapping)
        )
        model, device = move_model_to_device(model, modeling["device"])
        max_length = int(
            model_config.get("max_length", training.get("max_length", 128))
        )
        stride = int(model_config.get("stride", 0))
        train_features = build_ner_features(
            load_ner_records(train_path),
            tokenizer,
            mapping,
            max_length=max_length,
            stride=stride,
        )
        if not train_features:
            raise BertCrfTrainingError("NER 训练集为空")
        eval_features = build_ner_features(
            load_ner_records(eval_path),
            tokenizer,
            mapping,
            max_length=max_length,
            stride=stride,
        )
        batch_size = int(
            model_config.get(
                "batch_size", training.get("train_batch_size", 4)
            )
        )
        eval_batch_size = int(training.get("eval_batch_size", batch_size))
        return (
            torch,
            tokenizer,
            model,
            build_dataloader(
                BertCrfDataset(train_features),
                batch_size,
                shuffle=True,
            ),
            build_dataloader(
                BertCrfDataset(eval_features),
                eval_batch_size,
                shuffle=False,
            ),
            {
                "device": device,
                "seed": seed,
                "pretrained_model": pretrained,
                "label_mapping": mapping,
                "batch_size": batch_size,
                "max_length": max_length,
            },
        )

    @staticmethod
    def _optimizer(torch: Any, model: Any, config: Mapping[str, Any]) -> Any:
        head_names = ("classifier", "crf")
        encoder, head = [], []
        for name, parameter in model.named_parameters():
            (head if any(item in name for item in head_names) else encoder).append(
                parameter
            )
        return torch.optim.AdamW(
            [
                {
                    "params": encoder,
                    "lr": float(config.get("learning_rate", 2e-5)),
                },
                {
                    "params": head,
                    "lr": float(
                        config.get(
                            "task_layer_learning_rate",
                            config.get("learning_rate", 2e-5),
                        )
                    ),
                },
            ],
            weight_decay=float(config.get("weight_decay", 0.01)),
        )

    @staticmethod
    def _evaluate_loader(
        model: Any, loader: Any, device: Any
    ) -> dict[str, float]:
        import torch

        model.eval()
        losses: list[float] = []
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                loss, emissions = model(**batch)
                losses.append(float(loss.detach().cpu().item()))
                paths = model.decode(emissions, batch["attention_mask"])
                for path, gold, mask in zip(
                    paths, batch["labels"], batch["attention_mask"]
                ):
                    length = int(mask.sum().item())
                    predicted = torch.tensor(path[:length], device=device)
                    expected = gold[:length]
                    correct += int((predicted == expected).sum().item())
                    total += length
        model.train()
        return {
            "loss": sum(losses) / len(losses) if losses else 0.0,
            "token_accuracy": correct / total if total else 0.0,
        }

    def _split_path(
        self, dataset_version: str, task: str, *, evaluate: bool
    ) -> Path:
        root = Path(self.config.training["modeling"]["datasets_root"])
        directory = root / dataset_version / task
        if not evaluate:
            path = directory / "train.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"NER 训练数据不存在：{path}")
            return path
        for name in ("test.jsonl", "validation.jsonl", "dev.jsonl"):
            path = directory / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"NER 评估数据不存在：{directory}/"
            "test.jsonl|validation.jsonl|dev.jsonl"
        )

    def _artifact_dir(self, dataset_version: str, prefix: str) -> Path:
        root = Path(
            self.config.training["modeling"]["ner"]["checkpoint_dir"]
        )
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "-", dataset_version).strip("-")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return root / f"{prefix}-{safe or 'dataset'}-{timestamp}"

    @property
    def _schema_version(self) -> str:
        return str(self.config.schema_config.get("schema_version", "unknown"))

    @staticmethod
    def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
        path = artifact_dir / "model_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"模型 manifest 不存在：{path}")
        return json.loads(path.read_text(encoding="utf-8"))

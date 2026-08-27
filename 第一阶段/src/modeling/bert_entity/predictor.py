"""BERTEntity 统一关系预测器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from models import EntityPrediction, RelationPrediction
from src.modeling.common.device import move_model_to_device
from src.modeling.common.label_mapping import load_schema
from .candidate_builder import RelationCandidate, RelationCandidateBuilder
from .dataset import encode_entity_pair


class RelationModelNotLoadedError(RuntimeError):
    """关系模型尚未配置、依赖缺失或权重不兼容。"""


# 中文注释：关系模型推理器，负责加载 checkpoint、批量编码候选并输出关系预测。
class BertEntityPredictor:
    """复用 legacy marker/位置编码并输出正式关系预测。"""

    def __init__(
        self,
        *,
        relation_mapping: Mapping[str, int],
        checkpoint_path: str | Path | None = None,
        pretrained_model: str | None = None,
        tokenizer: Any = None,
        model: Any = None,
        classifier: Callable[
            [str, RelationCandidate], tuple[str, float]
        ] | None = None,
        candidate_builder: RelationCandidateBuilder | None = None,
        schema: dict[str, Any] | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        mask_entity: bool = False,
        device: str = "auto",
        model_version: str = "unversioned",
    ) -> None:
        self.schema = load_schema(schema)
        self.relation_mapping = dict(relation_mapping)
        self.id2relation = {
            value: key for key, value in self.relation_mapping.items()
        }
        self.negative_label = str(self.schema["negative_relation"])
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.model = model
        self.classifier = classifier
        self.candidate_builder = (
            candidate_builder or RelationCandidateBuilder(self.schema)
        )
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.mask_entity = mask_entity
        self.device_name = device
        self.device: Any = None
        self.model_version = model_version

    @property
    def loaded(self) -> bool:
        return self.classifier is not None or (
            self.model is not None and self.tokenizer is not None
        )

    # 中文注释：从本地 manifest/checkpoint 加载模型、tokenizer 和关系映射。
    def load(self) -> None:
        """一次性加载 legacy state_dict，并严格校验分类维度。"""

        if self.loaded:
            return
        if self.checkpoint_path is None:
            raise RelationModelNotLoadedError(
                "未配置 BERTEntity checkpoint_path"
            )
        if not self.checkpoint_path.is_file():
            raise RelationModelNotLoadedError(
                f"BERTEntity checkpoint 不存在：{self.checkpoint_path}"
            )
        if not self.pretrained_model:
            raise RelationModelNotLoadedError("未配置 pretrained_model")
        try:
            import torch
            from transformers import AutoTokenizer
            from .model import BertEntityForRelation
        except ImportError as exc:
            raise RelationModelNotLoadedError(
                "BERTEntity 运行需要 torch 和 transformers"
            ) from exc
        state = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        classifier = state_dict.get("fc.weight")
        if classifier is None or classifier.shape[0] != len(
            self.relation_mapping
        ):
            actual = classifier.shape[0] if classifier is not None else None
            raise RelationModelNotLoadedError(
                f"关系 checkpoint 分类维度为 {actual}，"
                f"当前正式映射需要 {len(self.relation_mapping)}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model, use_fast=False
        )
        self.model = BertEntityForRelation(
            self.pretrained_model, len(self.relation_mapping)
        )
        self.model.load_state_dict(state_dict)
        self.model, self.device = move_model_to_device(
            self.model, self.device_name
        )
        self.model.eval()

    def predict(
        self,
        text: str,
        entities: list[EntityPrediction],
    ) -> list[RelationPrediction]:
        """构造候选对并预测，过滤“无关系”。"""

        return self.predict_candidates(
            text, self.candidate_builder.build(entities)
        )

    # 中文注释：批量分类实体对候选，过滤负类后返回带置信度和证据范围的关系预测。
    def predict_candidates(
        self,
        text: str,
        candidates: list[RelationCandidate],
    ) -> list[RelationPrediction]:
        """预测已构造候选对，保留实体引用和证据字符区间。"""

        if not self.loaded:
            self.load()
        output: list[RelationPrediction] = []
        predictions: list[tuple[str, float]] = []
        if self.classifier is not None:
            predictions = [
                self.classifier(text, candidate) for candidate in candidates
            ]
        else:
            for start in range(0, len(candidates), self.batch_size):
                predictions.extend(self._classify_batch(
                    text,
                    candidates[start:start + self.batch_size],
                ))

        for candidate, (relation, confidence) in zip(
            candidates, predictions
        ):
            if relation == self.negative_label:
                continue
            output.append(RelationPrediction(
                relation_id=f"r{len(output) + 1}",
                head_id=candidate.head.entity_id,
                tail_id=candidate.tail.entity_id,
                relation_type=relation,
                confidence=confidence,
                evidence_start=min(
                    candidate.head.start, candidate.tail.start
                ),
                evidence_end=max(candidate.head.end, candidate.tail.end),
            ))
        return output

    def _classify(
        self, text: str, candidate: RelationCandidate
    ) -> tuple[str, float]:
        if self.classifier is not None:
            return self.classifier(text, candidate)
        return self._classify_batch(text, [candidate])[0]

    def _classify_batch(
        self,
        text: str,
        candidates: list[RelationCandidate],
    ) -> list[tuple[str, float]]:
        """Encode one candidate chunk and run a single batched forward pass."""

        if not candidates:
            return []
        import torch
        encoded = [
            encode_entity_pair(
                text,
                candidate.head,
                candidate.tail,
                self.tokenizer,
                self.max_length,
                mask_entity=self.mask_entity,
            )
            for candidate in candidates
        ]
        inputs = [
            torch.tensor(
                [item[name] for item in encoded],
                device=self.device,
                dtype=torch.long,
            )
            for name in ("token", "att_mask")
        ]
        inputs.extend(
            torch.tensor(
                [[item[name]] for item in encoded],
                device=self.device,
                dtype=torch.long,
            )
            for name in ("pos1", "pos2")
        )
        with torch.no_grad():
            probabilities = torch.softmax(self.model(*inputs), -1)
            scores, labels = probabilities.max(-1)
        return [
            (self.id2relation[int(label)], float(score))
            for label, score in zip(labels.tolist(), scores.tolist())
        ]

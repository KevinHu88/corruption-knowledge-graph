"""BERT-CRF 长文本实体预测及 token/字符偏移转换。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from models import EntityPrediction
from src.modeling.common.device import move_model_to_device
from src.modeling.common.offset_mapping import (
    CharacterSpan,
    TokenLabel,
    decode_bio_offsets,
    deduplicate_spans,
)

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """预测器尚未配置或加载权重。"""


# 中文注释：BERT-CRF 推理器，负责加载本地 artifact、滑窗推理和跨窗口实体合并。
class BertCrfPredictor:
    """复用 legacy 标签顺序、使用 fast tokenizer offset 的 NER 预测器。"""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        label_mapping: Mapping[str, int],
        tokenizer: Any = None,
        model: Any = None,
        max_length: int = 512,
        stride: int = 64,
        device: str = "auto",
        model_version: str = "unversioned",
        window_decoder: Callable[
            [dict[str, Any]], tuple[list[int], list[float]]
        ] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.label_mapping = dict(label_mapping)
        self.id2label = {value: key for key, value in self.label_mapping.items()}
        self.allowed_types = {"PER", "ORG", "POSITION", "MONEY"}
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.stride = stride
        self.device_name = device
        self.device: Any = None
        self.model_version = model_version
        self.window_decoder = window_decoder

    @property
    def loaded(self) -> bool:
        """是否已有可执行 tokenizer 和解码器/模型。"""

        return self.tokenizer is not None and (
            self.window_decoder is not None or self.model is not None
        )

    # 中文注释：依据 ModelManifest 加载权重、tokenizer 和标签映射，并移动到目标设备。
    def load(self) -> None:
        """从配置路径一次性加载 tokenizer、模型和权重。"""

        if self.loaded:
            return
        if self.checkpoint_path is None:
            raise ModelNotLoadedError("未配置 BERT-CRF checkpoint_path")
        if not self.checkpoint_path.is_dir():
            raise ModelNotLoadedError(
                f"BERT-CRF checkpoint 目录不存在：{self.checkpoint_path}"
            )
        try:
            from transformers import AutoConfig, AutoTokenizer
            from .model import BertCrfForNer
        except ImportError as exc:
            raise ModelNotLoadedError(
                "BERT-CRF 运行需要 torch 和 transformers"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, use_fast=True
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ModelNotLoadedError("BERT-CRF 必须使用 fast tokenizer")
        config = AutoConfig.from_pretrained(
            self.checkpoint_path,
            num_labels=len(self.label_mapping),
        )
        self.model = BertCrfForNer.from_pretrained(
            self.checkpoint_path, config=config
        )
        self.model, self.device = move_model_to_device(
            self.model, self.device_name
        )
        self.model.eval()

    # 中文注释：对长文本使用 tokenizer overflow/stride 分窗，解码后合并为全文字符实体。
    def predict(self, text: str) -> list[EntityPrediction]:
        """识别实体并保证所有结果满足原文左闭右开字符偏移。"""

        if not text:
            return []
        if not self.loaded:
            self.load()
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            padding=False,
        )
        windows = self._windows(encoded)
        spans: list[CharacterSpan] = []
        for window in windows:
            label_ids, confidences = self._decode_window(window)
            offsets = window["offset_mapping"]
            token_labels = [
                TokenLabel(
                    label=self.id2label.get(int(label_id), "O"),
                    start=int(offset[0]),
                    end=int(offset[1]),
                    confidence=float(confidence),
                )
                for label_id, confidence, offset in zip(
                    label_ids, confidences, offsets
                )
                if tuple(offset) != (0, 0)
            ]
            spans.extend(decode_bio_offsets(token_labels, self.allowed_types))
        clean = deduplicate_spans(spans)
        return [
            EntityPrediction(
                entity_id=f"e{index}",
                name=text[span.start:span.end],
                entity_type=span.entity_type,
                start=span.start,
                end=span.end,
                confidence=span.confidence,
            )
            for index, span in enumerate(clean, start=1)
            if text[span.start:span.end]
        ]

    @staticmethod
    def _windows(encoded: Mapping[str, Any]) -> list[dict[str, Any]]:
        offsets = encoded["offset_mapping"]
        nested = bool(offsets and isinstance(offsets[0], Sequence)
                      and offsets[0] and isinstance(offsets[0][0], Sequence))
        count = len(offsets) if nested else 1
        windows = []
        for index in range(count):
            windows.append({
                key: (value[index] if nested else value)
                for key, value in encoded.items()
                if key != "overflow_to_sample_mapping"
            })
        return windows

    def _decode_window(
        self, window: dict[str, Any]
    ) -> tuple[list[int], list[float]]:
        if self.window_decoder is not None:
            return self.window_decoder(window)
        try:
            import torch
        except ImportError as exc:
            raise ModelNotLoadedError("缺少 PyTorch") from exc
        inputs = {
            key: torch.tensor([value], device=self.device)
            for key, value in window.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            emissions = self.model(**inputs)[0]
            path = self.model.decode(
                emissions, inputs.get("attention_mask")
            )[0]
            probabilities = torch.softmax(emissions[0], -1)
            scores = [
                float(probabilities[index, label].item())
                for index, label in enumerate(path)
            ]
        return list(path), scores

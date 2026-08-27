"""BERT-CRF 数据读取、字符标签对齐与 DataLoader 构建。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ALLOWED_ENTITY_TYPES = {"PER", "ORG", "POSITION", "MONEY"}


class NerDatasetError(ValueError):
    """NER 数据格式或字符区间不合法。"""


# 中文注释：BERT-CRF 的内存特征数据集，为 PyTorch DataLoader 提供单条张量样本。
class BertCrfDataset:
    """保存已完成 tokenizer/标签对齐的 NER 特征。"""

    def __init__(self, features: Sequence[dict[str, Any]]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def load_ner_records(path: str | Path) -> list[dict[str, Any]]:
    """读取每行包含 text 和 entities 的 UTF-8 JSONL 数据集。"""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"NER 数据集不存在：{target}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NerDatasetError(
                f"NER JSONL 第 {line_number} 行不是合法 JSON：{target}"
            ) from exc
        if not isinstance(record, Mapping):
            raise NerDatasetError(f"NER JSONL 第 {line_number} 行必须是对象")
        records.append(dict(record))
    return records


# 中文注释：将 BIO JSONL 记录编码为带 overflow 窗口、offset 和对齐标签的模型特征。
def build_ner_features(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    label_mapping: Mapping[str, int],
    *,
    max_length: int,
    stride: int = 0,
) -> list[dict[str, list[int]]]:
    """利用 fast tokenizer 的 offset_mapping 将字符实体对齐到 BIO 标签。"""

    if not getattr(tokenizer, "is_fast", False):
        raise NerDatasetError("NER 字符偏移训练必须使用 fast tokenizer")
    outside_id = label_mapping.get("O")
    if outside_id is None:
        raise NerDatasetError("NER label mapping 缺少 O 标签")

    features: list[dict[str, list[int]]] = []
    for record_index, record in enumerate(records):
        text = record.get("text")
        entities = record.get("entities", [])
        if not isinstance(text, str) or not text:
            raise NerDatasetError(
                f"NER 第 {record_index + 1} 条记录缺少非空 text"
            )
        normalized = _normalize_entities(text, entities, record_index)
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            padding="max_length",
        )
        input_windows = _as_windows(encoded["input_ids"])
        attention_windows = _as_windows(encoded["attention_mask"])
        offset_windows = _as_offset_windows(encoded["offset_mapping"])
        token_type_windows = (
            _as_windows(encoded["token_type_ids"])
            if "token_type_ids" in encoded
            else [[0] * len(ids) for ids in input_windows]
        )
        for input_ids, attention, offsets, token_types in zip(
            input_windows,
            attention_windows,
            offset_windows,
            token_type_windows,
        ):
            labels = _labels_for_window(
                offsets, normalized, label_mapping, outside_id
            )
            features.append(
                {
                    "input_ids": [int(value) for value in input_ids],
                    "attention_mask": [int(value) for value in attention],
                    "token_type_ids": [int(value) for value in token_types],
                    "labels": labels,
                }
            )
    return features


def _normalize_entities(
    text: str,
    entities: Any,
    record_index: int,
) -> list[dict[str, Any]]:
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        raise NerDatasetError(
            f"NER 第 {record_index + 1} 条记录的 entities 必须是数组"
        )
    normalized: list[dict[str, Any]] = []
    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise NerDatasetError("NER entity 必须是对象")
        entity_type = str(entity.get("entity_type") or entity.get("type") or "")
        if entity_type not in ALLOWED_ENTITY_TYPES:
            raise NerDatasetError(f"不支持的实体类型：{entity_type}")
        try:
            start, end = int(entity["start"]), int(entity["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NerDatasetError("NER entity 缺少合法 start/end") from exc
        name = str(entity.get("name", text[start:end]))
        if start < 0 or end <= start or end > len(text) or text[start:end] != name:
            raise NerDatasetError(
                f"NER entity 字符偏移不匹配：record={record_index + 1}, "
                f"entity={entity_index + 1}"
            )
        normalized.append(
            {"start": start, "end": end, "entity_type": entity_type}
        )
    return sorted(normalized, key=lambda item: (item["start"], item["end"]))


def _labels_for_window(
    offsets: Sequence[Sequence[int]],
    entities: Sequence[Mapping[str, Any]],
    label_mapping: Mapping[str, int],
    outside_id: int,
) -> list[int]:
    labels: list[int] = []
    seen_entities: set[int] = set()
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        if start == end:
            labels.append(outside_id)
            continue
        matched_index = next(
            (
                index
                for index, entity in enumerate(entities)
                if start >= entity["start"] and end <= entity["end"]
            ),
            None,
        )
        if matched_index is None:
            labels.append(outside_id)
            continue
        entity_type = str(entities[matched_index]["entity_type"])
        prefix = "I" if matched_index in seen_entities else "B"
        label = f"{prefix}-{entity_type}"
        if label not in label_mapping:
            raise NerDatasetError(f"NER label mapping 缺少 {label}")
        labels.append(int(label_mapping[label]))
        seen_entities.add(matched_index)
    return labels


def _as_windows(values: Any) -> list[list[Any]]:
    values = values.tolist() if hasattr(values, "tolist") else values
    if not values:
        return []
    return values if isinstance(values[0], list) else [values]


def _as_offset_windows(values: Any) -> list[list[list[int]]]:
    values = values.tolist() if hasattr(values, "tolist") else values
    if not values:
        return []
    return (
        values
        if isinstance(values[0][0], (list, tuple))
        else [values]
    )


def collate_ner_features(
    batch: Sequence[Mapping[str, Sequence[int]]],
) -> dict[str, Any]:
    """将预处理特征转换为 long tensor。"""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("构建 NER batch 需要安装 torch") from exc
    return {
        key: torch.tensor([item[key] for item in batch], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "token_type_ids", "labels")
    }


# 中文注释：根据训练配置创建 NER DataLoader，并使用专用 collate 组合变长特征。
def build_dataloader(
    dataset: BertCrfDataset,
    batch_size: int,
    *,
    shuffle: bool,
) -> Any:
    """延迟导入 PyTorch 并创建 DataLoader。"""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("构建 NER DataLoader 需要安装 torch") from exc
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ner_features,
    )

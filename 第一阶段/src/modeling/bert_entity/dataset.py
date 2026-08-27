"""OpenNRE 兼容 JSONL 数据集与实体 marker 编码。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from models import EntityPrediction

HEAD_START, HEAD_END = "[unused0]", "[unused1]"
TAIL_START, TAIL_END = "[unused2]", "[unused3]"


class RelationInputError(ValueError):
    """实体区间或 marker 编码无法满足模型输入约束。"""


# 中文注释：把头尾实体边界编码为特殊标记和位置索引，供关系模型感知实体位置。
def encode_entity_pair(
    text: str,
    head: EntityPrediction,
    tail: EntityPrediction,
    tokenizer: Any,
    max_length: int,
    *,
    mask_entity: bool = False,
) -> dict[str, list[int] | int]:
    """复用 legacy [unused0..3] 方案，并始终保留两个实体。"""

    if text[head.start:head.end] != head.name:
        raise RelationInputError(f"头实体偏移不匹配：{head.entity_id}")
    if text[tail.start:tail.end] != tail.name:
        raise RelationInputError(f"尾实体偏移不匹配：{tail.entity_id}")
    if head.start == tail.start or not (
        head.end <= tail.start or tail.end <= head.start
    ):
        raise RelationInputError("BERTEntity 不支持重叠实体对")

    first, second = (
        (head, tail) if head.start < tail.start else (tail, head)
    )
    before = tokenizer.tokenize(text[:first.start])
    middle = tokenizer.tokenize(text[first.end:second.start])
    after = tokenizer.tokenize(text[second.end:])

    def marked(entity: EntityPrediction) -> list[str]:
        if mask_entity:
            return ["[unused4]" if entity is head else "[unused5]"]
        tokens = tokenizer.tokenize(text[entity.start:entity.end])
        if entity is head:
            return [HEAD_START, *tokens, HEAD_END]
        return [TAIL_START, *tokens, TAIL_END]

    first_tokens, second_tokens = marked(first), marked(second)
    fixed = 2 + len(first_tokens) + len(middle) + len(second_tokens)
    if fixed > max_length:
        raise RelationInputError(
            "两个实体及其中间文本超过 max_length，无法保留 legacy marker"
        )
    context_budget = max_length - fixed
    left_budget = min(len(before), context_budget // 2)
    right_budget = min(len(after), context_budget - left_budget)
    remaining = context_budget - left_budget - right_budget
    left_budget = min(len(before), left_budget + remaining)
    tokens = [
        "[CLS]",
        *(before[-left_budget:] if left_budget else []),
        *first_tokens,
        *middle,
        *second_tokens,
        *after[:right_budget],
        "[SEP]",
    ]
    pos1 = tokens.index("[unused4]" if mask_entity else HEAD_START)
    pos2 = tokens.index("[unused5]" if mask_entity else TAIL_START)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    attention = [1] * len(ids)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    ids += [pad_id] * (max_length - len(ids))
    attention += [0] * (max_length - len(attention))
    return {
        "token": ids,
        "att_mask": attention,
        "pos1": pos1,
        "pos2": pos2,
    }


# 中文注释：读取关系 JSONL 并保存标准化记录，为关系模型特征构建提供输入。
class BertEntityDataset:
    """读取 OpenNRE text/h/t/relation JSONL。"""

    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"关系数据集不存在：{target}")
        self.records = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def encode_relation_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    relation_mapping: Mapping[str, int],
    *,
    max_length: int,
    mask_entity: bool = False,
) -> dict[str, list[int] | int]:
    """将 OpenNRE text/h/t/relation 记录编码为 BERTEntity 训练特征。"""

    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise RelationInputError("关系记录缺少非空 text")
    head = _argument_to_entity(text, record.get("h"), "head")
    tail = _argument_to_entity(text, record.get("t"), "tail")
    relation = str(record.get("relation", ""))
    if relation not in relation_mapping:
        raise RelationInputError(f"关系标签不在正式 mapping 中：{relation}")
    feature = encode_entity_pair(
        text,
        head,
        tail,
        tokenizer,
        max_length,
        mask_entity=mask_entity,
    )
    feature["label"] = int(relation_mapping[relation])
    return feature


# 中文注释：将关系样本批量编码为 token、attention mask、实体位置和分类标签。
def build_relation_features(
    dataset: BertEntityDataset,
    tokenizer: Any,
    relation_mapping: Mapping[str, int],
    *,
    max_length: int,
    mask_entity: bool = False,
) -> list[dict[str, list[int] | int]]:
    """预编码关系数据集，尽早暴露实体偏移和标签错误。"""

    return [
        encode_relation_record(
            record,
            tokenizer,
            relation_mapping,
            max_length=max_length,
            mask_entity=mask_entity,
        )
        for record in dataset.records
    ]


def _argument_to_entity(
    text: str, argument: Any, role: str
) -> EntityPrediction:
    if not isinstance(argument, Mapping):
        raise RelationInputError(f"关系记录缺少 {role} 实体")
    position = argument.get("pos")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 2
    ):
        raise RelationInputError(f"{role}.pos 必须是 [start, end]")
    start, end = int(position[0]), int(position[1])
    name = str(argument.get("name", text[start:end]))
    entity_type = str(
        argument.get("entity_type") or argument.get("type") or ""
    )
    try:
        return EntityPrediction(
            entity_id=str(argument.get("id") or f"{role}-{start}-{end}"),
            name=name,
            entity_type=entity_type,
            start=start,
            end=end,
            confidence=1.0,
        )
    except Exception as exc:
        raise RelationInputError(f"{role} 实体字段不合法：{argument}") from exc


def collate_relation_features(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """将 BERTEntity 特征转换为模型所需 long tensor。"""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("构建关系 batch 需要安装 torch") from exc
    return {
        "token": torch.tensor(
            [item["token"] for item in batch], dtype=torch.long
        ),
        "att_mask": torch.tensor(
            [item["att_mask"] for item in batch], dtype=torch.long
        ),
        "pos1": torch.tensor(
            [[item["pos1"]] for item in batch], dtype=torch.long
        ),
        "pos2": torch.tensor(
            [[item["pos2"]] for item in batch], dtype=torch.long
        ),
        "label": torch.tensor(
            [item["label"] for item in batch], dtype=torch.long
        ),
    }


def build_dataloader(
    dataset: Any,
    batch_size: int,
    *,
    shuffle: bool,
    collate_fn: Any = None,
) -> Any:
    """延迟导入 PyTorch 并创建关系分类 DataLoader。"""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("构建关系 DataLoader 需要安装 torch") from exc
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

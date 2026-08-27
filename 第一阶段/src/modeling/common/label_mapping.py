"""从单一配置源加载实体与关系标签映射。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import load_yaml


class LabelMappingError(ValueError):
    """标签映射缺失、重复或与 schema 不一致。"""


def load_schema(schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """加载并校验项目关系 schema。"""

    data = dict(schema or load_yaml("schema.yaml"))
    if not isinstance(data.get("entity_types"), Mapping):
        raise LabelMappingError("schema.yaml 缺少 entity_types")
    if not isinstance(data.get("relation_types"), Mapping):
        raise LabelMappingError("schema.yaml 缺少 relation_types")
    if not data.get("negative_relation"):
        raise LabelMappingError("schema.yaml 缺少 negative_relation")
    return data


def _load_json_mapping(path: str | Path) -> dict[str, int]:
    target = Path(path)
    if not target.is_file():
        raise LabelMappingError(f"标签映射文件不存在：{target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise LabelMappingError(f"标签映射必须是 JSON 对象：{target}")
    return {str(key): int(value) for key, value in data.items()}


# 中文注释：加载或根据 schema 生成 BIO 标签映射，保证训练和推理使用同一标签编号。
def load_ner_label_mapping(
    *,
    label_order: Sequence[str] | None = None,
    label_map_path: str | Path | None = None,
) -> dict[str, int]:
    """按 artifact 映射或 training.yaml 中的 legacy 顺序加载 NER 标签。"""

    if label_map_path:
        mapping = _load_json_mapping(label_map_path)
    else:
        if not label_order:
            raise LabelMappingError("NER 缺少 label_order 或 label_map_path")
        mapping = {str(label): index for index, label in enumerate(label_order)}
    if len(set(mapping.values())) != len(mapping):
        raise LabelMappingError("NER 标签 ID 存在重复")
    return mapping


def relation_mapping_from_schema(
    schema: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """生成“无关系=0，正关系按 schema 顺序排列”的正式映射。"""

    data = load_schema(schema)
    labels = [
        str(data["negative_relation"]),
        *(str(name) for name in data["relation_types"]),
    ]
    return {label: index for index, label in enumerate(labels)}


# 中文注释：加载或根据 schema 生成关系类别映射，并确保负类包含在分类空间中。
def load_relation_mapping(
    *,
    schema: Mapping[str, Any] | None = None,
    relation_map_path: str | Path | None = None,
    require_schema_match: bool = True,
) -> dict[str, int]:
    """加载关系映射，并可严格要求其与正式 schema 完全一致。"""

    formal = relation_mapping_from_schema(schema)
    mapping = (
        _load_json_mapping(relation_map_path)
        if relation_map_path
        else formal
    )
    if require_schema_match and mapping != formal:
        missing = sorted(set(formal) - set(mapping))
        extra = sorted(set(mapping) - set(formal))
        raise LabelMappingError(
            f"关系映射与正式 schema 不一致；missing={missing}, extra={extra}"
        )
    return mapping

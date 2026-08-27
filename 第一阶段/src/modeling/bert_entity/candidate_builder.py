"""依据 schema.yaml 构造合法候选实体对。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models import EntityPrediction
from src.modeling.common.label_mapping import load_schema


@dataclass(frozen=True)
# 中文注释：单个关系分类候选，保存原文以及头尾实体预测。
class RelationCandidate:
    """送入 BERTEntity 的有向候选实体对。"""

    head: EntityPrediction
    tail: EntityPrediction
    allowed_relations: tuple[str, ...]


# 中文注释：根据 schema 中允许的头尾实体类型和方向生成关系分类候选对。
class RelationCandidateBuilder:
    """从单一 schema 配置推导实体类型组合与候选方向。"""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = load_schema(schema)
        self.rules = self.schema["relation_types"]

    # 中文注释：遍历实体组合、排除非法类型/方向后输出稳定排序的候选列表。
    def build(
        self,
        entities: list[EntityPrediction],
    ) -> list[RelationCandidate]:
        """生成不同实体之间至少支持一种正式关系的候选对。"""

        candidates: list[RelationCandidate] = []
        for head in entities:
            for tail in entities:
                if head.entity_id == tail.entity_id:
                    continue
                allowed = tuple(
                    relation
                    for relation, rule in self.rules.items()
                    if head.entity_type in rule["head_types"]
                    and tail.entity_type in rule["tail_types"]
                )
                if not allowed:
                    continue
                # 纯无向组合仅按字符顺序保留一次；存在有向规则则保留双向。
                has_directional = any(
                    bool(self.rules[name].get("directional", True))
                    for name in allowed
                )
                if not has_directional and head.start > tail.start:
                    continue
                candidates.append(RelationCandidate(head, tail, allowed))
        return candidates

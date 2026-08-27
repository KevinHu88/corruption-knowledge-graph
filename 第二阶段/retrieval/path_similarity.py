"""知识图谱路径签名与轻量相似度评分。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence


PathOrientation = Literal["forward", "reversed"]


@dataclass(frozen=True, slots=True)
class PathSignature:
    """只保留可跨实体比较的路径结构，不把姓名纳入相似度。"""

    relation_types: tuple[str, ...]
    entity_types: tuple[str, ...]
    directions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.relation_types:
            raise ValueError("路径至少需要一条关系")
        if len(self.entity_types) != len(self.relation_types) + 1:
            raise ValueError("实体类型数量必须比关系数量多 1")
        if self.directions and len(self.directions) != len(self.relation_types):
            raise ValueError("关系方向数量必须与关系数量一致")

    @property
    def relation_tokens(self) -> tuple[str, ...]:
        directions = self.directions or ("forward",) * len(self.relation_types)
        return tuple(
            f"{direction}:{relation_type}"
            for relation_type, direction in zip(
                self.relation_types,
                directions,
                strict=True,
            )
        )

    def reversed(self) -> "PathSignature":
        inverse = {"forward": "reverse", "reverse": "forward"}
        directions = self.directions or ("forward",) * len(self.relation_types)
        return PathSignature(
            relation_types=tuple(reversed(self.relation_types)),
            entity_types=tuple(reversed(self.entity_types)),
            directions=tuple(
                inverse.get(direction, direction)
                for direction in reversed(directions)
            ),
        )


@dataclass(frozen=True, slots=True)
class PathSimilarity:
    score: float
    relation_sequence_score: float
    entity_type_sequence_score: float
    relation_overlap_score: float
    length_score: float
    orientation: PathOrientation = "forward"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "score": self.score,
            "relation_sequence_score": self.relation_sequence_score,
            "entity_type_sequence_score": self.entity_type_sequence_score,
            "relation_overlap_score": self.relation_overlap_score,
            "length_score": self.length_score,
            "orientation": self.orientation,
        }


class PathSimilarityScorer:
    """用序列编辑相似度和多重集合重合度比较两条路径。"""

    def __init__(
        self,
        *,
        relation_sequence_weight: float = 0.50,
        entity_type_sequence_weight: float = 0.20,
        relation_overlap_weight: float = 0.20,
        length_weight: float = 0.10,
    ) -> None:
        weights = (
            relation_sequence_weight,
            entity_type_sequence_weight,
            relation_overlap_weight,
            length_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("路径相似度权重必须非负且总和大于 0")
        total = sum(weights)
        self.relation_sequence_weight = relation_sequence_weight / total
        self.entity_type_sequence_weight = entity_type_sequence_weight / total
        self.relation_overlap_weight = relation_overlap_weight / total
        self.length_weight = length_weight / total

    def score(
        self,
        anchor: PathSignature,
        candidate: PathSignature,
        *,
        allow_reverse: bool = True,
    ) -> PathSimilarity:
        direct = self._score_oriented(anchor, candidate, "forward")
        if not allow_reverse:
            return direct
        reversed_score = self._score_oriented(
            anchor,
            candidate.reversed(),
            "reversed",
        )
        return reversed_score if reversed_score.score > direct.score else direct

    def _score_oriented(
        self,
        anchor: PathSignature,
        candidate: PathSignature,
        orientation: PathOrientation,
    ) -> PathSimilarity:
        relation_sequence = self._sequence_similarity(
            anchor.relation_tokens,
            candidate.relation_tokens,
        )
        entity_type_sequence = self._sequence_similarity(
            anchor.entity_types,
            candidate.entity_types,
        )
        relation_overlap = self._multiset_jaccard(
            anchor.relation_types,
            candidate.relation_types,
        )
        length_score = min(
            len(anchor.relation_types),
            len(candidate.relation_types),
        ) / max(len(anchor.relation_types), len(candidate.relation_types))
        score = (
            relation_sequence * self.relation_sequence_weight
            + entity_type_sequence * self.entity_type_sequence_weight
            + relation_overlap * self.relation_overlap_weight
            + length_score * self.length_weight
        )
        return PathSimilarity(
            score=round(score, 6),
            relation_sequence_score=round(relation_sequence, 6),
            entity_type_sequence_score=round(entity_type_sequence, 6),
            relation_overlap_score=round(relation_overlap, 6),
            length_score=round(length_score, 6),
            orientation=orientation,
        )

    @classmethod
    def _sequence_similarity(
        cls,
        left: Sequence[str],
        right: Sequence[str],
    ) -> float:
        if not left and not right:
            return 1.0
        maximum = max(len(left), len(right))
        if maximum == 0:
            return 1.0
        return 1.0 - cls._levenshtein_distance(left, right) / maximum

    @staticmethod
    def _levenshtein_distance(
        left: Sequence[str],
        right: Sequence[str],
    ) -> int:
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for left_index, left_value in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_value in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1]
                        + (left_value != right_value),
                    )
                )
            previous = current
        return previous[-1]

    @staticmethod
    def _multiset_jaccard(
        left: Sequence[str],
        right: Sequence[str],
    ) -> float:
        left_counts = Counter(left)
        right_counts = Counter(right)
        union = sum((left_counts | right_counts).values())
        if union == 0:
            return 1.0
        intersection = sum((left_counts & right_counts).values())
        return intersection / union

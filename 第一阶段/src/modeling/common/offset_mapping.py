"""BIO token 标签到原文字符区间的可靠转换。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLabel:
    """单个有效 token 的标签、字符区间和置信度。"""

    label: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class CharacterSpan:
    """合并后的实体字符区间。"""

    entity_type: str
    start: int
    end: int
    confidence: float


# 中文注释：把 token 级 BIO 标签和 tokenizer offset 还原为全文字符实体区间。
def decode_bio_offsets(
    tokens: Sequence[TokenLabel],
    allowed_types: set[str],
) -> list[CharacterSpan]:
    """合并 BIO；孤立或类型不匹配的 I 标签按 B 标签修复。"""

    spans: list[CharacterSpan] = []
    current_type: str | None = None
    current_start = 0
    current_end = 0
    scores: list[float] = []

    def flush() -> None:
        nonlocal current_type, current_start, current_end, scores
        if current_type is not None and current_end > current_start:
            spans.append(
                CharacterSpan(
                    entity_type=current_type,
                    start=current_start,
                    end=current_end,
                    confidence=sum(scores) / len(scores),
                )
            )
        current_type = None
        scores = []

    for token in tokens:
        if token.end <= token.start:
            continue
        prefix, separator, entity_type = token.label.partition("-")
        if (
            not separator
            or entity_type not in allowed_types
            or prefix not in {"B", "I"}
        ):
            flush()
            continue
        can_continue = (
            prefix == "I"
            and current_type == entity_type
            and token.start <= current_end
        )
        if not can_continue:
            flush()
            current_type = entity_type
            current_start = token.start
            current_end = token.end
            scores = [token.confidence]
        else:
            current_end = max(current_end, token.end)
            scores.append(token.confidence)
    flush()
    return spans


# 中文注释：合并跨滑动窗口产生的重复实体跨度，优先保留置信度更高的预测。
def deduplicate_spans(
    spans: Sequence[CharacterSpan],
) -> list[CharacterSpan]:
    """对长文本重叠窗口中的相同实体保留最高置信度结果。"""

    unique: dict[tuple[str, int, int], CharacterSpan] = {}
    for span in spans:
        key = (span.entity_type, span.start, span.end)
        previous = unique.get(key)
        if previous is None or span.confidence > previous.confidence:
            unique[key] = span
    return sorted(
        unique.values(),
        key=lambda item: (item.start, item.end, item.entity_type),
    )

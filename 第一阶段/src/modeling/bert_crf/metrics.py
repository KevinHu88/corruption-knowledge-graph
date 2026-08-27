"""NER 指标计算。"""

from __future__ import annotations

from collections.abc import Iterable


def entity_prf(
    predicted: Iterable[tuple[str, int, int]],
    gold: Iterable[tuple[str, int, int]],
) -> dict[str, float]:
    """计算实体级 precision、recall、F1。"""

    pred, truth = set(predicted), set(gold)
    correct = len(pred & truth)
    precision = correct / len(pred) if pred else 0.0
    recall = correct / len(truth) if truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

"""关系分类指标。"""

from __future__ import annotations

from collections.abc import Sequence


def micro_prf(
    predicted: Sequence[str],
    gold: Sequence[str],
    negative_label: str,
) -> dict[str, float]:
    """计算排除“无关系”后的 micro P/R/F1。"""

    correct = sum(
        p == g and g != negative_label for p, g in zip(predicted, gold)
    )
    pred_positive = sum(p != negative_label for p in predicted)
    gold_positive = sum(g != negative_label for g in gold)
    precision = correct / pred_positive if pred_positive else 0.0
    recall = correct / gold_positive if gold_positive else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

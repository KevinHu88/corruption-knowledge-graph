"""关键词检索和离线向量共享的轻量文本特征。"""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    terms = re.findall(r"[a-z0-9_]+", lowered)
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        terms.extend(run)
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return terms

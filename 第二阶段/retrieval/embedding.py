"""文档向量检索使用的可插拔 Embedding 客户端。"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

from dotenv import dotenv_values

from 第二阶段.config import FIRST_STAGE_DIR
from 第二阶段.retrieval.text_features import tokenize


class EmbeddingClient(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class HashingEmbeddingClient:
    """无需模型下载的确定性稀疏哈希向量，供离线运行和测试使用。"""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 必须大于 0")
        self.dimensions = dimensions

    @property
    def cache_namespace(self) -> str:
        return f"hashing-v1-{self.dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        frequencies = Counter(tokenize(text))
        for term, frequency in frequencies.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(frequency))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class FirstStageEmbeddingClient:
    """读取第一阶段 Embedding 配置并调用 OpenAI 兼容接口。"""

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        model_type: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        values = dotenv_values(Path(FIRST_STAGE_DIR) / ".env")
        self.api_key = (
            api_key
            or os.getenv("EMBED_API_KEY")
            or str(values.get("EMBED_API_KEY") or "")
        ).strip()
        self.model = (
            model
            or os.getenv("EMBED_MODEL_NAME")
            or str(values.get("EMBED_MODEL_NAME") or "")
        ).strip()
        self.model_type = (
            model_type
            or os.getenv("EMBED_MODEL_TYPE")
            or str(values.get("EMBED_MODEL_TYPE") or "openai")
        ).strip().lower()
        configured_base_url = (
            base_url
            or os.getenv("EMBED_BASE_URL")
            or str(values.get("EMBED_BASE_URL") or "")
        ).strip()
        if not configured_base_url and self.model_type == "dashscope":
            configured_base_url = self.DASHSCOPE_BASE_URL
        self.base_url = configured_base_url
        if not self.api_key:
            raise ValueError("缺少 EMBED_API_KEY")
        if not self.model:
            raise ValueError("缺少 EMBED_MODEL_NAME")
        if client is not None:
            self._client = client
            self._owns_client = False
            return
        from openai import OpenAI

        options: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": 0,
        }
        if self.base_url:
            options["base_url"] = self.base_url
        self._client = OpenAI(**options)
        self._owns_client = True

    @property
    def cache_namespace(self) -> str:
        return f"external-{self.model_type}-{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model,
            input=list(texts),
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding 服务返回的向量数量不匹配")
        return vectors

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

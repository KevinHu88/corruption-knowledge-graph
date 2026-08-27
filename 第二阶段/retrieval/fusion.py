"""文档与图谱证据的合并、去重和简单排序。"""

from __future__ import annotations

import hashlib

from 第二阶段.schemas.models import Evidence


class EvidenceFusion:
    def fuse(
        self,
        document_evidence: list[Evidence],
        graph_evidence: list[Evidence],
        *,
        limit: int = 12,
    ) -> list[Evidence]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        deduplicated: dict[str, Evidence] = {}
        content_keys: dict[str, str] = {}
        for item in [*document_evidence, *graph_evidence]:
            content_key = hashlib.sha256(item.content.strip().encode("utf-8")).hexdigest()
            key = item.id or content_key
            existing_key = key if key in deduplicated else content_keys.get(content_key)
            existing = deduplicated.get(existing_key) if existing_key else None
            if existing is None or (item.score or 0.0) > (existing.score or 0.0):
                if existing_key and existing_key != key:
                    deduplicated.pop(existing_key, None)
                deduplicated[key] = item
                content_keys[content_key] = key
        result = list(deduplicated.values())
        result.sort(
            key=lambda item: (
                -(item.score or 0.0),
                0 if item.source_type == "graph" else 1,
                item.id,
            )
        )
        return result[:limit]

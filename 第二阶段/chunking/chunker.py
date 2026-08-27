"""基础字符/段落边界切块器。"""

from __future__ import annotations

import hashlib

from 第二阶段.schemas.models import Chunk, ParsedDocument


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须位于 0..chunk_size-1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            target_end = min(len(text), start + self.chunk_size)
            end = self._paragraph_boundary(text, start, target_end)
            content = text[start:end].strip()
            if content:
                digest = hashlib.sha256(
                    f"{document.document_id}:{index}:{start}:{end}".encode("utf-8")
                ).hexdigest()[:24]
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk-{digest}",
                        document_id=document.document_id,
                        content=content,
                        metadata={
                            **document.metadata,
                            "file_name": document.file_name,
                            "file_type": document.file_type,
                            "chunk_index": index,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                index += 1
            if end >= len(text):
                break
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end
        return chunks

    def _paragraph_boundary(self, text: str, start: int, target_end: int) -> int:
        if target_end >= len(text):
            return len(text)
        minimum = start + max(1, self.chunk_size // 2)
        candidates = [
            text.rfind("\n\n", minimum, target_end),
            text.rfind("\n", minimum, target_end),
            text.rfind("。", minimum, target_end),
        ]
        boundary = max(candidates)
        return boundary + (1 if boundary >= 0 else 0) if boundary >= minimum else target_end


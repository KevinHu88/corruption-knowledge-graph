"""可替换为 Redis/数据库的内存 Session 文档存储。"""

from __future__ import annotations

import hashlib

from 第二阶段.schemas.models import Chunk, ParsedDocument, UploadedDocument


class SessionDocumentStore:
    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self._uploaded: dict[str, UploadedDocument] = {}
        self._parsed: dict[str, ParsedDocument] = {}
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, dict[str, tuple[str, list[float]]]] = {}

    def add_document(self, document: UploadedDocument) -> None:
        self._uploaded[document.document_id] = document

    def add_parsed_document(self, document: ParsedDocument) -> None:
        self._parsed[document.document_id] = document

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def get_chunks(self, document_id: str | None = None) -> list[Chunk]:
        chunks = list(self._chunks.values())
        if document_id is not None:
            chunks = [item for item in chunks if item.document_id == document_id]
        return sorted(chunks, key=lambda item: (item.document_id, item.metadata.get("chunk_index", 0)))

    def get_documents(self) -> list[UploadedDocument]:
        return list(self._uploaded.values())

    def get_chunk_vector(
        self, namespace: str, chunk: Chunk
    ) -> list[float] | None:
        cached = self._vectors.get(namespace, {}).get(chunk.chunk_id)
        if cached is None:
            return None
        content_hash, vector = cached
        if content_hash != self._content_hash(chunk.content):
            return None
        return list(vector)

    def set_chunk_vector(
        self, namespace: str, chunk: Chunk, vector: list[float]
    ) -> None:
        self._vectors.setdefault(namespace, {})[chunk.chunk_id] = (
            self._content_hash(chunk.content),
            list(vector),
        )

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        self._uploaded.clear()
        self._parsed.clear()
        self._chunks.clear()
        self._vectors.clear()

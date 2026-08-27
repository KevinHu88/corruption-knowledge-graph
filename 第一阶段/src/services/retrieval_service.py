"""Domain mapping and failure policy for web retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from models import RawDocument
from src.services.tavily_service import TavilyRequestError, TavilyService


class RetrievalBatch(BaseModel):
    """Project-facing result for one configured source."""

    source_id: str
    documents: list[RawDocument] = Field(default_factory=list)
    searched_count: int = 0
    extracted_count: int = 0
    skipped_count: int = 0
    errors: list[str] = Field(default_factory=list)


class RetrievalService:
    """Compose Tavily operations and return only stable domain models."""

    def __init__(self, tavily: TavilyService) -> None:
        self.tavily = tavily

    def retrieve_source(
        self,
        source: Mapping[str, Any],
        *,
        today: date | None = None,
        extract_missing_content: bool = True,
        continue_on_extract_error: bool = True,
    ) -> RetrievalBatch:
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("retrieval source requires a non-empty source_id")
        results = self.tavily.search_source(source, today=today)
        extracted_by_url: dict[str, str] = {}
        errors: list[str] = []
        extracted_count = 0
        missing = [
            str(item.get("canonical_url") or item.get("url") or "")
            for item in results
            if not (item.get("raw_content") or item.get("content"))
        ]
        if extract_missing_content and missing:
            try:
                for start in range(0, len(missing), 20):
                    extracted = self.tavily.extract(missing[start:start + 20])
                    for item in extracted:
                        url = str(item.get("canonical_url") or item.get("url"))
                        content = str(item.get("raw_content") or "").strip()
                        if content:
                            extracted_by_url[url] = content
                            extracted_count += 1
            except TavilyRequestError as exc:
                if not continue_on_extract_error:
                    raise
                errors.append(f"source={source_id} extract failed: {exc}")

        documents: list[RawDocument] = []
        skipped = 0
        seen_content: set[str] = set()
        for item in results:
            canonical_url = str(
                item.get("canonical_url") or item.get("url") or ""
            ).strip()
            content = str(
                item.get("raw_content")
                or item.get("content")
                or extracted_by_url.get(canonical_url)
                or ""
            ).strip()
            if not canonical_url or not content:
                skipped += 1
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in seen_content:
                skipped += 1
                continue
            seen_content.add(content_hash)
            url_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
            documents.append(
                RawDocument(
                    doc_id=f"retrieval-{url_hash[:24]}",
                    doc_version_id=f"sha256:{content_hash}",
                    source_id=source_id,
                    source_url=canonical_url,
                    title=str(item.get("title") or "").strip() or None,
                    published_at=item.get("published_at"),
                    content_type="text/html",
                    raw_text=content,
                    content_sha256=content_hash,
                    metadata={
                        "retrieval_backend": "tavily",
                        "query": item.get("query"),
                        "matched_keyword": item.get("matched_keyword"),
                        "score": item.get("score"),
                        "request_id": item.get("request_id"),
                    },
                )
            )
        return RetrievalBatch(
            source_id=source_id,
            documents=documents,
            searched_count=len(results),
            extracted_count=extracted_count,
            skipped_count=skipped,
            errors=errors,
        )


__all__ = ["RetrievalBatch", "RetrievalService"]

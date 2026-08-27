"""Prefect flow that retrieves configured sources as RawDocument values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from prefect import flow
from pydantic import BaseModel, Field

from config import load_yaml
from models import RawDocument
from task.retrieval_tasks import retrieve_source_task


class RetrievalFlowResult(BaseModel):
    raw_documents: list[RawDocument] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    searched_count: int = 0
    skipped_count: int = 0
    errors: list[str] = Field(default_factory=list)


@flow(name="retrieval-flow")
def retrieval_flow(
    *,
    source_ids: Sequence[str] | None = None,
    sources: Sequence[Mapping[str, Any]] | None = None,
    today: date | None = None,
    extract_missing_content: bool = True,
    continue_on_error: bool = False,
) -> RetrievalFlowResult:
    """Select enabled sources, preserve partial results only when requested."""

    configured = list(sources or load_yaml("sources.yaml").get("sources", []))
    selected_ids = set(source_ids or ())
    selected = [
        item
        for item in configured
        if item.get("enabled", True)
        and (not selected_ids or str(item.get("source_id")) in selected_ids)
    ]
    if selected_ids:
        found = {str(item.get("source_id")) for item in selected}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"unknown or disabled retrieval sources: {missing}")

    documents: dict[str, RawDocument] = {}
    searched_count = 0
    skipped_count = 0
    errors: list[str] = []
    completed_sources: list[str] = []
    for source in selected:
        source_id = str(source.get("source_id"))
        try:
            batch = retrieve_source_task(
                source,
                today=today,
                extract_missing_content=extract_missing_content,
                continue_on_extract_error=continue_on_error,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            errors.append(f"source={source_id} failed: {exc}")
            continue
        completed_sources.append(source_id)
        searched_count += batch.searched_count
        skipped_count += batch.skipped_count
        errors.extend(batch.errors)
        for document in batch.documents:
            documents.setdefault(document.doc_id, document)
    return RetrievalFlowResult(
        raw_documents=list(documents.values()),
        source_ids=completed_sources,
        searched_count=searched_count,
        skipped_count=skipped_count,
        errors=errors,
    )


__all__ = ["RetrievalFlowResult", "retrieval_flow"]

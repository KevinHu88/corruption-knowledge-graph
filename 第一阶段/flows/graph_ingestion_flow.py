"""Prefect flow for Neo4j ingestion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prefect import flow

from models import CanonicalAnnotation, CaseDocument, SourceDocument
from src.services.neo4j_service import Neo4jBatchResult
from task.graph_tasks import graph_ingestion_task


# 中文注释：图谱写入编排，将规范标注及案件/来源映射交给 Neo4j Task 批量落库。
@flow(name="graph-ingestion-flow")
def graph_ingestion_flow(
    annotations: Sequence[CanonicalAnnotation],
    *,
    source_documents: Mapping[
        str, SourceDocument | Mapping[str, Any]
    ] | None = None,
    case_documents: Mapping[
        str, CaseDocument | Mapping[str, Any]
    ] | None = None,
    entity_uid_maps: Mapping[str, Mapping[str, str]] | None = None,
    continue_on_error: bool = False,
) -> Neo4jBatchResult:
    """Delegate graph normalization, validation, and upserts to the service."""

    return graph_ingestion_task(
        annotations,
        source_documents=source_documents,
        case_documents=case_documents,
        entity_uid_maps=entity_uid_maps,
        continue_on_error=continue_on_error,
    )


__all__ = ["graph_ingestion_flow"]

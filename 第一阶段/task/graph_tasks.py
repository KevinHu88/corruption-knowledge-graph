"""Thin Prefect wrapper around Neo4jService ingestion."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from prefect import task
from prefect.logging import get_run_logger

from models import CanonicalAnnotation, CaseDocument, SourceDocument
from src.services.neo4j_service import (
    Neo4jBatchResult,
    Neo4jConnectionError,
    Neo4jService,
    Neo4jWriteError,
)


# 中文注释：仅对 Neo4j 连接或瞬态写入异常重试，避免永久性 schema/数据错误被反复执行。
def _retry_transient_neo4j_error(task, task_run, state) -> bool:
    """Retry connection/session failures and driver transient write errors."""

    del task, task_run
    try:
        state.result()
    except Neo4jConnectionError:
        return True
    except Neo4jWriteError as exc:
        cause = exc.__cause__
        code = str(getattr(cause, "code", ""))
        return (
            type(cause).__name__ == "TransientError"
            or code.startswith("Neo.TransientError")
        )
    except Exception:
        return False
    return False


# 中文注释：Neo4j 写入的 Prefect 边界，任务结束时始终关闭本次创建的 Driver。
@task(
    name="graph-ingestion",
    retries=2,
    retry_delay_seconds=10,
    retry_condition_fn=_retry_transient_neo4j_error,
    timeout_seconds=1800,
)
def graph_ingestion_task(
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
    """Ingest canonical annotations and always close the task-local driver."""

    logger = get_run_logger()
    started = time.perf_counter()
    items = list(annotations)
    logger.info(
        "step=graph-ingestion input_annotations=%d continue_on_error=%s",
        len(items),
        continue_on_error,
    )
    service: Neo4jService | None = None
    try:
        service = Neo4jService()
        result = service.ingest_annotations_batch(
            items,
            source_documents=source_documents,
            case_documents=case_documents,
            entity_uid_maps=entity_uid_maps,
            continue_on_error=continue_on_error,
        )
    except Exception:
        logger.exception(
            "step=graph-ingestion status=failed input_annotations=%d",
            len(items),
        )
        raise
    finally:
        if service is not None:
            service.close()
    counters = result.counters
    logger.info(
        "step=graph-ingestion success=%d failed=%d skipped=%d "
        "nodes_created=%d relationships_created=%d output=neo4j "
        "elapsed=%.3fs",
        result.successful_batches,
        result.failed_batches,
        counters.records_skipped,
        counters.nodes_created,
        counters.relationships_created,
        time.perf_counter() - started,
    )
    return result


__all__ = ["graph_ingestion_task"]

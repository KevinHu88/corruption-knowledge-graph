"""Prefect boundary for recoverable retrieval requests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from prefect import task
from prefect.logging import get_run_logger

from config import EnvironmentSettings
from src.services.retrieval_service import RetrievalBatch, RetrievalService
from src.services.tavily_service import TavilyRequestError, TavilyService


def _retry_tavily_request(task, task_run, state) -> bool:
    del task, task_run
    try:
        state.result()
    except TavilyRequestError:
        return True
    except Exception:
        return False
    return False


@task(
    name="retrieve-source",
    retries=2,
    retry_delay_seconds=5,
    retry_condition_fn=_retry_tavily_request,
    timeout_seconds=180,
)
def retrieve_source_task(
    source: Mapping[str, Any],
    *,
    today: date | None = None,
    extract_missing_content: bool = True,
    continue_on_extract_error: bool = True,
) -> RetrievalBatch:
    """Run one source with Prefect owning retries and service cleanup."""

    logger = get_run_logger()
    source_id = str(source.get("source_id") or "<unknown>")
    logger.info("step=retrieval source_id=%s", source_id)
    settings = EnvironmentSettings()
    service = TavilyService(
        api_key=settings.tavily_api_key,
        timeout=float(source.get("timeout_seconds", 30)),
        max_retries=0,
    )
    try:
        result = RetrievalService(service).retrieve_source(
            source,
            today=today,
            extract_missing_content=extract_missing_content,
            continue_on_extract_error=continue_on_extract_error,
        )
    finally:
        service.close()
    logger.info(
        "step=retrieval source_id=%s documents=%d skipped=%d errors=%d",
        source_id,
        len(result.documents),
        result.skipped_count,
        len(result.errors),
    )
    return result


__all__ = ["retrieve_source_task"]

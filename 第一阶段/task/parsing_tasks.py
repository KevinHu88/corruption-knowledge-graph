"""Prefect wrappers for the asynchronous text-processing service."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prefect import task
from prefect.logging import get_run_logger

from models import ProcessedCase, RawDocument
from src.services.llm_service import LLMService
from src.services.text_processing_service import TextProcessingService


# 中文注释：异步文档处理任务，完成解析、清洗、相关性过滤和模型输入切块。
@task(
    name="process-documents",
    timeout_seconds=1800,
)
async def process_documents_task(
    documents: Sequence[RawDocument],
) -> list[ProcessedCase]:
    """Process raw documents without moving service instances between tasks."""

    logger = get_run_logger()
    started = time.perf_counter()
    items = list(documents)
    logger.info("step=text-processing input_documents=%d", len(items))
    llm = None
    try:
        # workflow.yaml 启用了模糊相关性窗口的 LLM 复核；服务实例必须在
        # 当前 task 内创建和关闭，避免 uncertain 窗口永远无法进入切块。
        llm = LLMService()
        result = await TextProcessingService(
            llm_service=llm
        ).process_documents(items)
    except Exception:
        logger.exception(
            "step=text-processing status=failed input_documents=%d",
            len(items),
        )
        raise
    finally:
        if llm is not None:
            llm.close()
    failed = sum(item.processing_status == "failed" for item in result)
    skipped = sum(item.processing_status == "irrelevant" for item in result)
    logger.info(
        "step=text-processing success=%d failed=%d skipped=%d elapsed=%.3fs",
        len(result) - failed - skipped,
        failed,
        skipped,
        time.perf_counter() - started,
    )
    return result


__all__ = ["process_documents_task"]

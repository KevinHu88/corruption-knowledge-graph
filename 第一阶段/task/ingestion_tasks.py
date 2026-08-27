"""Input-side tasks used by the top-level ingestion flow."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prefect import task
from prefect.logging import get_run_logger

from models import ModelExtractionResult
from src.services.inference_service import InferenceService


# 中文注释：本地双模型推理任务，对文本块依次执行实体识别和关系分类。
# 中文注释：输入批次规模不固定，不使用固定一小时截止长文档推理。
@task(name="model-inference")
def inference_task(texts: Sequence[str]) -> list[ModelExtractionResult]:
    """Run the existing end-to-end NER→RE service for a text batch."""

    logger = get_run_logger()
    started = time.perf_counter()
    items = list(texts)
    total = len(items)
    logger.info("step=model-inference input_samples=%d", total)
    service = InferenceService()
    try:
        logger.info("step=model-inference status=loading-models")
        load_started = time.perf_counter()
        service.load()
        logger.info(
            "step=model-inference status=models-loaded elapsed=%.3fs",
            time.perf_counter() - load_started,
        )

        result: list[ModelExtractionResult] = []
        for index, text in enumerate(items, start=1):
            item_started = time.perf_counter()
            logger.info(
                "step=model-inference progress=%d/%d status=started chars=%d",
                index,
                total,
                len(text),
            )
            try:
                extraction = service.extract(text)
            except Exception:
                logger.exception(
                    "step=model-inference progress=%d/%d status=failed "
                    "chars=%d elapsed=%.3fs",
                    index,
                    total,
                    len(text),
                    time.perf_counter() - item_started,
                )
                raise
            result.append(extraction)
            logger.info(
                "step=model-inference progress=%d/%d status=completed "
                "entities=%d relations=%d elapsed=%.3fs",
                index,
                total,
                len(extraction.entities),
                len(extraction.relations),
                time.perf_counter() - item_started,
            )
    except Exception:
        logger.exception(
            "step=model-inference status=failed input_samples=%d",
            total,
        )
        raise
    logger.info(
        "step=model-inference success=%d failed=0 skipped=0 elapsed=%.3fs",
        len(result),
        time.perf_counter() - started,
    )
    return result


__all__ = ["inference_task"]

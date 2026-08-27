"""Inference task progress logging tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import task.ingestion_tasks as ingestion_tasks


def test_inference_task_has_no_fixed_batch_timeout():
    assert ingestion_tasks.inference_task.timeout_seconds is None


def test_inference_task_logs_each_item_progress(monkeypatch):
    logger = MagicMock()
    service = MagicMock()
    expected = [
        SimpleNamespace(entities=[object()], relations=[]),
        SimpleNamespace(entities=[], relations=[object()]),
    ]
    service.extract.side_effect = expected
    service_factory = MagicMock(return_value=service)
    monkeypatch.setattr(ingestion_tasks, "get_run_logger", lambda: logger)
    monkeypatch.setattr(ingestion_tasks, "InferenceService", service_factory)

    result = ingestion_tasks.inference_task.fn(["第一条", "第二条"])

    service.load.assert_called_once_with()
    assert result == expected
    assert service.extract.call_count == 2
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert "step=model-inference status=loading-models" in messages
    assert any("progress=%d/%d status=started" in item for item in messages)
    assert any("progress=%d/%d status=completed" in item for item in messages)
    progress_values = [
        call.args[1:3]
        for call in logger.info.call_args_list
        if "progress=%d/%d" in call.args[0]
    ]
    assert progress_values == [(1, 2), (1, 2), (2, 2), (2, 2)]

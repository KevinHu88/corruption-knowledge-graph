"""文本处理 Prefect task 的 LLM 相关性复核接线测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import task.parsing_tasks as parsing_tasks


class StubLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def test_process_documents_task_injects_and_closes_llm(monkeypatch):
    llm = SimpleNamespace(closed=False)
    llm.close = lambda: setattr(llm, "closed", True)
    captured = {}

    class Service:
        def __init__(self, *, llm_service):
            captured["llm_service"] = llm_service

        async def process_documents(self, items):
            captured["items"] = items
            return []

    monkeypatch.setattr(parsing_tasks, "LLMService", lambda: llm)
    monkeypatch.setattr(parsing_tasks, "TextProcessingService", Service)
    monkeypatch.setattr(
        parsing_tasks, "get_run_logger", lambda: StubLogger()
    )

    result = asyncio.run(parsing_tasks.process_documents_task.fn([]))

    assert result == []
    assert captured == {"llm_service": llm, "items": []}
    assert llm.closed is True


def test_process_documents_task_closes_llm_on_error(monkeypatch):
    llm = SimpleNamespace(closed=False)
    llm.close = lambda: setattr(llm, "closed", True)

    class Service:
        def __init__(self, *, llm_service):
            assert llm_service is llm

        async def process_documents(self, items):
            del items
            raise RuntimeError("processing failed")

    monkeypatch.setattr(parsing_tasks, "LLMService", lambda: llm)
    monkeypatch.setattr(parsing_tasks, "TextProcessingService", Service)
    monkeypatch.setattr(
        parsing_tasks, "get_run_logger", lambda: StubLogger()
    )

    try:
        asyncio.run(parsing_tasks.process_documents_task.fn([]))
    except RuntimeError as exc:
        assert str(exc) == "processing failed"
    else:
        raise AssertionError("expected RuntimeError")

    assert llm.closed is True

"""统一 LLM 接口、第一阶段 Wrapper 与测试 Mock。"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from typing import Any

from 第二阶段.config import FIRST_STAGE_DIR


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """生成最终回答文本。"""


class MockLLMClient(LLMClient):
    """无网络、结果确定的测试客户端。"""

    def __init__(self, response: str = "Mock answer based on supplied evidence.") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FirstStageLLMClient(LLMClient):
    """复用第一阶段 LLMService.generate_text 的第二阶段 Wrapper。"""

    def __init__(self, service: Any | None = None) -> None:
        self._service = service
        self._owns_service = service is None

    @property
    def service(self) -> Any:
        if self._service is None:
            first_stage_path = str(FIRST_STAGE_DIR)
            if first_stage_path not in sys.path:
                sys.path.insert(0, first_stage_path)
            module = importlib.import_module("src.services.llm_service")
            self._service = module.LLMService()
        return self._service

    def generate(self, prompt: str) -> str:
        response = self.service.generate_text(
            "你是一个严格依据给定证据回答问题的助手。",
            prompt,
        )
        return str(response.content)

    def close(self) -> None:
        if self._owns_service and self._service is not None:
            self._service.close()
            self._service = None


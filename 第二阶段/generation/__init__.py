"""Prompt 与 LLM 统一调用接口。"""

from 第二阶段.generation.llm_client import (
    FirstStageLLMClient,
    LLMClient,
    MockLLMClient,
)
from 第二阶段.generation.prompt_builder import PromptBuilder

__all__ = ["FirstStageLLMClient", "LLMClient", "MockLLMClient", "PromptBuilder"]


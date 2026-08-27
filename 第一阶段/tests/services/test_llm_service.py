"""LLMService 单元测试；所有模型响应均来自 mock 客户端。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

from config import EnvironmentSettings
from src.services.llm_service import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMParseError,
    LLMResponseError,
    LLMService,
    LLMValidationError,
)


class StructuredResult(BaseModel):
    relevant: bool
    score: float = Field(ge=0, le=1)
    reason: str


def make_completion(
    content: str | None,
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    total_tokens: int = 18,
):
    return SimpleNamespace(
        id="chatcmpl-test",
        _request_id="req-test",
        model="qwen-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def make_service(
    response=None,
    *,
    side_effect=None,
    clock=None,
):
    client = MagicMock()
    client.chat.completions.create.return_value = response
    client.chat.completions.create.side_effect = side_effect
    settings = EnvironmentSettings(
        llm_api_key="test-key",
        llm_base_url=(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        llm_model_id="qwen-test",
        llm_request_timeout=30,
        llm_temperature=0.1,
        llm_max_tokens=512,
    )
    return LLMService(
        settings=settings,
        client=client,
        clock=clock or (lambda: 1.0),
    )


def test_plain_text_response():
    service = make_service(make_completion("普通文本"))

    response = service.generate_text("系统提示", "用户提示")

    assert response.content == "普通文本"
    assert response.provider == "qwen"
    assert response.model_name == "qwen-test"


def test_structured_response_from_plain_json():
    service = make_service(
        make_completion(
            '{"relevant": true, "score": 0.9, "reason": "相关"}'
        )
    )

    result = service.generate_structured(
        "系统提示", "用户提示", StructuredResult
    )

    assert isinstance(result, StructuredResult)
    assert result.relevant is True
    assert result.score == 0.9


def test_responses_api_structured_output_contract():
    client = MagicMock()
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed={
            "relevant": True,
            "score": 0.88,
            "reason": "存在明确关系",
        }
    )
    settings = EnvironmentSettings(
        llm_api_key="test-key",
        llm_model_id="gpt-5.4-mini",
        llm_request_timeout=30,
        llm_temperature=0.1,
        llm_max_tokens=512,
        llm_structured_api="auto",
    )
    service = LLMService(settings=settings, client=client)

    result = service.generate_structured_response(
        "系统提示",
        "用户提示",
        StructuredResult,
        max_tokens=256,
    )

    assert result.score == 0.88
    client.responses.parse.assert_called_once_with(
        model="gpt-5.4-mini",
        instructions="系统提示",
        input="用户提示",
        text_format=StructuredResult,
        max_output_tokens=256,
        reasoning={"effort": "none"},
        store=False,
        timeout=30.0,
    )


def test_structured_output_5xx_falls_back_to_responses_json():
    class ProxyServerError(RuntimeError):
        status_code = 500

    client = MagicMock()
    client.responses.parse.side_effect = ProxyServerError("unexpected EOF")
    client.responses.create.return_value = SimpleNamespace(
        output_text=(
            '{"relevant":true,"score":0.91,'
            '"reason":"存在明确案件事实"}'
        )
    )
    settings = EnvironmentSettings(
        llm_api_key="test-key",
        llm_model_id="gpt-5.4-mini",
        llm_request_timeout=30,
        llm_max_tokens=512,
        llm_structured_api="auto",
    )
    service = LLMService(settings=settings, client=client)

    result = service.generate_structured_response(
        "系统提示", "用户提示", StructuredResult
    )

    assert result.relevant is True
    assert result.score == 0.91
    call = client.responses.create.call_args.kwargs
    assert call["model"] == "gpt-5.4-mini"
    assert call["input"] == "用户提示"
    assert "JSON Schema" in call["instructions"]
    assert call["reasoning"] == {"effort": "none"}


def test_responses_5xx_falls_back_to_chat_completions_json():
    class ProxyServerError(RuntimeError):
        status_code = 500

    client = MagicMock()
    client.responses.parse.side_effect = ProxyServerError("unexpected EOF")
    client.responses.create.side_effect = ProxyServerError("unexpected EOF")
    client.chat.completions.create.return_value = make_completion(
        '{"relevant":true,"score":0.93,"reason":"包含案件事实"}'
    )
    settings = EnvironmentSettings(
        llm_api_key="test-key",
        llm_base_url="http://127.0.0.1:8317/v1",
        llm_model_id="gpt-5.4-mini",
        llm_request_timeout=30,
        llm_max_tokens=512,
        llm_structured_api="auto",
    )
    service = LLMService(settings=settings, client=client)

    result = service.generate_structured_response(
        "系统提示", "用户提示", StructuredResult
    )

    assert result.score == 0.93
    call = client.chat.completions.create.call_args.kwargs
    assert call["model"] == "gpt-5.4-mini"
    assert call["reasoning_effort"] == "low"
    assert call["verbosity"] == "low"
    assert call["max_completion_tokens"] == 512
    assert "temperature" not in call
    assert "max_tokens" not in call
    assert "JSON Schema" in call["messages"][0]["content"]


def test_missing_responses_parse_falls_back_to_chat_completions_json():
    client = MagicMock()
    client.responses.parse = None
    client.chat.completions.create.return_value = make_completion(
        '{"relevant":false,"score":0.12,"reason":"无具体案件事实"}'
    )
    settings = EnvironmentSettings(
        llm_api_key="test-key",
        llm_model_id="gpt-5.4-mini",
        llm_request_timeout=30,
        llm_max_tokens=512,
        llm_structured_api="auto",
    )
    service = LLMService(settings=settings, client=client)

    result = service.generate_structured_response(
        "系统提示", "用户提示", StructuredResult
    )

    assert result.relevant is False
    client.chat.completions.create.assert_called_once()


def test_chat_structured_api_bypasses_responses_endpoints():
    client = MagicMock()
    client.chat.completions.create.return_value = make_completion(
        '{"relevant":true,"score":0.82,"reason":"存在案件事实"}'
    )
    settings = EnvironmentSettings(
        llm_api_key="test-key",
        llm_model_id="gpt-5.4-mini",
        llm_request_timeout=180,
        llm_max_tokens=512,
        llm_structured_api="chat",
    )
    service = LLMService(settings=settings, client=client)

    result = service.generate_structured_response(
        "系统提示", "用户提示", StructuredResult
    )

    assert result.score == 0.82
    client.responses.parse.assert_not_called()
    client.responses.create.assert_not_called()
    client.chat.completions.create.assert_called_once()


def test_parse_markdown_json_code_block():
    content = (
        "```json\n"
        '{"relevant": false, "score": 0.1, '
        '"reason": "包含中文\\n和换行"}\n'
        "```"
    )

    parsed = LLMService.parse_json_response(content)

    assert parsed["reason"] == "包含中文\n和换行"


def test_parse_json_with_surrounding_explanation_and_nested_data():
    content = (
        "以下是结果：\n"
        '{"outer": {"items": [{"text": "带有 } 和 ] 的字符串"}]}}'
        "\n以上。"
    )

    parsed = LLMService.parse_json_response(content)

    assert parsed["outer"]["items"][0]["text"] == "带有 } 和 ] 的字符串"


def test_empty_response_raises_response_error():
    service = make_service(make_completion("  "))

    with pytest.raises(LLMResponseError, match="content 为空"):
        service.generate_text("系统提示", "用户提示")


def test_missing_choices_raises_response_error():
    service = make_service(
        SimpleNamespace(
            id="chatcmpl-test",
            model="qwen-test",
            choices=[],
            usage=None,
        )
    )

    with pytest.raises(LLMResponseError, match="缺少 choices"):
        service.generate_text("系统提示", "用户提示")


def test_invalid_json_raises_parse_error():
    with pytest.raises(LLMParseError):
        LLMService.parse_json_response('说明：{"broken": ]')


def test_pydantic_validation_failure():
    service = make_service(
        make_completion(
            '{"relevant": true, "score": 3, "reason": "越界"}'
        )
    )

    with pytest.raises(LLMValidationError) as error:
        service.generate_structured(
            "系统提示", "用户提示", StructuredResult
        )

    assert isinstance(error.value.__cause__, Exception)


def test_missing_api_configuration():
    settings = EnvironmentSettings(
        llm_api_key="",
        llm_base_url="https://example.test/v1",
        llm_model_id="test-model",
    )

    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        LLMService(settings=settings)


def test_network_exception_is_converted():
    original = openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.test/v1/chat")
    )
    service = make_service(side_effect=original)

    with pytest.raises(LLMConnectionError) as error:
        service.generate_text("系统提示", "用户提示")

    assert error.value.__cause__ is original


def test_usage_latency_and_request_metadata_are_recorded():
    ticks = iter([10.0, 10.25])
    service = make_service(
        make_completion(
            "完成",
            prompt_tokens=13,
            completion_tokens=5,
            total_tokens=18,
        ),
        clock=lambda: next(ticks),
    )

    response = service.generate_text("系统提示", "用户提示")

    assert response.request_id == "req-test"
    assert response.input_tokens == 13
    assert response.output_tokens == 5
    assert response.total_tokens == 18
    assert response.latency_seconds == pytest.approx(0.25)
    assert response.created_at.tzinfo is not None
    assert service.last_response == response
    assert service.call_history == (response,)

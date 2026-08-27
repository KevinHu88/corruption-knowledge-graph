"""项目统一大语言模型调用服务。

本模块只封装 OpenAI 兼容客户端、消息发送、响应解析和调用元数据。
Prompt 渲染、业务规则校验、工作流重试及数据持久化由其他模块负责。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, NoReturn, Protocol, TypeVar, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError

from config import EnvironmentSettings, load_project_config

logger = logging.getLogger(__name__)

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
ChatMessage = Mapping[str, Any]


class LLMServiceError(RuntimeError):
    """大语言模型服务的基础异常。"""


class LLMConfigurationError(LLMServiceError):
    """客户端依赖或运行配置无效。"""


class LLMConnectionError(LLMServiceError):
    """接口连接、超时或远端状态异常。"""


class LLMRateLimitError(LLMServiceError):
    """模型服务触发限流或配额限制。"""


class LLMResponseError(LLMServiceError):
    """模型返回的数据结构或文本内容无效。"""


class LLMParseError(LLMResponseError):
    """无法从模型文本中解析合法 JSON。"""


class LLMValidationError(LLMResponseError):
    """解析后的 JSON 不符合目标 Pydantic 模型。"""


# 中文注释：统一保存一次 LLM 调用的文本、模型、token 用量和响应标识。
class LLMResponse(BaseModel):
    """统一的大语言模型文本响应与调用元数据。"""

    provider: str
    model_name: str
    request_id: str | None = None
    content: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(ge=0)
    created_at: datetime


class _ChatCompletionsProtocol(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _ChatProtocol(Protocol):
    completions: _ChatCompletionsProtocol


class OpenAICompatibleClient(Protocol):
    chat: _ChatProtocol

    def close(self) -> None: ...


def _get_value(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    """同时读取 SDK 对象属性和 mock/兼容接口字典字段。"""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _infer_provider(base_url: str) -> str:
    """根据兼容接口地址生成稳定、非敏感的 provider 名称。"""

    hostname = (urlsplit(base_url).hostname or "").lower()
    if "dashscope.aliyuncs.com" in hostname:
        return "qwen"
    if "modelscope" in hostname:
        return "modelscope"
    if hostname in {"api.openai.com", ""}:
        return "openai"
    return hostname or "openai-compatible"


# 中文注释：OpenAI 兼容大模型适配器，统一处理客户端配置、调用、异常映射和结构化输出。
class LLMService:
    """OpenAI 及 OpenAI 兼容接口的统一同步调用服务。

    生产环境默认从 :mod:`config` 读取配置。测试可注入 ``client``，
    从而避免真实网络请求。
    """

    def __init__(
        self,
        *,
        settings: EnvironmentSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        resolved_settings = settings
        if resolved_settings is None:
            resolved_settings = load_project_config().environment

        self.api_key = (
            api_key
            if api_key is not None
            else resolved_settings.llm_api_key
        ).strip()
        self.base_url = (
            base_url
            if base_url is not None
            else resolved_settings.llm_base_url
        ).strip()
        self.model_name = (
            model
            if model is not None
            else resolved_settings.llm_model_id
        ).strip()
        self.timeout = (
            float(timeout)
            if timeout is not None
            else resolved_settings.llm_request_timeout
        )
        self.structured_api = resolved_settings.llm_structured_api
        self.reasoning_effort = resolved_settings.llm_reasoning_effort
        self.temperature = (
            float(temperature)
            if temperature is not None
            else resolved_settings.llm_temperature
        )
        self.max_tokens = (
            int(max_tokens)
            if max_tokens is not None
            else resolved_settings.llm_max_tokens
        )
        self.provider = provider or _infer_provider(self.base_url)
        self._clock = clock
        self._call_history: list[LLMResponse] = []
        self._openai_module: Any = None

        self._validate_defaults(client_injected=client is not None)

        if client is not None:
            self._client = client
            self._owns_client = False
            self._load_openai_module(required=False)
            return

        openai_module = self._load_openai_module(required=True)
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            # SDK 默认会自动重试；本项目由 Prefect 统一管理重试。
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        try:
            self._client = openai_module.OpenAI(**client_kwargs)
        except Exception as exc:
            raise LLMConfigurationError(
                "OpenAI 兼容客户端初始化失败"
            ) from exc
        self._owns_client = True

    @property
    def last_response(self) -> LLMResponse | None:
        """返回最近一次成功调用的统一响应。"""

        return self._call_history[-1] if self._call_history else None

    @property
    def call_history(self) -> tuple[LLMResponse, ...]:
        """返回当前服务实例内成功调用的只读快照。"""

        return tuple(self._call_history)

    def __enter__(self) -> "LLMService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """关闭由本服务创建的 OpenAI HTTP 客户端。"""

        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    # 中文注释：底层聊天调用入口，负责参数校验、SDK 调用、响应标准化和调用历史记录。
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """发送标准消息列表并返回统一文本响应和调用元数据。"""

        normalized_messages = self._validate_messages(messages)
        selected_model = (model or self.model_name).strip()
        selected_temperature = (
            self.temperature
            if temperature is None
            else float(temperature)
        )
        selected_max_tokens = (
            self.max_tokens
            if max_tokens is None
            else int(max_tokens)
        )
        selected_timeout = (
            self.timeout if timeout is None else float(timeout)
        )
        self._validate_call_options(
            model=selected_model,
            temperature=selected_temperature,
            max_tokens=selected_max_tokens,
            timeout=selected_timeout,
        )

        request_options: dict[str, Any] = {
            "model": selected_model,
            "messages": normalized_messages,
            "timeout": selected_timeout,
        }
        if selected_model.lower().startswith("gpt-5"):
            request_options.update(
                max_completion_tokens=selected_max_tokens,
                reasoning_effort=self.reasoning_effort,
                verbosity="low",
            )
        else:
            request_options.update(
                temperature=selected_temperature,
                max_tokens=selected_max_tokens,
            )

        started_at = self._clock()
        try:
            completion = self._client.chat.completions.create(
                **request_options
            )
        except Exception as exc:
            self._raise_converted_exception(exc)

        latency_seconds = max(0.0, self._clock() - started_at)
        response = self._parse_completion(
            completion,
            requested_model=selected_model,
            latency_seconds=latency_seconds,
        )
        self._call_history.append(response)
        logger.info(
            "LLM 调用成功 provider=%s model=%s request_id=%s "
            "input_tokens=%d output_tokens=%d total_tokens=%d "
            "latency_seconds=%.6f",
            response.provider,
            response.model_name,
            response.request_id,
            response.input_tokens,
            response.output_tokens,
            response.total_tokens,
            response.latency_seconds,
        )
        return response

    # 中文注释：面向业务层的纯文本生成封装，隐藏底层 completion 响应结构。
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """根据已由业务层准备好的系统 Prompt 和用户 Prompt 生成文本。"""

        if not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")
        if not user_prompt.strip():
            raise ValueError("user_prompt 不能为空")

        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # 中文注释：将 LLM 文本解析为 JSON，再使用指定 Pydantic 模型进行严格校验。
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> ResponseModelT:
        """生成 JSON，并校验为调用方传入的任意 Pydantic v2 模型。"""

        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError("response_model 必须是 BaseModel 的子类")

        response = self.generate_text(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        parsed = self.parse_json_response(response.content)

        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise LLMValidationError(
                f"模型响应不符合 {response_model.__name__}：{exc}"
            ) from exc

    def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> ResponseModelT:
        """使用 Responses API Structured Outputs 生成 Pydantic 对象。"""

        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt 和 user_prompt 不能为空")
        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError("response_model 必须是 BaseModel 的子类")
        if self.structured_api == "chat":
            return self._generate_structured_response_via_chat(
                system_prompt,
                user_prompt,
                response_model,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        parse = getattr(getattr(self._client, "responses", None), "parse", None)
        if not callable(parse):
            logger.warning(
                "客户端不支持 Responses API parse，降级为 Chat Completions "
                "JSON provider=%s model=%s",
                self.provider,
                (model or self.model_name).strip(),
            )
            return self._generate_structured_response_via_chat(
                system_prompt,
                user_prompt,
                response_model,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        try:
            response = parse(
                model=(model or self.model_name).strip(),
                instructions=system_prompt,
                input=user_prompt,
                text_format=response_model,
                max_output_tokens=int(max_tokens or self.max_tokens),
                reasoning={"effort": "none"},
                store=False,
                timeout=timeout or self.timeout,
            )
        except Exception as exc:
            if self._is_retryable_structured_output_error(exc):
                logger.warning(
                    "Responses Structured Outputs 失败，降级为 Responses JSON "
                    "provider=%s model=%s status_code=%s",
                    self.provider,
                    (model or self.model_name).strip(),
                    getattr(exc, "status_code", None),
                )
                return self._generate_structured_response_via_json(
                    system_prompt,
                    user_prompt,
                    response_model,
                    model=model,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            self._raise_converted_exception(exc)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMResponseError(
                "Responses API 未返回可解析的结构化输出"
            )
        return response_model.model_validate(parsed)

    @staticmethod
    def _is_retryable_structured_output_error(exc: Exception) -> bool:
        """仅对代理或上游 5xx 降级，认证、限流和请求错误仍原样抛出。"""

        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code >= 500

    def _generate_structured_response_via_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        *,
        model: str | None,
        max_tokens: int | None,
        timeout: float | None,
    ) -> ResponseModelT:
        """兼容不稳定代理的 Responses JSON 降级路径，结果仍由 Pydantic 校验。"""

        create = getattr(
            getattr(self._client, "responses", None), "create", None
        )
        if not callable(create):
            raise LLMConfigurationError(
                "当前 OpenAI 客户端不支持 Responses API create"
            )
        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        instructions = (
            f"{system_prompt}\n"
            "只输出一个合法 JSON 对象，不输出 Markdown 或解释。"
            f"必须满足以下 JSON Schema：{schema}"
        )
        try:
            response = create(
                model=(model or self.model_name).strip(),
                instructions=instructions,
                input=user_prompt,
                max_output_tokens=int(max_tokens or self.max_tokens),
                reasoning={"effort": "none"},
                store=False,
                timeout=timeout or self.timeout,
            )
        except Exception as exc:
            if self._is_retryable_structured_output_error(exc):
                logger.warning(
                    "Responses JSON 降级路径仍失败，继续降级为 Chat "
                    "Completions JSON provider=%s model=%s status_code=%s",
                    self.provider,
                    (model or self.model_name).strip(),
                    getattr(exc, "status_code", None),
                )
                return self._generate_structured_response_via_chat(
                    system_prompt,
                    user_prompt,
                    response_model,
                    model=model,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            self._raise_converted_exception(exc)
        content = getattr(response, "output_text", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(
                "Responses API JSON 降级路径未返回 output_text"
            )
        parsed = self.parse_json_response(content)
        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise LLMValidationError(
                f"模型响应不符合 {response_model.__name__}：{exc}"
            ) from exc

    def _generate_structured_response_via_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        *,
        model: str | None,
        max_tokens: int | None,
        timeout: float | None,
    ) -> ResponseModelT:
        """最终兼容路径：用 Chat Completions 生成并严格校验 JSON。"""

        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        instructions = (
            f"{system_prompt}\n"
            "只输出一个合法 JSON 对象，不输出 Markdown 或解释。"
            f"必须满足以下 JSON Schema：{schema}"
        )
        return self.generate_structured(
            instructions,
            user_prompt,
            response_model,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @staticmethod
    # 中文注释：从可能包含额外说明的响应中定位并解析第一个合法 JSON 对象或数组。
    def parse_json_response(content: str) -> dict[str, Any] | list[Any]:
        """定位并解码文本中的首个合法 JSON 对象或数组。

        使用 ``JSONDecoder.raw_decode`` 逐个候选起点尝试解码，不依赖
        首尾花括号截取，因此不会误截断嵌套对象、数组或字符串内容。
        """

        if not isinstance(content, str) or not content.strip():
            raise LLMParseError("模型响应为空，无法解析 JSON")

        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(content, index)
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return cast(dict[str, Any] | list[Any], value)

        raise LLMParseError("模型响应中未找到合法 JSON 对象或数组")

    def health_check(self) -> LLMResponse:
        """通过最小生成请求检查密钥、接口地址和模型是否可用。"""

        return self.generate_text(
            "你是接口健康检查助手。",
            "仅回复 OK。",
            temperature=0.0,
            max_tokens=2,
        )

    def _validate_defaults(self, *, client_injected: bool) -> None:
        if not self.model_name:
            raise LLMConfigurationError("缺少 LLM_MODEL_ID")
        if not client_injected and not self.api_key:
            raise LLMConfigurationError("缺少 LLM_API_KEY")
        self._validate_call_options(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    @staticmethod
    def _validate_call_options(
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        if not model:
            raise LLMConfigurationError("模型名称不能为空")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")

    @staticmethod
    def _validate_messages(
        messages: Sequence[ChatMessage],
    ) -> list[dict[str, Any]]:
        if not messages:
            raise ValueError("messages 不能为空")

        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] 必须是映射对象")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"messages[{index}].role 不能为空")
            if content is None:
                raise ValueError(f"messages[{index}].content 不能为空")
            normalized.append(dict(message))
        return normalized

    def _parse_completion(
        self,
        completion: Any,
        *,
        requested_model: str,
        latency_seconds: float,
    ) -> LLMResponse:
        choices = _get_value(completion, "choices")
        if not isinstance(choices, Sequence) or isinstance(
            choices, (str, bytes)
        ) or not choices:
            raise LLMResponseError("模型响应缺少 choices")

        message = _get_value(choices[0], "message")
        if message is None:
            raise LLMResponseError("模型响应缺少 choices[0].message")

        content = self._normalize_content(_get_value(message, "content"))
        if not content.strip():
            raise LLMResponseError("模型响应的 message.content 为空")

        usage = _get_value(completion, "usage")
        input_tokens = self._token_count(
            usage, "prompt_tokens", "input_tokens"
        )
        output_tokens = self._token_count(
            usage, "completion_tokens", "output_tokens"
        )
        total_tokens = self._token_count(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        request_id = (
            _get_value(completion, "_request_id")
            or _get_value(completion, "request_id")
            or _get_value(completion, "id")
        )
        returned_model = _get_value(
            completion, "model", requested_model
        )

        return LLMResponse(
            provider=self.provider,
            model_name=str(returned_model or requested_model),
            request_id=str(request_id) if request_id else None,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            created_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes)
        ):
            parts: list[str] = []
            for part in content:
                text = _get_value(part, "text")
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _token_count(usage: Any, *names: str) -> int:
        for name in names:
            value = _get_value(usage, name)
            if isinstance(value, int) and not isinstance(value, bool):
                return max(0, value)
        return 0

    def _load_openai_module(self, *, required: bool) -> Any:
        try:
            import openai
        except ImportError as exc:
            if required:
                raise LLMConfigurationError(
                    "缺少 openai 依赖，请执行：pip install openai"
                ) from exc
            return None

        self._openai_module = openai
        return openai

    def _raise_converted_exception(self, exc: Exception) -> NoReturn:
        openai_module = self._openai_module
        if openai_module is None:
            raise LLMServiceError("OpenAI 兼容接口调用失败") from exc

        if isinstance(exc, openai_module.RateLimitError):
            raise LLMRateLimitError("大语言模型接口触发限流") from exc
        if isinstance(
            exc,
            (
                openai_module.APITimeoutError,
                openai_module.APIConnectionError,
            ),
        ):
            raise LLMConnectionError(
                "无法连接大语言模型接口或请求超时"
            ) from exc
        if isinstance(
            exc,
            (
                openai_module.AuthenticationError,
                openai_module.PermissionDeniedError,
            ),
        ):
            raise LLMConfigurationError(
                "大语言模型接口认证或权限校验失败"
            ) from exc
        if isinstance(exc, openai_module.APIStatusError):
            raise LLMConnectionError(
                f"大语言模型接口返回 HTTP {exc.status_code}"
            ) from exc
        if isinstance(exc, openai_module.APIError):
            raise LLMServiceError("大语言模型接口调用失败") from exc
        raise LLMServiceError("OpenAI 兼容客户端调用失败") from exc

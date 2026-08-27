"""Tavily 搜索服务。

该模块只负责：
1. 调用 Tavily Search/Extract API；
2. 将返回值标准化为项目可消费的字典；
3. 按 ``configs/sources.yaml`` 的单个 source 配置执行增量检索。

安装依赖::

    pip install tavily-python
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "spm",
    "from",
    "source",
}
_TRACKING_QUERY_PREFIXES = ("utm_",)


class TavilyServiceError(RuntimeError):
    """Tavily 服务的统一异常。"""


class TavilyConfigurationError(TavilyServiceError):
    """Tavily 配置无效或依赖缺失。"""


class TavilyRequestError(TavilyServiceError):
    """Tavily 请求在重试后仍然失败。"""


class TavilyClientProtocol(Protocol):
    """便于测试时注入 fake client，同时兼容官方 TavilyClient。"""

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...

    def extract(
        self, urls: str | Sequence[str], **kwargs: Any
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


# 中文注释：规范化 URL 以形成稳定去重键，避免同一页面因查询参数或尾斜杠重复出现。
def _canonicalize_url(url: str) -> str:
    """移除 fragment 和常见跟踪参数，生成稳定的去重键。"""

    value = url.strip()
    if not value:
        return ""

    parts = urlsplit(value)
    filtered_query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS
        and not key.lower().startswith(_TRACKING_QUERY_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            urlencode(sorted(filtered_query)),
            "",
        )
    )


def _iso_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


# 中文注释：Tavily 外部检索适配器，负责搜索、来源约束、结果规范化、去重和退避重试。
class TavilyService:
    """面向采集工作流的 Tavily SDK 薄封装。

    ``client`` 参数用于单元测试；生产环境通常只需传入 API key，
    未显式传入时会读取 ``TAVILY_API_KEY``。
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: TavilyClientProtocol | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        if max_retries < 0:
            raise ValueError("max_retries 不能小于 0")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds 不能小于 0")

        self.timeout = min(float(timeout), 120.0)
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self._sleep = sleep

        if client is not None:
            self._client = client
            self._owns_client = False
            return

        resolved_key = api_key or os.getenv("TAVILY_API_KEY", "")
        if not resolved_key:
            raise TavilyConfigurationError(
                "缺少 Tavily API key，请设置 TAVILY_API_KEY"
            )

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise TavilyConfigurationError(
                "缺少依赖 tavily-python，请执行：pip install tavily-python"
            ) from exc

        self._client = TavilyClient(api_key=resolved_key)
        self._owns_client = True

    def __enter__(self) -> "TavilyService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """释放官方 SDK 持有的 HTTP session。"""

        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    # 中文注释：执行单次普通查询，并把第三方响应转换为稳定的项目结果结构。
    def search(
        self,
        query: str,
        *,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        max_results: int = 10,
        search_depth: str = "advanced",
        topic: str = "general",
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        include_raw_content: bool | str = "text",
    ) -> list[dict[str, Any]]:
        """执行一次搜索，并返回标准化、去重后的结果列表。"""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results 必须在 1 到 20 之间")

        kwargs: dict[str, Any] = {
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "include_answer": False,
            "include_images": False,
            "timeout": self.timeout,
        }
        if include_domains:
            kwargs["include_domains"] = list(dict.fromkeys(include_domains))
        if exclude_domains:
            kwargs["exclude_domains"] = list(dict.fromkeys(exclude_domains))
        if start := _iso_date(start_date):
            kwargs["start_date"] = start
        if end := _iso_date(end_date):
            kwargs["end_date"] = end

        payload = self._call_with_retry(
            "search", lambda: self._client.search(cleaned_query, **kwargs)
        )
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise TavilyRequestError("Tavily 返回的 results 不是列表")

        return self._normalize_and_deduplicate(
            raw_results,
            query=cleaned_query,
            request_id=payload.get("request_id"),
            response_time=payload.get("response_time"),
        )

    # 中文注释：根据来源配置依次搜索多个关键词，执行日期过滤、URL 去重、排序和截断。
    def search_source(
        self,
        source: Mapping[str, Any],
        *,
        today: date | None = None,
    ) -> list[dict[str, Any]]:
        """按照 ``sources.yaml`` 中的一项 source 配置执行增量检索。

        每个关键词单独检索，最终按规范化 URL 去重并按相关度降序截断。
        """

        if not source.get("enabled", True):
            return []

        keywords = [
            str(keyword).strip()
            for keyword in source.get("keywords", [])
            if str(keyword).strip()
        ]
        if not keywords:
            raise TavilyConfigurationError(
                f"source {source.get('source_id', '<unknown>')} 未配置 keywords"
            )

        max_results = int(source.get("max_results", 20))
        if not 1 <= max_results <= 20:
            raise TavilyConfigurationError(
                "sources.yaml 中的 max_results 必须在 1 到 20 之间"
            )

        overlap_days = max(0, int(source.get("overlap_days", 0)))
        end_date = today or date.today()
        start_date = end_date - timedelta(days=overlap_days)
        domains = [
            str(domain).strip()
            for domain in source.get("domains", [])
            if str(domain).strip()
        ]
        rate_limit_seconds = max(
            0.0, float(source.get("rate_limit_seconds", 0))
        )

        collected: list[dict[str, Any]] = []
        for index, keyword in enumerate(keywords):
            if index and rate_limit_seconds:
                self._sleep(rate_limit_seconds)

            results = self.search(
                keyword,
                include_domains=domains or None,
                max_results=max_results,
                search_depth=str(source.get("search_depth", "advanced")),
                topic=str(source.get("topic", "general")),
                start_date=start_date,
                end_date=end_date,
                include_raw_content=source.get(
                    "include_raw_content", "text"
                ),
            )
            for result in results:
                result["source_id"] = source.get("source_id")
                result["matched_keyword"] = keyword
            collected.extend(results)

        deduplicated = self._deduplicate(collected)
        deduplicated.sort(
            key=lambda item: float(item.get("score") or 0.0), reverse=True
        )
        return deduplicated[:max_results]

    # 中文注释：调用 Tavily extract 获取候选 URL 的正文内容，并限制单次请求规模。
    def extract(
        self,
        urls: str | Sequence[str],
        *,
        extract_depth: str = "basic",
        output_format: str = "text",
    ) -> list[dict[str, Any]]:
        """提取指定 URL 的正文；一次最多处理 20 个 URL。"""

        url_list = [urls] if isinstance(urls, str) else list(urls)
        url_list = [url.strip() for url in url_list if url.strip()]
        if not url_list:
            return []
        if len(url_list) > 20:
            raise ValueError("Tavily Extract 一次最多接收 20 个 URL")

        payload = self._call_with_retry(
            "extract",
            lambda: self._client.extract(
                urls=url_list,
                extract_depth=extract_depth,
                format=output_format,
                include_images=False,
                timeout=self.timeout,
            ),
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise TavilyRequestError("Tavily 返回的 results 不是列表")

        return [
            {
                "url": str(item.get("url") or "").strip(),
                "canonical_url": _canonicalize_url(
                    str(item.get("url") or "")
                ),
                "raw_content": item.get("raw_content") or "",
            }
            for item in results
            if isinstance(item, Mapping) and item.get("url")
        ]

    # 中文注释：为临时请求故障提供指数退避；后续接入 Prefect 时需避免与 Task 重试叠加。
    def _call_with_retry(
        self,
        operation: str,
        request: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = request()
                if not isinstance(response, dict):
                    raise TypeError("Tavily 响应不是字典")
                return response
            except Exception as exc:  # SDK 使用多种 requests 自定义异常
                last_error = exc
                if attempt >= self.max_retries:
                    break

                delay = self.retry_base_seconds * (2**attempt)
                logger.warning(
                    "Tavily %s 失败，将在 %.1f 秒后重试（%d/%d）：%s",
                    operation,
                    delay,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if delay:
                    self._sleep(delay)

        raise TavilyRequestError(
            f"Tavily {operation} 请求失败，已重试 {self.max_retries} 次"
        ) from last_error

    @classmethod
    def _normalize_and_deduplicate(
        cls,
        results: Sequence[Any],
        *,
        query: str,
        request_id: Any,
        response_time: Any,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []

        for item in results:
            if not isinstance(item, Mapping):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue

            normalized.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": url,
                    "canonical_url": _canonicalize_url(url),
                    "content": item.get("content") or "",
                    "raw_content": item.get("raw_content") or "",
                    "score": item.get("score"),
                    "published_at": (
                        item.get("published_date")
                        or item.get("published_at")
                    ),
                    "query": query,
                    "request_id": request_id,
                    "response_time": response_time,
                }
            )

        return cls._deduplicate(normalized)

    @staticmethod
    def _deduplicate(
        results: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}

        for result in results:
            key = result.get("canonical_url") or result.get("url")
            if not key:
                continue
            existing = unique.get(str(key))
            if existing is None or float(result.get("score") or 0.0) > float(
                existing.get("score") or 0.0
            ):
                unique[str(key)] = result

        return list(unique.values())

from __future__ import annotations

from datetime import date

from src.services.tavily_service import TavilyService


class FakeTavilyClient:
    def __init__(self) -> None:
        self.search_calls = []

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return {
            "request_id": "request-1",
            "response_time": 0.12,
            "results": [
                {
                    "title": "案件通报",
                    "url": "https://example.gov.cn/a?utm_source=test&id=1#top",
                    "content": "某案件内容",
                    "raw_content": "完整正文",
                    "score": 0.9,
                },
                {
                    "title": "重复页面",
                    "url": "https://example.gov.cn/a?id=1",
                    "content": "重复内容",
                    "score": 0.5,
                },
            ],
        }

    def extract(self, urls, **kwargs):
        return {
            "results": [
                {"url": urls[0], "raw_content": "提取后的正文"}
            ]
        }


def test_search_normalizes_and_deduplicates_urls():
    service = TavilyService(client=FakeTavilyClient(), max_retries=0)

    results = service.search("受贿", include_domains=["example.gov.cn"])

    assert len(results) == 1
    assert results[0]["canonical_url"] == "https://example.gov.cn/a?id=1"
    assert results[0]["request_id"] == "request-1"


def test_search_source_uses_yaml_shaped_configuration():
    client = FakeTavilyClient()
    service = TavilyService(client=client, max_retries=0, sleep=lambda _: None)

    results = service.search_source(
        {
            "source_id": "court",
            "enabled": True,
            "domains": ["example.gov.cn"],
            "keywords": ["受贿", "行贿"],
            "overlap_days": 14,
            "max_results": 10,
            "rate_limit_seconds": 2,
        },
        today=date(2026, 7, 28),
    )

    assert len(results) == 1
    assert len(client.search_calls) == 2
    assert client.search_calls[0][1]["start_date"] == "2026-07-14"
    assert client.search_calls[0][1]["end_date"] == "2026-07-28"
    assert client.search_calls[0][1]["include_domains"] == [
        "example.gov.cn"
    ]


def test_extract_returns_normalized_content():
    service = TavilyService(client=FakeTavilyClient(), max_retries=0)

    result = service.extract("https://example.gov.cn/a")

    assert result[0]["raw_content"] == "提取后的正文"

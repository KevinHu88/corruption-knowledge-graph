from fastapi.testclient import TestClient

from 第二阶段.api.app import create_app
from 第二阶段.api.dependencies import build_container
from 第二阶段.config import QAConfig
from 第二阶段.evaluation.runner import (
    DEFAULT_DATASET_PATH,
    load_cases,
    run_evaluation,
)
from 第二阶段.generation.llm_client import MockLLMClient


def test_mock_evaluation_dataset_passes() -> None:
    cases = load_cases(DEFAULT_DATASET_PATH)
    container = build_container(
        QAConfig(api_mode="mock"),
        llm_client=MockLLMClient("模拟回答：已根据给定证据完成作答。"),
    )

    with TestClient(create_app(container)) as client:
        report = run_evaluation(client, cases)

    assert report["summary"] == {
        "total": 7,
        "passed": 7,
        "failed": 0,
        "pass_rate": 1.0,
    }


def test_evaluation_reports_wrong_route() -> None:
    cases = [
        {
            "id": "wrong-route",
            "question": "张某与哪些人存在关系？",
            "expected": {
                "status_code": 200,
                "route": "DOCUMENT",
                "source_types": ["graph"],
                "answer_nonempty": True,
            },
        }
    ]
    container = build_container(
        QAConfig(api_mode="mock"),
        llm_client=MockLLMClient("模拟回答"),
    )

    with TestClient(create_app(container)) as client:
        report = run_evaluation(client, cases)

    assert report["summary"]["failed"] == 1
    assert report["results"][0]["failures"] == [
        "route: expected DOCUMENT, got GRAPH"
    ]

from fastapi.testclient import TestClient

from 第二阶段.api.app import create_app
from 第二阶段.api.dependencies import build_container
from 第二阶段.generation.llm_client import LLMClient, MockLLMClient
from 第二阶段.retrieval.graph_retriever import GraphRetriever
from 第二阶段.tests.api.conftest import (
    APIMockGraphRepository,
    create_session,
    make_test_config,
)
from 第二阶段.tests.test_graph_retriever import DuplicateAnonymousPersonRepository


def test_document_question_returns_answer_route_evidence_sources(api_context) -> None:
    session_id = create_session(api_context.client)
    api_context.client.post(
        f"/sessions/{session_id}/documents",
        files={"file": ("report.txt", "报告描述该项目由李某审批。", "text/plain")},
    )
    response = api_context.client.post(
        f"/sessions/{session_id}/questions",
        json={"question": "上传报告中如何描述该项目？"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "DOCUMENT"
    assert payload["answer"] == "mock-api-answer"
    assert {item["source_type"] for item in payload["evidence"]} == {"document"}
    assert payload["sources"] == [
        {"source_type": "document", "name": "report.txt"}
    ]
    assert payload["evidence"][0]["metadata"]["retrieval"]["mode"] == "hybrid"
    assert "file_path" not in payload["evidence"][0]["metadata"]


def test_graph_question_uses_mock_graph_without_upload(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/questions",
        json={"question": "张某与哪些人存在关系？"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "GRAPH"
    assert {item["source_type"] for item in payload["evidence"]} == {"graph"}
    claim = next(item for item in payload["evidence"] if item["id"].startswith("graph-claim"))
    assert claim["metadata"]["relationship"] == "请托"
    assert claim["metadata"]["relationship_properties"]["doc_id"] == "doc-api"


def test_graph_question_accepts_case_id(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/questions",
        json={"question": "张某与哪些人存在关系？", "case_id": "case-api"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert any(
        item["metadata"].get("case", {}).get("case_id") == "case-api"
        for item in payload["evidence"]
        if item["source_type"] == "graph"
    )


def test_default_mock_api_exposes_frontend_path_demo() -> None:
    with _failure_client(llm=MockLLMClient("mock path answer")) as client:
        session_id = create_session(client)
        response = client.post(
            f"/sessions/{session_id}/questions",
            json={
                "question": "查找谢晚林与刘某之间的相似路径",
                "case_id": "mock-case",
            },
        )

    assert response.status_code == 200
    paths = [
        item
        for item in response.json()["evidence"]
        if item["metadata"].get("kind") in {"path", "similar_path"}
    ]
    assert {item["metadata"]["kind"] for item in paths} == {
        "path",
        "similar_path",
    }
    anchor = next(item for item in paths if item["metadata"]["kind"] == "path")
    assert anchor["metadata"]["directions"] == ["forward", "reverse"]
    assert [
        entity["entity_type"] for entity in anchor["metadata"]["path_entities"]
    ] == ["PER", "MONEY", "PER"]


def test_ambiguous_entity_returns_candidate_cases() -> None:
    llm = MockLLMClient("must-not-be-used")
    container = build_container(
        make_test_config(),
        graph_retriever=GraphRetriever(DuplicateAnonymousPersonRepository()),
        llm_client=llm,
    )
    with TestClient(create_app(container)) as client:
        session_id = create_session(client)
        response = client.post(
            f"/sessions/{session_id}/questions",
            json={"question": "罗某与哪些人存在关系？"},
        )

    assert response.status_code == 409
    assert response.json()["entity_name"] == "罗某"
    assert response.json()["candidate_case_ids"] == ["case-a", "case-b"]
    assert llm.prompts == []


def test_hybrid_question_returns_document_and_graph_evidence(api_context) -> None:
    session_id = create_session(api_context.client)
    api_context.client.post(
        f"/sessions/{session_id}/documents",
        files={"file": ("source.txt", "原文证据是张某请托李某。", "text/plain")},
    )
    response = api_context.client.post(
        f"/sessions/{session_id}/questions",
        json={"question": "张某与李某是什么关系，原文证据是什么？"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "HYBRID"
    assert {item["source_type"] for item in payload["evidence"]} == {
        "document",
        "graph",
    }


class FailingGraphRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        case_id: str | None = None,
        search_scope: str = "same_case",
        selected_case_ids: list[str] | None = None,
    ):
        del query, top_k, case_id, search_scope, selected_case_ids
        raise ConnectionError("neo4j offline")


class FailingLLMClient(LLMClient):
    def generate(self, prompt: str) -> str:
        del prompt
        raise TimeoutError("llm timeout")


def _failure_client(*, graph=None, llm=None) -> TestClient:
    container = build_container(
        make_test_config(),
        graph_retriever=graph,
        llm_client=llm,
    )
    return TestClient(create_app(container))


def test_graph_failure_returns_503() -> None:
    with _failure_client(
        graph=FailingGraphRetriever(), llm=MockLLMClient("unused")
    ) as client:
        session_id = create_session(client)
        response = client.post(
            f"/sessions/{session_id}/questions",
            json={"question": "张某与哪些人存在关系？"},
        )
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Knowledge graph service is unavailable."
        }


def test_llm_failure_returns_503() -> None:
    with _failure_client(
        graph=GraphRetriever(APIMockGraphRepository()),
        llm=FailingLLMClient(),
    ) as client:
        session_id = create_session(client)
        response = client.post(
            f"/sessions/{session_id}/questions",
            json={"question": "张某与哪些人存在关系？"},
        )
        assert response.status_code == 503
        assert response.json() == {
            "detail": "LLM generation service is unavailable."
        }


def test_blank_question_returns_400(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/questions", json={"question": "   "}
    )
    assert response.status_code == 400


def test_all_cases_scope_requires_anchor_case_id(api_context) -> None:
    session_id = create_session(api_context.client)

    response = api_context.client.post(
        f"/sessions/{session_id}/questions",
        json={
            "question": "查找张某与李某之间的相似路径",
            "search_scope": "all_cases",
        },
    )

    assert response.status_code == 422


def test_selected_cases_scope_requires_case_list(api_context) -> None:
    session_id = create_session(api_context.client)

    response = api_context.client.post(
        f"/sessions/{session_id}/questions",
        json={
            "question": "查找张某与李某之间的相似路径",
            "case_id": "case-api",
            "search_scope": "selected_cases",
        },
    )

    assert response.status_code == 422

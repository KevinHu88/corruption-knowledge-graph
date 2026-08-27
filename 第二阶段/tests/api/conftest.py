from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from 第二阶段.api.app import create_app
from 第二阶段.api.dependencies import build_container
from 第二阶段.config import QAConfig
from 第二阶段.generation.llm_client import MockLLMClient
from 第二阶段.retrieval.graph_retriever import GraphRetriever


class APIMockGraphRepository:
    entity = {
        "entity_uid": "u-zhang",
        "name": "张某",
        "entity_type": "PER",
        "case_id": "case-api",
    }

    def find_entities_in_text(
        self, text: str, limit: int = 10, *, case_id: str | None = None
    ):
        del limit
        matches_case = case_id is None or case_id == self.entity["case_id"]
        return [{"entity": self.entity}] if "张某" in text and matches_case else []

    def find_entity_by_name(
        self, name: str, limit: int = 10, *, case_id: str | None = None
    ):
        del limit
        matches_case = case_id is None or case_id == self.entity["case_id"]
        return [{"entity": self.entity}] if name == "张某" and matches_case else []

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        del limit
        if entity_uid != "u-zhang" or (
            case_id is not None and case_id != self.entity["case_id"]
        ):
            return []
        return [
            {
                "claim": {
                    "claim_id": "claim-api",
                    "relation_type": "请托",
                    "status": "HUMAN_VERIFIED",
                    "case_id": "case-api",
                    "doc_id": "doc-api",
                    "text_id": "text-api",
                    "evidence_text": "张某请托李某。",
                },
                "head": self.entity,
                "tail": {
                    "entity_uid": "u-li",
                    "name": "李某",
                    "entity_type": "PER",
                },
                "evidence": [{"text": "张某请托李某。", "text_id": "text-api"}],
                "document": {"title": "图谱案件材料", "doc_id": "doc-api"},
                "case": {"case_id": "case-api"},
            }
        ]


def make_test_config() -> QAConfig:
    return QAConfig(
        chunk_size=80,
        chunk_overlap=10,
        max_context_chars=2000,
        max_upload_size=256,
        question_max_chars=500,
        api_mode="mock",
    )


@pytest.fixture
def api_context():
    llm = MockLLMClient("mock-api-answer")
    container = build_container(
        make_test_config(),
        graph_retriever=GraphRetriever(APIMockGraphRepository()),
        llm_client=llm,
    )
    app = create_app(container)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, container=container, llm=llm)


def create_session(client: TestClient) -> str:
    response = client.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]

from 第二阶段.retrieval.document_retriever import DocumentRetriever
from 第二阶段.schemas.models import Chunk
from 第二阶段.storage.session_document_store import SessionDocumentStore


def test_document_retrieval_returns_relevant_evidence() -> None:
    store = SessionDocumentStore()
    store.add_chunks(
        [
            Chunk("c1", "d1", "张某请托李某办理项目审批。", {"file_name": "a.txt"}),
            Chunk("c2", "d1", "今日天气晴朗，适合出行。", {"file_name": "a.txt"}),
            Chunk("c3", "d2", "王某担任某机构负责人。", {"file_name": "b.txt"}),
        ]
    )
    result = DocumentRetriever(store).retrieve("谁办理项目审批", top_k=2)
    assert result
    assert result[0].id == "c1"
    assert result[0].source_type == "document"


class FakeSemanticEmbeddingClient:
    cache_namespace = "fake-semantic-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        mapping = {
            "采购违规": [1.0, 0.0],
            "多家公司事先约定报价顺序。": [1.0, 0.0],
            "今日天气晴朗。": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


def test_vector_recall_finds_semantic_match_without_bm25_overlap() -> None:
    store = SessionDocumentStore()
    store.add_chunks(
        [
            Chunk("semantic", "d1", "多家公司事先约定报价顺序。"),
            Chunk("weather", "d1", "今日天气晴朗。"),
        ]
    )

    result = DocumentRetriever(
        store,
        embedding_client=FakeSemanticEmbeddingClient(),
        vector_min_score=0.5,
    ).retrieve("采购违规", top_k=1)

    assert result[0].id == "semantic"
    scores = result[0].metadata["retrieval"]
    assert scores["mode"] == "hybrid"
    assert scores["bm25_score"] == 0.0
    assert scores["vector_score"] == 1.0


def test_chunk_vectors_are_cached_across_retriever_instances() -> None:
    store = SessionDocumentStore()
    store.add_chunks(
        [
            Chunk("semantic", "d1", "多家公司事先约定报价顺序。"),
            Chunk("weather", "d1", "今日天气晴朗。"),
        ]
    )
    embedding = FakeSemanticEmbeddingClient()

    DocumentRetriever(store, embedding_client=embedding).retrieve("采购违规")
    DocumentRetriever(store, embedding_client=embedding).retrieve("采购违规")

    assert [len(call) for call in embedding.calls] == [2, 1, 1]


class FailingEmbeddingClient:
    cache_namespace = "failing"

    def embed(self, texts):
        del texts
        raise ConnectionError("embedding offline")


def test_vector_failure_falls_back_to_bm25() -> None:
    store = SessionDocumentStore()
    store.add_chunks([Chunk("c1", "d1", "项目审批由李某负责。")])

    result = DocumentRetriever(
        store,
        embedding_client=FailingEmbeddingClient(),
    ).retrieve("项目审批")

    assert result[0].id == "c1"
    assert result[0].metadata["retrieval"]["mode"] == "bm25"

from types import SimpleNamespace

from 第二阶段.retrieval.embedding import (
    FirstStageEmbeddingClient,
    HashingEmbeddingClient,
    cosine_similarity,
)


def test_hashing_embedding_is_deterministic_and_normalized() -> None:
    client = HashingEmbeddingClient(dimensions=32)
    first, second = client.embed(["张某办理审批", "张某办理审批"])

    assert first == second
    assert len(first) == 32
    assert round(cosine_similarity(first, second), 6) == 1.0


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )


def test_first_stage_embedding_client_preserves_input_order() -> None:
    api = FakeEmbeddingsAPI()
    client = FirstStageEmbeddingClient(
        api_key="test-key",
        model="test-model",
        model_type="dashscope",
        client=SimpleNamespace(embeddings=api),
    )

    vectors = client.embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert api.calls == [{"model": "test-model", "input": ["first", "second"]}]

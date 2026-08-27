import pytest

from 第二阶段.retrieval.path_similarity import (
    PathSignature,
    PathSimilarityScorer,
)


def test_identical_path_signatures_have_full_similarity() -> None:
    signature = PathSignature(
        relation_types=("请托", "帮助谋利"),
        entity_types=("PER", "PER", "PER"),
        directions=("forward", "forward"),
    )

    result = PathSimilarityScorer().score(signature, signature)

    assert result.score == 1.0
    assert result.orientation == "forward"


def test_reversed_equivalent_path_can_be_matched() -> None:
    anchor = PathSignature(
        relation_types=("请托", "帮助谋利"),
        entity_types=("PER", "PER", "ORG"),
        directions=("forward", "forward"),
    )
    reversed_candidate = PathSignature(
        relation_types=("帮助谋利", "请托"),
        entity_types=("ORG", "PER", "PER"),
        directions=("reverse", "reverse"),
    )

    result = PathSimilarityScorer().score(anchor, reversed_candidate)

    assert result.score == 1.0
    assert result.orientation == "reversed"


def test_different_relation_and_entity_sequences_reduce_similarity() -> None:
    anchor = PathSignature(
        relation_types=("请托", "帮助谋利"),
        entity_types=("PER", "PER", "ORG"),
    )
    candidate = PathSignature(
        relation_types=("任职于",),
        entity_types=("PER", "ORG"),
    )

    result = PathSimilarityScorer().score(anchor, candidate)

    assert result.score < 0.55


def test_path_signature_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        PathSignature(relation_types=("请托",), entity_types=("PER",))

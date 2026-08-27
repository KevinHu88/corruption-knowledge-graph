from 第二阶段.retrieval.reranker import HybridReranker
from 第二阶段.schemas.models import Chunk


def test_reranker_combines_bm25_vector_and_coverage() -> None:
    lexical = Chunk("lexical", "d1", "项目审批流程")
    semantic = Chunk("semantic", "d1", "多家公司事先约定报价顺序")
    reranker = HybridReranker(
        bm25_weight=0.2,
        vector_weight=0.7,
        coverage_weight=0.1,
    )

    result = reranker.rerank(
        "采购违规",
        [lexical, semantic],
        bm25_scores={"lexical": 1.0},
        vector_scores={"lexical": 0.1, "semantic": 0.95},
        top_k=2,
    )

    assert [item.chunk.chunk_id for item in result] == ["semantic", "lexical"]
    assert result[0].score > result[1].score

from 第二阶段.retrieval.fusion import EvidenceFusion
from 第二阶段.schemas.models import Evidence


def test_fusion_deduplicates_same_content_with_different_ids() -> None:
    lower = Evidence("doc-1", "document", "相同证据", 0.4)
    higher = Evidence("graph-1", "graph", "相同证据", 0.9)
    result = EvidenceFusion().fuse([lower], [higher])
    assert result == [higher]


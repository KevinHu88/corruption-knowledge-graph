import pytest

from 第二阶段.graph.graph_repository import GraphRepository


class CapturingAdapter:
    def __init__(self) -> None:
        self.cypher = ""
        self.parameters = {}
        self.max_records = 0

    def find_entities(self, **filters):
        del filters
        return []

    def execute_read(self, cypher, parameters=None, *, max_records=100):
        self.cypher = cypher
        self.parameters = dict(parameters or {})
        self.max_records = max_records
        return []


def test_find_simple_paths_builds_bounded_claim_path_query() -> None:
    adapter = CapturingAdapter()

    GraphRepository(adapter).find_simple_paths(
        "person-a",
        "person-b",
        case_id="case-1",
        max_hops=3,
        limit=7,
    )

    assert "[:HEAD|TAIL*2..6]" in adapter.cypher
    assert "single(other IN path_entities" in adapter.cypher
    assert "claim.status <> 'REJECTED'" in adapter.cypher
    assert "type(relationship)] AS path_relationship_types" in adapter.cypher
    assert adapter.parameters["case_id"] == "case-1"
    assert adapter.max_records == 7


def test_find_path_candidates_applies_structure_prefilter() -> None:
    adapter = CapturingAdapter()

    GraphRepository(adapter).find_path_candidates(
        ["请托", "帮助谋利"],
        start_entity_type="PER",
        end_entity_type="ORG",
        exclude_claim_ids=["claim-anchor"],
        case_ids=["case-1", "case-2"],
        max_hops=2,
        limit=30,
    )

    assert "[:HEAD|TAIL*2..4]" in adapter.cypher
    assert "claim.relation_type IN $relation_types" in adapter.cypher
    assert adapter.parameters["exclude_claim_ids"] == ["claim-anchor"]
    assert adapter.parameters["relation_types"] == ["请托", "帮助谋利"]
    assert adapter.parameters["case_ids"] == ["case-1", "case-2"]
    assert "claim.case_id IN $case_ids" in adapter.cypher


@pytest.mark.parametrize("max_hops", [0, 6])
def test_path_hops_are_hard_bounded(max_hops: int) -> None:
    with pytest.raises(ValueError):
        GraphRepository(CapturingAdapter()).find_simple_paths(
            "a",
            "b",
            max_hops=max_hops,
        )

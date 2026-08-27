from types import SimpleNamespace

from 第二阶段.graph.first_stage_adapter import FirstStageGraphAdapter
from 第二阶段.graph.graph_repository import GraphRepository


class FakeFirstStageService:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def find_entities(self, **filters):
        self.calls.append(("find_entities", filters))
        return []

    def execute_read_query(self, cypher, parameters, **options):
        self.calls.append((cypher, parameters, options))
        return SimpleNamespace(records=[])

    def close(self):
        self.closed = True


def test_repository_uses_only_read_adapter_operations() -> None:
    service = FakeFirstStageService()
    adapter = FirstStageGraphAdapter(service)
    repository = GraphRepository(adapter)
    repository.find_entity_by_name("张某")
    repository.get_one_hop_subgraph("u1")
    cypher_calls = [call[0] for call in service.calls if call[0] != "find_entities"]
    assert cypher_calls
    assert all("MATCH" in query for query in cypher_calls)
    assert not any(word in " ".join(cypher_calls) for word in ("CREATE", "MERGE", "DELETE", "SET"))


def test_repository_passes_case_scope_to_entity_and_claim_queries() -> None:
    service = FakeFirstStageService()
    repository = GraphRepository(FirstStageGraphAdapter(service))

    repository.find_entity_by_name("罗某", case_id="case-a")
    repository.find_entities_in_text("罗某与谁有关？", case_id="case-a")
    repository.get_one_hop_subgraph("luo-case-a", case_id="case-a")

    find_call = next(call for call in service.calls if call[0] == "find_entities")
    assert find_call[1]["case_id"] == "case-a"
    read_calls = [call for call in service.calls if call[0] != "find_entities"]
    assert read_calls
    assert all(call[1]["case_id"] == "case-a" for call in read_calls)
    assert all("$case_id" in call[0] for call in read_calls)

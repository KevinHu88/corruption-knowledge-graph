import pytest

from 第二阶段.exceptions import AmbiguousEntityError
from 第二阶段.retrieval.graph_retriever import GraphRetriever


class MockGraphRepository:
    def find_entities_in_text(
        self, text: str, limit: int = 10, *, case_id: str | None = None
    ):
        del text, limit
        if case_id and case_id != "case-1":
            return []
        return [{"entity": {"entity_uid": "u1", "name": "张某", "entity_type": "PER"}}]

    def find_entity_by_name(
        self, name: str, limit: int = 10, *, case_id: str | None = None
    ):
        del name, limit, case_id
        return []

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        del entity_uid, limit
        if case_id and case_id != "case-1":
            return []
        return [
            {
                "claim": {
                    "claim_id": "claim-1",
                    "relation_type": "请托",
                    "status": "HUMAN_VERIFIED",
                    "evidence_text": "张某请托李某。",
                    "doc_id": "doc-1",
                },
                "head": {"entity_uid": "u1", "name": "张某", "entity_type": "PER"},
                "tail": {"entity_uid": "u2", "name": "李某", "entity_type": "PER"},
                "evidence": [{"text": "张某请托李某。"}],
                "document": {"title": "案件材料"},
                "case": {"case_id": "case-1"},
            }
        ]


def test_graph_retriever_returns_claim_evidence() -> None:
    evidence = GraphRetriever(MockGraphRepository()).retrieve("张某与谁存在关系？")
    claim = next(item for item in evidence if item.id == "graph-claim-claim-1")
    assert claim.source_type == "graph"
    assert "请托" in claim.content
    assert claim.metadata["relationship"] == "请托"
    assert claim.metadata["relationship_properties"]["evidence_text"] == "张某请托李某。"


class DuplicateAnonymousPersonRepository:
    entities = [
        {
            "entity": {
                "entity_uid": "luo-case-a",
                "name": "罗某",
                "entity_type": "PER",
                "case_id": "case-a",
            }
        },
        {
            "entity": {
                "entity_uid": "luo-case-b",
                "name": "罗某",
                "entity_type": "PER",
                "case_id": "case-b",
            }
        },
    ]

    def __init__(self) -> None:
        self.requested_case_ids: list[str | None] = []

    def find_entities_in_text(
        self, text: str, limit: int = 10, *, case_id: str | None = None
    ):
        del text, limit
        self.requested_case_ids.append(case_id)
        return [
            record
            for record in self.entities
            if case_id is None or record["entity"]["case_id"] == case_id
        ]

    def find_entity_by_name(
        self, name: str, limit: int = 10, *, case_id: str | None = None
    ):
        del name, limit
        self.requested_case_ids.append(case_id)
        return []

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        del limit
        self.requested_case_ids.append(case_id)
        entity_case_id = "case-a" if entity_uid == "luo-case-a" else "case-b"
        if case_id is not None and case_id != entity_case_id:
            return []
        return [
            {
                "claim": {
                    "claim_id": f"claim-{entity_case_id}",
                    "relation_type": "任职",
                    "status": "HUMAN_VERIFIED",
                    "case_id": entity_case_id,
                    "evidence_text": f"罗某在{entity_case_id}任职。",
                },
                "head": {
                    "entity_uid": entity_uid,
                    "name": "罗某",
                    "case_id": entity_case_id,
                },
                "tail": {
                    "entity_uid": f"org-{entity_case_id}",
                    "name": f"单位-{entity_case_id}",
                    "case_id": entity_case_id,
                },
                "evidence": [{"text": f"罗某在{entity_case_id}任职。"}],
                "case": {"case_id": entity_case_id},
            }
        ]


def test_duplicate_anonymous_person_requires_case_id() -> None:
    repository = DuplicateAnonymousPersonRepository()

    with pytest.raises(AmbiguousEntityError) as exc_info:
        GraphRetriever(repository).retrieve("罗某与哪些人存在关系？")

    assert exc_info.value.entity_name == "罗某"
    assert exc_info.value.candidate_case_ids == ["case-a", "case-b"]


def test_case_id_filters_duplicate_anonymous_person() -> None:
    repository = DuplicateAnonymousPersonRepository()

    evidence = GraphRetriever(repository).retrieve(
        "罗某与哪些人存在关系？",
        case_id="case-b",
    )

    assert any(item.id == "graph-claim-claim-case-b" for item in evidence)
    assert not any("case-a" in item.id for item in evidence)
    assert repository.requested_case_ids
    assert set(repository.requested_case_ids) == {"case-b"}


class CrossCaseClaimRepository(MockGraphRepository):
    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        first = super().get_one_hop_subgraph(
            entity_uid,
            limit,
            case_id=case_id,
        )
        if case_id == "case-1":
            return first
        second = {
            **first[0],
            "claim": {
                **first[0]["claim"],
                "claim_id": "claim-2",
                "case_id": "case-2",
            },
            "case": {"case_id": "case-2"},
        }
        first[0]["claim"]["case_id"] = "case-1"
        return [first[0], second]


def test_cross_case_claims_for_merged_entity_require_case_id() -> None:
    with pytest.raises(AmbiguousEntityError) as exc_info:
        GraphRetriever(CrossCaseClaimRepository()).retrieve("张某与谁存在关系？")

    assert exc_info.value.entity_name == "张某"
    assert exc_info.value.candidate_case_ids == ["case-1", "case-2"]


def _path_record(
    entity_names: list[str],
    relation_types: list[str],
    claim_prefix: str,
    case_id: str = "case-1",
):
    entities = [
        {
            "entity_uid": f"{claim_prefix}-entity-{index}",
            "name": name,
            "entity_type": "PER",
            "case_id": case_id,
        }
        for index, name in enumerate(entity_names)
    ]
    claims = [
        {
            "claim_id": f"{claim_prefix}-claim-{index}",
            "relation_type": relation_type,
            "status": "HUMAN_VERIFIED",
            "case_id": case_id,
            "doc_id": "doc-1",
            "evidence_text": f"路径证据 {index}",
        }
        for index, relation_type in enumerate(relation_types)
    ]
    return {
        "path_entities": entities,
        "path_claims": claims,
        "directions": ["forward"] * len(claims),
        "hop_count": len(claims),
    }


class MultiPathRepository:
    entities = [
        {
            "entity": {
                "entity_uid": "person-zhang",
                "name": "张某",
                "entity_type": "PER",
                "case_id": "case-1",
            }
        },
        {
            "entity": {
                "entity_uid": "person-wang",
                "name": "王某",
                "entity_type": "PER",
                "case_id": "case-1",
            }
        },
    ]

    def find_entities_in_text(
        self, text: str, limit: int = 10, *, case_id: str | None = None
    ):
        del text, limit
        return self.entities if case_id in {None, "case-1"} else []

    def find_entity_by_name(
        self, name: str, limit: int = 10, *, case_id: str | None = None
    ):
        del name, limit, case_id
        return []

    def find_simple_paths(
        self,
        start_uid: str,
        end_uid: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
        max_hops: int = 3,
    ):
        del start_uid, end_uid, limit, case_id, max_hops
        return [
            _path_record(["张某", "王某"], ["请托"], "direct"),
            _path_record(
                ["张某", "李某", "王某"],
                ["请托", "帮助谋利"],
                "indirect",
            ),
        ]

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ):
        del entity_uid, limit, case_id
        raise AssertionError("路径命中后不应退回一跳检索")


def test_graph_retriever_extracts_multiple_bounded_paths() -> None:
    evidence = GraphRetriever(MultiPathRepository()).retrieve(
        "张某与王某之间有哪些路径？",
        case_id="case-1",
    )

    assert len(evidence) == 2
    assert {item.metadata["hop_count"] for item in evidence} == {1, 2}
    assert all(item.metadata["kind"] == "path" for item in evidence)
    assert any("帮助谋利" in item.content for item in evidence)


def test_graph_retriever_parses_real_neo4j_relationship_type_sequence() -> None:
    record = _path_record(["张某", "李某"], ["请托"], "direction")
    record.pop("directions")
    record["path_relationship_types"] = ["TAIL", "HEAD"]

    path = GraphRetriever._parse_path(record)

    assert path is not None
    assert path.directions == ["reverse"]


def test_graph_path_key_is_identical_when_path_is_reversed() -> None:
    record = _path_record(
        ["张某", "10万元", "李某"],
        ["收受金额", "支付金额"],
        "canonical",
    )
    reversed_record = {
        **record,
        "path_entities": list(reversed(record["path_entities"])),
        "path_claims": list(reversed(record["path_claims"])),
        "directions": ["forward", "reverse"],
    }

    forward = GraphRetriever._parse_path(record)
    reversed_path = GraphRetriever._parse_path(reversed_record)

    assert forward is not None
    assert reversed_path is not None
    assert forward.key == reversed_path.key
    assert GraphRetriever._path_id(forward) == GraphRetriever._path_id(
        reversed_path
    )


class SimilarPathRepository(MultiPathRepository):
    def find_simple_paths(self, *args, **kwargs):
        del args, kwargs
        return [
            _path_record(
                ["张某", "李某", "王某"],
                ["请托", "帮助谋利"],
                "anchor",
            )
        ]

    def find_path_candidates(
        self,
        relation_types: list[str],
        *,
        start_entity_type: str | None = None,
        end_entity_type: str | None = None,
        exclude_claim_ids: list[str] | None = None,
        case_id: str | None = None,
        case_ids: list[str] | None = None,
        max_hops: int = 3,
        limit: int = 100,
    ):
        assert relation_types == ["请托", "帮助谋利"]
        assert start_entity_type == "PER"
        assert end_entity_type == "PER"
        assert exclude_claim_ids == ["anchor-claim-0", "anchor-claim-1"]
        assert case_id is None
        assert case_ids == ["case-1"]
        del max_hops, limit
        return [
            _path_record(
                ["赵某", "钱某", "孙某"],
                ["请托", "帮助谋利"],
                "candidate",
            )
        ]


def test_graph_retriever_ranks_similar_paths_from_anchor() -> None:
    evidence = GraphRetriever(SimilarPathRepository()).retrieve(
        "查找张某与王某之间的相似路径",
        case_id="case-1",
    )

    similar = next(
        item for item in evidence if item.metadata["kind"] == "similar_path"
    )
    assert similar.metadata["similarity"]["score"] == 1.0
    assert similar.metadata["similarity"]["orientation"] == "forward"
    assert "赵某" in similar.content


class ReversedAnchorCandidateRepository(MultiPathRepository):
    anchors = [
        _path_record(
            ["张某", "10万元", "王某"],
            ["收受金额", "支付金额"],
            "anchor-money",
        ),
        _path_record(
            ["张某", "某公司", "王某"],
            ["实际控制", "利益输送"],
            "anchor-company",
        ),
    ]

    def find_simple_paths(self, *args, **kwargs):
        del args, kwargs
        return self.anchors

    def find_path_candidates(self, relation_types: list[str], **kwargs):
        del relation_types, kwargs
        return [
            {
                **record,
                "path_entities": list(reversed(record["path_entities"])),
                "path_claims": list(reversed(record["path_claims"])),
                "directions": ["forward", "reverse"],
            }
            for record in self.anchors
        ]


def test_reversed_anchor_paths_are_not_returned_as_similar_candidates() -> None:
    evidence = GraphRetriever(ReversedAnchorCandidateRepository()).retrieve(
        "查找张某与王某之间的相似路径",
        case_id="case-1",
    )

    assert len(evidence) == 2
    assert all(item.metadata["kind"] == "path" for item in evidence)


class CrossCaseSimilarPathRepository(SimilarPathRepository):
    def __init__(self) -> None:
        self.requested_case_ids: list[str] | None = None

    def find_path_candidates(self, relation_types: list[str], **kwargs):
        assert relation_types == ["请托", "帮助谋利"]
        self.requested_case_ids = kwargs.get("case_ids")
        return [
            _path_record(
                ["跨案甲", "跨案乙", "跨案丙"],
                ["请托", "帮助谋利"],
                "cross-case",
                case_id="case-2",
            )
        ]


def test_selected_cases_scope_retrieves_only_requested_cases() -> None:
    repository = CrossCaseSimilarPathRepository()

    evidence = GraphRetriever(repository).retrieve(
        "查找张某与王某之间的相似路径",
        case_id="case-1",
        search_scope="selected_cases",
        selected_case_ids=["case-2"],
    )

    assert repository.requested_case_ids == ["case-2"]
    similar = next(
        item for item in evidence if item.metadata["kind"] == "similar_path"
    )
    assert similar.metadata["search_scope"] == "selected_cases"
    assert similar.metadata["candidate_case_id"] == "case-2"


def test_all_cases_scope_uses_unbounded_case_prefilter() -> None:
    repository = CrossCaseSimilarPathRepository()

    evidence = GraphRetriever(repository).retrieve(
        "查找张某与王某之间的相似路径",
        case_id="case-1",
        search_scope="all_cases",
    )

    assert repository.requested_case_ids == []
    assert any(
        item.metadata.get("candidate_case_id") == "case-2"
        for item in evidence
    )


class UnknownCaseSimilarPathRepository(SimilarPathRepository):
    def find_path_candidates(self, relation_types: list[str], **kwargs):
        del relation_types, kwargs
        record = _path_record(
            ["未归档甲", "未归档乙", "未归档丙"],
            ["请托", "帮助谋利"],
            "unknown-case",
        )
        for item in [*record["path_entities"], *record["path_claims"]]:
            item.pop("case_id", None)
        return [record]


def test_cross_case_scope_rejects_candidate_without_case_boundary() -> None:
    evidence = GraphRetriever(UnknownCaseSimilarPathRepository()).retrieve(
        "查找张某与王某之间的相似路径",
        case_id="case-1",
        search_scope="all_cases",
    )

    assert all(item.metadata["kind"] == "path" for item in evidence)


def test_cross_case_scope_requires_anchor_case_id() -> None:
    with pytest.raises(ValueError, match="锚点 case_id"):
        GraphRetriever(CrossCaseSimilarPathRepository()).retrieve(
            "查找张某与王某之间的相似路径",
            search_scope="all_cases",
        )

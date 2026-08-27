"""Neo4jService 的纯 mock/fake Driver 单元测试。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import BaseModel

from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    CaseDocument,
    ClaimStatus,
    EntityMention,
    EntityType,
    GraphClaim,
    RelationMention,
    RelationType,
    SourceDocument,
)
from src.services.neo4j_service import (
    Neo4jAuthenticationError,
    Neo4jConfigurationError,
    Neo4jConnectionError,
    Neo4jConversionError,
    Neo4jQueryError,
    Neo4jSchemaError,
    Neo4jService,
    Neo4jServiceConfig,
    Neo4jUnsafeQueryError,
    Neo4jValidationError,
    sanitize_properties,
    serialize_neo4j_value,
)


class FakeCounters:
    nodes_created = 1
    nodes_deleted = 0
    relationships_created = 1
    relationships_deleted = 0
    properties_set = 2
    labels_added = 1
    indexes_added = 1
    constraints_added = 1


class FakeSummary:
    counters = FakeCounters()


class FakeRecord(dict):
    def data(self) -> dict[str, object]:
        return dict(self)


class FakeResult:
    def __init__(self, records: list[dict[str, object]] | None = None) -> None:
        self.records = [FakeRecord(item) for item in records or []]

    def __iter__(self):
        return iter(self.records)

    def keys(self) -> list[str]:
        return list(self.records[0]) if self.records else []

    def consume(self) -> FakeSummary:
        return FakeSummary()


class FakeTx:
    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    def run(self, query, parameters=None, **kwargs):
        params = dict(parameters or {})
        params.update(kwargs)
        query_text = str(query)
        self.driver.queries.append((query_text, params))
        if self.driver.run_error is not None:
            raise self.driver.run_error
        if "dbms.components" in query_text:
            return FakeResult([{"version": "5.fake"}])
        if "head_count" in query_text:
            return FakeResult(self.driver.validation_records)
        if "WHERE NOT" in query_text:
            return FakeResult(self.driver.orphan_records)
        if "RETURN c.case_id AS case_id" in query_text:
            return FakeResult([{"case_id": params.get("case_id")}])
        if "RETURN d.doc_version_id" in query_text:
            return FakeResult(
                [{"doc_version_id": params.get("doc_version_id")}]
            )
        if "RETURN s.text_uid AS text_uid" in query_text:
            return FakeResult([{"text_uid": params.get("text_uid")}])
        if "RETURN row.entity_id AS entity_id" in query_text:
            return FakeResult(
                [
                    {
                        "entity_id": row["entity_id"],
                        "entity_uid": row["entity_uid"],
                        "mention_uid": row["mention_uid"],
                    }
                    for row in params.get("rows", [])
                ]
            )
        if "RETURN c.claim_id AS claim_id" in query_text:
            return FakeResult(
                [
                    {"claim_id": row["claim_id"]}
                    for row in params.get("rows", [])
                ]
            )
        return FakeResult(self.driver.generic_records)


class FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args) -> None:
        return None

    def execute_write(self, function, *args):
        if self.driver.transaction_error is not None:
            raise self.driver.transaction_error
        return function(FakeTx(self.driver), *args)

    def execute_read(self, function, *args):
        if self.driver.transaction_error is not None:
            raise self.driver.transaction_error
        return function(FakeTx(self.driver), *args)


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, object]]] = []
        self.sessions: list[dict[str, object]] = []
        self.closed = 0
        self.verify_error: Exception | None = None
        self.run_error: Exception | None = None
        self.transaction_error: Exception | None = None
        self.generic_records: list[dict[str, object]] = []
        self.validation_records: list[dict[str, object]] = []
        self.orphan_records: list[dict[str, object]] = []

    def verify_connectivity(self) -> None:
        if self.verify_error is not None:
            raise self.verify_error

    def get_server_info(self):
        return SimpleNamespace(address="localhost:7687", agent="Neo4j/5.fake")

    def session(self, **kwargs) -> FakeSession:
        self.sessions.append(kwargs)
        return FakeSession(self)

    def close(self) -> None:
        self.closed += 1


def _service(
    driver: FakeDriver | None = None, **overrides: object
) -> Neo4jService:
    values: dict[str, object] = {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "secret",
        "database": "kg",
        "batch_size": 2,
    }
    values.update(overrides)
    return Neo4jService(
        Neo4jServiceConfig(**values), driver=driver or FakeDriver()
    )


def _documents() -> tuple[SourceDocument, CaseDocument]:
    source = SourceDocument(
        doc_id="d1",
        doc_version_id="dv1",
        source_id="court",
        title="案件",
    )
    case = CaseDocument(
        case_id="case1",
        doc_id="d1",
        doc_version_id="dv1",
        title="案件",
        source_id="court",
        raw_text="李某请托张某。",
        clean_text="李某请托张某。",
    )
    return source, case


def _annotation(
    *,
    status: AnnotationStatus = AnnotationStatus.APPROVED,
    with_relation: bool = True,
) -> CanonicalAnnotation:
    entities = [
        EntityMention(
            entity_id="e1",
            name="李某",
            type=EntityType.PER,
            start=0,
            end=2,
        ),
        EntityMention(
            entity_id="e2",
            name="张某",
            type=EntityType.PER,
            start=4,
            end=6,
        ),
    ]
    relations = (
        [
            RelationMention(
                relation_id="r1",
                head_id="e1",
                tail_id="e2",
                type=RelationType.ENTRUST,
                evidence_start=0,
                evidence_end=6,
                extraction_source="HUMAN",
            )
        ]
        if with_relation
        else []
    )
    return CanonicalAnnotation(
        annotation_id="a1",
        case_id="case1",
        doc_id="d1",
        text_id="t1",
        text="李某请托张某。",
        entities=entities,
        relations=relations,
        annotation_source="HUMAN",
        schema_version="relation_v2.0",
        status=status,
    )


@pytest.mark.parametrize("field", ["uri", "username", "password", "database"])
def test_missing_configuration_is_rejected(field: str) -> None:
    values = {
        "uri": "neo4j://localhost",
        "username": "neo4j",
        "password": "secret",
        "database": "kg",
    }
    values[field] = ""
    with pytest.raises(Neo4jConfigurationError, match=field):
        Neo4jService(
            Neo4jServiceConfig(**values), driver=FakeDriver()
        )


def test_health_check_and_explicit_database() -> None:
    driver = FakeDriver()
    result = _service(driver).health_check()
    assert result.connected is True
    assert result.server_version == "5.fake"
    assert driver.sessions[0]["database"] == "kg"
    assert driver.sessions[0]["fetch_size"] == 1000


@pytest.mark.parametrize(
    ("exception_name", "expected"),
    [
        ("AuthError", Neo4jAuthenticationError),
        ("ServiceUnavailable", Neo4jConnectionError),
        ("SessionExpired", Neo4jConnectionError),
    ],
)
def test_health_errors_are_converted(exception_name: str, expected: type) -> None:
    error_type = type(exception_name, (Exception,), {})
    driver = FakeDriver()
    driver.verify_error = error_type("failure")
    with pytest.raises(expected):
        _service(driver).health_check()


def test_close_is_idempotent_and_context_manager_closes() -> None:
    driver = FakeDriver()
    with _service(driver) as service:
        assert service is not None
    service.close()
    assert driver.closed == 1


def test_initialize_schema_has_constraints_indexes_and_no_drop() -> None:
    driver = FakeDriver()
    result = _service(driver).initialize_schema()
    cypher = "\n".join(query for query, _ in driver.queries)
    assert "CREATE CONSTRAINT case_id_unique IF NOT EXISTS" in cypher
    assert "CREATE INDEX entity_name_idx IF NOT EXISTS" in cypher
    assert "DROP" not in cypher
    assert len(result.items) >= 17


def test_schema_error_is_converted() -> None:
    driver = FakeDriver()
    driver.run_error = RuntimeError("bad")
    with pytest.raises(Neo4jSchemaError):
        _service(driver).initialize_schema()


def test_schema_snapshot_uses_standard_commands() -> None:
    driver = FakeDriver()
    driver.generic_records = [{"name": "item"}]
    snapshot = _service(driver).get_schema_snapshot()
    cypher = "\n".join(query for query, _ in driver.queries)
    assert "CALL db.labels()" in cypher
    assert "CALL db.relationshipTypes()" in cypher
    assert "SHOW CONSTRAINTS" in cypher
    assert "SHOW INDEXES" in cypher
    assert snapshot["constraints"] == [{"name": "item"}]


def test_upsert_case_and_document_use_unique_merge_keys() -> None:
    driver = FakeDriver()
    service = _service(driver)
    source, case = _documents()
    service.upsert_case(case)
    service.upsert_source_document(source, case_id="case1")
    cypher = "\n".join(query for query, _ in driver.queries)
    assert "MERGE (c:Case {case_id: $case_id})" in cypher
    assert "MERGE (d:SourceDocument {doc_version_id: $doc_version_id})" in cypher
    assert "BELONGS_TO_CASE" in cypher
    assert "案件" not in cypher


def test_upsert_text_span_validates_offsets_and_relationship() -> None:
    driver = FakeDriver()
    service = _service(driver)
    with pytest.raises(Neo4jValidationError):
        service.upsert_text_span(
            annotation_id="a",
            text_id="t",
            case_id="c",
            doc_id="d",
            doc_version_id="dv",
            text="短文",
            end=3,
        )
    service.upsert_text_span(
        annotation_id="a",
        text_id="t",
        case_id="c",
        doc_id="d",
        doc_version_id="dv",
        text="短文",
    )
    assert "FROM_DOCUMENT" in driver.queries[-1][0]


@pytest.mark.parametrize(
    ("entity_type", "label"),
    [
        (EntityType.PER, "Person"),
        (EntityType.ORG, "Organization"),
        (EntityType.POSITION, "Position"),
        (EntityType.MONEY, "Money"),
    ],
)
def test_entity_type_uses_fixed_label(
    entity_type: EntityType, label: str
) -> None:
    driver = FakeDriver()
    service = _service(driver)
    entity = EntityMention(
        entity_id="e",
        name="甲",
        type=entity_type,
        start=0,
        end=1,
    )
    result = service.upsert_entities(
        [entity],
        case_id="case1",
        annotation_id="a1",
        text_id="t1",
        text="甲",
    )
    assert f":Entity:{label}" in driver.queries[-1][0]
    assert result.mention_uid_map["e"] == "a1:e"


def test_entity_uid_map_wins_and_default_is_case_scoped() -> None:
    driver = FakeDriver()
    service = _service(driver)
    entity = _annotation().entities[0]
    explicit = service.upsert_entities(
        [entity],
        case_id="case1",
        annotation_id="a1",
        text_id="t1",
        text=_annotation().text,
        entity_uid_map={"e1": "resolved-person"},
    )
    assert explicit.entity_uid_map["e1"] == "resolved-person"
    default = service.upsert_entities(
        [entity],
        case_id="other-case",
        annotation_id="a2",
        text_id="t1",
        text=_annotation().text,
    )
    assert default.entity_uid_map["e1"] != "resolved-person"


def test_entity_offset_and_unknown_type_are_rejected() -> None:
    service = _service()
    entity = _annotation().entities[0].model_copy(update={"name": "王某"})
    with pytest.raises(Neo4jValidationError):
        service.upsert_entities(
            [entity],
            case_id="c",
            annotation_id="a",
            text_id="t",
            text=_annotation().text,
        )
    unknown = EntityMention.model_construct(
        entity_id="x", name="甲", type="BAD", start=0, end=1
    )
    with pytest.raises(Neo4jValidationError):
        service.upsert_entities(
            [unknown],
            case_id="c",
            annotation_id="a",
            text_id="t",
            text="甲",
        )


def _claim(**updates: object) -> GraphClaim:
    values: dict[str, object] = {
        "claim_id": "claim1",
        "case_id": "case1",
        "doc_id": "d1",
        "text_id": "t1",
        "head_entity_id": "e1",
        "tail_entity_id": "e2",
        "relation": RelationType.ENTRUST,
        "evidence_text": "李某请托张某",
        "evidence_start": 0,
        "evidence_end": 6,
        "status": ClaimStatus.MODEL_PREDICTED,
    }
    values.update(updates)
    try:
        return GraphClaim(**values)
    except Exception:
        return GraphClaim.model_construct(**values)


def test_claim_upsert_uses_unwind_fixed_edges_and_state_case() -> None:
    driver = FakeDriver()
    result = _service(driver).upsert_claims(
        [_claim()],
        annotation_id="a1",
        entity_uid_map={"e1": "u1", "e2": "u2"},
        entity_types={"e1": "PER", "e2": "PER"},
    )
    query, params = driver.queries[-1]
    assert "UNWIND $rows" in query
    assert all(edge in query for edge in ("[:HEAD]", "[:TAIL]", "[:SUPPORTED_BY]"))
    assert "HUMAN_VERIFIED" in query and "REJECTED" in query
    assert result.successful_claim_ids == ["claim1"]
    assert params["rows"][0]["head_entity_uid"] == "u1"


@pytest.mark.parametrize(
    "updates",
    [
        {"relation": RelationType.NO_RELATION},
        {"head_entity_id": "e1", "tail_entity_id": "e1"},
        {"evidence_text": "", "evidence_end": 0},
    ],
)
def test_invalid_claims_are_rejected(updates: dict[str, object]) -> None:
    with pytest.raises(Neo4jValidationError):
        _service().upsert_claims(
            [_claim(**updates)],
            annotation_id="a1",
            entity_types={"e1": "PER", "e2": "PER"},
        )


def test_claim_schema_type_constraint_is_checked() -> None:
    with pytest.raises(Neo4jValidationError, match="头实体类型"):
        _service().upsert_claims(
            [_claim()],
            annotation_id="a1",
            entity_types={"e1": "ORG", "e2": "PER"},
        )


def test_rejected_claim_can_be_skipped_by_configuration() -> None:
    service = _service(FakeDriver(), store_rejected_claims=False)
    result = service.upsert_claims(
        [_claim(status=ClaimStatus.REJECTED)],
        annotation_id="a1",
        entity_types={"e1": "PER", "e2": "PER"},
    )
    assert result.skipped_claim_ids == ["claim1"]
    assert result.successful_claim_ids == []


def test_complete_annotation_is_one_managed_transaction() -> None:
    driver = FakeDriver()
    source, case = _documents()
    result = _service(driver).ingest_annotation(
        _annotation(), source, case
    )
    assert result.success
    assert result.successful_claim_ids == ["a1:r1"]
    assert len(driver.sessions) == 1
    cypher = "\n".join(query for query, _ in driver.queries)
    assert "BELONGS_TO_CASE" in cypher
    assert "CONTAINS_MENTION" in cypher
    assert "SUPPORTED_BY" in cypher


def test_annotation_traceability_metadata_is_persisted() -> None:
    driver = FakeDriver()
    source, case = _documents()
    annotation = _annotation().model_copy(
        update={
            "metadata": {
                "chunk_id": "chunk-1",
                "dataset_version": "mydata-v1",
                "dataset_splits": ["train"],
                "source_files": ["train.jsonl"],
                "relation_provenance": {
                    "r1": {
                        "dataset_splits": ["train"],
                        "source_files": ["train.jsonl"],
                        "source_rows": ["train.jsonl:1"],
                        "original_relation_types": ["请托"],
                    }
                },
            }
        }
    )
    _service(driver).ingest_annotation(annotation, source, case)
    claim_rows = next(
        parameters["rows"]
        for query, parameters in driver.queries
        if "RETURN c.claim_id AS claim_id" in query
    )
    span_properties = next(
        parameters["properties"]
        for query, parameters in driver.queries
        if "RETURN s.text_uid AS text_uid" in query
    )
    assert claim_rows[0]["properties"]["chunk_id"] == "chunk-1"
    assert claim_rows[0]["properties"]["source_rows"] == [
        "train.jsonl:1"
    ]
    assert span_properties["chunk_id"] == "chunk-1"
    assert span_properties["dataset_splits"] == ["train"]


def test_empty_annotation_and_review_policy() -> None:
    driver = FakeDriver()
    source, case = _documents()
    result = _service(driver).ingest_annotation(
        _annotation(with_relation=False).model_copy(update={"entities": []}),
        source,
        case,
    )
    assert result.successful_claim_ids == []
    with pytest.raises(Neo4jValidationError, match="尚未审核"):
        _service(
            FakeDriver(), allow_unreviewed_annotations=False
        ).ingest_annotation(
            _annotation(status=AnnotationStatus.GENERATED), source, case
        )


def test_batching_and_continue_on_error() -> None:
    annotations = [
        _annotation().model_copy(
            update={"annotation_id": f"a{i}", "case_id": f"case{i}"}
        )
        for i in range(3)
    ]
    driver = FakeDriver()
    service = _service(driver)
    result = service.ingest_annotations_batch(
        annotations,
        case_documents={
            f"case{i}": {
                **_documents()[1].model_dump(),
                "case_id": f"case{i}",
            }
            for i in range(3)
        },
    )
    assert result.total_batches == 2
    assert result.failed_batches == 0


def test_batch_continue_on_error_records_every_failed_batch() -> None:
    driver = FakeDriver()
    driver.transaction_error = RuntimeError("transaction failed")
    service = _service(driver)
    annotations = [
        _annotation().model_copy(update={"annotation_id": f"a{i}"})
        for i in range(3)
    ]
    result = service.ingest_annotations_batch(
        annotations, continue_on_error=True
    )
    assert result.total_batches == 2
    assert result.failed_batches == 2
    assert all(not batch.success for batch in result.batches)


def test_batch_stops_on_error_by_default() -> None:
    driver = FakeDriver()
    driver.transaction_error = RuntimeError("transaction failed")
    with pytest.raises(Exception, match="批次 0"):
        _service(driver).ingest_annotations_batch([_annotation()])


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (n)",
        "MERGE (n:X {id: 1})",
        "MATCH (n) DELETE n",
        "MATCH (n) SET n.x = 1",
        "LOAD CSV FROM 'x' AS row RETURN row",
        "MATCH (n) RETURN n; MATCH (m) RETURN m",
        "CALL dbms.components()",
    ],
)
def test_unsafe_read_queries_are_rejected(cypher: str) -> None:
    with pytest.raises(Neo4jUnsafeQueryError):
        _service().execute_read_query(cypher, validated=True)


def test_unvalidated_query_is_rejected() -> None:
    with pytest.raises(Neo4jUnsafeQueryError):
        _service().execute_read_query(
            "MATCH (n) RETURN n", validated=False
        )


def test_read_query_is_parameterized_bounded_and_truncated() -> None:
    driver = FakeDriver()
    driver.generic_records = [{"x": 1}, {"x": 2}, {"x": 3}]
    result = _service(driver).execute_read_query(
        "MATCH (n {name: $name}) RETURN n AS x",
        {"name": "张某"},
        max_records=2,
    )
    query, params = driver.queries[-1]
    assert "$name" in query and "张某" not in query
    assert params["name"] == "张某"
    assert result.truncated is True
    assert len(result.records) == 2


def test_read_query_error_is_converted() -> None:
    driver = FakeDriver()
    driver.run_error = RuntimeError("bad query")
    with pytest.raises(Neo4jQueryError):
        _service(driver).execute_read_query("MATCH (n) RETURN n")


def test_get_claim_returns_record_or_none() -> None:
    driver = FakeDriver()
    service = _service(driver)
    assert service.get_claim("missing") is None
    driver.generic_records = [{"claim": {"claim_id": "c1"}}]
    assert service.get_claim("c1") == {"claim": {"claim_id": "c1"}}
    assert driver.queries[-1][1] == {"claim_id": "c1"}


def test_find_entities_is_parameterized_and_bounded() -> None:
    driver = FakeDriver()
    driver.generic_records = [{"e": {"entity_uid": "u1"}}]
    service = _service(driver, max_query_limit=50)
    result = service.find_entities(
        name="张某", entity_type="PER", case_id="case1", limit=999
    )
    query, params = driver.queries[-1]
    assert "张某" not in query
    assert params["name"] == "张某"
    assert params["limit"] == 50
    assert result
    with pytest.raises(Neo4jValidationError):
        service.find_entities(entity_type="BAD")


def test_list_entity_claims_preserves_direction_and_excludes_rejected() -> None:
    driver = FakeDriver()
    service = _service(driver)
    service.list_entity_claims("u1", role="head", limit=10)
    query, params = driver.queries[-1]
    assert "head.entity_uid = $entity_uid" in query
    assert "c.status <> 'REJECTED'" in query
    assert params["include_rejected"] is False


def test_get_case_graph_uses_three_bounded_queries() -> None:
    driver = FakeDriver()
    driver.generic_records = [{"value": 1}]
    result = _service(driver).get_case_graph("case1", limit=10)
    assert len(driver.queries) == 3
    assert result["case_id"] == "case1"
    assert result["entities"] == [{"value": 1}]


class FakeNode(dict):
    element_id = "n1"
    labels = {"Entity", "Person"}


class FakeRelationship(dict):
    element_id = "r1"
    type = "HEAD"
    start_node_element_id = "n1"
    end_node_element_id = "n2"


class FakePath:
    nodes = [FakeNode(name="张某")]
    relationships = [FakeRelationship()]


def test_neo4j_native_values_are_serialized() -> None:
    node = serialize_neo4j_value(FakeNode(name="张某"))
    relationship = serialize_neo4j_value(FakeRelationship(weight=1))
    path = serialize_neo4j_value(FakePath())
    assert node["element_id"] == "n1"
    assert node["labels"] == ["Entity", "Person"]
    assert relationship["type"] == "HEAD"
    assert path["relationships"][0]["end_node_element_id"] == "n2"


class ExampleModel(BaseModel):
    status: ClaimStatus
    created_at: datetime


def test_sanitize_properties_supports_types_and_filters_secrets() -> None:
    value = sanitize_properties(
        {
            "status": ClaimStatus.HUMAN_VERIFIED,
            "created_at": datetime(2026, 1, 1),
            "path": Path("a/b"),
            "uuid": UUID("12345678-1234-5678-1234-567812345678"),
            "none": None,
            "metadata": {"名称": "中文", "password": "bad"},
            "api_key": "bad",
            "model": ExampleModel(
                status=ClaimStatus.MODEL_PREDICTED,
                created_at=datetime(2026, 1, 1),
            ),
        }
    )
    assert value["status"] == "HUMAN_VERIFIED"
    assert value["path"] == str(Path("a/b"))
    assert "none" not in value and "api_key" not in value
    assert "\\u" not in value["metadata"]
    assert "bad" not in value["metadata"]
    assert json.loads(value["model"])["status"] == "MODEL_PREDICTED"


def test_unsupported_property_type_raises() -> None:
    with pytest.raises(Neo4jConversionError):
        sanitize_properties({"bad": object()})


@pytest.mark.parametrize(
    ("record", "code"),
    [
        (
            {
                "claim_id": "c1",
                "relation_type": "请托",
                "head_count": 0,
                "tail_count": 1,
                "evidence_count": 1,
            },
            "invalid_head_count",
        ),
        (
            {
                "claim_id": "c1",
                "relation_type": "请托",
                "head_count": 1,
                "tail_count": 0,
                "evidence_count": 1,
            },
            "invalid_tail_count",
        ),
        (
            {
                "claim_id": "c1",
                "relation_type": "请托",
                "head_count": 1,
                "tail_count": 1,
                "evidence_count": 0,
            },
            "missing_evidence",
        ),
        (
            {
                "claim_id": "c1",
                "relation_type": "无关系",
                "head_count": 1,
                "tail_count": 1,
                "evidence_count": 1,
            },
            "negative_relation_claim",
        ),
    ],
)
def test_ingestion_validation_reports_structure(
    record: dict[str, object], code: str
) -> None:
    driver = FakeDriver()
    driver.validation_records = [record]
    result = _service(driver).validate_ingestion(claim_ids=["c1"])
    assert result.valid is False
    assert any(item.code == code for item in result.errors)


def test_complete_claim_validation_passes() -> None:
    driver = FakeDriver()
    driver.validation_records = [
        {
            "claim_id": "c1",
            "relation_type": "请托",
            "head_count": 1,
            "tail_count": 1,
            "evidence_count": 1,
        }
    ]
    result = _service(driver).validate_ingestion(claim_ids=["c1"])
    assert result.valid is True
    assert result.errors == []

"""LabelStudioService 的离线 fake SDK 和双向转换测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from config import load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    EntityMention,
    EntityType,
    RelationMention,
    RelationType,
)
from src.services.label_studio_service import (
    LabelStudioConfigurationError,
    LabelStudioConnectionError,
    LabelStudioConversionError,
    LabelStudioProjectError,
    LabelStudioService,
    LabelStudioTaskNotFoundError,
)


class ImportResponse(BaseModel):
    task_count: int
    task_ids: list[int]
    import_id: int


class FakeProjects:
    def __init__(self, project=None, error=None, import_error=None):
        self.project = project or {"id": 7, "title": "审核项目"}
        self.error = error
        self.import_error = import_error
        self.import_calls = []

    def get(self, *, id):
        if self.error:
            raise self.error
        return {**self.project, "id": id}

    def import_tasks(self, *, id, request, return_task_ids):
        if self.import_error:
            raise self.import_error
        if self.error:
            raise self.error
        self.import_calls.append((id, request, return_task_ids))
        start = 100 + sum(len(item[1]) for item in self.import_calls[:-1])
        return ImportResponse(
            task_count=len(request),
            task_ids=list(range(start, start + len(request))),
            import_id=len(self.import_calls),
        )


class FakeTasks:
    def __init__(self, tasks=None, error=None):
        self.tasks = list(tasks or [])
        self.error = error
        self.list_calls = []

    def get(self, *, id):
        if self.error:
            raise self.error
        for task in self.tasks:
            if task["id"] == id:
                return task
        raise FakeApiError(404)

    def list(self, **kwargs):
        if self.error:
            raise self.error
        self.list_calls.append(kwargs)
        page = kwargs["page"]
        size = kwargs["page_size"]
        start = (page - 1) * size
        return {
            "tasks": self.tasks[start:start + size],
            "total": len(self.tasks),
        }


class FakeApiError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def fake_client(project=None, tasks=None, error=None, import_error=None):
    return SimpleNamespace(
        projects=FakeProjects(project, error, import_error),
        tasks=FakeTasks(tasks, error),
    )


def service(client=None, **kwargs):
    return LabelStudioService(
        project_config=load_project_config(),
        client=client or fake_client(),
        base_url="http://label-studio.test",
        api_key="test-key",
        project_id=7,
        **kwargs,
    )


def annotation(
    *,
    entities=None,
    relations=None,
    annotation_id="a1",
):
    text = "张三请托李四支付人民币二十万元"
    return CanonicalAnnotation(
        annotation_id=annotation_id,
        case_id="case-1",
        doc_id="doc-1",
        text_id="text-1",
        text=text,
        entities=entities
        if entities is not None
        else [
            EntityMention(
                entity_id="e1", name="张三", type=EntityType.PER,
                start=0, end=2, confidence=0.9,
            ),
            EntityMention(
                entity_id="e2", name="李四", type=EntityType.PER,
                start=4, end=6, confidence=0.8,
            ),
        ],
        relations=relations
        if relations is not None
        else [
            RelationMention(
                relation_id="r1", head_id="e1", tail_id="e2",
                type=RelationType.ENTRUST, confidence=0.7,
                evidence_start=0, evidence_end=6,
                extraction_source="DEEP_MODEL",
            )
        ],
        annotation_source="DEEP_MODEL",
        schema_version="relation_v2.0",
        status=AnnotationStatus.PENDING_REVIEW,
    )


def reviewed_task(result, *, annotations=None):
    return {
        "id": 101,
        "project": 7,
        "data": {
            "text": "张三请托李四支付人民币二十万元",
            "annotation_id": "a1",
            "case_id": "case-1",
            "doc_id": "doc-1",
            "text_id": "text-1",
            "schema_version": "relation_v2.0",
        },
        "predictions": [{"model_version": "ner-re-v1"}],
        "annotations": annotations
        if annotations is not None
        else [{
            "id": 501,
            "updated_at": "2026-07-28T10:00:00Z",
            "completed_by": {"id": 8},
            "was_cancelled": False,
            "result": result,
        }],
    }


def entity_result(id, start, end, text, label):
    return {
        "id": id,
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": start, "end": end, "text": text,
            "labels": [label],
        },
    }


def relation_result(head, tail, label, id="r1"):
    return {
        "id": id,
        "from_id": head,
        "to_id": tail,
        "type": "relation",
        "direction": "right",
        "labels": [label],
    }


def test_configuration_missing_is_clear(monkeypatch):
    monkeypatch.delenv("LABEL_STUDIO_URL", raising=False)
    monkeypatch.delenv("LABEL_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("LABEL_STUDIO_PROJECT_ID", raising=False)

    with pytest.raises(LabelStudioConfigurationError):
        LabelStudioService(
            project_config=load_project_config(),
            client=fake_client(),
        )


def test_health_check_and_missing_project_conversion():
    result = service().health_check()
    assert result.connected is True
    assert result.project_id == 7
    assert result.project_title == "审核项目"

    missing = service(client=fake_client(error=FakeApiError(404)))
    with pytest.raises(LabelStudioProjectError):
        missing.health_check()


def test_task_data_and_four_entity_types_conversion():
    text = "张三在甲公司任局长收十万元"
    entities = [
        EntityMention(
            entity_id="e1", name="张三", type="PER", start=0, end=2
        ),
        EntityMention(
            entity_id="e2", name="甲公司", type="ORG", start=3, end=6
        ),
        EntityMention(
            entity_id="e3", name="局长", type="POSITION", start=7, end=9
        ),
        EntityMention(
            entity_id="e4", name="十万元", type="MONEY", start=10, end=13
        ),
    ]
    value = annotation(entities=[], relations=[]).model_copy(
        update={"text": text, "entities": entities}
    )
    api = service()

    data = api.build_task_payload(value)
    results = api.build_prediction_results(value)

    assert data["annotation_id"] == "a1"
    assert data["case_id"] == "case-1"
    assert [item["value"]["labels"][0] for item in results] == [
        "PER", "ORG", "POSITION", "MONEY"
    ]
    assert all(
        text[item["value"]["start"]:item["value"]["end"]]
        == item["value"]["text"]
        for item in results
    )


def test_relation_direction_and_empty_annotation_prediction():
    api = service()
    results = api.build_prediction_results(annotation())
    relation = results[-1]

    assert relation["from_id"] == "e1"
    assert relation["to_id"] == "e2"
    assert relation["labels"] == ["请托"]
    assert relation["direction"] == "right"
    assert api.build_prediction_results(
        annotation(entities=[], relations=[])
    ) == []


def test_import_task_overlays_entities_and_relation_on_chunk_text():
    payload = service().build_import_task(
        annotation(), model_version="gpt-5.4-mini"
    )

    assert payload["data"]["text"] == "张三请托李四支付人民币二十万元"
    assert len(payload["predictions"]) == 1
    prediction = payload["predictions"][0]
    assert prediction["model_version"] == "gpt-5.4-mini"
    assert [item["type"] for item in prediction["result"]] == [
        "labels", "labels", "relation"
    ]
    assert prediction["result"][-1] == {
        "from_id": "e1",
        "to_id": "e2",
        "type": "relation",
        "direction": "right",
        "labels": ["请托"],
    }


def test_import_task_uses_empty_prediction_only_when_annotation_is_empty():
    payload = service().build_import_task(
        annotation(entities=[], relations=[]),
        model_version="gpt-5.4-mini",
    )

    assert payload["data"]["text"]
    assert payload["predictions"][0]["result"] == []


def test_invalid_entity_or_missing_relation_reference_is_rejected():
    api = service()
    bad_entity = EntityMention.model_construct(
        entity_id="e1", name="错误", type=EntityType.PER,
        start=0, end=2, confidence=None, normalized_name=None,
    )
    invalid_data = annotation(entities=[], relations=[]).__dict__.copy()
    invalid_data["entities"] = [bad_entity]
    invalid = CanonicalAnnotation.model_construct(**invalid_data)
    with pytest.raises(LabelStudioConversionError):
        api.build_prediction_results(invalid)

    bad_relation = RelationMention(
        relation_id="r1", head_id="missing", tail_id="e2",
        type="请托",
    )
    relation_data = annotation(relations=[]).__dict__.copy()
    relation_data["relations"] = [bad_relation]
    invalid_relation = CanonicalAnnotation.model_construct(**relation_data)
    with pytest.raises(LabelStudioConversionError):
        api.build_prediction_results(invalid_relation)


def test_batch_import_mapping_split_and_empty_short_circuit():
    client = fake_client()
    api = service(client=client, batch_size=2)
    client.projects.project["label_config"] = api.build_label_config()
    values = [annotation(annotation_id=f"a{i}") for i in range(5)]

    result = api.import_annotations(values)

    assert [len(call[1]) for call in client.projects.import_calls] == [2, 2, 1]
    assert result.imported_count == 5
    assert result.failed_count == 0
    assert {item.annotation_id: item.task_id for item in result.mappings} == {
        "a0": 100, "a1": 101, "a2": 102, "a3": 103, "a4": 104,
    }
    calls = len(client.projects.import_calls)
    assert api.import_annotations([]).requested_count == 0
    assert len(client.projects.import_calls) == calls


def test_import_sdk_error_is_not_reported_as_success():
    client = fake_client(import_error=RuntimeError("failed"))
    api = service(client=client)
    client.projects.project["label_config"] = api.build_label_config()
    result = api.import_annotations([annotation()])

    assert result.imported_count == 0
    assert result.failed_count == 1
    assert result.batches[0].error


def test_oversized_prediction_is_omitted_from_import_task():
    api = service()
    api.max_prediction_results_per_task = 1

    payload = api.build_import_task(annotation())

    assert "predictions" not in payload
    assert payload["data"]["text"]


def test_generated_label_config_matches_schema():
    api = service()

    result = api.validate_project_label_config({
        "label_config": api.build_label_config(),
    })

    assert result.valid is True


def test_import_connection_error_propagates_for_task_retry():
    class FakeConnectionError(RuntimeError):
        pass

    api = service(client=fake_client(error=FakeConnectionError("offline")))

    with pytest.raises(LabelStudioConnectionError):
        api.import_annotations([annotation()])


def test_reviewed_entity_relation_round_trip_and_human_changes():
    results = [
        entity_result("new-e1", 0, 2, "张三", "PER"),
        entity_result("new-e2", 4, 6, "李四", "PER"),
        relation_result("new-e2", "new-e1", "请托", "new-r1"),
    ]
    converted = service().convert_reviewed_task(reviewed_task(results))

    assert converted.status == AnnotationStatus.APPROVED
    assert converted.annotation_source == "HUMAN"
    assert [item.entity_id for item in converted.entities] == [
        "new-e1", "new-e2"
    ]
    assert converted.relations[0].head_id == "new-e2"
    assert converted.relations[0].tail_id == "new-e1"
    assert converted.relations[0].extraction_source == "HUMAN"
    assert converted.metadata["label_studio_task_id"] == 101


def test_human_deleted_predictions_are_not_retained():
    final_result = [entity_result("e1", 0, 2, "张三", "PER")]
    converted = service().convert_reviewed_task(reviewed_task(final_result))

    assert len(converted.entities) == 1
    assert converted.relations == []


def test_invalid_human_boundary_and_missing_region_are_rejected():
    api = service()
    with pytest.raises(LabelStudioConversionError):
        api.convert_reviewed_task(
            reviewed_task([entity_result("e1", 0, 99, "张三", "PER")])
        )

    with pytest.raises(LabelStudioConversionError):
        api.convert_reviewed_task(
            reviewed_task([
                entity_result("e1", 0, 2, "张三", "PER"),
                relation_result("e1", "missing", "请托"),
            ])
        )


def test_cancelled_and_empty_annotations_ignored_latest_valid_selected():
    result_old = [entity_result("old", 0, 2, "张三", "PER")]
    result_new = [entity_result("new", 4, 6, "李四", "PER")]
    annotations = [
        {"id": 1, "updated_at": "2026-01-01T00:00:00Z",
         "was_cancelled": False, "result": result_old},
        {"id": 2, "updated_at": "2026-02-01T00:00:00Z",
         "was_cancelled": True, "result": result_old},
        {"id": 3, "updated_at": "2026-03-01T00:00:00Z",
         "was_cancelled": False, "result": []},
        {"id": 4, "updated_at": "2026-04-01T00:00:00Z",
         "was_cancelled": False, "result": result_new},
    ]
    task = reviewed_task([], annotations=annotations)
    api = service(client=fake_client(tasks=[task]))

    reviewed = api.fetch_reviewed_tasks()

    assert len(reviewed) == 1
    assert reviewed[0].annotation["id"] == 4


def test_no_human_annotation_does_not_treat_prediction_as_review():
    task = reviewed_task([], annotations=[])
    task["predictions"] = [{
        "model_version": "model-v1",
        "result": [entity_result("e1", 0, 2, "张三", "PER")],
    }]
    api = service(client=fake_client(tasks=[task]))

    assert api.fetch_reviewed_tasks() == []


def test_sync_reviewed_annotations_returns_data_without_persistence():
    task = reviewed_task([
        entity_result("e1", 0, 2, "张三", "PER"),
        entity_result("e2", 4, 6, "李四", "PER"),
        relation_result("e1", "e2", "请托"),
    ])
    api = service(client=fake_client(tasks=[task]))

    result = api.sync_reviewed_annotations(task_ids=[101])

    assert result.task_ids == [101]
    assert result.failed_task_ids == []
    assert len(result.annotations) == 1
    assert result.annotations[0].status == AnnotationStatus.APPROVED


def test_sync_skips_missing_requested_task_and_reports_failure():
    task = reviewed_task([
        entity_result("e1", 0, 2, "张三", "PER"),
    ])
    api = service(client=fake_client(tasks=[task]))

    result = api.sync_reviewed_annotations(task_ids=[101, 999])

    assert result.task_ids == [101]
    assert result.failed_task_ids == [999]
    assert result.errors == ["Label Studio 任务不存在：task_id=999"]
    assert len(result.annotations) == 1


def test_direct_missing_task_has_specific_error():
    with pytest.raises(LabelStudioTaskNotFoundError, match="task_id=999"):
        service().get_task(999)


def test_task_pagination_and_model_dump_sdk_object():
    tasks = [
        reviewed_task([entity_result(f"e{i}", 0, 2, "张三", "PER")])
        | {"id": i}
        for i in range(1, 6)
    ]
    client = fake_client(tasks=tasks)
    api = service(client=client)

    output = api.list_tasks(page_size=2, max_items=5)

    assert len(output) == 5
    assert [item["page"] for item in client.tasks.list_calls] == [1, 2, 3]


def label_config(api, *, remove_entity=None, remove_relation=None):
    entities = [
        item for item in ("PER", "ORG", "POSITION", "MONEY")
        if item != remove_entity
    ]
    relations = [
        item for item in api.schema["relation_types"]
        if item != remove_relation
    ]
    return (
        '<View><Relations name="relation" toName="label">'
        + "".join(f'<Relation value="{item}"/>' for item in relations)
        + '</Relations><Labels name="label" toName="text">'
        + "".join(f'<Label value="{item}"/>' for item in entities)
        + '</Labels><Text name="text" value="$text"/></View>'
    )


def test_project_label_config_xml_validation():
    api = service()
    valid = api.validate_project_label_config(
        {"label_config": label_config(api)}
    )
    assert valid.valid is True

    with pytest.raises(LabelStudioProjectError, match="missing_entities"):
        api.validate_project_label_config(
            {"label_config": label_config(api, remove_entity="MONEY")}
        )
    relation = next(iter(api.schema["relation_types"]))
    with pytest.raises(LabelStudioProjectError, match="missing_relations"):
        api.validate_project_label_config(
            {"label_config": label_config(api, remove_relation=relation)}
        )

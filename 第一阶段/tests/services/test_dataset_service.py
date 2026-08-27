"""DatasetService 的离线单元测试。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from config import ProjectConfig, load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    EntityMention,
    EntityType,
    RelationMention,
    RelationType,
)
from src.services.dataset_service import (
    DatasetConfigurationError,
    DatasetConversionError,
    DatasetSchemaError,
    DatasetService,
    DatasetServiceConfig,
    DatasetValidationError,
    DatasetVersionExistsError,
)


def _service(tmp_path: Path, **overrides: object) -> DatasetService:
    values: dict[str, object] = dict(
        output_dir=tmp_path,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        random_seed=7,
        negative_ratio=1.0,
        max_negatives_per_text=10,
    )
    values.update(overrides)
    config = DatasetServiceConfig(**values)
    project = load_project_config()
    return DatasetService(config=config, project_config=project)


def _annotation(
    index: int,
    *,
    case_id: str | None = None,
    status: AnnotationStatus = AnnotationStatus.APPROVED,
    relation: bool = True,
) -> CanonicalAnnotation:
    text = "李某请托张某帮助某公司承揽项目。"
    entities = [
        EntityMention(
            entity_id="e1",
            name="李某",
            type=EntityType.PER,
            start=0,
            end=2,
            confidence=1,
        ),
        EntityMention(
            entity_id="e2",
            name="张某",
            type=EntityType.PER,
            start=4,
            end=6,
            confidence=1,
        ),
        EntityMention(
            entity_id="e3",
            name="某公司",
            type=EntityType.ORG,
            start=8,
            end=11,
            confidence=1,
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
                evidence_end=8,
                extraction_source="HUMAN",
            )
        ]
        if relation
        else []
    )
    return CanonicalAnnotation(
        annotation_id=f"a{index}",
        case_id=case_id or f"case{index}",
        doc_id=f"doc{index}",
        text_id=f"text{index}",
        text=text,
        entities=entities,
        relations=relations,
        annotation_source="HUMAN",
        schema_version="relation_v2.0",
        status=status,
        updated_at=datetime(2026, 1, 1) + timedelta(days=index),
    )


def test_validation_accepts_only_approved(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = service.validate_annotations(
        [
            _annotation(1),
            _annotation(2, status=AnnotationStatus.PENDING_REVIEW),
        ],
        strict=False,
    )
    assert [item.annotation_id for item in result.valid_annotations] == ["a1"]
    assert result.skipped_annotations == ["a2"]
    assert result.issues[0].code == "not_human_approved"


def test_strict_validation_raises(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(DatasetValidationError, match="not_human_approved"):
        service.validate_annotations(
            [_annotation(1, status=AnnotationStatus.GENERATED)]
        )


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("annotation_id", "empty_annotation_id"),
        ("case_id", "empty_case_id"),
        ("text_id", "empty_text_id"),
        ("text", "empty_text"),
    ],
)
def test_validation_rejects_empty_required_fields(
    tmp_path: Path, field: str, code: str
) -> None:
    service = _service(tmp_path)
    valid = _annotation(1)
    invalid = valid.model_construct(**{**valid.__dict__, field: ""})
    result = service.validate_annotations([invalid], strict=False)
    assert any(issue.code == code for issue in result.errors)


def test_validation_detects_mismatched_offset_constructed_model(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    valid = _annotation(1)
    bad_entity = valid.entities[0].model_copy(update={"name": "王某"})
    invalid = valid.model_construct(
        **{
            **valid.__dict__,
            "entities": [bad_entity, *valid.entities[1:]],
        }
    )
    result = service.validate_annotations([invalid], strict=False)
    assert any(issue.code == "entity_text_mismatch" for issue in result.issues)


def test_validation_detects_relation_direction(tmp_path: Path) -> None:
    service = _service(tmp_path)
    valid = _annotation(1)
    wrong = valid.relations[0].model_copy(
        update={
            "head_id": "e3",
            "tail_id": "e1",
            "type": RelationType.EMPLOYED_BY,
        }
    )
    invalid = valid.model_construct(
        **{**valid.__dict__, "relations": [wrong]}
    )
    result = service.validate_annotations([invalid], strict=False)
    assert any(
        issue.code == "relation_direction_or_type" for issue in result.issues
    )


def test_validation_detects_dangling_duplicate_and_negative_relations(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    valid = _annotation(1)
    dangling = valid.relations[0].model_copy(update={"tail_id": "missing"})
    negative = valid.relations[0].model_copy(
        update={"relation_id": "r2", "type": RelationType.NO_RELATION}
    )
    invalid = valid.model_construct(
        **{
            **valid.__dict__,
            "relations": [dangling, dangling, negative],
        }
    )
    result = service.validate_annotations([invalid], strict=False)
    codes = {issue.code for issue in result.errors}
    assert {"dangling_relation", "duplicate_relation_id", "negative_as_positive"} <= codes


def test_validation_detects_overlapping_and_unknown_entity(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    valid = _annotation(1)
    overlapping = EntityMention.model_construct(
        entity_id="e4",
        name=valid.text[1:3],
        type="UNKNOWN",
        start=1,
        end=3,
        confidence=1.0,
    )
    invalid = valid.model_construct(
        **{**valid.__dict__, "entities": [*valid.entities, overlapping]}
    )
    result = service.validate_annotations([invalid], strict=False)
    codes = {issue.code for issue in result.errors}
    assert {"overlapping_entities", "unknown_entity_type"} <= codes


def test_deduplicate_prefers_newer_record(tmp_path: Path) -> None:
    service = _service(tmp_path)
    old = _annotation(1)
    newer = old.model_copy(
        update={
            "annotation_id": "a-new",
            "updated_at": old.updated_at + timedelta(days=1),
        }
    )
    result = service.deduplicate_annotations([old, newer])
    assert [item.annotation_id for item in result.annotations] == ["a-new"]
    assert result.removed[0].kept_annotation_id == "a-new"


def test_deduplicate_by_annotation_text_and_content_is_stable(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    first = _annotation(1)
    annotation_duplicate = first.model_copy()
    text_duplicate = first.model_copy(
        update={"annotation_id": "x", "case_id": "x"}
    )
    content_duplicate = first.model_copy(
        update={
            "annotation_id": "y",
            "case_id": "y",
            "text_id": "y",
        }
    )
    result = service.deduplicate_annotations(
        [content_duplicate, text_duplicate, annotation_duplicate, first]
    )
    assert result.duplicate_count == 3
    assert len(result.deduplicated_annotations) == 1


def test_split_is_case_isolated_and_reproducible(tmp_path: Path) -> None:
    service = _service(tmp_path)
    annotations = [_annotation(index) for index in range(10)]
    first = service.split_annotations(annotations)
    second = service.split_annotations(list(reversed(annotations)))
    first_cases = {
        name: {item.case_id for item in getattr(first, name)}
        for name in ("train", "validation", "test")
    }
    second_cases = {
        name: {item.case_id for item in getattr(second, name)}
        for name in ("train", "validation", "test")
    }
    assert first_cases == second_cases
    assert first_cases["train"].isdisjoint(first_cases["validation"])
    assert first_cases["train"].isdisjoint(first_cases["test"])
    assert first_cases["validation"].isdisjoint(first_cases["test"])


def test_frozen_test_cases_stay_in_test(tmp_path: Path) -> None:
    service = _service(tmp_path)
    split = service.split_annotations(
        [_annotation(index) for index in range(6)],
        previous_manifest={"frozen_test_case_ids": ["case2"]},
    )
    assert {item.case_id for item in split.test} == {"case2"}


def test_rebuild_test_set_ignores_frozen_cases(tmp_path: Path) -> None:
    service = _service(tmp_path)
    annotations = [_annotation(index) for index in range(6)]
    split = service.split_annotations(
        annotations,
        previous_manifest={"frozen_test_case_ids": ["case2"]},
        rebuild_test_set=True,
    )
    assert len(split.test_case_ids) == 1
    assert split.test_case_ids != ["case2"]


def test_small_dataset_warns_and_bad_ratios_fail(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.split_annotations([_annotation(1)]).warnings
    with pytest.raises(DatasetConfigurationError):
        _service(
            tmp_path,
            train_ratio=0.5,
            validation_ratio=0.2,
            test_ratio=0.2,
        )


def test_bio_conversion_is_character_level(tmp_path: Path) -> None:
    service = _service(tmp_path)
    sample = service.convert_annotation_to_bio(_annotation(1))
    assert sample.tokens == list(sample.text)
    assert sample.labels[:6] == [
        "B-PER",
        "I-PER",
        "O",
        "O",
        "B-PER",
        "I-PER",
    ]
    assert len(sample.labels) == len(sample.text)
    assert service.label2id["O"] == 0
    assert list(service.label2id) == [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-POSITION",
        "I-POSITION",
        "B-MONEY",
        "I-MONEY",
    ]


def test_bio_preserves_spaces_punctuation_and_same_name_offsets(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    text = "张某 张某。"
    annotation = _annotation(1, relation=False).model_copy(
        update={
            "text": text,
            "entities": [
                EntityMention(
                    entity_id="e1",
                    name="张某",
                    type=EntityType.PER,
                    start=0,
                    end=2,
                ),
                EntityMention(
                    entity_id="e2",
                    name="张某",
                    type=EntityType.PER,
                    start=3,
                    end=5,
                ),
            ],
        }
    )
    sample = service.convert_annotation_to_bio(annotation)
    assert sample.tokens == ["张", "某", " ", "张", "某", "。"]
    assert sample.labels == [
        "B-PER",
        "I-PER",
        "O",
        "B-PER",
        "I-PER",
        "O",
    ]


def test_bio_overlap_raises_conversion_error(tmp_path: Path) -> None:
    service = _service(tmp_path)
    valid = _annotation(1, relation=False)
    overlap = valid.entities[0].model_copy(
        update={"entity_id": "e9", "start": 1, "end": 3, "name": valid.text[1:3]}
    )
    invalid = valid.model_construct(
        **{**valid.__dict__, "entities": [valid.entities[0], overlap]}
    )
    with pytest.raises(DatasetConversionError):
        service.convert_annotation_to_bio(invalid)


def test_positive_relation_sample_has_opennre_shape(tmp_path: Path) -> None:
    service = _service(tmp_path)
    sample = service.build_positive_relation_samples([_annotation(1)])[0]
    assert sample.relation == "请托"
    assert sample.h["name"] == "李某"
    assert sample.h["pos"] == [0, 2]
    assert sample.t["pos"] == [4, 6]


def test_trace_fields_can_be_disabled(tmp_path: Path) -> None:
    service = _service(tmp_path, include_trace_fields=False)
    sample = service.build_positive_relation_samples([_annotation(1)])[0]
    dumped = sample.model_dump(exclude_none=True)
    assert "case_id" not in dumped
    assert "annotation_id" not in dumped


def test_candidates_respect_schema_and_exclude_self(tmp_path: Path) -> None:
    service = _service(tmp_path)
    candidates = service.generate_relation_candidates(_annotation(1))
    assert all(item.head_id != item.tail_id for item in candidates)
    assert any(
        item.head_id == "e1"
        and item.tail_id == "e2"
        and "请托" in item.allowed_relations
        for item in candidates
    )
    assert not any(
        item.head_id == "e3" and item.tail_id == "e1"
        for item in candidates
    )


def test_negative_samples_do_not_collide_with_positive(tmp_path: Path) -> None:
    service = _service(tmp_path)
    annotation = _annotation(1)
    negatives = service.generate_negative_relation_samples([annotation])
    assert all(sample.relation == "无关系" for sample in negatives)
    assert all(
        not (sample.h["id"] == "e1" and sample.t["id"] == "e2")
        for sample in negatives
    )


def test_negative_sampling_is_deterministic_and_honors_cap(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        negative_ratio=10,
        max_negatives_per_text=1,
    )
    annotation = _annotation(1)
    first = service.generate_negative_relation_samples([annotation])
    second = service.generate_negative_relation_samples([annotation])
    assert first == second
    assert len(first) == 1


def test_undirected_positive_reverse_is_not_negative(tmp_path: Path) -> None:
    service = _service(tmp_path, negative_ratio=10)
    annotation = _annotation(1)
    friend = annotation.relations[0].model_copy(
        update={"type": RelationType.FRIEND}
    )
    annotation = annotation.model_copy(update={"relations": [friend]})
    negatives = service.generate_negative_relation_samples([annotation])
    assert all(
        {sample.h["id"], sample.t["id"]} != {"e1", "e2"}
        for sample in negatives
    )


def test_relation_mapping_has_negative_zero_and_schema_order(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    assert service.relation2id["无关系"] == 0
    assert service.relation2id["任职于"] == 1
    assert len(service.relation2id) == 22
    assert "关联人物" not in service.relation2id


def test_existing_rel2id_conflict_is_reported(tmp_path: Path) -> None:
    service = _service(tmp_path)
    relation_dir = tmp_path / "relation"
    relation_dir.mkdir()
    (relation_dir / "rel2id.json").write_text(
        '{"关联人物": 0}\n', encoding="utf-8"
    )
    split = service.split_annotations(
        [_annotation(index) for index in range(3)]
    )
    with pytest.raises(DatasetSchemaError, match="schema"):
        service.export_relation_dataset(split, tmp_path)


def test_fingerprint_is_input_order_independent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    annotations = [_annotation(index) for index in range(4)]
    assert service.calculate_dataset_fingerprint(
        annotations
    ) == service.calculate_dataset_fingerprint(list(reversed(annotations)))


def test_create_version_exports_required_and_trainer_compatible_files(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    result = service.create_dataset_version(
        [_annotation(index) for index in range(5)],
        dataset_version="dataset-test",
    )
    root = Path(result.output_dir)
    required = [
        "manifest.json",
        "statistics.json",
        "source_annotations.jsonl",
        "ner/train.txt",
        "ner/validation.txt",
        "ner/test.txt",
        "ner/train.jsonl",
        "ner/validation.jsonl",
        "ner/test.jsonl",
        "ner/label2id.json",
        "ner/id2label.json",
        "relation/train.jsonl",
        "relation/validation.jsonl",
        "relation/test.jsonl",
        "relation/rel2id.json",
        "relation/id2rel.json",
    ]
    assert all((root / name).is_file() for name in required)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_fingerprint"] == result.manifest.dataset_fingerprint
    assert "ner/train.txt" in manifest["file_checksums"]
    assert result.dataset.status == "READY"


def test_existing_version_refuses_overwrite_by_default(tmp_path: Path) -> None:
    service = _service(tmp_path)
    annotations = [_annotation(index) for index in range(3)]
    service.create_dataset_version(annotations, dataset_version="fixed")
    with pytest.raises(DatasetVersionExistsError):
        service.create_dataset_version(annotations, dataset_version="fixed")


def test_explicit_overwrite_replaces_complete_version(tmp_path: Path) -> None:
    service = _service(tmp_path)
    original = [_annotation(index) for index in range(3)]
    service.create_dataset_version(original, dataset_version="fixed")
    result = service.create_dataset_version(
        [_annotation(index) for index in range(4)],
        dataset_version="fixed",
        overwrite=True,
    )
    root = Path(result.output_dir)
    assert (root / "manifest.json").is_file()
    assert not list(tmp_path.glob(".fixed.backup-*"))


def test_manifest_checksums_match_exported_bytes(tmp_path: Path) -> None:
    import hashlib

    service = _service(tmp_path)
    result = service.create_dataset_version(
        [_annotation(index) for index in range(4)],
        dataset_version="checksums",
    )
    root = Path(result.output_dir)
    expected = hashlib.sha256((root / "ner/train.txt").read_bytes()).hexdigest()
    assert result.manifest.checksums["ner/train.txt"] == expected
    source_lines = (root / "source_annotations.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert all(isinstance(json.loads(line), dict) for line in source_lines)


def test_statistics_are_consistent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    split = service.split_annotations(
        [_annotation(index) for index in range(5)]
    )
    statistics = service.calculate_dataset_statistics(split)
    assert statistics.annotation_count == 5
    assert statistics.case_count == 5
    assert statistics.entity_count == 15
    assert statistics.relation_positive_count == 5

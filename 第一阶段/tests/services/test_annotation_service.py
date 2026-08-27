"""AnnotationService 的合法转换、去重与复核标记测试。"""

from __future__ import annotations

import json

from models import (
    AnnotationStatus,
    EntityMention,
    EntityPrediction,
    EntityType,
    ModelExtractionResult,
    RelationMention,
    RelationPrediction,
    RelationType,
)
from src.services.annotation_service import (
    AnnotationService,
    LLMPreannotationEntityPayload,
    LLMPreannotationPayload,
    LLMPreannotationRelationPayload,
    LLMPreannotationResult,
)


def test_annotation_deduplicates_and_marks_low_confidence():
    text = "张三请托李四"
    extraction = ModelExtractionResult(
        text=text,
        entities=[
            EntityPrediction(
                entity_id="e1", name="张三", entity_type="PER",
                start=0, end=2, confidence=0.9,
            ),
            EntityPrediction(
                entity_id="e1-dup", name="张三", entity_type="PER",
                start=0, end=2, confidence=0.9,
            ),
            EntityPrediction(
                entity_id="e2", name="李四", entity_type="PER",
                start=4, end=6, confidence=0.5,
            ),
        ],
        relations=[
            RelationPrediction(
                relation_id="r1", head_id="e1", tail_id="e2",
                relation_type="请托", confidence=0.5,
                evidence_start=0, evidence_end=6,
            ),
            RelationPrediction(
                relation_id="r2", head_id="e1-dup", tail_id="e2",
                relation_type="请托", confidence=0.5,
                evidence_start=0, evidence_end=6,
            ),
        ],
        ner_model_version="ner-v1",
        relation_model_version="re-v1",
        inference_seconds=0.1,
    )

    annotation = AnnotationService().to_canonical(
        extraction,
        annotation_id="a1",
        case_id="c1",
        doc_id="d1",
        text_id="t1",
    )

    assert len(annotation.entities) == 2
    assert len(annotation.relations) == 1
    assert annotation.status == AnnotationStatus.PENDING_REVIEW
    assert annotation.annotation_source == "DEEP_MODEL"
    assert annotation.metadata["review_required"] is True
    assert annotation.metadata["ner_model_version"] == "ner-v1"


class FakeStructuredLLM:
    model_name = "gpt-5.4-mini"

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def generate_structured_response(
        self, system_prompt, user_prompt, response_model, **kwargs
    ):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_model": response_model,
            **kwargs,
        })
        if isinstance(self.result, response_model):
            return self.result
        return response_model.model_validate(self.result.model_dump())


def test_llm_preannotation_cleans_invalid_entities_and_relations():
    text = "张三请托李四支付十万元"
    llm = FakeStructuredLLM(LLMPreannotationResult(
        entities=[
            EntityMention(
                entity_id="e1", name="张三", type=EntityType.PER,
                start=0, end=2, confidence=0.98,
            ),
            EntityMention(
                entity_id="e1-copy", name="张三", type=EntityType.PER,
                start=0, end=2, confidence=0.8,
            ),
            EntityMention(
                entity_id="e2", name="李四", type=EntityType.PER,
                start=4, end=6, confidence=0.96,
            ),
            EntityMention(
                entity_id="e3", name="十万元", type=EntityType.MONEY,
                start=8, end=11, confidence=0.95,
            ),
            EntityMention(
                entity_id="bad-offset", name="王五", type=EntityType.PER,
                start=2, end=4, confidence=0.9,
            ),
        ],
        relations=[
            RelationMention(
                relation_id="r1", head_id="e1", tail_id="e2",
                type=RelationType.ENTRUST, confidence=0.94,
                evidence_start=0, evidence_end=6,
            ),
            RelationMention(
                relation_id="r1-copy", head_id="e1", tail_id="e2",
                type=RelationType.ENTRUST, confidence=0.7,
            ),
            RelationMention(
                relation_id="bad-ref", head_id="bad-offset", tail_id="e2",
                type=RelationType.ENTRUST, confidence=0.9,
            ),
            RelationMention(
                relation_id="bad-direction", head_id="e3", tail_id="e1",
                type=RelationType.PAYS_MONEY, confidence=0.9,
            ),
            RelationMention(
                relation_id="r2", head_id="e2", tail_id="e3",
                type=RelationType.PAYS_MONEY, confidence=0.91,
                evidence_start=0, evidence_end=99,
            ),
        ],
    ))

    annotation = AnnotationService(
        llm_service=llm  # type: ignore[arg-type]
    ).preannotate_with_llm(
        text,
        annotation_id="ann-llm-1",
        case_id="case-1",
        doc_id="doc-1",
        text_id="chunk-1",
    )

    assert [item.entity_id for item in annotation.entities] == [
        "e1", "e2", "e3"
    ]
    assert [item.relation_id for item in annotation.relations] == [
        "r1", "r2"
    ]
    assert annotation.relations[1].evidence_start is None
    assert annotation.relations[1].evidence_end is None
    assert all(
        item.extraction_source == "LLM" for item in annotation.relations
    )
    assert annotation.annotation_source == "LLM"
    assert annotation.status == AnnotationStatus.PENDING_REVIEW
    assert annotation.metadata["llm_model"] == "gpt-5.4-mini"
    assert annotation.metadata["review_required"] is True
    assert llm.calls[0]["response_model"] is LLMPreannotationPayload
    assert "张三请托李四支付十万元" in str(llm.calls[0]["user_prompt"])
    assert "relation_types" in str(llm.calls[0]["user_prompt"])


def test_llm_preannotation_all_unreliable_results_become_empty():
    llm = FakeStructuredLLM(LLMPreannotationResult(
        entities=[
            EntityMention(
                entity_id="e1", name="李四", type=EntityType.PER,
                start=0, end=2, confidence=0.9,
            )
        ],
        relations=[],
    ))

    annotation = AnnotationService(
        llm_service=llm  # type: ignore[arg-type]
    ).preannotate_with_llm(
        "张三没有明确关系",
        annotation_id="ann-empty",
        case_id="case-1",
        doc_id="doc-1",
        text_id="chunk-empty",
    )

    assert annotation.entities == []
    assert annotation.relations == []


def test_llm_compact_payload_filters_unknown_business_types():
    llm = FakeStructuredLLM(LLMPreannotationPayload(
        entities=[
            LLMPreannotationEntityPayload(
                entity_id="e1", name="张三", type="PERSON",
                start=0, end=2, confidence=0.9, normalized_name=None,
            )
        ],
        relations=[
            LLMPreannotationRelationPayload(
                relation_id="r1", head_id="e1", tail_id="e2",
                type="未知关系", confidence=0.9,
                evidence_start=0, evidence_end=2,
            )
        ],
    ))

    annotation = AnnotationService(
        llm_service=llm  # type: ignore[arg-type]
    ).preannotate_with_llm(
        "张三没有明确关系",
        annotation_id="ann-invalid-types",
        case_id="case-1",
        doc_id="doc-1",
        text_id="chunk-invalid-types",
    )

    assert annotation.entities == []
    assert annotation.relations == []


def test_llm_transport_schema_stays_compact_and_enum_free():
    schema = LLMPreannotationPayload.model_json_schema()
    serialized = json.dumps(schema, ensure_ascii=False)

    assert len(serialized) < 2200
    assert '"enum"' not in serialized
    assert "RelationType" not in schema.get("$defs", {})

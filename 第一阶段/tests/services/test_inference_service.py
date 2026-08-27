"""InferenceService 串联测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from models import EntityPrediction, RelationPrediction
from src.services.inference_service import InferenceService


def test_inference_service_runs_ner_then_re():
    entity = EntityPrediction(
        entity_id="e1", name="张三", entity_type="PER",
        start=0, end=2, confidence=0.9,
    )
    relation = RelationPrediction(
        relation_id="r1", head_id="e1", tail_id="e2",
        relation_type="请托", confidence=0.8,
    )
    ner = MagicMock(model_version="ner-v1")
    ner.predict.return_value = [entity]
    candidate = object()
    re_model = MagicMock(model_version="re-v1")
    re_model.candidate_builder.build.return_value = [candidate]
    re_model.predict_candidates.return_value = [relation]
    service = InferenceService(ner, re_model)

    # 关系引用必须有效，因此补充第二个实体。
    second = EntityPrediction(
        entity_id="e2", name="李四", entity_type="PER",
        start=4, end=6, confidence=0.9,
    )
    ner.predict.return_value = [entity, second]
    result = service.extract("张三请托李四")

    assert result.entities == [entity, second]
    assert result.relations == [relation]
    re_model.predict_candidates.assert_called_once_with(
        "张三请托李四", [candidate]
    )
    assert result.metadata["candidate_count"] == 1

"""BERTEntity 候选构造与预测测试。"""

from __future__ import annotations

import torch

from models import EntityPrediction
from src.modeling.bert_entity.candidate_builder import (
    RelationCandidateBuilder,
)
from src.modeling.bert_entity.predictor import BertEntityPredictor
from src.modeling.bert_entity.dataset import encode_relation_record
from src.modeling.common.label_mapping import relation_mapping_from_schema


def entities():
    return [
        EntityPrediction(
            entity_id="e1", name="张三", entity_type="PER",
            start=0, end=2, confidence=0.9,
        ),
        EntityPrediction(
            entity_id="e2", name="李四", entity_type="PER",
            start=4, end=6, confidence=0.9,
        ),
    ]


def test_candidate_builder_includes_per_per_and_not_self():
    candidates = RelationCandidateBuilder().build(entities())

    assert candidates
    assert all(item.head.entity_id != item.tail.entity_id for item in candidates)
    assert any("合谋" in item.allowed_relations for item in candidates)


def test_predictor_receives_head_tail_and_filters_negative():
    received = []

    def classify(text, candidate):
        received.append((candidate.head.entity_id, candidate.tail.entity_id))
        if candidate.head.entity_id == "e1":
            return "请托", 0.88
        return "无关系", 0.91

    predictor = BertEntityPredictor(
        relation_mapping=relation_mapping_from_schema(),
        classifier=classify,
    )

    relations = predictor.predict("张三请托李四", entities())

    assert ("e1", "e2") in received
    assert all(item.relation_type != "无关系" for item in relations)
    assert relations[0].head_id == "e1"
    assert relations[0].tail_id == "e2"


def test_predictor_batches_model_forward_passes():
    class BatchModel:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, token, att_mask, pos1, pos2):
            self.batch_sizes.append(token.shape[0])
            return torch.tensor([[0.0, 1.0]] * token.shape[0])

    model = BatchModel()
    predictor = BertEntityPredictor(
        relation_mapping={"无关系": 0, "请托": 1},
        tokenizer=MockMarkerTokenizer(),
        model=model,
        max_length=32,
        batch_size=2,
    )

    relations = predictor.predict("张三请托李四", entities())

    assert model.batch_sizes == [2]
    assert len(relations) == 2


class MockMarkerTokenizer:
    pad_token_id = 0

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        return list(range(1, len(tokens) + 1))


def test_relation_training_record_uses_legacy_entity_markers():
    mapping = {"无关系": 0, "请托": 1}
    feature = encode_relation_record(
        {
            "text": "张三请托李四",
            "h": {"name": "张三", "type": "PER", "pos": [0, 2]},
            "t": {"name": "李四", "type": "PER", "pos": [4, 6]},
            "relation": "请托",
        },
        MockMarkerTokenizer(),
        mapping,
        max_length=32,
    )

    assert feature["pos1"] < feature["pos2"]
    assert feature["label"] == 1
    assert len(feature["token"]) == 32

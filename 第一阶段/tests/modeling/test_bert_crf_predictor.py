"""BERT-CRF predictor 的字符偏移测试。"""

from __future__ import annotations

import pytest

from src.modeling.bert_crf.predictor import (
    BertCrfPredictor,
    ModelNotLoadedError,
)
from src.modeling.bert_crf.dataset import build_ner_features
from src.modeling.bert_crf.model import LinearChainCRF


class MockFastTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [[101, 1, 2, 3, 4, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
            "token_type_ids": [[0, 0, 0, 0, 0, 0]],
            "offset_mapping": [
                [(0, 0), (0, 1), (1, 2), (2, 4), (4, 6), (0, 0)]
            ],
            "overflow_to_sample_mapping": [0],
        }


def test_predictor_preserves_character_offsets_and_types():
    labels = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
    }
    predictor = BertCrfPredictor(
        label_mapping=labels,
        tokenizer=MockFastTokenizer(),
        window_decoder=lambda _: (
            [0, 1, 2, 3, 4, 0],
            [1.0, 0.9, 0.8, 0.95, 0.85, 1.0],
        ),
    )
    text = "张三任职公司"

    entities = predictor.predict(text)

    assert [(item.name, item.start, item.end) for item in entities] == [
        ("张三", 0, 2),
        ("任职公司", 2, 6),
    ]
    assert all(text[item.start:item.end] == item.name for item in entities)
    assert {item.entity_type for item in entities} <= {
        "PER", "ORG", "POSITION", "MONEY"
    }


def test_illegal_i_label_is_repaired_as_entity_start():
    predictor = BertCrfPredictor(
        label_mapping={"O": 0, "I-PER": 1},
        tokenizer=MockFastTokenizer(),
        window_decoder=lambda _: (
            [0, 1, 1, 0, 0, 0],
            [1.0] * 6,
        ),
    )

    result = predictor.predict("张三任职公司")

    assert result[0].name == "张三"


def test_missing_checkpoint_has_clear_error(tmp_path):
    predictor = BertCrfPredictor(
        checkpoint_path=tmp_path / "missing",
        label_mapping={"O": 0},
    )

    with pytest.raises(ModelNotLoadedError, match="目录不存在"):
        predictor.load()


def test_training_features_align_character_entities_to_bio():
    labels = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
    }
    features = build_ner_features(
        [{
            "text": "张三任职公司",
            "entities": [
                {
                    "name": "张三",
                    "entity_type": "PER",
                    "start": 0,
                    "end": 2,
                },
                {
                    "name": "任职公司",
                    "entity_type": "ORG",
                    "start": 2,
                    "end": 6,
                },
            ],
        }],
        MockFastTokenizer(),
        labels,
        max_length=6,
    )

    assert features[0]["labels"] == [0, 1, 2, 3, 4, 0]


def test_crf_loss_and_decode_run_on_cpu():
    torch = pytest.importorskip("torch")
    crf = LinearChainCRF(3)
    emissions = torch.randn(2, 4, 3)
    tags = torch.tensor([[0, 1, 2, 0], [1, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

    likelihood = crf(emissions, tags, mask)
    paths = crf.decode(emissions, mask)

    assert likelihood.ndim == 0
    assert [len(path) for path in paths] == [4, 3]

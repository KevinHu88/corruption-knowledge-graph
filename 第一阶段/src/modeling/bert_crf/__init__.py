"""BERT-CRF 实体识别模型。"""

from .predictor import BertCrfPredictor, ModelNotLoadedError

__all__ = ["BertCrfPredictor", "ModelNotLoadedError"]

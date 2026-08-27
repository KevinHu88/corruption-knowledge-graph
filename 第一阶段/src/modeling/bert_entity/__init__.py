"""BERTEntity 关系分类模型。"""

from .candidate_builder import RelationCandidate, RelationCandidateBuilder
from .predictor import BertEntityPredictor, RelationModelNotLoadedError

__all__ = [
    "BertEntityPredictor",
    "RelationCandidate",
    "RelationCandidateBuilder",
    "RelationModelNotLoadedError",
]

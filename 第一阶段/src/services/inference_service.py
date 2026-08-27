"""BERT-CRF 到 BERTEntity 的端到端推理编排服务。"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Sequence
from typing import Any

from config import ProjectConfig, load_project_config
from models import ModelExtractionResult
from src.modeling.bert_crf.predictor import BertCrfPredictor
from src.modeling.bert_entity.candidate_builder import (
    RelationCandidateBuilder,
)
from src.modeling.bert_entity.predictor import BertEntityPredictor
from src.modeling.common.label_mapping import (
    load_ner_label_mapping,
    load_relation_mapping,
)


class InferenceServiceError(RuntimeError):
    """端到端模型加载或推理失败。"""


# 中文注释：统一协调 BERT-CRF 实体识别、实体对候选构建和 BERTEntity 关系分类。
class InferenceService:
    """延迟加载且只初始化一次的 NER → RE 推理服务。"""

    def __init__(
        self,
        ner_predictor: BertCrfPredictor | None = None,
        relation_predictor: BertEntityPredictor | None = None,
        *,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.config = project_config or load_project_config()
        self.ner_predictor = ner_predictor
        self.relation_predictor = relation_predictor
        self._load_lock = threading.Lock()
        self._loaded = (
            ner_predictor is not None and relation_predictor is not None
        )

    # 中文注释：按需加载两类本地模型，避免模块导入时就占用模型和设备资源。
    def load(self) -> None:
        """根据 training.yaml 创建并加载两个预测器，重复调用无副作用。"""

        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                modeling = self.config.training["modeling"]
                ner = modeling["ner"]
                relation = modeling["relation"]
                if self.ner_predictor is None:
                    self.ner_predictor = BertCrfPredictor(
                        checkpoint_path=ner.get("checkpoint_path"),
                        label_mapping=load_ner_label_mapping(
                            label_order=ner.get("label_order"),
                            label_map_path=ner.get("label_map_path"),
                        ),
                        max_length=int(ner["max_length"]),
                        stride=int(ner["stride"]),
                        device=str(modeling["device"]),
                    )
                    self.ner_predictor.load()
                if self.relation_predictor is None:
                    builder = RelationCandidateBuilder(
                        self.config.schema_config
                    )
                    self.relation_predictor = BertEntityPredictor(
                        relation_mapping=load_relation_mapping(
                            schema=self.config.schema_config,
                            relation_map_path=relation.get(
                                "relation_map_path"
                            ),
                        ),
                        checkpoint_path=relation.get("checkpoint_path"),
                        pretrained_model=relation.get("pretrained_model"),
                        candidate_builder=builder,
                        schema=self.config.schema_config,
                        max_length=int(relation["max_length"]),
                        batch_size=int(relation.get("batch_size", 16)),
                        mask_entity=bool(relation["mask_entity"]),
                        device=str(modeling["device"]),
                    )
                    self.relation_predictor.load()
            except Exception as exc:
                raise InferenceServiceError(
                    f"模型加载失败：{exc}"
                ) from exc
            self._loaded = True

    # 中文注释：处理单段文本，先抽取实体，再基于实体候选预测关系并返回统一结果。
    def extract(self, text: str) -> ModelExtractionResult:
        """对单段文本执行 NER、候选构造和关系分类。"""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text 不能为空")
        self.load()
        started = time.perf_counter()
        try:
            entities = self.ner_predictor.predict(text)
            candidates = (
                self.relation_predictor.candidate_builder.build(entities)
            )
            relations = self.relation_predictor.predict_candidates(
                text, candidates
            )
        except Exception as exc:
            raise InferenceServiceError(f"模型推理失败：{exc}") from exc
        return ModelExtractionResult(
            text=text,
            entities=entities,
            relations=relations,
            ner_model_version=self.ner_predictor.model_version,
            relation_model_version=self.relation_predictor.model_version,
            inference_seconds=time.perf_counter() - started,
            metadata={"candidate_count": len(candidates)},
        )

    # 中文注释：批量推理入口，保持输出顺序与输入文本顺序一致。
    def extract_batch(
        self,
        texts: Sequence[str],
    ) -> list[ModelExtractionResult]:
        """复用已加载模型顺序处理一个文本批次。"""

        self.load()
        return [self.extract(text) for text in texts]


def main() -> None:
    """完整 NER → RE 命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    result = InferenceService().extract(args.text)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

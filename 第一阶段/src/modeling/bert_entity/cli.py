"""BERTEntity 单独推理命令行入口。"""

from __future__ import annotations

import argparse
import json

from config import load_project_config
from models import EntityPrediction
from src.modeling.common.label_mapping import load_relation_mapping
from .predictor import BertEntityPredictor


def main() -> None:
    """读取文本与实体 JSON 后执行关系预测。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--entities-json", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    project = load_project_config()
    modeling = project.training["modeling"]
    relation = modeling["relation"]
    entities = [
        EntityPrediction.model_validate(item)
        for item in json.loads(args.entities_json)
    ]
    predictor = BertEntityPredictor(
        relation_mapping=load_relation_mapping(
            schema=project.schema_config,
            relation_map_path=relation.get("relation_map_path"),
        ),
        checkpoint_path=args.checkpoint or relation.get("checkpoint_path"),
        pretrained_model=relation["pretrained_model"],
        max_length=relation["max_length"],
        batch_size=relation.get("batch_size", 16),
        mask_entity=relation["mask_entity"],
        device=modeling["device"],
    )
    print(json.dumps(
        [item.model_dump() for item in predictor.predict(args.text, entities)],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

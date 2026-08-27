"""BERT-CRF 命令行推理入口。"""

from __future__ import annotations

import argparse
import json

from config import load_project_config
from src.modeling.common.label_mapping import load_ner_label_mapping
from .predictor import BertCrfPredictor


def main() -> None:
    """读取 training.yaml 并执行单条文本预测。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_project_config().training["modeling"]
    ner = config["ner"]
    predictor = BertCrfPredictor(
        checkpoint_path=args.checkpoint or ner.get("checkpoint_path"),
        label_mapping=load_ner_label_mapping(
            label_order=ner["label_order"],
            label_map_path=ner.get("label_map_path"),
        ),
        max_length=ner["max_length"],
        stride=ner["stride"],
        device=config["device"],
    )
    print(json.dumps(
        [item.model_dump() for item in predictor.predict(args.text)],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

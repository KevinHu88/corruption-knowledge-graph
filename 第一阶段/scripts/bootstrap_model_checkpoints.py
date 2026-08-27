"""Create loadable bootstrap NER and relation checkpoints.

The generated task heads are intentionally untrained.  These artifacts make the
inference pipeline runnable for integration testing; production promotion still
requires training on an approved, versioned project dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertConfig, BertModel, BertTokenizerFast

from config import BASE_DIR, load_project_config
from src.modeling.bert_crf.model import BertCrfForNer
from src.modeling.bert_entity.model import BertEntityForRelation
from src.modeling.common.label_mapping import (
    load_ner_label_mapping,
    relation_mapping_from_schema,
)


DEFAULT_BASE_MODEL = "artifacts/models/base/bootstrap-chinese-bert-tiny-v1"
DEFAULT_VERSION = "bootstrap-untrained-v1"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _create_local_base_model(path: Path, *, seed: int) -> None:
    """Create a compact offline BERT with a practical Chinese character vocab."""

    path.mkdir(parents=True, exist_ok=True)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    ascii_tokens = [chr(code) for code in range(33, 127)]
    cjk_punctuation = [chr(code) for code in range(0x3000, 0x3040)]
    cjk_characters = [chr(code) for code in range(0x4E00, 0xA000)]
    fullwidth_tokens = [chr(code) for code in range(0xFF01, 0xFF5F)]
    vocab = list(
        dict.fromkeys(
            specials
            + ascii_tokens
            + cjk_punctuation
            + cjk_characters
            + fullwidth_tokens
        )
    )
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(path)

    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        pad_token_id=0,
    )
    BertModel(config).save_pretrained(path)


def build_checkpoints(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    version: str = DEFAULT_VERSION,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Build deterministic, loadable artifacts with untrained task heads."""

    config = load_project_config()
    modeling = config.training["modeling"]
    ner_dir = BASE_DIR / modeling["ner"]["checkpoint_dir"] / version
    relation_dir = (
        BASE_DIR / modeling["relation"]["checkpoint_dir"] / version
    )
    targets = (ner_dir, relation_dir)
    if not overwrite and any(path.exists() for path in targets):
        existing = ", ".join(str(path) for path in targets if path.exists())
        raise FileExistsError(f"checkpoint 已存在，拒绝覆盖：{existing}")
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    base_path = Path(base_model)
    if not base_path.is_absolute():
        base_path = BASE_DIR / base_path
    if not (base_path / "config.json").is_file():
        _create_local_base_model(base_path, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("NER checkpoint 需要 fast tokenizer")

    ner_mapping = load_ner_label_mapping(
        label_order=modeling["ner"]["label_order"]
    )
    ner_model = BertCrfForNer.from_pretrained(
        base_path,
        num_labels=len(ner_mapping),
        id2label={value: key for key, value in ner_mapping.items()},
        label2id=ner_mapping,
    )
    ner_model.save_pretrained(ner_dir)
    tokenizer.save_pretrained(ner_dir)
    _write_json(ner_dir / "label_map.json", ner_mapping)

    relation_mapping = relation_mapping_from_schema(config.schema_config)
    relation_model = BertEntityForRelation(
        str(base_path),
        len(relation_mapping),
    )
    relation_checkpoint = relation_dir / "model.pth.tar"
    torch.save(
        {
            "state_dict": relation_model.state_dict(),
            "relation_mapping": relation_mapping,
            "pretrained_model": str(base_path),
            "bootstrap_untrained": True,
            "random_seed": seed,
        },
        relation_checkpoint,
    )
    tokenizer.save_pretrained(relation_dir)
    _write_json(relation_dir / "relation_map.json", relation_mapping)

    metadata = {
        "artifact_status": "bootstrap_untrained",
        "base_model": str(base_path.relative_to(BASE_DIR)),
        "random_seed": seed,
        "warning": "任务头未经项目审核数据训练，仅用于联调和启动验证。",
    }
    for path in targets:
        _write_json(path / "bootstrap_metadata.json", metadata)
    return ner_dir, relation_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ner_dir, relation_checkpoint = build_checkpoints(
        base_model=args.base_model,
        version=args.version,
        overwrite=args.overwrite,
    )
    print(f"NER checkpoint: {ner_dir}")
    print(f"Relation checkpoint: {relation_checkpoint}")


if __name__ == "__main__":
    main()

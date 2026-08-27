"""模型版本清单读写。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# 中文注释：模型产物的统一描述文件，绑定架构、数据集、schema、权重、tokenizer 和标签映射。
class ModelManifest(BaseModel):
    """模型 artifact 的可复现实验清单。"""

    task_type: Literal["ner", "relation"]
    model_version: str
    dataset_version: str
    schema_version: str
    architecture: str
    pretrained_model: str
    checkpoint_file: str
    tokenizer_dir: str
    mapping_file: str
    random_seed: int = 42
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    role: Literal["CHALLENGER", "CHAMPION", "ARCHIVED"] = "CHALLENGER"
    created_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        """从 model_manifest.json 加载清单。"""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str | Path) -> Path:
        """保存清单，不创建任何虚假权重文件。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

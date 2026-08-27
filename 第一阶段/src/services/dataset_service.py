"""从人工审核标注构建可复现 NER 与关系抽取数据集。

本模块只负责校验、去重、切分、样本转换与版本化落盘，不调用外部
Service、不执行模型训练，也不修改输入的 ``CanonicalAnnotation``。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from config import ProjectConfig, load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    DatasetVersion,
    EntityMention,
    RelationMention,
)
from src.modeling.common.label_mapping import relation_mapping_from_schema

logger = logging.getLogger(__name__)


class DatasetServiceError(RuntimeError):
    """数据集构建服务基础异常。"""


class DatasetConfigurationError(DatasetServiceError):
    """数据集配置或 schema 不可用。"""


class DatasetValidationError(DatasetServiceError):
    """人工标注未通过数据集级校验。"""


class DatasetSplitError(DatasetServiceError):
    """数据集切分配置或结果不合法。"""


class DatasetExportError(DatasetServiceError):
    """数据集文件无法安全导出。"""


class DatasetConversionError(DatasetServiceError):
    """BIO 或 OpenNRE 样本无法从统一标注确定性转换。"""


class DatasetSchemaError(DatasetServiceError):
    """schema 与已有标签映射不一致。"""


class DatasetWriteError(DatasetExportError):
    """数据集文件或目录写入失败。"""


class DatasetVersionExistsError(DatasetExportError):
    """目标数据集版本已经存在。"""


class ValidationIssue(BaseModel):
    """单条标注的校验问题。"""

    annotation_id: str
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class AnnotationValidationResult(BaseModel):
    """批量标注校验结果。"""

    valid_annotations: list[CanonicalAnnotation] = Field(default_factory=list)
    skipped_annotations: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @computed_field
    @property
    def error_count(self) -> int:
        """返回错误数量。"""

        return sum(issue.severity == "error" for issue in self.issues)

    @computed_field
    @property
    def errors(self) -> list[ValidationIssue]:
        """返回全部错误。"""

        return [item for item in self.issues if item.severity == "error"]

    @computed_field
    @property
    def warnings(self) -> list[ValidationIssue]:
        """返回全部警告。"""

        return [item for item in self.issues if item.severity == "warning"]

    @computed_field
    @property
    def valid_count(self) -> int:
        """返回有效标注数。"""

        return len(self.valid_annotations)

    @computed_field
    @property
    def invalid_count(self) -> int:
        """返回无效标注数。"""

        return len(self.skipped_annotations)


class RemovedAnnotation(BaseModel):
    """去重时被移除的标注及原因。"""

    annotation_id: str
    kept_annotation_id: str
    reason: str


class DeduplicationResult(BaseModel):
    """标注去重结果。"""

    annotations: list[CanonicalAnnotation] = Field(default_factory=list)
    removed: list[RemovedAnnotation] = Field(default_factory=list)

    @computed_field
    @property
    def deduplicated_annotations(self) -> list[CanonicalAnnotation]:
        """返回去重后的标注，兼容业务命名。"""

        return self.annotations

    @computed_field
    @property
    def removed_records(self) -> list[RemovedAnnotation]:
        """返回移除记录。"""

        return self.removed

    @computed_field
    @property
    def duplicate_count(self) -> int:
        """返回移除的重复标注数量。"""

        return len(self.removed)


class DatasetSplit(BaseModel):
    """按案件隔离后的三份数据。"""

    train: list[CanonicalAnnotation] = Field(default_factory=list)
    validation: list[CanonicalAnnotation] = Field(default_factory=list)
    test: list[CanonicalAnnotation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def train_case_ids(self) -> list[str]:
        """返回训练案件 ID。"""

        return sorted({item.case_id for item in self.train})

    @computed_field
    @property
    def validation_case_ids(self) -> list[str]:
        """返回验证案件 ID。"""

        return sorted({item.case_id for item in self.validation})

    @computed_field
    @property
    def test_case_ids(self) -> list[str]:
        """返回测试案件 ID。"""

        return sorted({item.case_id for item in self.test})


class BioSample(BaseModel):
    """一条字符级 BIO 样本。"""

    annotation_id: str
    case_id: str
    text_id: str
    text: str
    tokens: list[str]
    labels: list[str]


class RelationCandidate(BaseModel):
    """一个符合 schema 类型约束的关系候选实体对。"""

    annotation_id: str
    case_id: str
    text_id: str
    head_id: str
    tail_id: str
    allowed_relations: list[str]
    distance_chars: int
    same_sentence: bool


class RelationSample(BaseModel):
    """兼容 OpenNRE 的关系分类样本。"""

    text: str
    h: dict[str, Any]
    t: dict[str, Any]
    relation: str
    annotation_id: str | None = None
    case_id: str | None = None
    text_id: str | None = None
    relation_id: str | None = None
    negative_reason: str | None = None
    sampling_strategy: str | None = None


class DatasetStatistics(BaseModel):
    """数据集统计信息。"""

    annotation_count: int
    case_count: int
    text_count: int
    total_characters: int
    entity_count: int
    relation_positive_count: int
    relation_negative_count: int
    entity_distribution: dict[str, int]
    relation_distribution: dict[str, int]
    split_entity_counts: dict[str, int]
    split_relation_counts: dict[str, int]
    split_annotation_counts: dict[str, int]
    split_case_counts: dict[str, int]
    text_length: dict[str, float]
    entity_length: dict[str, float]
    average_entities_per_text: float
    average_relations_per_text: float
    no_entity_text_count: int
    no_relation_text_count: int
    positive_negative_ratio: float | None
    rare_relations: list[str]
    missing_train_relations: list[str]
    validation_or_test_only_relations: list[str]
    case_sets_disjoint: bool
    leakage_detected: bool
    warnings: list[str] = Field(default_factory=list)


# 中文注释：数据集版本的可追溯清单，记录 schema、来源、切分、统计和文件校验和。
class DatasetManifest(BaseModel):
    """不可变数据集版本的复现清单。"""

    dataset_version: str
    dataset_fingerprint: str
    schema_version: str
    created_at: datetime
    random_seed: int
    split_ratios: dict[str, float]
    frozen_test_case_ids: list[str]
    train_case_ids: list[str]
    validation_case_ids: list[str]
    test_case_ids: list[str]
    source_annotation_ids: list[str]
    label2id: dict[str, int]
    relation2id: dict[str, int]
    negative_sampling: dict[str, Any]
    files: dict[str, str]
    configuration: dict[str, Any]
    file_checksums: dict[str, str]
    statistics: DatasetStatistics
    python_version: str
    git_commit: str | None = None

    @computed_field
    @property
    def checksums(self) -> dict[str, str]:
        """返回文件校验和，兼容 manifest 约定字段名。"""

        return self.file_checksums


# 中文注释：数据集构建的标准结果，返回版本目录、manifest 和去重/切分摘要。
class DatasetBuildResult(BaseModel):
    """一次成功数据集构建的统一返回。"""

    dataset: DatasetVersion
    output_dir: str
    manifest: DatasetManifest
    validation: AnnotationValidationResult
    deduplication: DeduplicationResult

    @computed_field
    @property
    def statistics(self) -> DatasetStatistics:
        """直接返回本次构建统计。"""

        return self.manifest.statistics


# 中文注释：数据集构建参数的强类型视图，由 workflow.yaml 转换而来。
class DatasetServiceConfig(BaseModel):
    """DatasetService 的可注入配置。"""

    output_dir: Path = Path("artifacts/datasets")
    train_ratio: float = Field(default=0.70, ge=0, le=1)
    validation_ratio: float = Field(default=0.15, ge=0, le=1)
    test_ratio: float = Field(default=0.15, ge=0, le=1)
    random_seed: int = 42
    strict_validation: bool = True
    freeze_test_set: bool = True
    include_trace_fields: bool = True
    negative_sampling_strategy: Literal[
        "random", "hard", "hard_and_random"
    ] = "hard_and_random"
    negative_ratio: float = Field(default=1.0, ge=0)
    max_negatives_per_text: int = Field(default=20, ge=0)
    max_entity_distance_chars: int | None = Field(default=None, gt=0)
    same_sentence_only: bool = False
    allow_reverse_negative_pairs: bool = False
    overwrite_schema_files: bool = False


# 中文注释：负责把已审核规范标注转换成可复现、可追溯的 NER 与关系训练数据集。
class DatasetService:
    """将已审核统一标注冻结为 NER/RE 训练数据集。"""

    def __init__(
        self,
        config: DatasetServiceConfig | None = None,
        *,
        project_config: ProjectConfig | None = None,
    ) -> None:
        """初始化服务并复用项目 schema 与 workflow 配置。"""

        self.project_config = project_config or load_project_config()
        self.schema = self.project_config.schema_config
        if not isinstance(self.schema.get("entity_types"), Mapping):
            raise DatasetConfigurationError("schema.yaml 缺少 entity_types")
        if not isinstance(self.schema.get("relation_types"), Mapping):
            raise DatasetConfigurationError("schema.yaml 缺少 relation_types")
        if not self.schema.get("negative_relation"):
            raise DatasetConfigurationError("schema.yaml 缺少 negative_relation")
        self.config = config or self._config_from_project()
        total = (
            self.config.train_ratio
            + self.config.validation_ratio
            + self.config.test_ratio
        )
        if abs(total - 1.0) > 1e-8:
            raise DatasetConfigurationError("train/validation/test 比例之和必须为 1")
        self.entity_types = tuple(str(item) for item in self.schema["entity_types"])
        self.relation_rules: dict[str, Mapping[str, Any]] = {
            str(name): rule
            for name, rule in self.schema["relation_types"].items()
        }
        self.negative_relation = str(self.schema["negative_relation"])
        self.relation2id = relation_mapping_from_schema(self.schema)
        self.label2id = self._build_bio_mapping()

    def _config_from_project(self) -> DatasetServiceConfig:
        data = dict(self.project_config.workflow.get("dataset", {}))
        negative = dict(data.get("negative_sampling", {}))
        return DatasetServiceConfig(
            output_dir=Path(data.get("output_dir", "artifacts/datasets")),
            train_ratio=float(data.get("train_ratio", 0.70)),
            validation_ratio=float(data.get("validation_ratio", 0.15)),
            test_ratio=float(data.get("test_ratio", 0.15)),
            random_seed=int(data.get("random_seed", 42)),
            strict_validation=bool(data.get("strict_validation", True)),
            freeze_test_set=bool(
                data.get(
                    "freeze_test_set",
                    data.get("keep_frozen_test_set", True),
                )
            ),
            include_trace_fields=bool(data.get("include_trace_fields", True)),
            negative_sampling_strategy=str(
                negative.get(
                    "strategy",
                    data.get("relation", {}).get(
                        "negative_sampling_strategy", "hard_and_random"
                    ),
                )
            ),
            negative_ratio=float(
                negative.get(
                    "negative_ratio",
                    data.get("relation", {}).get("negative_ratio", 1.0),
                )
            ),
            max_negatives_per_text=int(
                negative.get(
                    "max_negatives_per_text",
                    data.get("relation", {}).get(
                        "max_negative_samples_per_text", 20
                    ),
                )
            ),
            max_entity_distance_chars=negative.get(
                "max_entity_distance_chars"
            ),
            same_sentence_only=bool(
                negative.get("same_sentence_only", False)
            ),
            allow_reverse_negative_pairs=bool(
                negative.get("allow_reverse_negative_pairs", False)
            ),
            overwrite_schema_files=bool(
                data.get("overwrite_schema_files", False)
            ),
        )

    # 中文注释：构建前统一校验审核状态、offset、schema、关系方向、证据和重复记录。
    def validate_annotations(
        self,
        annotations: Sequence[CanonicalAnnotation],
        *,
        strict: bool | None = None,
    ) -> AnnotationValidationResult:
        """逐条校验审核状态、字符偏移、schema 约束和重复记录。"""

        result = AnnotationValidationResult()
        for annotation in annotations:
            issues = self._validate_one(annotation)
            result.issues.extend(issues)
            if any(issue.severity == "error" for issue in issues):
                result.skipped_annotations.append(
                    str(getattr(annotation, "annotation_id", "unknown"))
                )
            else:
                result.valid_annotations.append(
                    self._sorted_annotation(annotation)
                )
        should_raise = (
            self.config.strict_validation if strict is None else strict
        )
        if should_raise and result.error_count:
            details = "; ".join(
                f"{issue.annotation_id}:{issue.code}"
                for issue in result.issues
                if issue.severity == "error"
            )
            raise DatasetValidationError(f"标注校验失败：{details}")
        return result

    def _validate_one(
        self, annotation: CanonicalAnnotation
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        annotation_id = str(getattr(annotation, "annotation_id", "unknown"))

        def error(code: str, message: str) -> None:
            issues.append(
                ValidationIssue(
                    annotation_id=annotation_id, code=code, message=message
                )
            )

        status = getattr(annotation, "status", None)
        status_value = getattr(status, "value", status)
        if status_value != AnnotationStatus.APPROVED.value:
            error("not_human_approved", "仅 APPROVED 标注可进入数据集")
        for field_name in ("annotation_id", "case_id", "text_id"):
            if not str(getattr(annotation, field_name, "")).strip():
                error(f"empty_{field_name}", f"{field_name} 不能为空")
        text = getattr(annotation, "text", "")
        if not isinstance(text, str) or not text:
            error("empty_text", "标注文本为空")
            return issues
        entities = list(getattr(annotation, "entities", []))
        ids: set[str] = set()
        spans: list[tuple[int, int, str]] = []
        for entity in entities:
            entity_id = str(getattr(entity, "entity_id", ""))
            if not entity_id:
                error("empty_entity_id", "实体 ID 为空")
            elif entity_id in ids:
                error("duplicate_entity_id", f"实体 ID 重复：{entity_id}")
            ids.add(entity_id)
            start, end = int(entity.start), int(entity.end)
            if start < 0 or end <= start or end > len(text):
                error("invalid_entity_offset", f"实体偏移非法：{entity_id}")
                continue
            if text[start:end] != entity.name:
                error("entity_text_mismatch", f"实体文本不匹配：{entity_id}")
            entity_type = self._entity_type(entity)
            if entity_type not in self.entity_types:
                error("unknown_entity_type", f"未知实体类型：{entity_type}")
            spans.append((start, end, entity_id))
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                error(
                    "overlapping_entities",
                    f"实体区间重叠：{previous[2]} 与 {current[2]}",
                )
        entity_map = {str(item.entity_id): item for item in entities}
        relation_ids: set[str] = set()
        relation_keys: set[tuple[str, str, str]] = set()
        for relation in getattr(annotation, "relations", []):
            relation_id = str(getattr(relation, "relation_id", ""))
            relation_type = self._relation_type(relation)
            if not relation_id:
                error("empty_relation_id", "关系 ID 为空")
            elif relation_id in relation_ids:
                error("duplicate_relation_id", f"关系 ID 重复：{relation_id}")
            relation_ids.add(relation_id)
            if relation_type == self.negative_relation:
                error("negative_as_positive", "人工正关系中不得出现无关系")
            rule = self.relation_rules.get(relation_type)
            if rule is None:
                error("unknown_relation_type", f"未知关系类型：{relation_type}")
            head = entity_map.get(str(relation.head_id))
            tail = entity_map.get(str(relation.tail_id))
            if head is None or tail is None:
                error("dangling_relation", f"关系引用实体不存在：{relation_id}")
                continue
            if relation.head_id == relation.tail_id:
                error("self_relation", f"关系不能指向自身：{relation_id}")
            if rule is not None and (
                self._entity_type(head) not in rule.get("head_types", [])
                or self._entity_type(tail) not in rule.get("tail_types", [])
            ):
                error(
                    "relation_direction_or_type",
                    f"关系方向或实体类型不符合 schema：{relation_id}",
                )
            head_id, tail_id = str(relation.head_id), str(relation.tail_id)
            if rule is not None and not bool(rule.get("directional", True)):
                head_id, tail_id = sorted((head_id, tail_id))
            key = (head_id, tail_id, relation_type)
            if key in relation_keys:
                error("duplicate_relation", f"关系重复：{relation_id}")
            relation_keys.add(key)
            if relation.evidence_start is not None:
                if (
                    relation.evidence_end is None
                    or relation.evidence_start < 0
                    or relation.evidence_end > len(text)
                    or relation.evidence_end <= relation.evidence_start
                ):
                    error(
                        "invalid_evidence_offset",
                        f"关系证据偏移非法：{relation_id}",
                    )
        return issues

    # 中文注释：按稳定业务键去除重复标注，并保留被移除记录及原因用于审计。
    def deduplicate_annotations(
        self, annotations: Sequence[CanonicalAnnotation]
    ) -> DeduplicationResult:
        """按 annotation/text/content 三层稳定键去重并保留最优版本。"""

        normalized = [
            self._deduplicate_within_annotation(item) for item in annotations
        ]
        ranked = sorted(normalized, key=self._dedup_rank)
        seen_annotation: dict[str, CanonicalAnnotation] = {}
        seen_text: dict[str, CanonicalAnnotation] = {}
        seen_content: dict[str, CanonicalAnnotation] = {}
        kept: list[CanonicalAnnotation] = []
        removed: list[RemovedAnnotation] = []
        for annotation in ranked:
            content_key = hashlib.sha256(
                annotation.text.encode("utf-8")
            ).hexdigest()
            duplicate = (
                seen_annotation.get(annotation.annotation_id)
                or seen_text.get(annotation.text_id)
                or seen_content.get(content_key)
            )
            if duplicate is not None:
                removed.append(
                    RemovedAnnotation(
                        annotation_id=annotation.annotation_id,
                        kept_annotation_id=duplicate.annotation_id,
                        reason="duplicate_annotation_or_text",
                    )
                )
                continue
            kept.append(annotation)
            seen_annotation[annotation.annotation_id] = annotation
            seen_text[annotation.text_id] = annotation
            seen_content[content_key] = annotation
        kept.sort(key=self._annotation_sort_key)
        removed.sort(key=lambda item: item.annotation_id)
        return DeduplicationResult(annotations=kept, removed=removed)

    def _deduplicate_within_annotation(
        self, annotation: CanonicalAnnotation
    ) -> CanonicalAnnotation:
        """确定性移除单条文本内完全相同的实体和关系记录。"""

        entities: dict[tuple[Any, ...], EntityMention] = {}
        for entity in sorted(
            annotation.entities,
            key=lambda item: (item.start, item.end, item.entity_id),
        ):
            key = (
                entity.entity_id,
                entity.start,
                entity.end,
                entity.name,
                self._entity_type(entity),
            )
            entities.setdefault(key, entity)
        relations: dict[tuple[Any, ...], RelationMention] = {}
        for relation in sorted(
            annotation.relations,
            key=lambda item: (
                self._relation_type(item),
                item.head_id,
                item.tail_id,
                item.relation_id,
            ),
        ):
            relation_type = self._relation_type(relation)
            head_id, tail_id = relation.head_id, relation.tail_id
            rule = self.relation_rules.get(relation_type, {})
            if not bool(rule.get("directional", True)):
                head_id, tail_id = sorted((head_id, tail_id))
            key = (head_id, tail_id, relation_type)
            relations.setdefault(key, relation)
        return annotation.model_copy(
            update={
                "entities": list(entities.values()),
                "relations": list(relations.values()),
            }
        )

    @staticmethod
    def _dedup_rank(annotation: CanonicalAnnotation) -> tuple[Any, ...]:
        reviewed = annotation.metadata.get("reviewed_at")
        completeness = len(annotation.entities) + len(annotation.relations)
        return (
            -_timestamp_number(str(reviewed or "")),
            -annotation.updated_at.timestamp(),
            -completeness,
            annotation.annotation_id,
        )

    # 中文注释：以 case_id 为隔离单位执行可复现切分，防止同一案件泄漏到不同数据集。
    def split_annotations(
        self,
        annotations: Sequence[CanonicalAnnotation],
        *,
        previous_manifest: DatasetManifest | Mapping[str, Any] | None = None,
        rebuild_test_set: bool = False,
    ) -> DatasetSplit:
        """按 case_id 切分，支持沿用历史冻结测试案件。"""

        by_case: dict[str, list[CanonicalAnnotation]] = defaultdict(list)
        for annotation in annotations:
            by_case[annotation.case_id].append(annotation)
        case_ids = sorted(by_case)
        if not case_ids:
            raise DatasetSplitError("没有可切分的有效案件")
        warnings: list[str] = []
        if len(case_ids) < 3:
            warnings.append("案件数量少于 3，无法保证三个子集均非空")
        frozen = set()
        if (
            previous_manifest is not None
            and self.config.freeze_test_set
            and not rebuild_test_set
        ):
            data = (
                previous_manifest.model_dump()
                if isinstance(previous_manifest, BaseModel)
                else dict(previous_manifest)
            )
            frozen = set(
                data.get("frozen_test_case_ids")
                or data.get("test_case_ids")
                or []
            ) & set(case_ids)
        rng = random.Random(self.config.random_seed)
        remaining = [case_id for case_id in case_ids if case_id not in frozen]
        rng.shuffle(remaining)
        if frozen:
            validation_share = self.config.validation_ratio / max(
                self.config.train_ratio + self.config.validation_ratio, 1e-12
            )
            validation_count = round(len(remaining) * validation_share)
            if len(remaining) >= 2 and self.config.validation_ratio > 0:
                validation_count = max(1, min(validation_count, len(remaining) - 1))
            validation_cases = set(remaining[:validation_count])
            train_cases = set(remaining[validation_count:])
            test_cases = frozen
        else:
            test_count = round(len(case_ids) * self.config.test_ratio)
            validation_count = round(
                len(case_ids) * self.config.validation_ratio
            )
            if len(case_ids) >= 3:
                test_count = max(1, test_count)
                validation_count = max(1, validation_count)
                if test_count + validation_count >= len(case_ids):
                    validation_count = 1
                    test_count = 1
            test_cases = set(remaining[:test_count])
            validation_cases = set(
                remaining[test_count : test_count + validation_count]
            )
            train_cases = set(
                remaining[test_count + validation_count :]
            )

        def collect(case_set: set[str]) -> list[CanonicalAnnotation]:
            return sorted(
                (
                    annotation
                    for case_id in sorted(case_set)
                    for annotation in by_case[case_id]
                ),
                key=self._annotation_sort_key,
            )

        return DatasetSplit(
            train=collect(train_cases),
            validation=collect(validation_cases),
            test=collect(test_cases),
            warnings=warnings,
        )

    # 中文注释：把规范实体 offset 转换为 BERT-CRF 使用的 BIO 标注样本。
    def convert_annotation_to_bio(
        self, annotation: CanonicalAnnotation
    ) -> BioSample:
        """将一条标注转换为逐字符 BIO 序列。"""

        labels = ["O"] * len(annotation.text)
        previous_end = -1
        for entity in sorted(
            annotation.entities,
            key=lambda item: (item.start, item.end, item.entity_id),
        ):
            if entity.start < previous_end:
                raise DatasetConversionError(
                    f"实体重叠，无法转换 BIO：{annotation.annotation_id}"
                )
            entity_type = self._entity_type(entity)
            if entity_type not in self.entity_types:
                raise DatasetConversionError(f"未知实体类型：{entity_type}")
            labels[entity.start] = f"B-{entity_type}"
            for index in range(entity.start + 1, entity.end):
                labels[index] = f"I-{entity_type}"
            previous_end = entity.end
        return BioSample(
            annotation_id=annotation.annotation_id,
            case_id=annotation.case_id,
            text_id=annotation.text_id,
            text=annotation.text,
            tokens=list(annotation.text),
            labels=labels,
        )

    def export_bio_dataset(
        self,
        split: DatasetSplit,
        output_dir: str | Path,
    ) -> dict[str, str]:
        """导出字符级 BIO、映射和兼容现有 NER trainer 的 JSONL。"""

        root = Path(output_dir) / "ner"
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name in ("train", "validation", "test"):
            annotations = getattr(split, name)
            samples = [self.convert_annotation_to_bio(item) for item in annotations]
            text = "\n\n".join(
                "\n".join(
                    f"{token} {label}"
                    for token, label in zip(sample.tokens, sample.labels)
                )
                for sample in samples
            )
            if text:
                text += "\n"
            bio_path = root / f"{name}.txt"
            self._write_text(bio_path, text)
            trainer_records = [
                {
                    "annotation_id": item.annotation_id,
                    "case_id": item.case_id,
                    "text_id": item.text_id,
                    "text": item.text,
                    "entities": [
                        {
                            "entity_id": entity.entity_id,
                            "name": entity.name,
                            "entity_type": self._entity_type(entity),
                            "start": entity.start,
                            "end": entity.end,
                        }
                        for entity in item.entities
                    ],
                }
                for item in annotations
            ]
            self._write_jsonl(root / f"{name}.jsonl", trainer_records)
            paths[name] = str(bio_path)
        self._write_json(root / "label2id.json", self.label2id)
        self._write_json(
            root / "id2label.json",
            {str(value): key for key, value in self.label2id.items()},
        )
        return paths

    def build_positive_relation_samples(
        self, annotations: Sequence[CanonicalAnnotation]
    ) -> list[RelationSample]:
        """将人工正关系转换为 OpenNRE 样本。"""

        samples: list[RelationSample] = []
        for annotation in sorted(annotations, key=self._annotation_sort_key):
            entity_map = {
                entity.entity_id: entity for entity in annotation.entities
            }
            for relation in sorted(
                annotation.relations,
                key=lambda item: (
                    self._relation_type(item),
                    item.head_id,
                    item.tail_id,
                    item.relation_id,
                ),
            ):
                if self._relation_type(relation) == self.negative_relation:
                    raise DatasetValidationError("正关系中不得出现无关系")
                samples.append(
                    self._relation_sample(
                        annotation,
                        entity_map[relation.head_id],
                        entity_map[relation.tail_id],
                        self._relation_type(relation),
                        relation=relation,
                    )
                )
        return samples

    def generate_relation_candidates(
        self, annotation: CanonicalAnnotation
    ) -> list[RelationCandidate]:
        """依据 schema 的实体类型组合生成合法候选对。"""

        entities = sorted(
            annotation.entities,
            key=lambda item: (item.start, item.end, item.entity_id),
        )
        candidates: list[RelationCandidate] = []
        for head in entities:
            for tail in entities:
                if head.entity_id == tail.entity_id:
                    continue
                allowed = [
                    name
                    for name, rule in self.relation_rules.items()
                    if self._entity_type(head) in rule.get("head_types", [])
                    and self._entity_type(tail) in rule.get("tail_types", [])
                ]
                if head.entity_id > tail.entity_id:
                    allowed = [
                        name
                        for name in allowed
                        if bool(
                            self.relation_rules[name].get(
                                "directional", True
                            )
                        )
                    ]
                if not allowed:
                    continue
                distance = self._entity_distance(head, tail)
                same_sentence = self._same_sentence(
                    annotation.text, head, tail
                )
                if (
                    self.config.max_entity_distance_chars is not None
                    and distance > self.config.max_entity_distance_chars
                ):
                    continue
                if self.config.same_sentence_only and not same_sentence:
                    continue
                candidates.append(
                    RelationCandidate(
                        annotation_id=annotation.annotation_id,
                        case_id=annotation.case_id,
                        text_id=annotation.text_id,
                        head_id=head.entity_id,
                        tail_id=tail.entity_id,
                        allowed_relations=allowed,
                        distance_chars=distance,
                        same_sentence=same_sentence,
                    )
                )
        return candidates

    def generate_negative_relation_samples(
        self,
        annotations: Sequence[CanonicalAnnotation],
    ) -> list[RelationSample]:
        """从未标注候选对中按固定种子生成“无关系”负样本。"""

        output: list[RelationSample] = []
        for annotation in sorted(annotations, key=self._annotation_sort_key):
            entity_map = {
                entity.entity_id: entity for entity in annotation.entities
            }
            positives = {
                (relation.head_id, relation.tail_id)
                for relation in annotation.relations
            }
            reverse_positives = (
                set()
                if self.config.allow_reverse_negative_pairs
                else {
                    (relation.tail_id, relation.head_id)
                    for relation in annotation.relations
                }
            )
            undirected_positives = {
                tuple(sorted((relation.head_id, relation.tail_id)))
                for relation in annotation.relations
                if not bool(
                    self.relation_rules[self._relation_type(relation)].get(
                        "directional", True
                    )
                )
            }
            candidates = [
                candidate
                for candidate in self.generate_relation_candidates(annotation)
                if (candidate.head_id, candidate.tail_id) not in positives
                and (candidate.head_id, candidate.tail_id)
                not in reverse_positives
                and tuple(sorted((candidate.head_id, candidate.tail_id)))
                not in undirected_positives
            ]
            positive_count = len(annotation.relations)
            target = min(
                self.config.max_negatives_per_text,
                round(positive_count * self.config.negative_ratio),
            )
            if target <= 0:
                continue
            chosen = self._choose_negative_candidates(
                candidates,
                target,
                annotation.text_id,
                positive_entity_ids={
                    relation.head_id
                    for relation in annotation.relations
                }
                | {
                    relation.tail_id
                    for relation in annotation.relations
                },
            )
            for candidate in chosen:
                output.append(
                    self._relation_sample(
                        annotation,
                        entity_map[candidate.head_id],
                        entity_map[candidate.tail_id],
                        self.negative_relation,
                        negative_reason=(
                            "schema_compatible_pair_without_human_relation"
                        ),
                        sampling_strategy=self.config.negative_sampling_strategy,
                    )
                )
        return output

    # 中文注释：组合正样本和受 schema 约束的负样本，生成关系分类训练记录。
    def build_relation_dataset(
        self, annotations: Sequence[CanonicalAnnotation]
    ) -> list[RelationSample]:
        """合并、去重并稳定排序关系正负样本。"""

        samples = [
            *self.build_positive_relation_samples(annotations),
            *self.generate_negative_relation_samples(annotations),
        ]
        unique: dict[tuple[Any, ...], RelationSample] = {}
        for sample in samples:
            key = (
                sample.text_id,
                tuple(sample.h["pos"]),
                tuple(sample.t["pos"]),
                sample.relation,
            )
            unique.setdefault(key, sample)
        return sorted(
            unique.values(),
            key=lambda item: (
                item.case_id or "",
                item.text_id or "",
                item.h["pos"][0],
                item.t["pos"][0],
                self.relation2id[item.relation],
            ),
        )

    def export_relation_dataset(
        self,
        split: DatasetSplit,
        output_dir: str | Path,
    ) -> dict[str, str]:
        """导出 OpenNRE JSONL 与正式关系映射。"""

        root = Path(output_dir) / "relation"
        root.mkdir(parents=True, exist_ok=True)
        mapping_path = root / "rel2id.json"
        if mapping_path.is_file():
            try:
                current_mapping = json.loads(
                    mapping_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetSchemaError(
                    f"已有 rel2id.json 无法解析：{mapping_path}"
                ) from exc
            if (
                current_mapping != self.relation2id
                and not self.config.overwrite_schema_files
            ):
                missing = sorted(set(self.relation2id) - set(current_mapping))
                extra = sorted(set(current_mapping) - set(self.relation2id))
                changed = sorted(
                    key
                    for key in set(current_mapping) & set(self.relation2id)
                    if current_mapping[key] != self.relation2id[key]
                )
                raise DatasetSchemaError(
                    "已有 rel2id.json 与 schema 不一致；"
                    f"missing={missing}, extra={extra}, changed={changed}"
                )
        paths: dict[str, str] = {}
        for name in ("train", "validation", "test"):
            samples = self.build_relation_dataset(getattr(split, name))
            path = root / f"{name}.jsonl"
            self._write_jsonl(
                path, [sample.model_dump(exclude_none=True) for sample in samples]
            )
            paths[name] = str(path)
        self._write_json(mapping_path, self.relation2id)
        self._write_json(
            root / "id2rel.json",
            {str(value): key for key, value in self.relation2id.items()},
        )
        return paths

    def calculate_dataset_statistics(
        self, split: DatasetSplit
    ) -> DatasetStatistics:
        """统计实体、关系、正负样本、案件和文本长度分布。"""

        all_annotations = [*split.train, *split.validation, *split.test]
        entity_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        lengths: list[int] = []
        entity_lengths: list[int] = []
        no_entity_count = 0
        no_relation_count = 0
        for annotation in all_annotations:
            lengths.append(len(annotation.text))
            entity_lengths.extend(
                entity.end - entity.start for entity in annotation.entities
            )
            no_entity_count += not annotation.entities
            no_relation_count += not annotation.relations
            entity_counts.update(
                self._entity_type(entity) for entity in annotation.entities
            )
            relation_counts.update(
                self._relation_type(relation)
                for relation in annotation.relations
            )
        negative_count = sum(
            len(self.generate_negative_relation_samples(items))
            for items in (split.train, split.validation, split.test)
        )
        case_sets = {
            name: {item.case_id for item in getattr(split, name)}
            for name in ("train", "validation", "test")
        }
        split_entity_counts = {
            name: sum(
                len(item.entities) for item in getattr(split, name)
            )
            for name in ("train", "validation", "test")
        }
        split_relation_counts = {
            name: sum(
                len(item.relations) for item in getattr(split, name)
            )
            for name in ("train", "validation", "test")
        }
        train_relations = {
            self._relation_type(relation)
            for item in split.train
            for relation in item.relations
        }
        validation_test_relations = {
            self._relation_type(relation)
            for item in [*split.validation, *split.test]
            for relation in item.relations
        }
        formal_relations = set(self.relation_rules)
        rare_focus = {"合谋", "转请托", "利益输送", "代收代持", "协助实施"}
        case_disjoint = (
            case_sets["train"].isdisjoint(case_sets["validation"])
            and case_sets["train"].isdisjoint(case_sets["test"])
            and case_sets["validation"].isdisjoint(case_sets["test"])
        )
        positive_count = sum(relation_counts.values())
        rare_relations = sorted(
            relation
            for relation in rare_focus
            if relation_counts.get(relation, 0) < 5
        )
        missing_train = sorted(formal_relations - train_relations)
        warnings = []
        if rare_relations:
            warnings.append(f"稀缺关系：{','.join(rare_relations)}")
        if missing_train:
            warnings.append(
                f"训练集未覆盖关系：{','.join(missing_train)}"
            )
        return DatasetStatistics(
            annotation_count=len(all_annotations),
            case_count=len({item.case_id for item in all_annotations}),
            text_count=len({item.text_id for item in all_annotations}),
            total_characters=sum(lengths),
            entity_count=sum(entity_counts.values()),
            relation_positive_count=positive_count,
            relation_negative_count=negative_count,
            entity_distribution=dict(sorted(entity_counts.items())),
            relation_distribution=dict(sorted(relation_counts.items())),
            split_entity_counts=split_entity_counts,
            split_relation_counts=split_relation_counts,
            split_annotation_counts={
                name: len(getattr(split, name))
                for name in ("train", "validation", "test")
            },
            split_case_counts={
                name: len(case_sets[name])
                for name in ("train", "validation", "test")
            },
            text_length={
                "min": float(min(lengths, default=0)),
                "max": float(max(lengths, default=0)),
                "mean": (
                    float(sum(lengths) / len(lengths)) if lengths else 0.0
                ),
            },
            entity_length={
                "min": float(min(entity_lengths, default=0)),
                "max": float(max(entity_lengths, default=0)),
                "mean": (
                    float(sum(entity_lengths) / len(entity_lengths))
                    if entity_lengths
                    else 0.0
                ),
            },
            average_entities_per_text=(
                sum(entity_counts.values()) / len(all_annotations)
                if all_annotations
                else 0.0
            ),
            average_relations_per_text=(
                positive_count / len(all_annotations)
                if all_annotations
                else 0.0
            ),
            no_entity_text_count=no_entity_count,
            no_relation_text_count=no_relation_count,
            positive_negative_ratio=(
                positive_count / negative_count if negative_count else None
            ),
            rare_relations=rare_relations,
            missing_train_relations=missing_train,
            validation_or_test_only_relations=sorted(
                validation_test_relations - train_relations
            ),
            case_sets_disjoint=case_disjoint,
            leakage_detected=not case_disjoint,
            warnings=warnings,
        )

    # 中文注释：根据规范化标注内容生成稳定指纹，用于版本追溯和重复构建识别。
    def calculate_dataset_fingerprint(
        self, annotations: Sequence[CanonicalAnnotation]
    ) -> str:
        """计算与输入顺序无关、包含 schema 与关键参数的数据集指纹。"""

        normalized = [
            self._annotation_payload(annotation)
            for annotation in sorted(
                annotations, key=self._annotation_sort_key
            )
        ]
        payload = {
            "schema": self.schema,
            "annotations": normalized,
            "split": {
                "train": self.config.train_ratio,
                "validation": self.config.validation_ratio,
                "test": self.config.test_ratio,
                "seed": self.config.random_seed,
            },
            "negative_sampling": self._negative_config(),
        }
        return hashlib.sha256(
            self._canonical_json(payload).encode("utf-8")
        ).hexdigest()

    # 中文注释：汇总 schema、切分、统计、配置和 checksum，形成数据集版本清单。
    def build_manifest(
        self,
        *,
        dataset_version: str,
        fingerprint: str,
        split: DatasetSplit,
        statistics: DatasetStatistics,
        source_annotation_ids: Sequence[str],
        file_checksums: Mapping[str, str] | None = None,
    ) -> DatasetManifest:
        """构造数据集复现 manifest。"""

        return DatasetManifest(
            dataset_version=dataset_version,
            dataset_fingerprint=fingerprint,
            schema_version=str(self.schema.get("schema_version", "unknown")),
            created_at=datetime.now(),
            random_seed=self.config.random_seed,
            split_ratios={
                "train": self.config.train_ratio,
                "validation": self.config.validation_ratio,
                "test": self.config.test_ratio,
            },
            frozen_test_case_ids=sorted(
                {item.case_id for item in split.test}
            ),
            train_case_ids=split.train_case_ids,
            validation_case_ids=split.validation_case_ids,
            test_case_ids=split.test_case_ids,
            source_annotation_ids=sorted(source_annotation_ids),
            label2id=self.label2id,
            relation2id=self.relation2id,
            negative_sampling=self._negative_config(),
            files={
                "ner_train": "ner/train.txt",
                "ner_validation": "ner/validation.txt",
                "ner_test": "ner/test.txt",
                "relation_train": "relation/train.jsonl",
                "relation_validation": "relation/validation.jsonl",
                "relation_test": "relation/test.jsonl",
                "rel2id": "relation/rel2id.json",
                "statistics": "statistics.json",
            },
            configuration={
                "split_by": "case_id",
                "train_ratio": self.config.train_ratio,
                "validation_ratio": self.config.validation_ratio,
                "test_ratio": self.config.test_ratio,
                "freeze_test_set": self.config.freeze_test_set,
                "include_trace_fields": self.config.include_trace_fields,
                "negative_sampling": self._negative_config(),
            },
            file_checksums=dict(sorted((file_checksums or {}).items())),
            statistics=statistics,
            python_version=platform.python_version(),
            git_commit=self._git_commit(),
        )

    # 中文注释：完整构建入口；校验、去重、切分、导出后通过临时目录原子发布新版本。
    def create_dataset_version(
        self,
        annotations: Sequence[CanonicalAnnotation],
        *,
        dataset_version: str | None = None,
        previous_manifest: DatasetManifest | Mapping[str, Any] | None = None,
        rebuild_test_set: bool = False,
        strict: bool | None = None,
        overwrite: bool = False,
    ) -> DatasetBuildResult:
        """经校验、去重和案件级切分后原子创建完整数据集版本。"""

        validation = self.validate_annotations(annotations, strict=strict)
        deduplication = self.deduplicate_annotations(
            validation.valid_annotations
        )
        if not deduplication.annotations:
            raise DatasetValidationError("没有可用于构建数据集的有效标注")
        fingerprint = self.calculate_dataset_fingerprint(
            deduplication.annotations
        )
        version = dataset_version or (
            f"dataset-{datetime.now().strftime('%Y%m%d%H%M%S')}-"
            f"{fingerprint[:10]}"
        )
        self._validate_version_name(version)
        root = self.config.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / version).resolve()
        if target.parent != root:
            raise DatasetExportError("数据集版本路径越出 output_dir")
        if target.exists() and not overwrite:
            raise DatasetVersionExistsError(f"数据集版本已存在：{target}")
        split = self.split_annotations(
            deduplication.annotations,
            previous_manifest=previous_manifest,
            rebuild_test_set=rebuild_test_set,
        )
        temp = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=root))
        try:
            bio_paths = self.export_bio_dataset(split, temp)
            relation_paths = self.export_relation_dataset(split, temp)
            source_path = temp / "source_annotations.jsonl"
            self._write_jsonl(
                source_path,
                [
                    self._annotation_payload(item)
                    for item in deduplication.annotations
                ],
            )
            statistics = self.calculate_dataset_statistics(split)
            self._write_json(
                temp / "statistics.json",
                statistics.model_dump(mode="json"),
            )
            checksums = self._directory_checksums(temp)
            manifest = self.build_manifest(
                dataset_version=version,
                fingerprint=fingerprint,
                split=split,
                statistics=statistics,
                source_annotation_ids=[
                    item.annotation_id for item in deduplication.annotations
                ],
                file_checksums=checksums,
            )
            self._write_json(
                temp / "manifest.json", manifest.model_dump(mode="json")
            )
            if target.exists():
                self._replace_existing_version(temp, target, root)
            else:
                os.replace(temp, target)
        except Exception as exc:
            if temp.exists():
                shutil.rmtree(temp)
            if isinstance(exc, DatasetServiceError):
                raise
            raise DatasetWriteError(
                f"数据集版本导出失败：{version}"
            ) from exc
        logger.info(
            "数据集版本创建完成 version=%s fingerprint=%s path=%s",
            version,
            fingerprint,
            target,
        )
        dataset = DatasetVersion(
            dataset_version=version,
            schema_version=str(self.schema.get("schema_version", "unknown")),
            train_case_ids=sorted({item.case_id for item in split.train}),
            validation_case_ids=sorted(
                {item.case_id for item in split.validation}
            ),
            test_case_ids=sorted({item.case_id for item in split.test}),
            train_size=len(split.train),
            validation_size=len(split.validation),
            test_size=len(split.test),
            bio_train_uri=str(target / "ner" / Path(bio_paths["train"]).name),
            bio_validation_uri=str(
                target / "ner" / Path(bio_paths["validation"]).name
            ),
            bio_test_uri=str(target / "ner" / Path(bio_paths["test"]).name),
            re_train_uri=str(
                target / "relation" / Path(relation_paths["train"]).name
            ),
            re_validation_uri=str(
                target
                / "relation"
                / Path(relation_paths["validation"]).name
            ),
            re_test_uri=str(
                target / "relation" / Path(relation_paths["test"]).name
            ),
            status="READY",
            created_at=manifest.created_at,
        )
        return DatasetBuildResult(
            dataset=dataset,
            output_dir=str(target),
            manifest=manifest,
            validation=validation,
            deduplication=deduplication,
        )

    def _build_bio_mapping(self) -> dict[str, int]:
        labels = ["O"]
        for entity_type in self.entity_types:
            labels.extend((f"B-{entity_type}", f"I-{entity_type}"))
        return {label: index for index, label in enumerate(labels)}

    @staticmethod
    def _entity_type(entity: EntityMention) -> str:
        value = getattr(entity, "type", "")
        return str(getattr(value, "value", value))

    @staticmethod
    def _relation_type(relation: RelationMention) -> str:
        value = getattr(relation, "type", "")
        return str(getattr(value, "value", value))

    @staticmethod
    def _annotation_sort_key(
        annotation: CanonicalAnnotation,
    ) -> tuple[str, str, str]:
        return annotation.case_id, annotation.text_id, annotation.annotation_id

    def _sorted_annotation(
        self, annotation: CanonicalAnnotation
    ) -> CanonicalAnnotation:
        return annotation.model_copy(
            update={
                "entities": sorted(
                    annotation.entities,
                    key=lambda item: (
                        item.start,
                        item.end,
                        item.entity_id,
                    ),
                ),
                "relations": sorted(
                    annotation.relations,
                    key=lambda item: (
                        self._relation_type(item),
                        item.head_id,
                        item.tail_id,
                        item.relation_id,
                    ),
                ),
            }
        )

    @staticmethod
    def _entity_distance(
        head: EntityMention, tail: EntityMention
    ) -> int:
        if head.end <= tail.start:
            return tail.start - head.end
        if tail.end <= head.start:
            return head.start - tail.end
        return 0

    @staticmethod
    def _same_sentence(
        text: str, head: EntityMention, tail: EntityMention
    ) -> bool:
        left = min(head.end, tail.end)
        right = max(head.start, tail.start)
        return not any(char in "。！？；\n" for char in text[left:right])

    def _relation_sample(
        self,
        annotation: CanonicalAnnotation,
        head: EntityMention,
        tail: EntityMention,
        relation_type: str,
        *,
        relation: RelationMention | None = None,
        negative_reason: str | None = None,
        sampling_strategy: str | None = None,
    ) -> RelationSample:
        trace = self.config.include_trace_fields
        return RelationSample(
            text=annotation.text,
            h={
                "id": head.entity_id,
                "name": head.name,
                "pos": [head.start, head.end],
                "type": self._entity_type(head),
            },
            t={
                "id": tail.entity_id,
                "name": tail.name,
                "pos": [tail.start, tail.end],
                "type": self._entity_type(tail),
            },
            relation=relation_type,
            annotation_id=annotation.annotation_id if trace else None,
            case_id=annotation.case_id if trace else None,
            text_id=annotation.text_id if trace else None,
            relation_id=relation.relation_id if relation and trace else None,
            negative_reason=negative_reason if trace else None,
            sampling_strategy=sampling_strategy if trace else None,
        )

    def _choose_negative_candidates(
        self,
        candidates: Sequence[RelationCandidate],
        target: int,
        salt: str,
        *,
        positive_entity_ids: set[str],
    ) -> list[RelationCandidate]:
        hard = sorted(
            candidates,
            key=lambda item: (
                not item.same_sentence,
                not (
                    item.head_id in positive_entity_ids
                    or item.tail_id in positive_entity_ids
                ),
                item.distance_chars,
                item.head_id,
                item.tail_id,
            ),
        )
        strategy = self.config.negative_sampling_strategy
        if strategy == "hard":
            return hard[:target]
        rng = random.Random(
            f"{self.config.random_seed}:{salt}"
        )
        if strategy == "random":
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            return shuffled[:target]
        hard_count = min(len(hard), (target + 1) // 2)
        selected = hard[:hard_count]
        remaining = [item for item in candidates if item not in selected]
        rng.shuffle(remaining)
        return [*selected, *remaining[: target - hard_count]]

    def _negative_config(self) -> dict[str, Any]:
        return {
            "strategy": self.config.negative_sampling_strategy,
            "negative_ratio": self.config.negative_ratio,
            "max_negatives_per_text": self.config.max_negatives_per_text,
            "max_entity_distance_chars": (
                self.config.max_entity_distance_chars
            ),
            "same_sentence_only": self.config.same_sentence_only,
            "allow_reverse_negative_pairs": (
                self.config.allow_reverse_negative_pairs
            ),
        }

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _annotation_payload(
        self, annotation: CanonicalAnnotation
    ) -> dict[str, Any]:
        return self._sorted_annotation(annotation).model_dump(mode="json")

    @staticmethod
    def _validate_version_name(version: str) -> None:
        if not version or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for char in version
        ):
            raise DatasetExportError(
                "dataset_version 仅允许字母、数字、点、下划线和连字符"
            )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def _write_json(cls, path: Path, value: Any) -> None:
        cls._write_text(
            path,
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )

    @classmethod
    def _write_jsonl(
        cls, path: Path, records: Sequence[Mapping[str, Any]]
    ) -> None:
        cls._write_text(
            path,
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
        )

    @staticmethod
    def _directory_checksums(root: Path) -> dict[str, str]:
        checksums: dict[str, str] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            checksums[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        return checksums

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    @staticmethod
    def _replace_existing_version(
        temporary: Path, target: Path, root: Path
    ) -> None:
        """显式覆盖时先备份旧版本，发布失败则恢复。"""

        resolved = target.resolve()
        if resolved.parent != root.resolve() or resolved == root.resolve():
            raise DatasetExportError("拒绝删除 output_dir 外的路径")
        backup = target.with_name(
            f".{target.name}.backup-{os.getpid()}"
        )
        if backup.exists():
            raise DatasetWriteError(f"覆盖备份路径已存在：{backup}")
        os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)


def _timestamp_number(value: str) -> float:
    """将 ISO 时间转换为排序数值，非法值按最旧处理。"""

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OSError):
        return float("-inf")

"""Stage planning and upstream business preconditions shared by all entry points."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from config import BASE_DIR, PreflightFeature, ProjectConfig, load_project_config
from models import AnnotationStatus, CanonicalAnnotation
from src.services.dataset_service import DatasetManifest


StageName = Literal[
    "retrieval",
    "annotation",
    "review-sync",
    "dataset-build",
    "training",
    "graph-ingestion",
]


class StageRoutePlan(BaseModel):
    """Validated description of the stages and integrations a request needs."""

    command: str
    stages: list[StageName] = Field(default_factory=list)
    required_features: list[PreflightFeature] = Field(default_factory=list)


class StagePreconditionError(ValueError):
    """Stable error raised before a child flow creates side effects."""

    def __init__(self, stage: str, code: str, message: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(f"stage={stage} code={code}: {message}")


class StageRoutingService:
    """Build execution plans and validate cross-field/upstream state."""

    def __init__(
        self,
        *,
        project_config: ProjectConfig | None = None,
        datasets_root: str | Path | None = None,
    ) -> None:
        self.config = project_config or load_project_config()
        configured_root = self.config.training["modeling"]["datasets_root"]
        root = Path(datasets_root or configured_root)
        self.datasets_root = root if root.is_absolute() else BASE_DIR / root

    def prepare(self, command: str, request: BaseModel) -> StageRoutePlan:
        """Perform steps 1–3: select stages, validate request, validate upstream."""

        stages = self._selected_stages(command, request)
        self._validate_parameters(command, request, stages)
        self._validate_upstream(command, request, stages)
        return StageRoutePlan(
            command=command,
            stages=stages,
            required_features=self._required_features(command, request),
        )

    @staticmethod
    def require_approved_annotations(
        annotations: Sequence[CanonicalAnnotation],
        *,
        stage: str,
    ) -> None:
        """Reject empty or non-human-approved annotation collections."""

        if not annotations:
            raise StagePreconditionError(
                stage, "missing_annotations", "该阶段需要上游标注结果"
            )
        invalid = [
            item.annotation_id
            for item in annotations
            if item.status != AnnotationStatus.APPROVED
        ]
        if invalid:
            preview = ", ".join(invalid[:5])
            raise StagePreconditionError(
                stage,
                "annotations_not_approved",
                f"仅 APPROVED 标注可进入该阶段，未通过标注：{preview}",
            )

    def validate_dataset_artifact(self, dataset_version: str) -> DatasetManifest:
        """Validate manifest identity, schema compatibility, files and checksums."""

        root = self.datasets_root / dataset_version
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise StagePreconditionError(
                "training",
                "dataset_manifest_missing",
                f"数据集 manifest 不存在：{manifest_path}",
            )
        try:
            manifest = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise StagePreconditionError(
                "training", "dataset_manifest_invalid", "数据集 manifest 无法校验"
            ) from exc
        if manifest.dataset_version != dataset_version:
            raise StagePreconditionError(
                "training",
                "dataset_version_mismatch",
                "请求版本与 manifest.dataset_version 不一致",
            )
        expected_schema = str(self.config.schema_config.get("schema_version", ""))
        if expected_schema and manifest.schema_version != expected_schema:
            raise StagePreconditionError(
                "training",
                "dataset_schema_mismatch",
                f"数据集 schema={manifest.schema_version}，运行时 schema={expected_schema}",
            )
        for relative_name, expected_hash in manifest.file_checksums.items():
            artifact = (root / relative_name).resolve()
            if not artifact.is_relative_to(root.resolve()):
                raise StagePreconditionError(
                    "training",
                    "dataset_artifact_path_invalid",
                    f"数据集文件路径越界：{relative_name}",
                )
            if not artifact.is_file():
                raise StagePreconditionError(
                    "training",
                    "dataset_artifact_missing",
                    f"数据集文件不存在：{relative_name}",
                )
            digest = hashlib.sha256()
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise StagePreconditionError(
                    "training",
                    "dataset_checksum_mismatch",
                    f"数据集文件校验失败：{relative_name}",
                )
        return manifest

    def validate_model_artifacts(self) -> None:
        """Check inference checkpoints before the annotation flow starts."""

        modeling = self.config.training.get("modeling", {})
        ner = modeling.get("ner", {})
        relation = modeling.get("relation", {})
        ner_value = str(ner.get("checkpoint_path") or "").strip()
        relation_value = str(relation.get("checkpoint_path") or "").strip()
        if not ner_value:
            raise StagePreconditionError(
                "annotation",
                "ner_checkpoint_not_configured",
                "training.modeling.ner.checkpoint_path 未配置",
            )
        ner_path = Path(ner_value)
        ner_path = ner_path if ner_path.is_absolute() else BASE_DIR / ner_path
        if not ner_path.is_dir():
            raise StagePreconditionError(
                "annotation",
                "ner_checkpoint_missing",
                f"NER checkpoint 目录不存在：{ner_path}",
            )
        required_ner = [ner_path / "config.json"]
        if not all(path.is_file() for path in required_ner) or not any(
            (ner_path / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            raise StagePreconditionError(
                "annotation",
                "ner_checkpoint_incomplete",
                "NER checkpoint 缺少 config.json 或模型权重",
            )
        if not relation_value:
            raise StagePreconditionError(
                "annotation",
                "relation_checkpoint_not_configured",
                "training.modeling.relation.checkpoint_path 未配置",
            )
        relation_path = Path(relation_value)
        relation_path = (
            relation_path
            if relation_path.is_absolute()
            else BASE_DIR / relation_path
        )
        if not relation_path.is_file():
            raise StagePreconditionError(
                "annotation",
                "relation_checkpoint_missing",
                f"关系模型 checkpoint 不存在：{relation_path}",
            )

    @staticmethod
    def _selected_stages(command: str, request: BaseModel) -> list[StageName]:
        if command != "ingestion":
            return [command]  # type: ignore[list-item]
        ordered: tuple[tuple[str, StageName], ...] = (
            ("run_retrieval", "retrieval"),
            ("run_annotation", "annotation"),
            ("run_review_sync", "review-sync"),
            ("run_dataset_build", "dataset-build"),
            ("run_training", "training"),
            ("run_graph_ingestion", "graph-ingestion"),
        )
        return [stage for flag, stage in ordered if getattr(request, flag)]

    def _validate_parameters(
        self,
        command: str,
        request: BaseModel,
        stages: Sequence[StageName],
    ) -> None:
        if command == "annotation" and not (
            request.jobs or request.raw_documents
        ):
            raise StagePreconditionError(
                "annotation",
                "missing_input",
                "必须提供 jobs 或 raw_documents",
            )
        if command == "dataset-build" and not request.annotations:
            raise StagePreconditionError(
                "dataset-build", "missing_annotations", "annotations 不能为空"
            )
        if command == "graph-ingestion" and not request.annotations:
            raise StagePreconditionError(
                "graph-ingestion", "missing_annotations", "annotations 不能为空"
            )
        if command != "ingestion" or request.dry_run:
            return
        if not stages:
            raise StagePreconditionError(
                "ingestion", "no_stage_selected", "至少启用一个 run_* 阶段"
            )
        if request.run_annotation and not (
            request.annotation_jobs
            or request.raw_documents
            or request.run_retrieval
        ):
            raise StagePreconditionError(
                "annotation",
                "missing_input",
                "run_annotation 需要 annotation_jobs、raw_documents 或 run_retrieval",
            )
        if request.publish_for_review and not request.run_annotation:
            raise StagePreconditionError(
                "annotation",
                "publish_without_annotation",
                "publish_for_review 需要启用 run_annotation",
            )
        self._validate_human_review_phase(request)
        if request.run_training and not (
            request.run_dataset_build or request.dataset_version
        ):
            raise StagePreconditionError(
                "training",
                "missing_dataset",
                "run_training 需要 run_dataset_build 或 dataset_version",
            )

    @staticmethod
    def _validate_human_review_phase(request: BaseModel) -> None:
        """Keep review submission and review consumption in separate flow runs."""

        if request.run_annotation and request.run_review_sync:
            raise StagePreconditionError(
                "ingestion",
                "human_review_phase_conflict",
                "run_annotation 与 run_review_sync 必须在两次独立 Flow 中执行",
            )
        if request.publish_for_review:
            forbidden = [
                name
                for name in (
                    "run_review_sync",
                    "run_dataset_build",
                    "run_training",
                    "run_graph_ingestion",
                )
                if getattr(request, name)
            ]
            if forbidden:
                raise StagePreconditionError(
                    "ingestion",
                    "review_submission_must_stop",
                    "提交人工审核后本次 Flow 必须结束，禁止阶段："
                    + ", ".join(forbidden),
                )
        if request.run_review_sync:
            forbidden = [
                name
                for name in ("run_retrieval", "run_annotation")
                if getattr(request, name)
            ]
            if request.publish_for_review:
                forbidden.append("publish_for_review")
            if forbidden:
                raise StagePreconditionError(
                    "ingestion",
                    "review_consumption_must_be_separate",
                    "审核结果回传阶段不能重新检索或标注："
                    + ", ".join(dict.fromkeys(forbidden)),
                )

    def _validate_upstream(
        self,
        command: str,
        request: BaseModel,
        stages: Sequence[StageName],
    ) -> None:
        del stages
        if command in {"dataset-build", "graph-ingestion"}:
            self.require_approved_annotations(
                request.annotations, stage=command
            )
        if command == "training":
            self.validate_dataset_artifact(request.dataset_version)
        if command == "annotation":
            self.validate_model_artifacts()
        if command == "retrieval":
            self._validate_retrieval_sources(request.source_ids)
        if command != "ingestion" or request.dry_run:
            return
        if request.run_retrieval:
            self._validate_retrieval_sources(request.retrieval_source_ids)
        needs_approved = request.run_dataset_build or request.run_graph_ingestion
        if needs_approved and not request.run_review_sync:
            if request.run_annotation:
                raise StagePreconditionError(
                    "ingestion",
                    "review_required",
                    "模型标注不能直接进入数据集或图谱；请启用 run_review_sync",
                )
            self.require_approved_annotations(
                request.annotations,
                stage=(
                    "dataset-build"
                    if request.run_dataset_build
                    else "graph-ingestion"
                ),
            )
        if request.run_annotation:
            self.validate_model_artifacts()
        if request.run_training and not request.run_dataset_build:
            self.validate_dataset_artifact(request.dataset_version)

    def _validate_retrieval_sources(
        self, source_ids: Sequence[str] | None
    ) -> None:
        if not source_ids:
            return
        configured = {
            str(item.get("source_id"))
            for item in self.config.sources.get("sources", [])
            if item.get("enabled", True)
        }
        missing = sorted(set(source_ids) - configured)
        if missing:
            raise StagePreconditionError(
                "retrieval",
                "unknown_sources",
                f"来源不存在或已禁用：{missing}",
            )

    @staticmethod
    def _required_features(
        command: str, request: BaseModel
    ) -> list[PreflightFeature]:
        if command == "retrieval":
            return ["tavily"]
        if command == "review-sync":
            return ["label_studio"]
        if command == "graph-ingestion":
            return ["neo4j"]
        if command == "annotation":
            return ["label_studio"] if request.publish_for_review else []
        if command != "ingestion" or request.dry_run:
            return []
        features: list[PreflightFeature] = []
        if request.run_retrieval:
            features.append("tavily")
        if request.publish_for_review or request.run_review_sync:
            features.append("label_studio")
        if request.run_graph_ingestion:
            features.append("neo4j")
        return features


__all__ = [
    "StagePreconditionError",
    "StageRoutePlan",
    "StageRoutingService",
]

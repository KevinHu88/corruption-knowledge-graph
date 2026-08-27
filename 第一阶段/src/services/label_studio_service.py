"""Label Studio SDK 交互及 CanonicalAnnotation 双向格式转换服务。"""

from __future__ import annotations

import importlib.metadata
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from config import ProjectConfig, load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    EntityMention,
    EntityType,
    RelationMention,
    RelationType,
)

logger = logging.getLogger(__name__)


class LabelStudioHealthResult(BaseModel):
    """Label Studio 连接健康检查结果。"""

    connected: bool
    project_id: int
    project_title: str | None = None
    sdk_version: str
    latency_seconds: float = Field(ge=0)


class LabelStudioTaskReference(BaseModel):
    """业务 annotation 与 Label Studio task 的映射。"""

    annotation_id: str
    task_id: int
    project_id: int


class LabelStudioImportBatchResult(BaseModel):
    """单次 SDK 批量导入结果。"""

    batch_index: int
    requested_count: int
    imported_count: int = 0
    import_id: int | None = None
    queued: bool = False
    task_ids: list[int] = Field(default_factory=list)
    mappings: list[LabelStudioTaskReference] = Field(default_factory=list)
    error: str | None = None


# 中文注释：批量发布结果，保留每批成功/失败情况以及生成的 Label Studio 任务引用。
class LabelStudioImportResult(BaseModel):
    """一次分批导入的统一汇总。"""

    project_id: int
    requested_count: int
    imported_count: int
    queued_count: int = 0
    failed_count: int
    batches: list[LabelStudioImportBatchResult] = Field(default_factory=list)
    mappings: list[LabelStudioTaskReference] = Field(default_factory=list)
    latency_seconds: float = Field(ge=0)


class LabelStudioReviewedTask(BaseModel):
    """包含已选定人工 annotation 的 Label Studio 任务。"""

    task_id: int
    project_id: int
    task: dict[str, Any]
    annotation: dict[str, Any]


# 中文注释：人工审核同步结果，汇总已转换标注和失败任务 ID。
class LabelStudioSyncResult(BaseModel):
    """人工审核结果回转汇总，不包含任何数据库写入。"""

    annotations: list[CanonicalAnnotation] = Field(default_factory=list)
    task_ids: list[int] = Field(default_factory=list)
    failed_task_ids: list[int] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_seconds: float = Field(ge=0)


class LabelStudioLabelConfigResult(BaseModel):
    """项目标注 XML 与 schema 的一致性检查结果。"""

    valid: bool
    missing_entity_labels: list[str] = Field(default_factory=list)
    missing_relation_labels: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LabelStudioServiceError(RuntimeError):
    """Label Studio 服务基础异常。"""


class LabelStudioConfigurationError(LabelStudioServiceError):
    """连接参数或标注控件配置缺失。"""


class LabelStudioConnectionError(LabelStudioServiceError):
    """网络不可达或请求超时。"""


class LabelStudioAuthenticationError(LabelStudioServiceError):
    """API Key 无效或没有访问权限。"""


class LabelStudioProjectError(LabelStudioServiceError):
    """项目不存在或项目标注配置错误。"""


class LabelStudioTaskNotFoundError(LabelStudioServiceError):
    """指定的 Label Studio 任务不存在。"""


class LabelStudioImportError(LabelStudioServiceError):
    """任务或 prediction 导入失败。"""


class LabelStudioResponseError(LabelStudioServiceError):
    """SDK 返回结构无法识别。"""


class LabelStudioConversionError(LabelStudioServiceError):
    """业务标注与 Label Studio JSON 转换失败。"""


# 中文注释：Label Studio 外部适配器，隔离 SDK、项目配置、预测导入和审核结果转换。
class LabelStudioService:
    """新版 ``LabelStudio`` SDK 的薄封装和标注格式转换器。"""

    def __init__(
        self,
        *,
        project_config: ProjectConfig | None = None,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: int | None = None,
        timeout: float | None = None,
        batch_size: int | None = None,
        model_version: str | None = None,
    ) -> None:
        self.project_config = project_config or load_project_config()
        environment = self.project_config.environment
        config = self.project_config.workflow.get("label_studio", {})
        self.base_url = (base_url or environment.label_studio_url).strip()
        self.api_key = (api_key or environment.label_studio_api_key).strip()
        self.project_id = (
            project_id
            or environment.label_studio_project_id
            or config.get("project_id")
        )
        self.timeout = float(
            timeout
            if timeout is not None
            else environment.label_studio_timeout
        )
        self.batch_size = int(
            batch_size
            if batch_size is not None
            else config.get(
                "batch_size", environment.label_studio_batch_size
            )
        )
        self.model_version = str(
            model_version
            or environment.label_studio_model_version
            or config.get(
                "prediction_model_version", "deep_model_annotation_v1"
            )
        )
        self.max_prediction_results_per_task = int(
            config.get("max_prediction_results_per_task", 500)
        )
        self.text_data_key = str(config.get("text_data_key", "text"))
        self.text_to_name = str(config.get("text_to_name", "text"))
        self.entity_from_name = str(
            config.get("entity_from_name", "label")
        )
        self.review_strategy = str(
            config.get("review_strategy", "latest_non_cancelled")
        )
        self.schema = self.project_config.schema_config
        self._validate_configuration()
        if client is None:
            try:
                from label_studio_sdk import LabelStudio
            except ImportError as exc:
                raise LabelStudioConfigurationError(
                    "缺少 label-studio-sdk；请安装支持 LabelStudio 客户端的新版 SDK"
                ) from exc
            try:
                self.client = LabelStudio(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                self._raise_sdk_exception(exc)
        else:
            self.client = client

    # 中文注释：验证服务连通性、认证和目标项目是否可访问。
    def health_check(self) -> LabelStudioHealthResult:
        """只读检查连接、密钥和默认项目访问权限。"""

        started = time.perf_counter()
        project = self.get_project()
        data = _sdk_object_to_dict(project)
        return LabelStudioHealthResult(
            connected=True,
            project_id=int(data.get("id", self.project_id)),
            project_title=(
                str(data.get("title")) if data.get("title") is not None else None
            ),
            sdk_version=_sdk_version(),
            latency_seconds=time.perf_counter() - started,
        )

    def get_project(self, project_id: int | None = None) -> Any:
        """获取指定或默认项目。"""

        selected = self._project_id(project_id)
        try:
            project = self.client.projects.get(id=selected)
        except Exception as exc:
            self._raise_sdk_exception(
                exc, project_context=True
            )
        if project is None:
            raise LabelStudioProjectError(f"项目不存在：{selected}")
        return project

    # 中文注释：把规范标注转换为 Label Studio 数据字段和预测结果所需的任务 payload。
    def build_task_payload(
        self, annotation: CanonicalAnnotation
    ) -> dict[str, Any]:
        """构造包含业务关联键的 Label Studio task data。"""

        return {
            self.text_data_key: annotation.text,
            "annotation_id": annotation.annotation_id,
            "case_id": annotation.case_id,
            "doc_id": annotation.doc_id,
            "text_id": annotation.text_id,
            "schema_version": annotation.schema_version,
        }

    def build_prediction_results(
        self, annotation: CanonicalAnnotation
    ) -> list[dict[str, Any]]:
        """将实体和正关系转换为 Label Studio prediction result。"""

        return [
            *self._annotation_to_entity_results(annotation),
            *self._annotation_to_relation_results(annotation),
        ]

    def build_import_task(
        self,
        annotation: CanonicalAnnotation,
        *,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        """组合 task data 和单条预标注 prediction。"""

        payload: dict[str, Any] = {
            "data": self.build_task_payload(annotation),
        }
        results = self.build_prediction_results(annotation)
        if len(results) > self.max_prediction_results_per_task:
            logger.warning(
                "预标注结果过多，仅发布无预标注文本 "
                "annotation_id=%s results=%d limit=%d",
                annotation.annotation_id,
                len(results),
                self.max_prediction_results_per_task,
            )
            return payload
        payload["predictions"] = [
            {
                "model_version": model_version or self.model_version,
                "score": self._calculate_prediction_score(annotation),
                "result": results,
            }
        ]
        return payload

    # 中文注释：按配置批量导入规范标注，并记录每批结果；调用方需检查部分失败。
    def import_annotations(
        self,
        annotations: Sequence[CanonicalAnnotation],
        *,
        project_id: int | None = None,
        model_version: str | None = None,
        batch_size: int | None = None,
    ) -> LabelStudioImportResult:
        """按批导入任务并返回 annotation_id→task_id 映射。"""

        selected = self._project_id(project_id)
        if not annotations:
            return LabelStudioImportResult(
                project_id=selected,
                requested_count=0,
                imported_count=0,
                failed_count=0,
                latency_seconds=0.0,
            )
        self.validate_project_label_config(project_id=selected)
        size = int(batch_size or self.batch_size)
        if size <= 0:
            raise LabelStudioConfigurationError("batch_size 必须大于 0")
        started = time.perf_counter()
        batches: list[LabelStudioImportBatchResult] = []
        mappings: list[LabelStudioTaskReference] = []
        imported = 0
        queued = 0
        for batch_index, batch in enumerate(_chunked(annotations, size), 1):
            payloads = [
                self.build_import_task(
                    item, model_version=model_version
                )
                for item in batch
            ]
            try:
                response = self.client.projects.import_tasks(
                    id=selected,
                    request=payloads,
                    return_task_ids=True,
                )
                data = _sdk_object_to_dict(response)
                task_ids = _extract_task_ids(data)
                import_id = _optional_int(
                    data.get("import", data.get("import_id"))
                )
                count = int(data.get("task_count", len(task_ids)))
                is_queued = (
                    import_id is not None
                    and "task_count" not in data
                    and not task_ids
                )
                batch_mappings = [
                    LabelStudioTaskReference(
                        annotation_id=annotation.annotation_id,
                        task_id=task_id,
                        project_id=selected,
                    )
                    for annotation, task_id in zip(batch, task_ids)
                ]
                imported += count
                if is_queued:
                    queued += len(batch)
                mappings.extend(batch_mappings)
                result = LabelStudioImportBatchResult(
                    batch_index=batch_index,
                    requested_count=len(batch),
                    imported_count=count,
                    import_id=import_id,
                    queued=is_queued,
                    task_ids=task_ids,
                    mappings=batch_mappings,
                )
            except Exception as exc:
                converted = self._converted_exception(
                    exc, import_context=True
                )
                # Transport failures must reach the Prefect task so its
                # retry policy can run. Per-item conversion/import responses
                # remain representable as partial batch failures.
                if isinstance(converted, LabelStudioConnectionError):
                    raise converted from exc
                result = LabelStudioImportBatchResult(
                    batch_index=batch_index,
                    requested_count=len(batch),
                    error=str(converted),
                )
            batches.append(result)
            logger.info(
                "Label Studio 导入 project_id=%s batch=%s requested=%s "
                "imported=%s failed=%s task_ids=%s",
                selected,
                batch_index,
                result.requested_count,
                result.imported_count,
                result.requested_count - result.imported_count,
                result.task_ids,
            )
        return LabelStudioImportResult(
            project_id=selected,
            requested_count=len(annotations),
            imported_count=imported,
            queued_count=queued,
            failed_count=max(0, len(annotations) - imported - queued),
            batches=batches,
            mappings=mappings,
            latency_seconds=time.perf_counter() - started,
        )

    def list_tasks(
        self,
        *,
        project_id: int | None = None,
        page_size: int = 100,
        max_items: int = 1000,
        only_annotated: bool = False,
    ) -> list[dict[str, Any]]:
        """分页读取任务，并对最大读取量设置硬上限。"""

        selected = self._project_id(project_id)
        if page_size <= 0 or max_items < 0:
            raise ValueError("page_size 必须大于 0，max_items 不能为负")
        output: list[dict[str, Any]] = []
        page = 1
        while len(output) < max_items:
            try:
                response = self.client.tasks.list(
                    project=selected,
                    page=page,
                    page_size=page_size,
                    fields="all",
                    only_annotated=only_annotated,
                )
            except Exception as exc:
                self._raise_sdk_exception(exc)
            items, total = _extract_task_page(response)
            for item in items:
                task = _sdk_object_to_dict(item)
                if (
                    not only_annotated
                    or self._select_review_annotation(task) is not None
                ):
                    output.append(task)
                    if len(output) >= max_items:
                        break
            if not items or len(items) < page_size:
                break
            if total is not None and page * page_size >= total:
                break
            page += 1
        return output

    def get_task(self, task_id: int) -> dict[str, Any]:
        """获取包含 data、predictions、annotations 和 project 的任务。"""

        try:
            response = self.client.tasks.get(id=int(task_id))
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                raise LabelStudioTaskNotFoundError(
                    f"Label Studio 任务不存在：task_id={task_id}"
                ) from exc
            self._raise_sdk_exception(exc)
        task = _sdk_object_to_dict(response)
        task.setdefault("id", int(task_id))
        task.setdefault("task_id", int(task_id))
        if "data" not in task:
            raise LabelStudioResponseError(
                f"任务返回缺少 data：task_id={task_id}"
            )
        return task

    # 中文注释：分页读取项目任务，并选择具有有效人工标注的记录。
    def fetch_reviewed_tasks(
        self,
        *,
        task_ids: Sequence[int] | None = None,
        project_id: int | None = None,
        max_items: int = 1000,
    ) -> list[LabelStudioReviewedTask]:
        """获取并选择每个任务最新的非取消人工 annotation。"""

        reviewed, _ = self._fetch_reviewed_tasks(
            task_ids=task_ids,
            project_id=project_id,
            max_items=max_items,
        )
        return reviewed

    def _fetch_reviewed_tasks(
        self,
        *,
        task_ids: Sequence[int] | None,
        project_id: int | None,
        max_items: int,
    ) -> tuple[list[LabelStudioReviewedTask], list[tuple[int, str]]]:
        """读取人工审核任务，单独汇总已删除的指定任务。"""

        selected = self._project_id(project_id)
        missing: list[tuple[int, str]] = []
        if task_ids is None:
            tasks = self.list_tasks(
                project_id=selected,
                max_items=max_items,
                only_annotated=True,
            )
        else:
            tasks = []
            for task_id in task_ids[:max_items]:
                try:
                    tasks.append(self.get_task(task_id))
                except LabelStudioTaskNotFoundError as exc:
                    missing.append((int(task_id), str(exc)))
                    logger.warning(
                        "跳过不存在的 Label Studio 任务 task_id=%s",
                        task_id,
                    )
        output: list[LabelStudioReviewedTask] = []
        for task in tasks:
            annotation = self._select_review_annotation(task)
            if annotation is None:
                continue
            output.append(
                LabelStudioReviewedTask(
                    task_id=int(task.get("id", task.get("task_id"))),
                    project_id=int(task.get("project", selected)),
                    task=task,
                    annotation=annotation,
                )
            )
        return output, missing

    # 中文注释：把 Label Studio 的实体/关系结果转换回 APPROVED CanonicalAnnotation。
    def convert_reviewed_task(
        self,
        reviewed: LabelStudioReviewedTask | Mapping[str, Any],
    ) -> CanonicalAnnotation:
        """将最终人工 result 转换为 APPROVED CanonicalAnnotation。"""

        if isinstance(reviewed, LabelStudioReviewedTask):
            task = reviewed.task
            annotation = reviewed.annotation
            project_id = reviewed.project_id
            task_id = reviewed.task_id
        else:
            task = _sdk_object_to_dict(reviewed)
            annotation = self._select_review_annotation(task)
            if annotation is None:
                raise LabelStudioConversionError("任务没有有效人工审核结果")
            project_id = int(task.get("project", self.project_id))
            task_id = int(task.get("id", task.get("task_id")))
        data = _sdk_object_to_dict(task.get("data", {}))
        required = {
            self.text_data_key,
            "annotation_id",
            "case_id",
            "doc_id",
            "text_id",
            "schema_version",
        }
        missing = sorted(required - set(data))
        if missing:
            raise LabelStudioConversionError(
                f"任务 data 缺少业务字段：{missing}"
            )
        text = str(data[self.text_data_key])
        results = annotation.get("result")
        if not isinstance(results, list):
            raise LabelStudioConversionError("人工 annotation.result 不是数组")
        entities = self._parse_entity_results(results, text)
        relations = self._parse_relation_results(results, entities)
        reviewed_at = _parse_datetime(
            annotation.get("updated_at") or annotation.get("created_at")
        ) or datetime.now(timezone.utc)
        reviewer = annotation.get(
            "completed_by", annotation.get("updated_by")
        )
        if isinstance(reviewer, Mapping):
            reviewer = reviewer.get("id", reviewer.get("email"))
        prediction_version = None
        predictions = task.get("predictions") or []
        if predictions:
            prediction_version = _sdk_object_to_dict(
                predictions[-1]
            ).get("model_version")
        try:
            return CanonicalAnnotation(
                annotation_id=str(data["annotation_id"]),
                case_id=str(data["case_id"]),
                doc_id=str(data["doc_id"]),
                text_id=str(data["text_id"]),
                text=text,
                entities=entities,
                relations=relations,
                annotation_source="HUMAN",
                schema_version=str(data["schema_version"]),
                status=AnnotationStatus.APPROVED,
                updated_at=reviewed_at,
                metadata={
                    "label_studio_project_id": project_id,
                    "label_studio_task_id": task_id,
                    "label_studio_annotation_id": annotation.get("id"),
                    "reviewer_id": reviewer,
                    "reviewed_at": reviewed_at.isoformat(),
                    "prediction_model_version": prediction_version,
                },
            )
        except Exception as exc:
            raise LabelStudioConversionError(
                f"人工标注无法转换：task_id={task_id}"
            ) from exc

    # 中文注释：人审同步总入口，负责拉取、转换和汇总失败任务。
    def sync_reviewed_annotations(
        self,
        *,
        task_ids: Sequence[int] | None = None,
        project_id: int | None = None,
        max_items: int = 1000,
    ) -> LabelStudioSyncResult:
        """拉取并转换人工审核结果，不写数据库或触发训练。"""

        started = time.perf_counter()
        reviewed, missing = self._fetch_reviewed_tasks(
            task_ids=task_ids,
            project_id=project_id,
            max_items=max_items,
        )
        annotations: list[CanonicalAnnotation] = []
        successful: list[int] = []
        failed = [task_id for task_id, _ in missing]
        errors = [error for _, error in missing]
        for item in reviewed:
            try:
                annotations.append(self.convert_reviewed_task(item))
                successful.append(item.task_id)
            except LabelStudioConversionError as exc:
                failed.append(item.task_id)
                errors.append(str(exc))
        logger.info(
            "Label Studio 审核同步 project_id=%s success=%s failed=%s "
            "latency_seconds=%.6f",
            self._project_id(project_id),
            len(successful),
            len(failed),
            time.perf_counter() - started,
        )
        return LabelStudioSyncResult(
            annotations=annotations,
            task_ids=successful,
            failed_task_ids=failed,
            errors=errors,
            latency_seconds=time.perf_counter() - started,
        )

    # 中文注释：检查 Label Studio 项目标签配置是否覆盖当前实体与关系 schema。
    def build_label_config(self) -> str:
        """根据当前 schema 生成 Label Studio 实体与关系 XML。"""

        root = ElementTree.Element("View")
        labels = ElementTree.SubElement(
            root,
            "Labels",
            {"name": self.entity_from_name, "toName": self.text_to_name},
        )
        colors = {
            "PER": "#4CAF50",
            "ORG": "#FF9800",
            "POSITION": "#9C27B0",
            "MONEY": "#F44336",
        }
        for entity_type in self.schema["entity_types"]:
            ElementTree.SubElement(
                labels,
                "Label",
                {
                    "value": str(entity_type),
                    "background": colors.get(str(entity_type), "#607D8B"),
                },
            )
        relations = ElementTree.SubElement(
            root,
            "Relations",
            {"name": "relation", "toName": self.entity_from_name},
        )
        for relation_type in self.schema["relation_types"]:
            ElementTree.SubElement(
                relations, "Relation", {"value": str(relation_type)}
            )
        ElementTree.SubElement(
            root,
            "Text",
            {"name": self.text_to_name, "value": f"${self.text_data_key}"},
        )
        ElementTree.indent(root, space="  ")
        return ElementTree.tostring(root, encoding="unicode")

    def validate_project_label_config(
        self,
        project: Any = None,
        *,
        project_id: int | None = None,
        raise_on_error: bool = True,
    ) -> LabelStudioLabelConfigResult:
        """使用 XML 解析检查 Text、Labels、实体和关系标签。"""

        source = _sdk_object_to_dict(
            project if project is not None else self.get_project(project_id)
        )
        xml = source.get("label_config")
        if not isinstance(xml, str) or not xml.strip():
            raise LabelStudioProjectError("项目缺少 label_config")
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError as exc:
            raise LabelStudioProjectError("项目 label_config 不是合法 XML") from exc
        text_names = {
            node.attrib.get("name")
            for node in root.iter()
            if _xml_name(node.tag) == "Text"
        }
        label_nodes = [
            node
            for node in root.iter()
            if _xml_name(node.tag) == "Labels"
            and node.attrib.get("name") == self.entity_from_name
        ]
        errors: list[str] = []
        if self.text_to_name not in text_names:
            errors.append(f"缺少 Text name={self.text_to_name}")
        if not label_nodes:
            errors.append(f"缺少 Labels name={self.entity_from_name}")
        entity_values = {
            child.attrib.get("value")
            for node in label_nodes
            for child in node.iter()
            if _xml_name(child.tag) == "Label"
        }
        missing_entities = sorted(
            {"PER", "ORG", "POSITION", "MONEY"} - entity_values
        )
        relation_values = {
            node.attrib.get("value")
            for node in root.iter()
            if _xml_name(node.tag) == "Relation"
        }
        missing_relations = sorted(
            set(self.schema["relation_types"]) - relation_values
        )
        result = LabelStudioLabelConfigResult(
            valid=not errors and not missing_entities and not missing_relations,
            missing_entity_labels=missing_entities,
            missing_relation_labels=missing_relations,
            errors=errors,
        )
        if raise_on_error and not result.valid:
            raise LabelStudioProjectError(
                "项目 label_config 不兼容："
                f"errors={errors}, missing_entities={missing_entities}, "
                f"missing_relations={missing_relations}"
            )
        return result

    def find_task_by_annotation_id(
        self,
        annotation_id: str,
        *,
        project_id: int | None = None,
        max_items: int = 1000,
    ) -> dict[str, Any] | None:
        """小规模排查辅助；不作为批量导入幂等机制。"""

        for task in self.list_tasks(
            project_id=project_id, max_items=max_items
        ):
            if str(task.get("data", {}).get("annotation_id")) == annotation_id:
                return task
        return None

    def _annotation_to_entity_results(
        self, annotation: CanonicalAnnotation
    ) -> list[dict[str, Any]]:
        results = []
        for entity in annotation.entities:
            if (
                entity.start < 0
                or entity.end > len(annotation.text)
                or annotation.text[entity.start:entity.end] != entity.name
            ):
                raise LabelStudioConversionError(
                    f"实体字符位置不匹配：{entity.entity_id}"
                )
            results.append(
                {
                    "id": entity.entity_id,
                    "from_name": self.entity_from_name,
                    "to_name": self.text_to_name,
                    "type": "labels",
                    "value": {
                        "start": entity.start,
                        "end": entity.end,
                        "text": entity.name,
                        "labels": [entity.type.value],
                    },
                }
            )
        return results

    def _annotation_to_relation_results(
        self, annotation: CanonicalAnnotation
    ) -> list[dict[str, Any]]:
        entity_ids = {item.entity_id for item in annotation.entities}
        results = []
        for relation in annotation.relations:
            if relation.type == RelationType.NO_RELATION:
                continue
            if (
                relation.head_id not in entity_ids
                or relation.tail_id not in entity_ids
            ):
                raise LabelStudioConversionError(
                    f"关系引用不存在的实体：{relation.relation_id}"
                )
            results.append(
                {
                    "from_id": relation.head_id,
                    "to_id": relation.tail_id,
                    "type": "relation",
                    "direction": "right",
                    "labels": [relation.type.value],
                }
            )
        return results

    def _parse_entity_results(
        self,
        results: Sequence[Mapping[str, Any]],
        text: str,
    ) -> list[EntityMention]:
        entities = []
        for raw in results:
            item = _sdk_object_to_dict(raw)
            if item.get("type") != "labels":
                continue
            value = _sdk_object_to_dict(item.get("value", {}))
            labels = value.get("labels")
            if not isinstance(labels, list) or len(labels) != 1:
                raise LabelStudioConversionError(
                    "人工实体必须且只能包含一个 labels 值"
                )
            try:
                start, end = int(value["start"]), int(value["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LabelStudioConversionError("人工实体位置无效") from exc
            name = str(value.get("text", ""))
            if (
                start < 0
                or end <= start
                or end > len(text)
                or text[start:end] != name
            ):
                raise LabelStudioConversionError(
                    f"人工实体字符边界无效：region_id={item.get('id')}"
                )
            entity_id = str(item.get("id") or "")
            if not entity_id:
                raise LabelStudioConversionError("人工实体缺少稳定 region id")
            try:
                entities.append(
                    EntityMention(
                        entity_id=entity_id,
                        name=name,
                        type=EntityType(str(labels[0])),
                        start=start,
                        end=end,
                        confidence=None,
                    )
                )
            except ValueError as exc:
                raise LabelStudioConversionError(
                    f"非法人工实体类型：{labels[0]}"
                ) from exc
        return entities

    def _parse_relation_results(
        self,
        results: Sequence[Mapping[str, Any]],
        entities: Sequence[EntityMention],
    ) -> list[RelationMention]:
        entity_ids = {item.entity_id for item in entities}
        relations = []
        for raw in results:
            item = _sdk_object_to_dict(raw)
            if item.get("type") != "relation":
                continue
            labels = item.get("labels")
            if labels is None and isinstance(item.get("value"), Mapping):
                labels = item["value"].get("labels")
            if not isinstance(labels, list) or len(labels) != 1:
                raise LabelStudioConversionError(
                    "人工关系必须且只能包含一个 labels 值"
                )
            head_id, tail_id = str(item.get("from_id", "")), str(
                item.get("to_id", "")
            )
            if head_id not in entity_ids or tail_id not in entity_ids:
                raise LabelStudioConversionError(
                    f"人工关系引用不存在的 region：{head_id}->{tail_id}"
                )
            try:
                relation_type = RelationType(str(labels[0]))
            except ValueError as exc:
                raise LabelStudioConversionError(
                    f"非法人工关系类型：{labels[0]}"
                ) from exc
            if relation_type == RelationType.NO_RELATION:
                continue
            relations.append(
                RelationMention(
                    relation_id=str(
                        item.get("id")
                        or f"relation-{len(relations) + 1}"
                    ),
                    head_id=head_id,
                    tail_id=tail_id,
                    type=relation_type,
                    confidence=None,
                    evidence_start=None,
                    evidence_end=None,
                    extraction_source="HUMAN",
                )
            )
        return relations

    def _select_review_annotation(
        self, task: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        annotations = task.get("annotations")
        if not isinstance(annotations, list):
            return None
        valid = []
        for raw in annotations:
            item = _sdk_object_to_dict(raw)
            result = item.get("result")
            if item.get("was_cancelled") is True:
                continue
            if not isinstance(result, list) or not result:
                continue
            valid.append(item)
        if not valid:
            return None
        return max(valid, key=_annotation_sort_key)

    def _validate_configuration(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("LABEL_STUDIO_URL")
        if not self.api_key:
            missing.append("LABEL_STUDIO_API_KEY")
        if not self.project_id:
            missing.append("LABEL_STUDIO_PROJECT_ID")
        if not self.text_data_key:
            missing.append("text_data_key")
        if not self.text_to_name:
            missing.append("text_to_name")
        if not self.entity_from_name:
            missing.append("entity_from_name")
        if missing:
            raise LabelStudioConfigurationError(
                f"Label Studio 配置缺失：{missing}"
            )

    def _project_id(self, project_id: int | None) -> int:
        selected = int(project_id or self.project_id or 0)
        if selected <= 0:
            raise LabelStudioConfigurationError("project_id 必须大于 0")
        return selected

    @staticmethod
    def _calculate_prediction_score(
        annotation: CanonicalAnnotation,
    ) -> float:
        scores = [
            float(value)
            for value in [
                *(item.confidence for item in annotation.entities),
                *(item.confidence for item in annotation.relations),
            ]
            if value is not None
        ]
        return min(1.0, max(0.0, sum(scores) / len(scores))) if scores else 0.0

    def _converted_exception(
        self,
        exc: Exception,
        *,
        project_context: bool = False,
        import_context: bool = False,
    ) -> LabelStudioServiceError:
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            return LabelStudioAuthenticationError(
                "Label Studio API Key 无效或无访问权限"
            )
        if status == 404 and project_context:
            return LabelStudioProjectError(
                f"Label Studio 项目不存在：{self.project_id}"
            )
        name = type(exc).__name__.lower()
        if "timeout" in name or "connection" in name:
            return LabelStudioConnectionError(
                "Label Studio 连接失败或请求超时"
            )
        if import_context:
            return LabelStudioImportError("Label Studio 任务导入失败")
        if project_context:
            return LabelStudioProjectError("Label Studio 项目读取失败")
        return LabelStudioServiceError("Label Studio SDK 调用失败")

    def _raise_sdk_exception(
        self,
        exc: Exception,
        *,
        project_context: bool = False,
        import_context: bool = False,
    ) -> None:
        raise self._converted_exception(
            exc,
            project_context=project_context,
            import_context=import_context,
        ) from exc


def _sdk_object_to_dict(value: Any) -> dict[str, Any]:
    """兼容新版 Pydantic、旧式 dict() 和普通 mapping。"""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        if isinstance(data, Mapping):
            return dict(data)
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    raise LabelStudioResponseError(
        f"无法将 SDK 对象转换为字典：{type(value).__name__}"
    )


def _extract_task_page(response: Any) -> tuple[list[Any], int | None]:
    if isinstance(response, list):
        return response, None
    if isinstance(response, Iterable) and not isinstance(
        response, (str, bytes, Mapping)
    ):
        return list(response), None
    data = _sdk_object_to_dict(response)
    items = data.get("tasks", data.get("results", []))
    if not isinstance(items, list):
        raise LabelStudioResponseError("任务分页响应缺少 tasks/results")
    total = data.get("total", data.get("count"))
    return items, int(total) if total is not None else None


def _extract_task_ids(data: Mapping[str, Any]) -> list[int]:
    raw = data.get("task_ids", data.get("tasks", []))
    if not isinstance(raw, list):
        return []
    output = []
    for item in raw:
        if isinstance(item, Mapping):
            item = item.get("id")
        if item is not None:
            output.append(int(item))
    return output


def _annotation_sort_key(annotation: Mapping[str, Any]) -> tuple[float, int]:
    timestamp = _parse_datetime(
        annotation.get("updated_at") or annotation.get("created_at")
    )
    return (
        timestamp.timestamp() if timestamp else 0.0,
        int(annotation.get("id") or 0),
    )


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _chunked(
    values: Sequence[Any], size: int
) -> Iterable[Sequence[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sdk_version() -> str:
    try:
        return importlib.metadata.version("label-studio-sdk")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

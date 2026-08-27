"""模型抽取结果到 CanonicalAnnotation 的校验与转换。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from config import BASE_DIR, ProjectConfig, load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    EntityMention,
    EntityType,
    ModelExtractionResult,
    RelationMention,
    RelationType,
)
from src.services.llm_service import LLMService

ReviewModelT = TypeVar("ReviewModelT", bound=BaseModel)


class LLMPreannotationResult(BaseModel):
    """大模型对单个文本块的实体和关系抽取结果。"""

    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationMention] = Field(default_factory=list)


# 中文注释：仅作为 Structured Outputs 的紧凑传输协议；正式枚举和业务约束在响应后校验。
class LLMPreannotationEntityPayload(BaseModel):
    """兼容代理使用的精简实体传输对象。"""

    entity_id: str
    name: str
    type: str
    start: int
    end: int
    confidence: float | None
    normalized_name: str | None


class LLMPreannotationRelationPayload(BaseModel):
    """兼容代理使用的精简关系传输对象。"""

    relation_id: str
    head_id: str
    tail_id: str
    type: str
    confidence: float | None
    evidence_start: int | None
    evidence_end: int | None


class LLMPreannotationPayload(BaseModel):
    """避免把完整业务枚举和字段描述发送给兼容代理。"""

    entities: list[LLMPreannotationEntityPayload]
    relations: list[LLMPreannotationRelationPayload]


class AnnotationServiceError(ValueError):
    """模型结果无法安全转换为统一标注。"""


# 中文注释：负责把模型预测转换成规范标注，并集中执行实体、关系、offset 和 schema 校验。
class AnnotationService:
    """完成去重、schema 约束校验和复核标记。"""

    def __init__(
        self,
        *,
        project_config: ProjectConfig | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.config = project_config or load_project_config()
        self.schema = self.config.schema_config
        self.rules = self.schema["relation_types"]
        self.threshold = float(
            self.config.workflow["annotation"][
                "model_confidence_threshold"
            ]
        )
        self.llm_service = llm_service

    def preannotate_with_llm(
        self,
        text: str,
        *,
        annotation_id: str,
        case_id: str,
        doc_id: str,
        text_id: str,
    ) -> CanonicalAnnotation:
        """使用结构化大模型输出生成可人工审核的预标注。"""

        if self.llm_service is None:
            raise AnnotationServiceError("未配置 LLMService")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text 不能为空")
        payload = self.llm_service.generate_structured_response(
            "你是中文裁判文书实体和关系抽取器。"
            "仅输出符合给定 Pydantic schema 的结构化结果。",
            self._render_llm_preannotation_prompt(text),
            LLMPreannotationPayload,
            max_tokens=self.config.environment.llm_max_tokens,
        )
        result = self._validate_llm_payload(payload, text=text)
        entities: list[EntityMention] = []
        entity_ids: set[str] = set()
        entity_keys: set[tuple[int, int, str]] = set()
        for entity in result.entities:
            key = (entity.start, entity.end, entity.type.value)
            if (
                entity.entity_id in entity_ids
                or key in entity_keys
                or entity.end > len(text)
                or text[entity.start:entity.end] != entity.name
            ):
                continue
            entity_ids.add(entity.entity_id)
            entity_keys.add(key)
            entities.append(entity)

        entity_by_id = {item.entity_id: item for item in entities}
        relations: list[RelationMention] = []
        relation_keys: set[tuple[str, str, str]] = set()
        for relation in result.relations:
            head = entity_by_id.get(relation.head_id)
            tail = entity_by_id.get(relation.tail_id)
            rule = self.rules.get(relation.type.value)
            if not head or not tail or head.entity_id == tail.entity_id or not rule:
                continue
            if (
                head.type.value not in rule["head_types"]
                or tail.type.value not in rule["tail_types"]
            ):
                continue
            key = (head.entity_id, tail.entity_id, relation.type.value)
            if key in relation_keys:
                continue
            relation_keys.add(key)
            evidence_valid = (
                relation.evidence_start is not None
                and relation.evidence_end is not None
                and 0 <= relation.evidence_start < relation.evidence_end <= len(text)
            )
            relations.append(relation.model_copy(update={
                "evidence_start": (
                    relation.evidence_start if evidence_valid else None
                ),
                "evidence_end": relation.evidence_end if evidence_valid else None,
                "extraction_source": "LLM",
            }))

        return CanonicalAnnotation(
            annotation_id=annotation_id,
            case_id=case_id,
            doc_id=doc_id,
            text_id=text_id,
            text=text,
            entities=entities,
            relations=relations,
            annotation_source="LLM",
            schema_version=self.schema["schema_version"],
            status=AnnotationStatus.PENDING_REVIEW,
            metadata={
                "llm_model": self.llm_service.model_name,
                "prompt_version": "llm_preannotation_v1.0",
                "review_required": True,
                "review_reasons": ["大模型预标注待人工审核"],
            },
        )

    def _validate_llm_payload(
        self,
        payload: LLMPreannotationPayload,
        *,
        text: str,
    ) -> LLMPreannotationResult:
        """把紧凑传输对象转换为正式业务模型并丢弃非法枚举或字段。"""

        entities: list[EntityMention] = []
        for item in payload.entities:
            try:
                entities.append(EntityMention(
                    entity_id=item.entity_id,
                    name=item.name,
                    type=EntityType(item.type),
                    start=item.start,
                    end=item.end,
                    confidence=item.confidence,
                    normalized_name=item.normalized_name,
                ))
            except ValueError:
                continue

        relations: list[RelationMention] = []
        for item in payload.relations:
            evidence_valid = (
                item.evidence_start is not None
                and item.evidence_end is not None
                and 0 <= item.evidence_start < item.evidence_end <= len(text)
            )
            try:
                relations.append(RelationMention(
                    relation_id=item.relation_id,
                    head_id=item.head_id,
                    tail_id=item.tail_id,
                    type=RelationType(item.type),
                    confidence=item.confidence,
                    evidence_start=(
                        item.evidence_start if evidence_valid else None
                    ),
                    evidence_end=(
                        item.evidence_end if evidence_valid else None
                    ),
                    extraction_source="LLM",
                ))
            except ValueError:
                continue
        return LLMPreannotationResult(
            entities=entities,
            relations=relations,
        )

    def _render_llm_preannotation_prompt(self, text: str) -> str:
        configured = self.config.workflow.get("prompts", {}).get(
            "llm_preannotation", {}
        ).get("path", "prompts/llm_preannotation_prompt.jinja2")
        path = Path(str(configured))
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.is_file():
            raise AnnotationServiceError(f"LLM 预标注 Prompt 不存在：{path}")
        rendered = path.read_text(encoding="utf-8")
        values = {
            "text": text,
            "schema": json.dumps(self.schema, ensure_ascii=False),
        }
        for name, value in values.items():
            rendered = re.sub(
                r"{{\s*" + re.escape(name) + r"\s*}}",
                lambda _: value,
                rendered,
            )
        return rendered

    # 中文注释：规范化主入口；去重并校验模型实体/关系，附加风险标志后生成 CanonicalAnnotation。
    def to_canonical(
        self,
        extraction: ModelExtractionResult,
        *,
        annotation_id: str,
        case_id: str,
        doc_id: str,
        text_id: str,
    ) -> CanonicalAnnotation:
        """校验模型结果并生成 CanonicalAnnotation。"""

        reasons: list[str] = []
        flagged: list[dict[str, Any]] = []
        entity_key_to_id: dict[tuple[Any, ...], str] = {}
        old_to_new: dict[str, str] = {}
        entities: list[EntityMention] = []

        for prediction in extraction.entities:
            if extraction.text[
                prediction.start:prediction.end
            ] != prediction.name:
                reasons.append(f"实体偏移错误:{prediction.entity_id}")
                flagged.append(prediction.model_dump())
                continue
            key = (
                prediction.start,
                prediction.end,
                prediction.entity_type,
                prediction.name,
            )
            if key in entity_key_to_id:
                old_to_new[prediction.entity_id] = entity_key_to_id[key]
                continue
            new_id = prediction.entity_id
            entity_key_to_id[key] = new_id
            old_to_new[prediction.entity_id] = new_id
            entities.append(EntityMention(
                entity_id=new_id,
                name=prediction.name,
                type=EntityType(prediction.entity_type),
                start=prediction.start,
                end=prediction.end,
                confidence=prediction.confidence,
            ))
            if prediction.confidence < self.threshold:
                reasons.append(f"低置信度实体:{new_id}")

        entity_by_id = {item.entity_id: item for item in entities}
        relations: list[RelationMention] = []
        relation_keys: set[tuple[str, str, str]] = set()
        for prediction in extraction.relations:
            head_id = old_to_new.get(prediction.head_id)
            tail_id = old_to_new.get(prediction.tail_id)
            if not head_id or not tail_id or head_id == tail_id:
                reasons.append(f"关系实体引用无效:{prediction.relation_id}")
                flagged.append(prediction.model_dump())
                continue
            rule = self.rules.get(prediction.relation_type)
            if rule is None:
                reasons.append(f"非法关系标签:{prediction.relation_type}")
                flagged.append(prediction.model_dump())
                continue
            head, tail = entity_by_id[head_id], entity_by_id[tail_id]
            direction_valid = (
                head.type.value in rule["head_types"]
                and tail.type.value in rule["tail_types"]
            )
            if not direction_valid:
                reasons.append(f"关系方向不合法:{prediction.relation_id}")
                flagged.append(prediction.model_dump())
            if bool(rule.get("directional", True)):
                key = (head_id, tail_id, prediction.relation_type)
            else:
                first, second = sorted((head_id, tail_id))
                key = (first, second, prediction.relation_type)
            if key in relation_keys:
                continue
            relation_keys.add(key)
            relations.append(RelationMention(
                relation_id=prediction.relation_id,
                head_id=head_id,
                tail_id=tail_id,
                type=RelationType(prediction.relation_type),
                confidence=prediction.confidence,
                evidence_start=prediction.evidence_start,
                evidence_end=prediction.evidence_end,
                extraction_source="DEEP_MODEL",
            ))
            if prediction.confidence < self.threshold:
                reasons.append(
                    f"低置信度关系:{prediction.relation_id}"
                )

        review_required = bool(reasons)
        return CanonicalAnnotation(
            annotation_id=annotation_id,
            case_id=case_id,
            doc_id=doc_id,
            text_id=text_id,
            text=extraction.text,
            entities=entities,
            relations=relations,
            annotation_source="DEEP_MODEL",
            schema_version=self.schema["schema_version"],
            status=(
                AnnotationStatus.PENDING_REVIEW
                if review_required
                else AnnotationStatus.GENERATED
            ),
            metadata={
                "ner_model_version": extraction.ner_model_version,
                "relation_model_version": extraction.relation_model_version,
                "review_required": review_required,
                "review_reasons": sorted(set(reasons)),
                "flagged_predictions": flagged,
                "llm_review_available": self.llm_service is not None,
            },
        )

    # 中文注释：为低置信或冲突标注预留的 LLM 审核接口；当前生产 Flow 尚未调用。
    def request_llm_review(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ReviewModelT],
    ) -> ReviewModelT:
        """使用现有 LLMService 执行可选复核，不在此渲染 Prompt。"""

        if self.llm_service is None:
            raise AnnotationServiceError("未配置 LLMService")
        return self.llm_service.generate_structured(
            system_prompt, user_prompt, response_model
        )

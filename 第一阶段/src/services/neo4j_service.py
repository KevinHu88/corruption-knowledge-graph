"""项目统一 Neo4j 数据访问服务。

采用 Claim 中心的可追溯图结构，提供参数化幂等写入、受限只读查询、
schema 初始化和结果序列化。本模块不负责实体消歧、自然语言转 Cypher
或工作流级重试。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from config import ProjectConfig, load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    CaseDocument,
    ClaimStatus,
    EntityMention,
    GraphClaim,
    SourceDocument,
)

try:
    from neo4j import GraphDatabase, Query, RoutingControl
    from neo4j.exceptions import (
        AuthError,
        ClientError,
        ConstraintError,
        Neo4jError,
        ServiceUnavailable,
        SessionExpired,
        TransientError,
    )
except ImportError:  # 单元测试允许注入 fake driver，无需真实驱动。
    GraphDatabase = Query = RoutingControl = None  # type: ignore[assignment]
    AuthError = ClientError = ConstraintError = Neo4jError = Exception
    ServiceUnavailable = SessionExpired = TransientError = Exception

logger = logging.getLogger(__name__)

ENTITY_LABELS: dict[str, str] = {
    "PER": "Person",
    "ORG": "Organization",
    "POSITION": "Position",
    "MONEY": "Money",
}
SENSITIVE_PROPERTY_NAMES = {
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
}
READ_ONLY_CALL_ALLOWLIST = {
    "db.labels",
    "db.relationshiptypes",
    "db.propertykeys",
}
WRITE_KEYWORDS = {
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD",
    "FOREACH",
    "GRANT",
    "DENY",
    "REVOKE",
    "TERMINATE",
}


class Neo4jServiceError(RuntimeError):
    """Neo4j 服务基础异常。"""


class Neo4jConfigurationError(Neo4jServiceError):
    """连接或图谱配置缺失。"""


class Neo4jConnectionError(Neo4jServiceError):
    """数据库网络连接或会话失效。"""


class Neo4jAuthenticationError(Neo4jServiceError):
    """数据库认证失败或权限不足。"""


class Neo4jQueryError(Neo4jServiceError):
    """Cypher 或参数执行失败。"""


class Neo4jWriteError(Neo4jQueryError):
    """图谱写入失败。"""


class Neo4jSchemaError(Neo4jQueryError):
    """约束或索引操作失败。"""


class Neo4jConversionError(Neo4jServiceError):
    """属性或 Neo4j 原生结果转换失败。"""


class Neo4jValidationError(Neo4jServiceError):
    """写入前或写入后的图结构校验失败。"""


class Neo4jUnsafeQueryError(Neo4jQueryError):
    """只读入口收到未验证或明显危险的查询。"""


class Neo4jNotFoundError(Neo4jServiceError):
    """明确要求存在的图对象未找到。"""


# 中文注释：Neo4j 连接、批处理和图谱写入策略的强类型配置。
class Neo4jServiceConfig(BaseModel):
    """可注入的 Neo4j 连接和图模型配置。"""

    uri: str
    username: str
    password: str
    database: str = "neo4j"
    connection_timeout: float = Field(default=30, gt=0)
    max_connection_pool_size: int = Field(default=20, gt=0)
    max_transaction_retry_time: float = Field(default=30, ge=0)
    fetch_size: int = Field(default=1000, gt=0)
    batch_size: int = Field(default=500, gt=0)
    initialize_schema: bool = True
    store_rejected_claims: bool = True
    store_full_evidence_text: bool = True
    allow_unreviewed_annotations: bool = True
    merge_people_across_cases: bool = False
    max_text_warning_chars: int = Field(default=100000, gt=0)
    default_query_limit: int = Field(default=100, gt=0)
    max_query_limit: int = Field(default=1000, gt=0)


class Neo4jHealthResult(BaseModel):
    """连接健康检查结果。"""

    connected: bool
    database: str
    server_version: str | None = None
    server_address: str | None = None
    latency_seconds: float
    checked_at: datetime


class Neo4jWriteCounters(BaseModel):
    """Neo4j ResultSummary 写入计数。"""

    nodes_created: int = 0
    nodes_deleted: int = 0
    relationships_created: int = 0
    relationships_deleted: int = 0
    properties_set: int = 0
    labels_added: int = 0
    indexes_added: int = 0
    constraints_added: int = 0
    records_processed: int = 0
    records_skipped: int = 0
    records_failed: int = 0

    def add(self, other: "Neo4jWriteCounters") -> "Neo4jWriteCounters":
        """返回两个计数器之和。"""

        values = {
            name: getattr(self, name) + getattr(other, name)
            for name in type(self).model_fields
        }
        return Neo4jWriteCounters(**values)


class Neo4jSchemaItemResult(BaseModel):
    """单个约束或索引初始化结果。"""

    name: str
    kind: Literal["constraint", "index"]
    status: Literal["created", "already_exists", "failed"]
    error: str | None = None


class Neo4jSchemaInitResult(BaseModel):
    """项目 schema 初始化结果。"""

    items: list[Neo4jSchemaItemResult]
    counters: Neo4jWriteCounters


class Neo4jQueryResult(BaseModel):
    """普通 Python 对象形式的查询结果。"""

    records: list[dict[str, Any]]
    keys: list[str]
    summary: dict[str, Any]
    truncated: bool
    latency_seconds: float


# 中文注释：单次图谱写入结果，记录实体映射、Claim ID、警告和数据库计数器。
class Neo4jIngestionResult(BaseModel):
    """一次写入操作的结构化结果。"""

    success: bool
    annotation_ids: list[str] = Field(default_factory=list)
    entity_uid_map: dict[str, str] = Field(default_factory=dict)
    mention_uid_map: dict[str, str] = Field(default_factory=dict)
    successful_claim_ids: list[str] = Field(default_factory=list)
    skipped_claim_ids: list[str] = Field(default_factory=list)
    failed_claim_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    counters: Neo4jWriteCounters = Field(default_factory=Neo4jWriteCounters)


class Neo4jBatchItemResult(BaseModel):
    """一个 annotation 批次的结果。"""

    batch_index: int
    annotation_ids: list[str]
    success: bool
    result: Neo4jIngestionResult | None = None
    error: str | None = None


# 中文注释：批量图谱写入汇总，保留逐批成功/失败详情和累计写入计数。
class Neo4jBatchResult(BaseModel):
    """批量入库总体结果。"""

    batches: list[Neo4jBatchItemResult]
    total_batches: int
    successful_batches: int
    failed_batches: int
    counters: Neo4jWriteCounters


class Neo4jValidationIssue(BaseModel):
    """入库后图结构问题。"""

    code: str
    message: str
    object_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class Neo4jValidationResult(BaseModel):
    """入库后结构校验报告。"""

    valid: bool
    checked_claim_ids: list[str]
    issues: list[Neo4jValidationIssue]

    @computed_field
    @property
    def errors(self) -> list[Neo4jValidationIssue]:
        """返回错误项。"""

        return [item for item in self.issues if item.severity == "error"]

    @computed_field
    @property
    def warnings(self) -> list[Neo4jValidationIssue]:
        """返回警告项。"""

        return [item for item in self.issues if item.severity == "warning"]


# 中文注释：Claim 中心图谱访问层，管理 Driver、schema、幂等 upsert、查询和写入验证。
class Neo4jService:
    """长期复用官方 Driver 的 Claim 中心图谱访问服务。"""

    def __init__(
        self,
        config: Neo4jServiceConfig | None = None,
        *,
        project_config: ProjectConfig | None = None,
        driver: Any | None = None,
    ) -> None:
        """加载配置并初始化或接收可复用 Driver。"""

        self.project_config = project_config or load_project_config()
        self.config = config or self._config_from_project()
        self._validate_configuration()
        self.schema = self.project_config.schema_config
        self.relation_rules: dict[str, Mapping[str, Any]] = {
            str(name): rule
            for name, rule in self.schema.get("relation_types", {}).items()
        }
        self.negative_relation = str(
            self.schema.get("negative_relation", "无关系")
        )
        self._driver = driver if driver is not None else self._build_driver()
        self._closed = False

    def _config_from_project(self) -> Neo4jServiceConfig:
        environment = self.project_config.environment
        neo4j_config = dict(self.project_config.graph.get("neo4j", {}))
        graph_model = dict(
            self.project_config.graph.get("graph_model", {})
        )
        return Neo4jServiceConfig(
            uri=environment.neo4j_uri,
            username=environment.neo4j_username,
            password=environment.neo4j_password,
            database=(
                environment.neo4j_database
                or str(neo4j_config.get("database", "neo4j"))
            ),
            connection_timeout=environment.neo4j_connection_timeout,
            max_connection_pool_size=(
                environment.neo4j_max_connection_pool_size
            ),
            max_transaction_retry_time=(
                environment.neo4j_max_transaction_retry_time
            ),
            fetch_size=environment.neo4j_fetch_size,
            batch_size=int(neo4j_config.get("batch_size", 500)),
            initialize_schema=bool(
                neo4j_config.get("initialize_schema", True)
            ),
            store_rejected_claims=bool(
                graph_model.get("store_rejected_claims", True)
            ),
            store_full_evidence_text=bool(
                graph_model.get("store_full_evidence_text", True)
            ),
            allow_unreviewed_annotations=bool(
                graph_model.get("allow_unreviewed_annotations", True)
            ),
            merge_people_across_cases=bool(
                graph_model.get("merge_people_across_cases", False)
            ),
            max_text_warning_chars=int(
                graph_model.get("max_text_warning_chars", 100000)
            ),
        )

    def _validate_configuration(self) -> None:
        missing = [
            name
            for name in ("uri", "username", "password", "database")
            if not str(getattr(self.config, name, "")).strip()
        ]
        if missing:
            raise Neo4jConfigurationError(
                f"Neo4j 配置缺失：{', '.join(missing)}"
            )

    def _build_driver(self) -> Any:
        if GraphDatabase is None:
            raise Neo4jConfigurationError(
                "未安装官方 neo4j Python Driver；请安装 neo4j 包"
            )
        try:
            return GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                connection_timeout=self.config.connection_timeout,
                max_connection_pool_size=(
                    self.config.max_connection_pool_size
                ),
                max_transaction_retry_time=(
                    self.config.max_transaction_retry_time
                ),
            )
        except Exception as exc:
            self._raise_driver_error(exc, "初始化 Neo4j Driver")
            raise AssertionError("unreachable")

    def __enter__(self) -> "Neo4jService":
        """进入上下文并返回服务。"""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        """离开上下文时关闭 Driver。"""

        self.close()

    def close(self) -> None:
        """幂等关闭 Driver。"""

        if self._closed:
            return
        try:
            self._driver.close()
        except Exception as exc:
            self._raise_driver_error(exc, "关闭 Neo4j Driver")
        finally:
            self._closed = True

    # 中文注释：验证连接、认证和指定 database 的只读访问能力。
    def health_check(self) -> Neo4jHealthResult:
        """验证连接、认证和指定 database 的只读访问权限。"""

        started = time.perf_counter()
        try:
            self._driver.verify_connectivity()
            result = self._execute_read(
                "CALL dbms.components() "
                "YIELD versions RETURN versions[0] AS version",
                {},
            )
            record = result["records"][0] if result["records"] else {}
            server_info = getattr(self._driver, "get_server_info", None)
            info = server_info() if callable(server_info) else None
            address = getattr(info, "address", None)
            version = record.get("version") or getattr(
                info, "agent", None
            )
            return Neo4jHealthResult(
                connected=True,
                database=self.config.database,
                server_version=str(version) if version else None,
                server_address=str(address) if address else None,
                latency_seconds=time.perf_counter() - started,
                checked_at=datetime.now(),
            )
        except Neo4jServiceError:
            raise
        except Exception as exc:
            self._raise_driver_error(exc, "Neo4j 健康检查")
            raise AssertionError("unreachable")

    # 中文注释：使用 IF NOT EXISTS 幂等创建唯一约束和查询索引。
    def initialize_schema(self) -> Neo4jSchemaInitResult:
        """幂等创建本项目唯一约束和查询索引。"""

        definitions = _schema_definitions()
        items: list[Neo4jSchemaItemResult] = []
        counters = Neo4jWriteCounters()
        for name, kind, cypher in definitions:
            try:
                outcome = self._execute_write(cypher, {})
                item_counters = _extract_counters(outcome.get("summary"))
                counters = counters.add(item_counters)
                created = (
                    item_counters.constraints_added
                    if kind == "constraint"
                    else item_counters.indexes_added
                )
                items.append(
                    Neo4jSchemaItemResult(
                        name=name,
                        kind=kind,
                        status="created" if created else "already_exists",
                    )
                )
            except Neo4jServiceError as exc:
                items.append(
                    Neo4jSchemaItemResult(
                        name=name,
                        kind=kind,
                        status="failed",
                        error=str(exc),
                    )
                )
                raise Neo4jSchemaError(
                    f"初始化 schema 失败：{name}"
                ) from exc
        return Neo4jSchemaInitResult(items=items, counters=counters)

    def get_schema_snapshot(self) -> dict[str, Any]:
        """返回项目相关标签、关系、约束、索引和属性快照。"""

        labels = self._execute_read(
            "CALL db.labels() YIELD label RETURN label ORDER BY label", {}
        )
        relationships = self._execute_read(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType ORDER BY relationshipType",
            {},
        )
        constraints = self._execute_read(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, "
            "properties, entityType RETURN name, type, labelsOrTypes, "
            "properties, entityType",
            {},
        )
        indexes = self._execute_read(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, "
            "state, entityType RETURN name, type, labelsOrTypes, "
            "properties, state, entityType",
            {},
        )
        return {
            "labels": labels["records"],
            "relationship_types": relationships["records"],
            "constraints": constraints["records"],
            "indexes": indexes["records"],
        }

    def upsert_case(
        self, case: CaseDocument | Mapping[str, Any]
    ) -> Neo4jIngestionResult:
        """按 case_id 幂等写入案件节点。"""

        data = _as_mapping(case)
        case_id = str(data.get("case_id", "")).strip()
        if not case_id:
            raise Neo4jValidationError("case_id 不能为空")
        properties = sanitize_properties(
            {
                "title": data.get("title"),
                "case_type": _metadata(data).get("case_type"),
                "region": _metadata(data).get("region"),
                "published_at": data.get("published_at"),
                "updated_at": datetime.now(),
            }
        )
        query = (
            "MERGE (c:Case {case_id: $case_id}) "
            "ON CREATE SET c.created_at = $created_at "
            "SET c += $properties "
            "RETURN c.case_id AS case_id"
        )
        outcome = self._execute_write(
            query,
            {
                "case_id": case_id,
                "created_at": datetime.now().isoformat(),
                "properties": properties,
            },
        )
        return Neo4jIngestionResult(
            success=True,
            counters=_extract_counters(
                outcome.get("summary"), records_processed=1
            ),
        )

    def upsert_source_document(
        self,
        document: SourceDocument | Mapping[str, Any],
        *,
        case_id: str,
    ) -> Neo4jIngestionResult:
        """按 doc_version_id 写入来源文档并关联案件。"""

        data = _as_mapping(document)
        doc_version_id = str(data.get("doc_version_id", "")).strip()
        if not case_id or not doc_version_id:
            raise Neo4jValidationError(
                "case_id 和 doc_version_id 不能为空"
            )
        properties = sanitize_properties(
            {
                "doc_id": data.get("doc_id"),
                "source_id": data.get("source_id"),
                "title": data.get("title"),
                "raw_url": data.get("raw_url"),
                "canonical_url": data.get("canonical_url"),
                "published_at": data.get("published_at"),
                "content_hash": data.get("content_hash"),
                "raw_file_uri": data.get("raw_file_uri"),
                "updated_at": datetime.now(),
            }
        )
        query = (
            "MATCH (c:Case {case_id: $case_id}) "
            "MERGE (d:SourceDocument {doc_version_id: $doc_version_id}) "
            "ON CREATE SET d.created_at = $created_at "
            "SET d += $properties "
            "MERGE (d)-[:BELONGS_TO_CASE]->(c) "
            "RETURN d.doc_version_id AS doc_version_id"
        )
        outcome = self._execute_write(
            query,
            {
                "case_id": case_id,
                "doc_version_id": doc_version_id,
                "created_at": datetime.now().isoformat(),
                "properties": properties,
            },
        )
        if not outcome["records"]:
            raise Neo4jNotFoundError(f"案件不存在：{case_id}")
        return Neo4jIngestionResult(
            success=True,
            counters=_extract_counters(
                outcome.get("summary"), records_processed=1
            ),
        )

    def upsert_text_span(
        self,
        *,
        annotation_id: str,
        text_id: str,
        case_id: str,
        doc_id: str,
        doc_version_id: str,
        text: str,
        start: int = 0,
        end: int | None = None,
    ) -> Neo4jIngestionResult:
        """写入证据文本并建立 FROM_DOCUMENT 关系。"""

        actual_end = len(text) if end is None else end
        if (
            not text
            or start < 0
            or actual_end <= start
            or actual_end > len(text)
        ):
            raise Neo4jValidationError("TextSpan start/end 与文本不一致")
        text_uid = self._text_uid(annotation_id, text_id)
        warnings: list[str] = []
        if len(text) > self.config.max_text_warning_chars:
            warnings.append(
                f"TextSpan 文本较长：{len(text)} 字符，未截断"
            )
            logger.warning(
                "TextSpan 文本较长 text_uid=%s chars=%d",
                text_uid,
                len(text),
            )
        properties = sanitize_properties(
            {
                "text_id": text_id,
                "annotation_id": annotation_id,
                "case_id": case_id,
                "doc_id": doc_id,
                "text": text if self.config.store_full_evidence_text else None,
                "text_hash": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "start": start,
                "end": actual_end,
                "updated_at": datetime.now(),
            }
        )
        query = (
            "MATCH (d:SourceDocument {doc_version_id: $doc_version_id}) "
            "MERGE (s:TextSpan {text_uid: $text_uid}) "
            "ON CREATE SET s.created_at = $created_at "
            "SET s += $properties "
            "MERGE (s)-[:FROM_DOCUMENT]->(d) "
            "RETURN s.text_uid AS text_uid"
        )
        outcome = self._execute_write(
            query,
            {
                "doc_version_id": doc_version_id,
                "text_uid": text_uid,
                "created_at": datetime.now().isoformat(),
                "properties": properties,
            },
        )
        if not outcome["records"]:
            raise Neo4jNotFoundError(
                f"文档版本不存在：{doc_version_id}"
            )
        return Neo4jIngestionResult(
            success=True,
            warnings=warnings,
            counters=_extract_counters(
                outcome.get("summary"), records_processed=1
            ),
        )

    def upsert_entities(
        self,
        entities: Sequence[EntityMention],
        *,
        case_id: str,
        annotation_id: str,
        text_id: str,
        text: str,
        entity_uid_map: Mapping[str, str] | None = None,
    ) -> Neo4jIngestionResult:
        """按固定实体子标签批量写入 Entity 与 EntityMention。"""

        prepared = self._prepare_entity_rows(
            entities,
            case_id=case_id,
            annotation_id=annotation_id,
            text_id=text_id,
            text=text,
            entity_uid_map=entity_uid_map,
        )
        text_uid = self._text_uid(annotation_id, text_id)
        counters = Neo4jWriteCounters()
        returned: list[dict[str, Any]] = []
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in prepared:
            by_label.setdefault(row["entity_label"], []).append(row)
        for entity_label in sorted(by_label):
            rows = by_label[entity_label]
            query = self._entity_upsert_query(entity_label)
            outcome = self._execute_write(
                query, {"text_uid": text_uid, "rows": rows}
            )
            returned.extend(outcome["records"])
            counters = counters.add(
                _extract_counters(
                    outcome.get("summary"),
                    records_processed=len(rows),
                )
            )
        return Neo4jIngestionResult(
            success=True,
            entity_uid_map={
                row["entity_id"]: row["entity_uid"] for row in prepared
            },
            mention_uid_map={
                row["entity_id"]: row["mention_uid"] for row in prepared
            },
            counters=counters,
        )

    def upsert_claims(
        self,
        claims: Sequence[GraphClaim | Mapping[str, Any]],
        *,
        annotation_id: str | None = None,
        schema_version: str | None = None,
        entity_uid_map: Mapping[str, str] | None = None,
        entity_types: Mapping[str, str] | None = None,
        text_uid_map: Mapping[str, str] | None = None,
    ) -> Neo4jIngestionResult:
        """批量写入 Claim，并关联固定 HEAD、TAIL 和证据边。"""

        rows, skipped, warnings = self._prepare_claim_rows(
            claims,
            annotation_id=annotation_id,
            schema_version=schema_version,
            entity_uid_map=entity_uid_map,
            entity_types=entity_types,
            text_uid_map=text_uid_map,
        )
        if not rows:
            return Neo4jIngestionResult(
                success=True,
                skipped_claim_ids=skipped,
                warnings=warnings,
                counters=Neo4jWriteCounters(
                    records_skipped=len(skipped)
                ),
            )
        outcome = self._execute_write(
            self._claim_upsert_query(), {"rows": rows}
        )
        successful = sorted(
            str(record["claim_id"])
            for record in outcome["records"]
            if record.get("claim_id")
        )
        attempted = {row["claim_id"] for row in rows}
        failed = sorted(attempted - set(successful))
        return Neo4jIngestionResult(
            success=not failed,
            successful_claim_ids=successful,
            skipped_claim_ids=sorted(skipped),
            failed_claim_ids=failed,
            warnings=warnings,
            counters=_extract_counters(
                outcome.get("summary"),
                records_processed=len(successful),
                records_skipped=len(skipped),
                records_failed=len(failed),
            ),
        )

    # 中文注释：在单个 Managed Write Transaction 中原子写入一条完整规范标注。
    def ingest_annotation(
        self,
        annotation: CanonicalAnnotation,
        source_document: SourceDocument | Mapping[str, Any],
        case_document: CaseDocument | Mapping[str, Any],
        entity_uid_map: Mapping[str, str] | None = None,
    ) -> Neo4jIngestionResult:
        """在一个 Managed Write Transaction 中原子写入完整标注。"""

        payload = self._prepare_annotation_payload(
            annotation,
            source_document,
            case_document,
            entity_uid_map=entity_uid_map,
        )
        try:
            with self._session() as session:
                raw = session.execute_write(
                    self._upsert_annotation_tx, payload
                )
        except Exception as exc:
            if isinstance(exc, Neo4jServiceError):
                raise
            self._raise_driver_error(
                exc,
                f"写入 annotation {annotation.annotation_id}",
                write=True,
            )
            raise AssertionError("unreachable")
        return Neo4jIngestionResult.model_validate(raw)

    # 中文注释：按 batch_size 分批写入标注，每批独立事务并真实保留失败信息。
    def ingest_annotations_batch(
        self,
        annotations: Sequence[CanonicalAnnotation],
        *,
        source_documents: Mapping[str, SourceDocument | Mapping[str, Any]]
        | None = None,
        case_documents: Mapping[str, CaseDocument | Mapping[str, Any]]
        | None = None,
        entity_uid_maps: Mapping[str, Mapping[str, str]] | None = None,
        continue_on_error: bool = False,
    ) -> Neo4jBatchResult:
        """按配置分批写入，每批独立事务并保持失败信息真实可见。"""

        batches: list[Neo4jBatchItemResult] = []
        total_counters = Neo4jWriteCounters()
        for batch_index, chunk in enumerate(
            _chunked(annotations, self.config.batch_size)
        ):
            annotation_ids = [item.annotation_id for item in chunk]
            try:
                payloads = [
                    self._prepare_annotation_payload(
                        annotation,
                        self._resolve_source_document(
                            annotation, source_documents
                        ),
                        self._resolve_case_document(
                            annotation, case_documents
                        ),
                        entity_uid_map=(
                            (entity_uid_maps or {}).get(
                                annotation.annotation_id
                            )
                        ),
                    )
                    for annotation in chunk
                ]
                with self._session() as session:
                    results = session.execute_write(
                        self._upsert_annotation_batch_tx, payloads
                    )
                combined = Neo4jWriteCounters()
                successful_claims: list[str] = []
                entity_map: dict[str, str] = {}
                mention_map: dict[str, str] = {}
                warnings: list[str] = []
                for result in results:
                    parsed = Neo4jIngestionResult.model_validate(result)
                    combined = combined.add(parsed.counters)
                    successful_claims.extend(parsed.successful_claim_ids)
                    entity_map.update(parsed.entity_uid_map)
                    mention_map.update(parsed.mention_uid_map)
                    warnings.extend(parsed.warnings)
                ingestion = Neo4jIngestionResult(
                    success=True,
                    annotation_ids=annotation_ids,
                    entity_uid_map=entity_map,
                    mention_uid_map=mention_map,
                    successful_claim_ids=successful_claims,
                    warnings=warnings,
                    counters=combined,
                )
                total_counters = total_counters.add(combined)
                batches.append(
                    Neo4jBatchItemResult(
                        batch_index=batch_index,
                        annotation_ids=annotation_ids,
                        success=True,
                        result=ingestion,
                    )
                )
            except Exception as exc:
                total_counters = total_counters.add(
                    Neo4jWriteCounters(
                        records_failed=len(annotation_ids)
                    )
                )
                batches.append(
                    Neo4jBatchItemResult(
                        batch_index=batch_index,
                        annotation_ids=annotation_ids,
                        success=False,
                        error=str(exc),
                    )
                )
                logger.error(
                    "Neo4j 批次写入失败 database=%s batch=%d "
                    "annotations=%s",
                    self.config.database,
                    batch_index,
                    annotation_ids,
                )
                if not continue_on_error:
                    if isinstance(exc, Neo4jServiceError):
                        raise
                    raise Neo4jWriteError(
                        f"批次 {batch_index} 写入失败"
                    ) from exc
        return Neo4jBatchResult(
            batches=batches,
            total_batches=len(batches),
            successful_batches=sum(item.success for item in batches),
            failed_batches=sum(not item.success for item in batches),
            counters=total_counters,
        )

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        """按 claim_id 返回 Claim、头尾实体、证据、文档和案件。"""

        query = (
            "MATCH (c:Claim {claim_id: $claim_id}) "
            "OPTIONAL MATCH (c)-[:HEAD]->(h:Entity) "
            "OPTIONAL MATCH (c)-[:TAIL]->(t:Entity) "
            "OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:TextSpan) "
            "OPTIONAL MATCH (s)-[:FROM_DOCUMENT]->(d:SourceDocument) "
            "OPTIONAL MATCH (d)-[:BELONGS_TO_CASE]->(k:Case) "
            "RETURN c AS claim, h AS head, t AS tail, "
            "collect(DISTINCT s) AS evidence, d AS source_document, "
            "k AS case"
        )
        result = self._execute_read(query, {"claim_id": claim_id})
        return result["records"][0] if result["records"] else None

    def find_entities(
        self,
        *,
        entity_uid: str | None = None,
        name: str | None = None,
        normalized_name: str | None = None,
        entity_type: str | None = None,
        case_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按参数化条件查找实体，始终执行有界返回。"""

        actual_limit = self._bounded_limit(limit)
        if entity_type is not None and entity_type not in ENTITY_LABELS:
            raise Neo4jValidationError(f"未知实体类型：{entity_type}")
        query = (
            "MATCH (e:Entity) "
            "WHERE ($entity_uid IS NULL OR e.entity_uid = $entity_uid) "
            "AND ($name IS NULL OR e.name = $name) "
            "AND ($normalized_name IS NULL OR "
            "e.normalized_name = $normalized_name) "
            "AND ($entity_type IS NULL OR e.entity_type = $entity_type) "
            "AND ($case_id IS NULL OR e.case_id = $case_id) "
            "RETURN e ORDER BY e.entity_uid LIMIT $limit"
        )
        result = self._execute_read(
            query,
            {
                "entity_uid": entity_uid,
                "name": name,
                "normalized_name": normalized_name,
                "entity_type": entity_type,
                "case_id": case_id,
                "limit": actual_limit,
            },
        )
        return result["records"]

    def list_entity_claims(
        self,
        entity_uid: str,
        *,
        role: Literal["head", "tail", "both"] = "both",
        relation_type: str | None = None,
        status: ClaimStatus | str | None = None,
        case_id: str | None = None,
        include_rejected: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """查询实体作为 HEAD、TAIL 或两者参与的有向 Claim。"""

        if role not in {"head", "tail", "both"}:
            raise Neo4jValidationError(f"未知 Claim 角色：{role}")
        if relation_type and relation_type not in self.relation_rules:
            raise Neo4jValidationError(f"未知关系类型：{relation_type}")
        actual_limit = self._bounded_limit(limit)
        role_condition = {
            "head": "head.entity_uid = $entity_uid",
            "tail": "tail.entity_uid = $entity_uid",
            "both": (
                "(head.entity_uid = $entity_uid OR "
                "tail.entity_uid = $entity_uid)"
            ),
        }[role]
        query = (
            "MATCH (c:Claim)-[:HEAD]->(head:Entity) "
            "MATCH (c)-[:TAIL]->(tail:Entity) "
            f"WHERE {role_condition} "
            "AND ($relation_type IS NULL OR "
            "c.relation_type = $relation_type) "
            "AND ($status IS NULL OR c.status = $status) "
            "AND ($case_id IS NULL OR c.case_id = $case_id) "
            "AND ($include_rejected OR c.status <> 'REJECTED') "
            "RETURN c AS claim, head, tail, "
            "CASE WHEN head.entity_uid = $entity_uid "
            "THEN 'HEAD' ELSE 'TAIL' END AS entity_role "
            "ORDER BY c.claim_id SKIP $offset LIMIT $limit"
        )
        result = self._execute_read(
            query,
            {
                "entity_uid": entity_uid,
                "relation_type": relation_type,
                "status": _enum_value(status) if status else None,
                "case_id": case_id,
                "include_rejected": include_rejected,
                "offset": max(0, offset),
                "limit": actual_limit,
            },
        )
        return result["records"]

    def get_case_graph(
        self, case_id: str, *, limit: int | None = None
    ) -> dict[str, Any]:
        """返回案件内有界实体、Claim、证据和来源文档。"""

        actual_limit = self._bounded_limit(limit)
        entities = self._execute_read(
            "MATCH (e:Entity {case_id: $case_id}) "
            "RETURN e ORDER BY e.entity_uid LIMIT $limit",
            {"case_id": case_id, "limit": actual_limit},
        )["records"]
        claims = self._execute_read(
            "MATCH (c:Claim {case_id: $case_id})-[:HEAD]->(h:Entity) "
            "MATCH (c)-[:TAIL]->(t:Entity) "
            "OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:TextSpan) "
            "WHERE c.status IN ['HUMAN_VERIFIED', 'MODEL_PREDICTED'] "
            "RETURN c AS claim, h AS head, t AS tail, "
            "collect(DISTINCT s) AS evidence "
            "ORDER BY c.claim_id LIMIT $limit",
            {"case_id": case_id, "limit": actual_limit},
        )["records"]
        documents = self._execute_read(
            "MATCH (d:SourceDocument)-[:BELONGS_TO_CASE]->"
            "(:Case {case_id: $case_id}) "
            "RETURN d ORDER BY d.doc_version_id LIMIT $limit",
            {"case_id": case_id, "limit": actual_limit},
        )["records"]
        return {
            "case_id": case_id,
            "entities": entities,
            "claims": claims,
            "source_documents": documents,
        }

    # 中文注释：受保护的通用只读查询入口，会拒绝包含写操作的 Cypher。
    def execute_read_query(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        max_records: int = 100,
        timeout: float | None = None,
        validated: bool = True,
    ) -> Neo4jQueryResult:
        """执行经上游校验且通过本地防御检查的单条只读 Cypher。"""

        self._assert_read_only(cypher, validated=validated)
        if max_records <= 0 or max_records > self.config.max_query_limit:
            raise Neo4jValidationError(
                f"max_records 必须位于 1..{self.config.max_query_limit}"
            )
        started = time.perf_counter()
        result = self._execute_read(
            cypher,
            dict(parameters or {}),
            max_records=max_records,
            timeout=timeout,
        )
        return Neo4jQueryResult(
            records=result["records"],
            keys=result["keys"],
            summary=_serialize_summary(result["summary"]),
            truncated=result["truncated"],
            latency_seconds=time.perf_counter() - started,
        )

    # 中文注释：写入后检查 Claim 端点、证据及孤立节点，返回结构化问题列表。
    def validate_ingestion(
        self,
        *,
        annotation_id: str | None = None,
        case_id: str | None = None,
        claim_ids: Sequence[str] | None = None,
    ) -> Neo4jValidationResult:
        """检查 Claim 结构、证据链、孤立节点和非法标签。"""

        query = (
            "MATCH (c:Claim) "
            "WHERE ($annotation_id IS NULL OR "
            "c.annotation_id = $annotation_id) "
            "AND ($case_id IS NULL OR c.case_id = $case_id) "
            "AND (size($claim_ids) = 0 OR c.claim_id IN $claim_ids) "
            "OPTIONAL MATCH (c)-[hr:HEAD]->(:Entity) "
            "OPTIONAL MATCH (c)-[tr:TAIL]->(:Entity) "
            "OPTIONAL MATCH (c)-[sr:SUPPORTED_BY]->(:TextSpan) "
            "RETURN c.claim_id AS claim_id, c.relation_type AS relation_type, "
            "count(DISTINCT hr) AS head_count, "
            "count(DISTINCT tr) AS tail_count, "
            "count(DISTINCT sr) AS evidence_count"
        )
        result = self._execute_read(
            query,
            {
                "annotation_id": annotation_id,
                "case_id": case_id,
                "claim_ids": list(claim_ids or []),
            },
        )
        issues: list[Neo4jValidationIssue] = []
        checked: list[str] = []
        for record in result["records"]:
            claim_id = str(record.get("claim_id", ""))
            checked.append(claim_id)
            if int(record.get("head_count", 0)) != 1:
                issues.append(
                    Neo4jValidationIssue(
                        code="invalid_head_count",
                        message="Claim 必须且只能有一个 HEAD",
                        object_id=claim_id,
                    )
                )
            if int(record.get("tail_count", 0)) != 1:
                issues.append(
                    Neo4jValidationIssue(
                        code="invalid_tail_count",
                        message="Claim 必须且只能有一个 TAIL",
                        object_id=claim_id,
                    )
                )
            if int(record.get("evidence_count", 0)) < 1:
                issues.append(
                    Neo4jValidationIssue(
                        code="missing_evidence",
                        message="Claim 缺少 SUPPORTED_BY 证据",
                        object_id=claim_id,
                    )
                )
            relation_type = str(record.get("relation_type", ""))
            if relation_type == self.negative_relation:
                issues.append(
                    Neo4jValidationIssue(
                        code="negative_relation_claim",
                        message="图谱中不得存在无关系 Claim",
                        object_id=claim_id,
                    )
                )
            elif relation_type not in self.relation_rules:
                issues.append(
                    Neo4jValidationIssue(
                        code="invalid_relation_type",
                        message=f"非法关系类型：{relation_type}",
                        object_id=claim_id,
                    )
                )
        orphan_checks = [
            (
                "orphan_mention",
                "MATCH (m:EntityMention) WHERE NOT (m)-[:MENTION_OF]->"
                "(:Entity) RETURN m.mention_uid AS object_id LIMIT 100",
            ),
            (
                "orphan_text_span",
                "MATCH (s:TextSpan) WHERE NOT (s)-[:FROM_DOCUMENT]->"
                "(:SourceDocument) RETURN s.text_uid AS object_id LIMIT 100",
            ),
            (
                "orphan_document",
                "MATCH (d:SourceDocument) WHERE NOT "
                "(d)-[:BELONGS_TO_CASE]->(:Case) "
                "RETURN d.doc_version_id AS object_id LIMIT 100",
            ),
        ]
        for code, orphan_query in orphan_checks:
            orphan_result = self._execute_read(orphan_query, {})
            for record in orphan_result["records"]:
                issues.append(
                    Neo4jValidationIssue(
                        code=code,
                        message="发现孤立图对象",
                        object_id=str(record.get("object_id", "")),
                    )
                )
        return Neo4jValidationResult(
            valid=not any(item.severity == "error" for item in issues),
            checked_claim_ids=sorted(checked),
            issues=issues,
        )

    def _prepare_entity_rows(
        self,
        entities: Sequence[EntityMention],
        *,
        case_id: str,
        annotation_id: str,
        text_id: str,
        text: str,
        entity_uid_map: Mapping[str, str] | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_mentions: set[str] = set()
        for entity in sorted(
            entities,
            key=lambda item: (item.start, item.end, item.entity_id),
        ):
            entity_type = _enum_value(entity.type)
            label = ENTITY_LABELS.get(entity_type)
            if label is None:
                raise Neo4jValidationError(
                    f"未知实体类型：{entity_type}"
                )
            if (
                entity.start < 0
                or entity.end <= entity.start
                or entity.end > len(text)
                or text[entity.start : entity.end] != entity.name
            ):
                raise Neo4jValidationError(
                    f"实体偏移与原文不一致：{entity.entity_id}"
                )
            mention_uid = f"{annotation_id}:{entity.entity_id}"
            if mention_uid in seen_mentions:
                raise Neo4jValidationError(
                    f"重复 mention_uid：{mention_uid}"
                )
            seen_mentions.add(mention_uid)
            entity_uid = (
                str((entity_uid_map or {}).get(entity.entity_id, "")).strip()
                or self._build_entity_uid(
                    entity,
                    case_id=case_id,
                    text_id=text_id,
                )
            )
            rows.append(
                {
                    "entity_id": entity.entity_id,
                    "entity_uid": entity_uid,
                    "mention_uid": mention_uid,
                    "entity_label": label,
                    "entity_properties": sanitize_properties(
                        {
                            "entity_type": entity_type,
                            "name": entity.name,
                            "normalized_name": entity.normalized_name,
                            "case_id": case_id,
                            "updated_at": datetime.now(),
                        }
                    ),
                    "mention_properties": sanitize_properties(
                        {
                            "entity_id": entity.entity_id,
                            "annotation_id": annotation_id,
                            "text_id": text_id,
                            "name": entity.name,
                            "entity_type": entity_type,
                            "start": entity.start,
                            "end": entity.end,
                            "confidence": entity.confidence,
                            "created_at": datetime.now(),
                        }
                    ),
                }
            )
        return rows

    def _build_entity_uid(
        self,
        entity: EntityMention,
        *,
        case_id: str,
        text_id: str,
    ) -> str:
        entity_type = _enum_value(entity.type)
        if entity.normalized_name:
            scope = "" if self.config.merge_people_across_cases else case_id
            raw = f"{scope}:{entity.normalized_name}:{entity_type}"
        else:
            raw = f"{case_id}:{text_id}:{entity.entity_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _text_uid(annotation_id: str, text_id: str) -> str:
        if not annotation_id or not text_id:
            raise Neo4jValidationError(
                "annotation_id 和 text_id 不能为空"
            )
        return f"{annotation_id}:{text_id}"

    @staticmethod
    def _entity_upsert_query(entity_label: str) -> str:
        if entity_label not in set(ENTITY_LABELS.values()):
            raise Neo4jValidationError(
                f"不允许的实体节点标签：{entity_label}"
            )
        return (
            "MATCH (s:TextSpan {text_uid: $text_uid}) "
            "UNWIND $rows AS row "
            f"MERGE (e:Entity:{entity_label} "
            "{entity_uid: row.entity_uid}) "
            "ON CREATE SET e.created_at = datetime() "
            "SET e += row.entity_properties "
            "MERGE (m:EntityMention {mention_uid: row.mention_uid}) "
            "SET m += row.mention_properties "
            "MERGE (s)-[:CONTAINS_MENTION]->(m) "
            "MERGE (m)-[:MENTION_OF]->(e) "
            "RETURN row.entity_id AS entity_id, "
            "e.entity_uid AS entity_uid, m.mention_uid AS mention_uid"
        )

    def _prepare_claim_rows(
        self,
        claims: Sequence[GraphClaim | Mapping[str, Any]],
        *,
        annotation_id: str | None,
        schema_version: str | None,
        entity_uid_map: Mapping[str, str] | None,
        entity_types: Mapping[str, str] | None,
        text_uid_map: Mapping[str, str] | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        warnings: list[str] = []
        seen: set[str] = set()
        for claim in claims:
            data = _as_mapping(claim)
            claim_id = str(data.get("claim_id", "")).strip()
            if not claim_id:
                raise Neo4jValidationError("claim_id 不能为空")
            if claim_id in seen:
                raise Neo4jValidationError(
                    f"重复 claim_id：{claim_id}"
                )
            seen.add(claim_id)
            relation_type = _enum_value(
                data.get("relation_type") or data.get("relation")
            )
            if relation_type == self.negative_relation:
                raise Neo4jValidationError("无关系不得写入 Claim")
            rule = self.relation_rules.get(relation_type)
            if rule is None:
                raise Neo4jValidationError(
                    f"关系类型不在 schema：{relation_type}"
                )
            status = _enum_value(
                data.get("status") or ClaimStatus.MODEL_PREDICTED
            )
            if (
                status == ClaimStatus.REJECTED.value
                and not self.config.store_rejected_claims
            ):
                skipped.append(claim_id)
                continue
            head_id = str(
                data.get("head_entity_id") or data.get("head_id") or ""
            )
            tail_id = str(
                data.get("tail_entity_id") or data.get("tail_id") or ""
            )
            if not head_id or not tail_id or head_id == tail_id:
                raise Neo4jValidationError(
                    f"Claim 头尾实体非法：{claim_id}"
                )
            head_type = (entity_types or {}).get(head_id)
            tail_type = (entity_types or {}).get(tail_id)
            if head_type is not None and head_type not in rule["head_types"]:
                raise Neo4jValidationError(
                    f"Claim 头实体类型不合法：{claim_id}"
                )
            if tail_type is not None and tail_type not in rule["tail_types"]:
                raise Neo4jValidationError(
                    f"Claim 尾实体类型不合法：{claim_id}"
                )
            evidence_text = str(data.get("evidence_text") or "")
            evidence_start = int(data.get("evidence_start", -1))
            evidence_end = int(data.get("evidence_end", -1))
            if (
                evidence_start < 0
                or evidence_end <= evidence_start
                or not evidence_text
            ):
                raise Neo4jValidationError(
                    f"Claim 证据位置非法：{claim_id}"
                )
            text_id = str(data.get("text_id", ""))
            resolved_annotation_id = str(
                data.get("annotation_id") or annotation_id or ""
            )
            text_uid = (
                (text_uid_map or {}).get(text_id)
                or self._text_uid(resolved_annotation_id, text_id)
            )
            rows.append(
                {
                    "claim_id": claim_id,
                    "head_entity_uid": (
                        (entity_uid_map or {}).get(head_id) or head_id
                    ),
                    "tail_entity_uid": (
                        (entity_uid_map or {}).get(tail_id) or tail_id
                    ),
                    "head_types": list(rule["head_types"]),
                    "tail_types": list(rule["tail_types"]),
                    "text_uid": text_uid,
                    "status": status,
                    "properties": sanitize_properties(
                        {
                            "relation_type": relation_type,
                            "confidence": data.get("confidence"),
                            "extraction_source": data.get(
                                "extraction_source"
                            ),
                            "case_id": data.get("case_id"),
                            "doc_id": data.get("doc_id"),
                            "text_id": text_id,
                            "annotation_id": resolved_annotation_id,
                            "chunk_id": data.get("chunk_id"),
                            "schema_version": (
                                data.get("schema_version")
                                or schema_version
                                or self.schema.get("schema_version")
                            ),
                            "model_version": data.get("model_version"),
                            "dataset_version": data.get("dataset_version"),
                            "dataset_splits": data.get("dataset_splits"),
                            "source_files": data.get("source_files"),
                            "source_rows": data.get("source_rows"),
                            "original_relation_types": data.get(
                                "original_relation_types"
                            ),
                            "evidence_text": evidence_text,
                            "evidence_start": evidence_start,
                            "evidence_end": evidence_end,
                            "source_url": data.get("source_url"),
                            "created_at": (
                                data.get("created_at") or datetime.now()
                            ),
                            "updated_at": datetime.now(),
                        }
                    ),
                }
            )
        return rows, skipped, warnings

    @staticmethod
    def _claim_upsert_query() -> str:
        return (
            "UNWIND $rows AS row "
            "MATCH (h:Entity {entity_uid: row.head_entity_uid}) "
            "WHERE h.entity_type IN row.head_types "
            "MATCH (t:Entity {entity_uid: row.tail_entity_uid}) "
            "WHERE t.entity_type IN row.tail_types "
            "MATCH (s:TextSpan {text_uid: row.text_uid}) "
            "MERGE (c:Claim {claim_id: row.claim_id}) "
            "ON CREATE SET c += row.properties, c.status = row.status "
            "ON MATCH SET c += row.properties, "
            "c.status = CASE "
            "WHEN c.status = 'HUMAN_VERIFIED' THEN 'HUMAN_VERIFIED' "
            "WHEN c.status = 'REJECTED' "
            "AND row.status = 'MODEL_PREDICTED' THEN 'REJECTED' "
            "WHEN row.status = 'HUMAN_VERIFIED' "
            "THEN 'HUMAN_VERIFIED' "
            "WHEN row.status = 'REJECTED' THEN 'REJECTED' "
            "ELSE row.status END "
            "MERGE (c)-[:HEAD]->(h) "
            "MERGE (c)-[:TAIL]->(t) "
            "MERGE (c)-[:SUPPORTED_BY]->(s) "
            "MERGE (c)-[:FROM_ANNOTATION]->(s) "
            "RETURN c.claim_id AS claim_id"
        )

    # 中文注释：在进入事务前完成跨对象一致性校验，并生成纯数据实体行和 Claim 行。
    def _prepare_annotation_payload(
        self,
        annotation: CanonicalAnnotation,
        source_document: SourceDocument | Mapping[str, Any],
        case_document: CaseDocument | Mapping[str, Any],
        *,
        entity_uid_map: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        if (
            not self.config.allow_unreviewed_annotations
            and annotation.status != AnnotationStatus.APPROVED
        ):
            raise Neo4jValidationError(
                f"标注尚未审核通过：{annotation.annotation_id}"
            )
        source = _as_mapping(source_document)
        case = _as_mapping(case_document)
        if str(case.get("case_id")) != annotation.case_id:
            raise Neo4jValidationError("CaseDocument.case_id 与标注不一致")
        if str(source.get("doc_id")) != annotation.doc_id:
            raise Neo4jValidationError("SourceDocument.doc_id 与标注不一致")
        entity_rows = self._prepare_entity_rows(
            annotation.entities,
            case_id=annotation.case_id,
            annotation_id=annotation.annotation_id,
            text_id=annotation.text_id,
            text=annotation.text,
            entity_uid_map=entity_uid_map,
        )
        resolved_map = {
            row["entity_id"]: row["entity_uid"] for row in entity_rows
        }
        entity_types = {
            entity.entity_id: _enum_value(entity.type)
            for entity in annotation.entities
        }
        claim_status = (
            ClaimStatus.HUMAN_VERIFIED
            if annotation.status == AnnotationStatus.APPROVED
            else ClaimStatus.MODEL_PREDICTED
        )
        claims = []
        chunk_id = str(
            annotation.metadata.get("chunk_id") or annotation.annotation_id
        )
        relation_provenance = annotation.metadata.get(
            "relation_provenance", {}
        )
        for relation in annotation.relations:
            evidence_start = (
                relation.evidence_start
                if relation.evidence_start is not None
                else min(
                    next(
                        item.start
                        for item in annotation.entities
                        if item.entity_id == relation.head_id
                    ),
                    next(
                        item.start
                        for item in annotation.entities
                        if item.entity_id == relation.tail_id
                    ),
                )
            )
            evidence_end = (
                relation.evidence_end
                if relation.evidence_end is not None
                else max(
                    next(
                        item.end
                        for item in annotation.entities
                        if item.entity_id == relation.head_id
                    ),
                    next(
                        item.end
                        for item in annotation.entities
                        if item.entity_id == relation.tail_id
                    ),
                )
            )
            provenance = (
                relation_provenance.get(relation.relation_id, {})
                if isinstance(relation_provenance, Mapping)
                else {}
            )
            claims.append(
                {
                    "claim_id": (
                        f"{annotation.annotation_id}:{relation.relation_id}"
                    ),
                    "case_id": annotation.case_id,
                    "doc_id": annotation.doc_id,
                    "text_id": annotation.text_id,
                    "annotation_id": annotation.annotation_id,
                    "chunk_id": chunk_id,
                    "head_entity_id": relation.head_id,
                    "tail_entity_id": relation.tail_id,
                    "relation": relation.type,
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                    "evidence_text": annotation.text[
                        evidence_start:evidence_end
                    ],
                    "confidence": relation.confidence,
                    "status": claim_status,
                    "schema_version": annotation.schema_version,
                    "extraction_source": relation.extraction_source,
                    "model_version": annotation.metadata.get(
                        "relation_model_version"
                    ),
                    "dataset_version": annotation.metadata.get(
                        "dataset_version"
                    ),
                    "dataset_splits": provenance.get("dataset_splits"),
                    "source_files": provenance.get("source_files"),
                    "source_rows": provenance.get("source_rows"),
                    "original_relation_types": provenance.get(
                        "original_relation_types"
                    ),
                }
            )
        claim_rows, skipped, warnings = self._prepare_claim_rows(
            claims,
            annotation_id=annotation.annotation_id,
            schema_version=annotation.schema_version,
            entity_uid_map=resolved_map,
            entity_types=entity_types,
            text_uid_map={
                annotation.text_id: self._text_uid(
                    annotation.annotation_id, annotation.text_id
                )
            },
        )
        return {
            "annotation_id": annotation.annotation_id,
            "case": sanitize_properties(
                {
                    "case_id": annotation.case_id,
                    "title": case.get("title"),
                    "published_at": case.get("published_at"),
                    "case_type": _metadata(case).get("case_type"),
                    "region": _metadata(case).get("region"),
                }
            ),
            "document": sanitize_properties(
                {
                    "doc_id": source.get("doc_id"),
                    "doc_version_id": source.get("doc_version_id"),
                    "source_id": source.get("source_id"),
                    "title": source.get("title"),
                    "raw_url": source.get("raw_url"),
                    "canonical_url": source.get("canonical_url"),
                    "published_at": source.get("published_at"),
                    "content_hash": source.get("content_hash"),
                    "raw_file_uri": source.get("raw_file_uri"),
                    "schema_version": annotation.schema_version,
                    "dataset_version": annotation.metadata.get(
                        "dataset_version"
                    ),
                }
            ),
            "text_span": sanitize_properties(
                {
                    "text_uid": self._text_uid(
                        annotation.annotation_id, annotation.text_id
                    ),
                    "text_id": annotation.text_id,
                    "annotation_id": annotation.annotation_id,
                    "chunk_id": chunk_id,
                    "case_id": annotation.case_id,
                    "doc_id": annotation.doc_id,
                    "dataset_version": annotation.metadata.get(
                        "dataset_version"
                    ),
                    "dataset_splits": annotation.metadata.get(
                        "dataset_splits"
                    ),
                    "source_files": annotation.metadata.get("source_files"),
                    "text": (
                        annotation.text
                        if self.config.store_full_evidence_text
                        else None
                    ),
                    "text_hash": hashlib.sha256(
                        annotation.text.encode("utf-8")
                    ).hexdigest(),
                    "start": 0,
                    "end": len(annotation.text),
                }
            ),
            "entity_rows": entity_rows,
            "claim_rows": claim_rows,
            "skipped_claim_ids": skipped,
            "warnings": warnings,
        }

    def _upsert_annotation_tx(
        self, tx: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        case = payload["case"]
        document = payload["document"]
        text_span = payload["text_span"]
        summaries: list[Any] = []
        case_result = tx.run(
            "MERGE (c:Case {case_id: $case_id}) "
            "ON CREATE SET c.created_at = datetime() "
            "SET c += $properties RETURN c.case_id AS case_id",
            case_id=case["case_id"],
            properties=case,
        )
        _, summary = _consume_result(case_result)
        summaries.append(summary)
        document_result = tx.run(
            "MATCH (c:Case {case_id: $case_id}) "
            "MERGE (d:SourceDocument {doc_version_id: $doc_version_id}) "
            "ON CREATE SET d.created_at = datetime() "
            "SET d += $properties "
            "MERGE (d)-[:BELONGS_TO_CASE]->(c) "
            "RETURN d.doc_version_id AS doc_version_id",
            case_id=case["case_id"],
            doc_version_id=document["doc_version_id"],
            properties=document,
        )
        document_records, summary = _consume_result(document_result)
        summaries.append(summary)
        if not document_records:
            raise Neo4jNotFoundError(
                f"案件不存在：{case['case_id']}"
            )
        span_result = tx.run(
            "MATCH (d:SourceDocument {doc_version_id: $doc_version_id}) "
            "MERGE (s:TextSpan {text_uid: $text_uid}) "
            "ON CREATE SET s.created_at = datetime() "
            "SET s += $properties "
            "MERGE (s)-[:FROM_DOCUMENT]->(d) "
            "RETURN s.text_uid AS text_uid",
            doc_version_id=document["doc_version_id"],
            text_uid=text_span["text_uid"],
            properties=text_span,
        )
        span_records, summary = _consume_result(span_result)
        summaries.append(summary)
        if not span_records:
            raise Neo4jNotFoundError(
                f"来源文档不存在：{document['doc_version_id']}"
            )
        entity_records: list[dict[str, Any]] = []
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in payload["entity_rows"]:
            by_label.setdefault(row["entity_label"], []).append(row)
        for label in sorted(by_label):
            result = tx.run(
                self._entity_upsert_query(label),
                text_uid=text_span["text_uid"],
                rows=by_label[label],
            )
            records, summary = _consume_result(result)
            entity_records.extend(records)
            summaries.append(summary)
        claim_records: list[dict[str, Any]] = []
        if payload["claim_rows"]:
            result = tx.run(
                self._claim_upsert_query(),
                rows=payload["claim_rows"],
            )
            claim_records, summary = _consume_result(result)
            summaries.append(summary)
        counters = Neo4jWriteCounters()
        for summary in summaries:
            counters = counters.add(_extract_counters(summary))
        successful_claims = sorted(
            str(record["claim_id"])
            for record in claim_records
            if record.get("claim_id")
        )
        expected_claims = {
            row["claim_id"] for row in payload["claim_rows"]
        }
        failed = sorted(expected_claims - set(successful_claims))
        counters.records_processed = (
            3 + len(payload["entity_rows"]) + len(successful_claims)
        )
        counters.records_skipped = len(payload["skipped_claim_ids"])
        counters.records_failed = len(failed)
        return Neo4jIngestionResult(
            success=not failed,
            annotation_ids=[payload["annotation_id"]],
            entity_uid_map={
                row["entity_id"]: row["entity_uid"]
                for row in payload["entity_rows"]
            },
            mention_uid_map={
                row["entity_id"]: row["mention_uid"]
                for row in payload["entity_rows"]
            },
            successful_claim_ids=successful_claims,
            skipped_claim_ids=payload["skipped_claim_ids"],
            failed_claim_ids=failed,
            warnings=payload["warnings"],
            counters=counters,
        ).model_dump()

    def _upsert_annotation_batch_tx(
        self, tx: Any, payloads: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """在同一批次事务内依次写入已准备的纯数据 payload。"""

        return [self._upsert_annotation_tx(tx, payload) for payload in payloads]

    def _resolve_source_document(
        self,
        annotation: CanonicalAnnotation,
        documents: Mapping[str, SourceDocument | Mapping[str, Any]]
        | None,
    ) -> SourceDocument | Mapping[str, Any]:
        if documents:
            document = (
                documents.get(annotation.doc_id)
                or documents.get(
                    str(annotation.metadata.get("doc_version_id", ""))
                )
            )
            if document is not None:
                return document
        return {
            "doc_id": annotation.doc_id,
            "doc_version_id": str(
                annotation.metadata.get(
                    "doc_version_id",
                    f"{annotation.doc_id}:unknown",
                )
            ),
            "source_id": str(annotation.metadata.get("source_id", "unknown")),
            "title": str(annotation.metadata.get("title", "")),
        }

    @staticmethod
    def _resolve_case_document(
        annotation: CanonicalAnnotation,
        documents: Mapping[str, CaseDocument | Mapping[str, Any]] | None,
    ) -> CaseDocument | Mapping[str, Any]:
        if documents and annotation.case_id in documents:
            return documents[annotation.case_id]
        return {
            "case_id": annotation.case_id,
            "doc_id": annotation.doc_id,
            "doc_version_id": str(
                annotation.metadata.get(
                    "doc_version_id",
                    f"{annotation.doc_id}:unknown",
                )
            ),
            "title": str(annotation.metadata.get("title", "")),
            "source_id": str(annotation.metadata.get("source_id", "unknown")),
            "raw_text": annotation.text,
            "clean_text": annotation.text,
            "metadata": {},
        }

    def _session(self) -> Any:
        if self._closed:
            raise Neo4jConnectionError("Neo4j Driver 已关闭")
        return self._driver.session(
            database=self.config.database,
            fetch_size=self.config.fetch_size,
        )

    def _execute_write(
        self, cypher: str, parameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            with self._session() as session:
                return session.execute_write(
                    _run_transaction_query,
                    cypher,
                    dict(parameters),
                    None,
                )
        except Exception as exc:
            if isinstance(exc, Neo4jServiceError):
                raise
            self._raise_driver_error(exc, "执行 Neo4j 写入", write=True)
            raise AssertionError("unreachable")

    def _execute_read(
        self,
        cypher: str,
        parameters: Mapping[str, Any],
        *,
        max_records: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            with self._session() as session:
                return session.execute_read(
                    _run_transaction_query,
                    cypher,
                    dict(parameters),
                    max_records,
                    timeout,
                )
        except Exception as exc:
            if isinstance(exc, Neo4jServiceError):
                raise
            self._raise_driver_error(exc, "执行 Neo4j 只读查询")
            raise AssertionError("unreachable")

    def _bounded_limit(self, limit: int | None) -> int:
        value = self.config.default_query_limit if limit is None else limit
        if value <= 0:
            raise Neo4jValidationError("limit 必须大于 0")
        return min(value, self.config.max_query_limit)

    @staticmethod
    def _assert_read_only(cypher: str, *, validated: bool) -> None:
        if not validated:
            raise Neo4jUnsafeQueryError("查询尚未通过上游安全校验")
        if not cypher or not cypher.strip():
            raise Neo4jUnsafeQueryError("Cypher 不能为空")
        cleaned = _strip_cypher_literals_and_comments(cypher)
        if ";" in cleaned:
            raise Neo4jUnsafeQueryError("只读接口不允许多语句或分号")
        tokens = set(re.findall(r"\b[A-Za-z_]+\b", cleaned.upper()))
        dangerous = sorted(tokens & WRITE_KEYWORDS)
        if dangerous:
            raise Neo4jUnsafeQueryError(
                f"只读接口检测到写入关键词：{', '.join(dangerous)}"
            )
        call_match = re.search(
            r"\bCALL\s+([A-Za-z0-9_.]+)", cleaned, flags=re.IGNORECASE
        )
        if call_match:
            procedure = call_match.group(1).lower()
            if procedure not in READ_ONLY_CALL_ALLOWLIST:
                raise Neo4jUnsafeQueryError(
                    f"CALL 过程不在只读白名单：{procedure}"
                )

    def _raise_driver_error(
        self, exc: Exception, operation: str, *, write: bool = False
    ) -> None:
        name = type(exc).__name__
        code = str(getattr(exc, "code", ""))
        if name == "AuthError" or "Security.Unauthorized" in code:
            raise Neo4jAuthenticationError(
                f"{operation}失败：认证失败"
            ) from exc
        if (
            name in {"ServiceUnavailable", "SessionExpired"}
            or "SessionExpired" in name
        ):
            raise Neo4jConnectionError(
                f"{operation}失败：连接或会话不可用"
            ) from exc
        if "Forbidden" in code or "Authorization" in code:
            raise Neo4jAuthenticationError(
                f"{operation}失败：权限不足"
            ) from exc
        if write:
            raise Neo4jWriteError(f"{operation}失败：{name}") from exc
        raise Neo4jQueryError(f"{operation}失败：{name}") from exc


# 中文注释：将业务对象属性转换为 Neo4j 可存储的安全基础类型，过滤不支持的嵌套值。
def sanitize_properties(
    value: BaseModel | Mapping[str, Any],
    *,
    metadata_warning_chars: int = 10000,
) -> dict[str, Any]:
    """转换为 Neo4j 可接受属性，并移除空值和敏感字段。"""

    source = (
        value.model_dump()
        if isinstance(value, BaseModel)
        else dict(value)
    )
    result: dict[str, Any] = {}
    for key, item in source.items():
        lowered = str(key).lower()
        if lowered in SENSITIVE_PROPERTY_NAMES or any(
            token in lowered
            for token in ("password", "api_key", "access_token")
        ):
            continue
        if item is None:
            continue
        try:
            result[str(key)] = _sanitize_property_value(item)
        except (TypeError, ValueError) as exc:
            raise Neo4jConversionError(
                f"属性无法转换：{key}"
            ) from exc
        if (
            str(key).lower() == "metadata"
            and len(str(result[str(key)])) > metadata_warning_chars
        ):
            logger.warning(
                "Neo4j metadata 较长 chars=%d",
                len(str(result[str(key)])),
            )
    return result


def _sanitize_property_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _sanitize_property_value(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, BaseModel):
        return json.dumps(
            sanitize_properties(value),
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(value, Mapping):
        return json.dumps(
            {
                str(key): _json_safe(item)
                for key, item in value.items()
                if item is not None
                and str(key).lower() not in SENSITIVE_PROPERTY_NAMES
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(value, (list, tuple)):
        if any(
            isinstance(item, (Mapping, BaseModel))
            for item in value
        ):
            return json.dumps(
                [_json_safe(item) for item in value],
                ensure_ascii=False,
                sort_keys=True,
            )
        return [_sanitize_property_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Neo4j 不支持的属性类型：{type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if item is not None
            and str(key).lower() not in SENSITIVE_PROPERTY_NAMES
            and not any(
                token in str(key).lower()
                for token in ("password", "api_key", "access_token")
            )
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def serialize_neo4j_value(value: Any) -> Any:
    """递归转换 Node、Relationship、Path、Record 和时间类型。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return {
            "nodes": [
                serialize_neo4j_value(item) for item in value.nodes
            ],
            "relationships": [
                serialize_neo4j_value(item)
                for item in value.relationships
            ],
        }
    if hasattr(value, "type") and (
        hasattr(value, "start_node")
        or hasattr(value, "start_node_element_id")
    ):
        return _serialize_relationship(value)
    if hasattr(value, "labels") and hasattr(value, "items"):
        return _serialize_node(value)
    if isinstance(value, Mapping):
        return {
            str(key): serialize_neo4j_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [serialize_neo4j_value(item) for item in value]
    if hasattr(value, "data") and callable(value.data):
        return serialize_neo4j_value(value.data())
    if hasattr(value, "iso_format") and callable(value.iso_format):
        return value.iso_format()
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if hasattr(value, "months") and hasattr(value, "days"):
        return str(value)
    raise Neo4jConversionError(
        f"无法序列化 Neo4j 值：{type(value).__name__}"
    )


def _serialize_node(value: Any) -> dict[str, Any]:
    properties = dict(value.items()) if hasattr(value, "items") else dict(value)
    return {
        "element_id": str(getattr(value, "element_id", "")),
        "labels": sorted(str(item) for item in getattr(value, "labels", [])),
        "properties": serialize_neo4j_value(properties),
    }


def _serialize_relationship(value: Any) -> dict[str, Any]:
    properties = dict(value.items()) if hasattr(value, "items") else {}
    relationship_type = getattr(value, "type", None)
    if callable(relationship_type):
        relationship_type = relationship_type()
    start = getattr(value, "start_node_element_id", None)
    end = getattr(value, "end_node_element_id", None)
    if start is None and getattr(value, "start_node", None) is not None:
        start = getattr(value.start_node, "element_id", None)
    if end is None and getattr(value, "end_node", None) is not None:
        end = getattr(value.end_node, "element_id", None)
    return {
        "element_id": str(getattr(value, "element_id", "")),
        "type": str(relationship_type or type(value).__name__),
        "start_node_element_id": str(start or ""),
        "end_node_element_id": str(end or ""),
        "properties": serialize_neo4j_value(properties),
    }


def _run_transaction_query(
    tx: Any,
    cypher: str,
    parameters: Mapping[str, Any],
    max_records: int | None,
    timeout: float | None = None,
) -> dict[str, Any]:
    query: Any = cypher
    if timeout is not None and Query is not None:
        query = Query(cypher, timeout=timeout)
    result = tx.run(query, parameters)
    keys = list(result.keys()) if hasattr(result, "keys") else []
    records: list[dict[str, Any]] = []
    truncated = False
    for index, record in enumerate(result):
        if max_records is not None and index >= max_records:
            truncated = True
            break
        data = record.data() if hasattr(record, "data") else dict(record)
        records.append(serialize_neo4j_value(data))
    summary = result.consume() if hasattr(result, "consume") else None
    return {
        "records": records,
        "keys": keys,
        "summary": summary,
        "truncated": truncated,
    }


def _consume_result(result: Any) -> tuple[list[dict[str, Any]], Any]:
    keys = list(result.keys()) if hasattr(result, "keys") else []
    records = []
    for record in result:
        data = record.data() if hasattr(record, "data") else dict(record)
        records.append(serialize_neo4j_value(data))
    summary = result.consume() if hasattr(result, "consume") else None
    del keys
    return records, summary


def _extract_counters(
    summary: Any,
    *,
    records_processed: int = 0,
    records_skipped: int = 0,
    records_failed: int = 0,
) -> Neo4jWriteCounters:
    counters = getattr(summary, "counters", summary)

    def count(name: str) -> int:
        return int(getattr(counters, name, 0) or 0)

    return Neo4jWriteCounters(
        nodes_created=count("nodes_created"),
        nodes_deleted=count("nodes_deleted"),
        relationships_created=count("relationships_created"),
        relationships_deleted=count("relationships_deleted"),
        properties_set=count("properties_set"),
        labels_added=count("labels_added"),
        indexes_added=count("indexes_added"),
        constraints_added=count("constraints_added"),
        records_processed=records_processed,
        records_skipped=records_skipped,
        records_failed=records_failed,
    )


def _serialize_summary(summary: Any) -> dict[str, Any]:
    """提取不暴露驱动对象的查询摘要。"""

    if summary is None:
        return {}
    counters = _extract_counters(summary)
    result_available_after = getattr(
        summary, "result_available_after", None
    )
    result_consumed_after = getattr(
        summary, "result_consumed_after", None
    )
    query_type = getattr(summary, "query_type", None)
    return {
        "counters": counters.model_dump(),
        "result_available_after": result_available_after,
        "result_consumed_after": result_consumed_after,
        "query_type": str(query_type) if query_type is not None else None,
    }


def _schema_definitions() -> list[tuple[str, str, str]]:
    constraints = [
        ("case_id_unique", "Case", "case_id"),
        (
            "source_document_version_unique",
            "SourceDocument",
            "doc_version_id",
        ),
        ("text_span_uid_unique", "TextSpan", "text_uid"),
        ("entity_uid_unique", "Entity", "entity_uid"),
        ("mention_uid_unique", "EntityMention", "mention_uid"),
        ("claim_id_unique", "Claim", "claim_id"),
    ]
    indexes = [
        ("entity_normalized_name_idx", "Entity", "normalized_name"),
        ("entity_name_idx", "Entity", "name"),
        ("entity_type_idx", "Entity", "entity_type"),
        ("claim_relation_type_idx", "Claim", "relation_type"),
        ("claim_status_idx", "Claim", "status"),
        ("claim_case_id_idx", "Claim", "case_id"),
        ("claim_annotation_id_idx", "Claim", "annotation_id"),
        ("source_doc_id_idx", "SourceDocument", "doc_id"),
        ("source_id_idx", "SourceDocument", "source_id"),
        ("text_annotation_id_idx", "TextSpan", "annotation_id"),
        ("text_case_id_idx", "TextSpan", "case_id"),
    ]
    result = [
        (
            name,
            "constraint",
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE",
        )
        for name, label, property_name in constraints
    ]
    result.extend(
        (
            name,
            "index",
            f"CREATE INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{property_name})",
        )
        for name, label, property_name in indexes
    )
    return result


def _strip_cypher_literals_and_comments(cypher: str) -> str:
    without_comments = re.sub(r"/\*.*?\*/", " ", cypher, flags=re.DOTALL)
    without_comments = re.sub(r"//[^\n]*", " ", without_comments)
    without_strings = re.sub(
        r"'(?:\\.|''|[^'])*'|\"(?:\\.|\"\"|[^\"])*\"",
        " ",
        without_comments,
    )
    return without_strings


def _as_mapping(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    raise Neo4jConversionError(
        f"期望 Pydantic 模型或 Mapping，实际为 {type(value).__name__}"
    )


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _chunked(
    values: Sequence[Any], size: int
) -> Iterable[Sequence[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]

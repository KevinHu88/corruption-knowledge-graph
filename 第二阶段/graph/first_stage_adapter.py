"""将第一阶段 Neo4jService 适配为第二阶段只读接口。"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from typing import Any

from 第二阶段.config import FIRST_STAGE_DIR


class FirstStageAdapterError(RuntimeError):
    """第一阶段模块加载或调用失败。"""


class FirstStageGraphAdapter:
    """只 READ/IMPORT/CALL 第一阶段 Neo4jService，不执行任何写方法。"""

    def __init__(self, service: Any | None = None) -> None:
        self._service = service
        self._owns_service = service is None

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = self._load_service()
        return self._service

    def _load_service(self) -> Any:
        if not FIRST_STAGE_DIR.is_dir():
            raise FirstStageAdapterError(f"第一阶段目录不存在：{FIRST_STAGE_DIR}")
        first_stage_path = str(FIRST_STAGE_DIR)
        if first_stage_path not in sys.path:
            sys.path.insert(0, first_stage_path)
        existing_config = sys.modules.get("config")
        if existing_config is not None:
            config_file = str(getattr(existing_config, "__file__", ""))
            if config_file and not config_file.startswith(first_stage_path):
                raise FirstStageAdapterError(
                    "顶层 config 模块已被其他项目占用；请通过第二阶段包入口启动"
                )
        try:
            module = importlib.import_module("src.services.neo4j_service")
            return module.Neo4jService()
        except Exception as exc:
            raise FirstStageAdapterError("无法初始化第一阶段 Neo4jService") from exc

    def find_entities(self, **filters: Any) -> list[dict[str, Any]]:
        return list(self.service.find_entities(**filters))

    def list_entity_claims(self, entity_uid: str, **filters: Any) -> list[dict[str, Any]]:
        return list(self.service.list_entity_claims(entity_uid, **filters))

    def execute_read(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        result = self.service.execute_read_query(
            cypher,
            parameters or {},
            max_records=max_records,
            validated=True,
        )
        records = result.records if hasattr(result, "records") else result
        return list(records)

    def close(self) -> None:
        if self._owns_service and self._service is not None:
            self._service.close()
            self._service = None

    def __enter__(self) -> "FirstStageGraphAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


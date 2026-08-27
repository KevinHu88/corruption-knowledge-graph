"""Unified programmatic and command-line entry point for project flows.

Examples:
    python main.py ingestion --input request.json
    python main.py annotation --json "{\"jobs\": [...]}"
    Get-Content request.json | python main.py dataset-build --input -
    '{"dry_run":true}' | python main.py ingestion --input -
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from flows.annotation_flow import (
    AnnotationFlowResult,
    AnnotationJob,
    annotation_flow,
)
from flows.dataset_build_flow import dataset_build_flow
from flows.graph_ingestion_flow import graph_ingestion_flow
from flows.ingestion_flow import ingestion_flow
from flows.review_sync_flow import review_sync_flow
from flows.retrieval_flow import RetrievalFlowResult, retrieval_flow
from flows.training_flow import TrainingFlowResult, training_flow
from models import (
    CanonicalAnnotation,
    CaseDocument,
    IngestionFlowResult,
    RawDocument,
    SourceDocument,
)
from src.services.dataset_service import DatasetBuildResult, DatasetManifest
from src.services.label_studio_service import LabelStudioSyncResult
from src.services.neo4j_service import Neo4jBatchResult
from config import PreflightError, require_preflight
from src.services.stage_routing_service import (
    StagePreconditionError,
    StageRoutePlan,
    StageRoutingService,
)


CommandName = Literal[
    "annotation",
    "retrieval",
    "review-sync",
    "dataset-build",
    "training",
    "graph-ingestion",
    "ingestion",
]
FlowResult = (
    AnnotationFlowResult
    | RetrievalFlowResult
    | LabelStudioSyncResult
    | DatasetBuildResult
    | TrainingFlowResult
    | Neo4jBatchResult
    | IngestionFlowResult
)


class RetrievalRequest(BaseModel):
    """Inputs accepted by ``retrieval_flow``."""

    source_ids: list[str] | None = None
    today: date | None = None
    extract_missing_content: bool = True
    continue_on_error: bool = False


# 中文注释：annotation 命令的输入协议，可接收已构造的标注任务或待处理原始文档。
class AnnotationRequest(BaseModel):
    """Inputs accepted by ``annotation_flow``."""

    jobs: list[AnnotationJob] = Field(default_factory=list)
    raw_documents: list[RawDocument] = Field(default_factory=list)
    publish_for_review: bool = False
    project_id: int | None = None
    model_version: str | None = None


# 中文注释：review-sync 命令的输入协议，用于限定 Label Studio 项目和待同步任务。
class ReviewSyncRequest(BaseModel):
    """Inputs accepted by ``review_sync_flow``."""

    task_ids: list[int] | None = None
    project_id: int | None = None
    max_items: int = Field(default=1000, gt=0)


# 中文注释：dataset-build 命令的输入协议，要求提供可用于构建版本数据集的规范标注。
class DatasetBuildRequest(BaseModel):
    """Inputs accepted by ``dataset_build_flow``."""

    annotations: list[CanonicalAnnotation]
    dataset_version: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    previous_manifest: DatasetManifest | dict[str, Any] | None = None
    rebuild_test_set: bool = False
    strict: bool | None = None
    overwrite: bool = False


# 中文注释：training 命令的输入协议，指定数据集版本以及需要训练的模型类型。
class TrainingRequest(BaseModel):
    """Inputs accepted by ``training_flow``."""

    dataset_version: str = Field(
        min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    task_types: list[Literal["ner", "relation"]] = Field(
        default_factory=lambda: ["ner", "relation"], min_length=1
    )


# 中文注释：graph-ingestion 命令的输入协议，同时携带标注及其案件、来源文档映射。
class GraphIngestionRequest(BaseModel):
    """Inputs accepted by ``graph_ingestion_flow``."""

    annotations: list[CanonicalAnnotation]
    source_documents: dict[
        str, SourceDocument | dict[str, Any]
    ] | None = None
    case_documents: dict[
        str, CaseDocument | dict[str, Any]
    ] | None = None
    entity_uid_maps: dict[str, dict[str, str]] | None = None
    continue_on_error: bool = False


# 中文注释：总流程输入协议，使用阶段开关组合标注、人审、数据集、训练和图谱写入。
class IngestionRequest(BaseModel):
    """Inputs accepted by the top-level ``ingestion_flow``."""

    batch_id: str | None = None
    case_id: str | None = None
    annotation_jobs: list[AnnotationJob] = Field(default_factory=list)
    raw_documents: list[RawDocument] = Field(default_factory=list)
    annotations: list[CanonicalAnnotation] = Field(default_factory=list)
    run_retrieval: bool = False
    retrieval_source_ids: list[str] | None = None
    retrieval_date: date | None = None
    extract_missing_content: bool = True
    continue_on_retrieval_error: bool = False
    run_annotation: bool = True
    publish_for_review: bool = False
    run_review_sync: bool = False
    run_dataset_build: bool = False
    run_training: bool = False
    run_graph_ingestion: bool = False
    project_id: int | None = None
    review_task_ids: list[int] | None = None
    review_max_items: int = Field(default=1000, gt=0)
    dataset_version: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    previous_manifest: DatasetManifest | dict[str, Any] | None = None
    rebuild_test_set: bool = False
    overwrite_dataset: bool = False
    training_task_types: list[Literal["ner", "relation"]] = Field(
        default_factory=lambda: ["ner", "relation"], min_length=1
    )
    source_documents: dict[
        str, SourceDocument | dict[str, Any]
    ] | None = None
    case_documents: dict[
        str, CaseDocument | dict[str, Any]
    ] | None = None
    entity_uid_maps: dict[str, dict[str, str]] | None = None
    continue_on_graph_error: bool = False
    dry_run: bool = False


REQUEST_MODELS: dict[CommandName, type[BaseModel]] = {
    "retrieval": RetrievalRequest,
    "annotation": AnnotationRequest,
    "review-sync": ReviewSyncRequest,
    "dataset-build": DatasetBuildRequest,
    "training": TrainingRequest,
    "graph-ingestion": GraphIngestionRequest,
    "ingestion": IngestionRequest,
}


def prepare_stage(
    command: CommandName,
    payload: Mapping[str, Any] | BaseModel,
) -> tuple[BaseModel, StageRoutePlan]:
    """Validate request, upstream state and environment before dispatch."""

    data = (
        payload.model_dump()
        if isinstance(payload, BaseModel)
        else dict(payload)
    )
    request = REQUEST_MODELS[command].model_validate(data)
    plan = StageRoutingService().prepare(command, request)
    require_preflight(plan.required_features)
    return request, plan


def _model_arguments(model: BaseModel) -> dict[str, Any]:
    """Preserve nested Pydantic objects when forwarding validated inputs."""

    return {
        name: getattr(model, name)
        for name in model.__class__.model_fields
    }


# 中文注释：统一命令路由器；完成输入模型校验后，把请求交给对应 Flow 执行。
async def execute(
    command: CommandName,
    payload: Mapping[str, Any] | BaseModel,
) -> FlowResult:
    """Validate a payload and execute the selected Prefect flow."""

    request, _plan = prepare_stage(command, payload)
    if command == "annotation":
        return await annotation_flow(**_model_arguments(request))
    if command == "retrieval":
        return retrieval_flow(**_model_arguments(request))
    if command == "review-sync":
        return review_sync_flow(**_model_arguments(request))
    if command == "dataset-build":
        return dataset_build_flow(**_model_arguments(request))
    if command == "training":
        return training_flow(**_model_arguments(request))
    if command == "graph-ingestion":
        return graph_ingestion_flow(**_model_arguments(request))
    if command == "ingestion":
        return await ingestion_flow(**_model_arguments(request))
    raise ValueError(f"unsupported command: {command}")


# 中文注释：把 CLI 层的 IngestionRequest 转为 Flow 请求，隔离入口协议与编排协议。
async def run_ingestion(
    payload: Mapping[str, Any] | IngestionRequest,
) -> IngestionFlowResult:
    """Typed convenience API for the complete ingestion pipeline."""

    result = await execute("ingestion", payload)
    if not isinstance(result, IngestionFlowResult):
        raise TypeError("ingestion command returned an unexpected result")
    return result


# 中文注释：声明命令行参数，只处理启动方式，不承载任何领域业务逻辑。
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the project's Prefect tasks and flows.",
    )
    parser.add_argument(
        "command",
        choices=tuple(REQUEST_MODELS),
        help="Flow to execute.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        metavar="PATH",
        help="UTF-8 JSON request file; use '-' to read stdin.",
    )
    source.add_argument(
        "--json",
        dest="json_payload",
        help="Inline JSON request object.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write the JSON result to this file instead of stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON result.",
    )
    return parser


# 中文注释：按“文件、标准输入、内联 JSON”的优先级读取本次运行参数。
def _load_payload(
    *,
    input_path: str | None,
    json_payload: str | None,
) -> dict[str, Any]:
    if json_payload is not None:
        raw = json_payload
    elif input_path == "-":
        raw = sys.stdin.read()
    elif input_path is not None:
        raw = Path(input_path).read_text(encoding="utf-8")
    else:
        raise ValueError("one of --input or --json is required")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("request JSON must be an object")
    return value


# 中文注释：把 Pydantic 模型或普通结果统一序列化为可输出的 JSON。
def _serialize_result(result: FlowResult, *, pretty: bool) -> str:
    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2 if pretty else None,
    )


# 中文注释：将最终 JSON 写到指定文件；未指定路径时输出到终端。
def _write_result(content: str, output_path: str | None) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{content}\n", encoding="utf-8")
    else:
        sys.stdout.write(f"{content}\n")


# 中文注释：程序总入口，负责参数解析、启动异步 Flow、输出结果并映射进程退出码。
def main(argv: Sequence[str] | None = None) -> int:
    """CLI boundary; async flows are driven only at this outermost layer."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _load_payload(
            input_path=args.input,
            json_payload=args.json_payload,
        )
        request = REQUEST_MODELS[args.command].model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError):
        logging.exception("Invalid request or local input file")
        return 2
    try:
        result = asyncio.run(execute(args.command, request))
    except (ValidationError, PreflightError, StagePreconditionError):
        logging.exception("Stage preparation failed")
        return 2
    except Exception:
        logging.exception("Flow execution failed")
        return 1
    try:
        _write_result(
            _serialize_result(result, pretty=args.pretty),
            args.output,
        )
    except OSError:
        logging.exception("Unable to write result")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

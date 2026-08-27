"""项目统一配置加载模块。"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"

load_dotenv(BASE_DIR / ".env")


# 中文注释：读取单个 YAML 配置文件，并统一校验“文件存在、根节点为字典”这两个基础条件。
def load_yaml(filename: str) -> dict[str, Any]:
    """读取指定YAML配置文件。"""

    path = CONFIG_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data or {}


# 中文注释：集中声明只能从环境变量获取的敏感配置和部署参数，避免密钥散落在业务模块中。
class EnvironmentSettings(BaseModel):
    """需要从环境变量中读取的敏感配置。"""

    tavily_api_key: str = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    llm_api_key: str = Field(
        default_factory=lambda: (
            os.getenv("OPENAI_API_KEY", "")
            or os.getenv("LLM_API_KEY", "")
        )
    )

    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "")
    )

    llm_model_id: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_ID", "gpt-5.4-mini")
    )

    llm_request_timeout: float = Field(
        default_factory=lambda: float(
            os.getenv("LLM_REQUEST_TIMEOUT", "60")
        ),
        gt=0,
    )

    llm_structured_api: Literal["auto", "responses", "chat"] = Field(
        default_factory=lambda: os.getenv(
            "LLM_STRUCTURED_API", "auto"
        ).strip().lower()
    )

    llm_reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] = Field(
        default_factory=lambda: os.getenv(
            "LLM_REASONING_EFFORT", "low"
        ).strip().lower()
    )

    llm_temperature: float = Field(
        default_factory=lambda: float(
            os.getenv("LLM_TEMPERATURE", "0.1")
        ),
        ge=0,
        le=2,
    )

    llm_max_tokens: int = Field(
        default_factory=lambda: int(
            os.getenv("LLM_MAX_TOKENS", "2048")
        ),
        gt=0,
    )

    neo4j_uri: str = Field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )

    neo4j_username: str = Field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j")
    )

    neo4j_password: str = Field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )

    neo4j_database: str = Field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    neo4j_connection_timeout: float = Field(
        default_factory=lambda: float(
            os.getenv("NEO4J_CONNECTION_TIMEOUT", "30")
        ),
        gt=0,
    )

    neo4j_max_connection_pool_size: int = Field(
        default_factory=lambda: int(
            os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "20")
        ),
        gt=0,
    )

    neo4j_max_transaction_retry_time: float = Field(
        default_factory=lambda: float(
            os.getenv("NEO4J_MAX_TRANSACTION_RETRY_TIME", "30")
        ),
        ge=0,
    )

    neo4j_fetch_size: int = Field(
        default_factory=lambda: int(os.getenv("NEO4J_FETCH_SIZE", "1000")),
        gt=0,
    )

    label_studio_url: str = Field(
        default_factory=lambda: os.getenv("LABEL_STUDIO_URL", "")
    )

    label_studio_api_key: str = Field(
        default_factory=lambda: os.getenv("LABEL_STUDIO_API_KEY", "")
    )

    label_studio_project_id: int | None = Field(
        default_factory=lambda: (
            int(value)
            if (value := os.getenv("LABEL_STUDIO_PROJECT_ID", "")).strip()
            else None
        ),
        gt=0,
    )

    label_studio_timeout: float = Field(
        default_factory=lambda: float(
            os.getenv("LABEL_STUDIO_TIMEOUT", "20")
        ),
        gt=0,
    )

    label_studio_batch_size: int = Field(
        default_factory=lambda: int(
            os.getenv("LABEL_STUDIO_BATCH_SIZE", "100")
        ),
        gt=0,
    )

    label_studio_model_version: str = Field(
        default_factory=lambda: os.getenv("LABEL_STUDIO_MODEL_VERSION", "")
    )


PreflightFeature = Literal["llm", "tavily", "neo4j", "label_studio"]


class PreflightResult(BaseModel):
    """Non-secret readiness report for external integrations."""

    ok: bool
    checked_features: list[PreflightFeature] = Field(default_factory=list)
    missing_variables: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PreflightError(RuntimeError):
    """Raised before a flow starts when required deployment settings are absent."""


def run_preflight(
    features: Iterable[PreflightFeature],
    *,
    settings: EnvironmentSettings | None = None,
) -> PreflightResult:
    """Validate only the integrations required by the selected command."""

    environment = settings or EnvironmentSettings()
    selected = list(dict.fromkeys(features))
    missing: list[str] = []
    warnings: list[str] = []
    requirements: dict[PreflightFeature, tuple[tuple[str, Any], ...]] = {
        "llm": (("OPENAI_API_KEY", environment.llm_api_key),),
        "tavily": (("TAVILY_API_KEY", environment.tavily_api_key),),
        "neo4j": (
            ("NEO4J_URI", environment.neo4j_uri),
            ("NEO4J_PASSWORD", environment.neo4j_password),
        ),
        "label_studio": (
            ("LABEL_STUDIO_URL", environment.label_studio_url),
            ("LABEL_STUDIO_API_KEY", environment.label_studio_api_key),
            ("LABEL_STUDIO_PROJECT_ID", environment.label_studio_project_id),
        ),
    }
    for feature in selected:
        missing.extend(
            name for name, value in requirements[feature] if not value
        )
    legacy_key = os.getenv("LLM_API_KEY", "")
    canonical_key = os.getenv("OPENAI_API_KEY", "")
    if legacy_key and not canonical_key:
        warnings.append(
            "LLM_API_KEY is supported for compatibility; prefer OPENAI_API_KEY"
        )
    if legacy_key and canonical_key and legacy_key != canonical_key:
        warnings.append(
            "OPENAI_API_KEY takes precedence over a different LLM_API_KEY"
        )
    return PreflightResult(
        ok=not missing,
        checked_features=selected,
        missing_variables=list(dict.fromkeys(missing)),
        warnings=warnings,
    )


def require_preflight(
    features: Iterable[PreflightFeature],
    *,
    settings: EnvironmentSettings | None = None,
) -> PreflightResult:
    """Return readiness details or stop before any external side effect occurs."""

    result = run_preflight(features, settings=settings)
    if not result.ok:
        raise PreflightError(
            "missing required environment variables: "
            + ", ".join(result.missing_variables)
        )
    return result


# 中文注释：项目的总配置容器，把环境变量与五类 YAML 配置聚合后提供给各个 Service 和模型模块。
class ProjectConfig(BaseModel):
    """项目运行所需的完整配置。"""

    environment: EnvironmentSettings
    sources: dict[str, Any]
    schema_config: dict[str, Any]
    workflow: dict[str, Any]
    training: dict[str, Any]
    graph: dict[str, Any]


# 中文注释：统一配置入口；调用方无需分别解析 dotenv 和各个 YAML 文件。
def load_project_config() -> ProjectConfig:
    """加载完整项目配置。"""

    return ProjectConfig(
        environment=EnvironmentSettings(),
        sources=load_yaml("sources.yaml"),
        schema_config=load_yaml("schema.yaml"),
        workflow=load_yaml("workflow.yaml"),
        training=load_yaml("training.yaml"),
        graph=load_yaml("graph.yaml"),
    )

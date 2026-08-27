"""业务 Prompt 与当前模型流程的数据契约测试。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import BASE_DIR, load_project_config
from models import ConflictReviewResult


PROMPT_DIR = BASE_DIR / "prompts"


def variables(name: str) -> set[str]:
    content = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return set(re.findall(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}", content))


def content(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def test_relevance_prompt_matches_compact_relevance_payload():
    prompt = content("relevance_filter_prompt.jinja2")

    assert variables("relevance_filter_prompt.jinja2") == {
        "source_name", "title", "published_at", "url", "text"
    }
    assert all(
        field in prompt
        for field in ('"relevant"', '"score"', '"reason"')
    )
    assert '"evidence_spans"' not in prompt
    assert "不执行实体识别和关系抽取" in prompt


def test_canonical_prompt_consumes_deep_predictions_and_outputs_model_fields():
    prompt = content("canonical_annotation_prompt.jinja2")

    assert variables("canonical_annotation_prompt.jinja2") == {
        "annotation_id",
        "case_id",
        "doc_id",
        "text_id",
        "schema_version",
        "ner_model_version",
        "relation_model_version",
        "prompt_version",
        "text",
        "deep_model_result",
        "review_reasons",
        "schema",
    }
    assert "entity_type 转换为" in prompt
    assert "relation_type 转换为" in prompt
    assert '"type": "PER"' in prompt
    assert '"status": "GENERATED"' in prompt
    assert "不得从零重新抽取" in prompt
    assert "不得新增深度模型结果中不存在的关系" in prompt


def test_repair_prompt_has_complete_canonical_annotation_contract():
    prompt = content("annotation_repair_prompt.jinja2")

    assert variables("annotation_repair_prompt.jinja2") == {
        "text",
        "invalid_annotation",
        "validation_errors",
        "schema_version",
        "schema",
    }
    assert "只修复 validation_errors" in prompt
    assert "annotation_id、case_id、doc_id、text_id、text" in prompt
    assert "metadata.unresolved_validation_errors" in prompt
    assert "PENDING_REVIEW" in prompt


def test_conflict_prompt_matches_conflict_review_result_contract():
    prompt = content("conflict_review_prompt.jinja2")

    assert variables("conflict_review_prompt.jinja2") == {
        "text", "deep_model_result", "llm_result", "schema"
    }
    assert all(
        field in prompt
        for field in (
            '"decision"',
            '"review_required"',
            '"selected_entities"',
            '"selected_relations"',
            '"conflicts"',
            '"reason"',
        )
    )
    assert "models.ConflictReviewResult" in prompt
    assert "HUMAN_REVIEW 时 review_required 必须为 true" in prompt


def test_conflict_review_model_enforces_review_flag_and_entity_references():
    result = ConflictReviewResult(
        decision="HUMAN_REVIEW",
        review_required=True,
        reason="证据不足",
    )
    assert result.review_required is True

    with pytest.raises(ValidationError):
        ConflictReviewResult(
            decision="USE_DEEP_MODEL",
            review_required=True,
            reason="错误标记",
        )


def test_workflow_uses_v2_prompt_versions_and_existing_paths():
    prompts = load_project_config().workflow["prompts"]

    assert {
        name: item["version"] for name, item in prompts.items()
    } == {
        "llm_preannotation": "llm_preannotation_v1.0",
        "relevance_filter": "relevance_v2.0",
        "canonical_annotation": "annotation_v2.0",
        "annotation_repair": "repair_v2.0",
        "conflict_review": "conflict_v2.0",
    }
    for item in prompts.values():
        assert (BASE_DIR / item["path"]).is_file()

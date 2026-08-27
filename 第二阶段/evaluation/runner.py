"""执行 JSONL 问答评测集并生成结构化报告。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from fastapi.testclient import TestClient

from 第二阶段.api.app import create_app
from 第二阶段.api.dependencies import build_container
from 第二阶段.config import QAConfig
from 第二阶段.generation.llm_client import MockLLMClient

DEFAULT_DATASET_PATH = Path(__file__).with_name("mock_qa_eval.jsonl")


class HTTPClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...

    def delete(self, url: str, **kwargs: Any) -> Any: ...


def load_cases(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """读取 JSONL，并验证每条样本具备最小必需字段。"""

    dataset_path = Path(path)
    cases: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"评测集第 {line_number} 行不是合法 JSON"
            ) from exc
        missing = {"id", "question", "expected"} - set(case)
        if missing:
            raise ValueError(
                f"评测集第 {line_number} 行缺少字段：{sorted(missing)}"
            )
        cases.append(case)
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


def _evidence_case_ids(payload: dict[str, Any]) -> list[str]:
    case_ids: set[str] = set()
    for evidence in payload.get("evidence", []):
        metadata = evidence.get("metadata", {})
        value = (
            metadata.get("relationship_properties", {}).get("case_id")
            or metadata.get("entity", {}).get("case_id")
            or metadata.get("case", {}).get("case_id")
        )
        if value:
            case_ids.add(str(value))
    return sorted(case_ids)


def _retrieval_modes(payload: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(mode)
            for evidence in payload.get("evidence", [])
            if (
                mode := evidence.get("metadata", {})
                .get("retrieval", {})
                .get("mode")
            )
        }
    )


def _evaluate_expectations(
    status_code: int,
    payload: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected_status = int(expected.get("status_code", 200))
    if status_code != expected_status:
        failures.append(f"status_code: expected {expected_status}, got {status_code}")

    if "route" in expected and payload.get("route") != expected["route"]:
        failures.append(
            f"route: expected {expected['route']}, got {payload.get('route')}"
        )

    actual_source_types = sorted(
        {item.get("source_type") for item in payload.get("evidence", [])}
        - {None}
    )
    expected_source_types = sorted(expected.get("source_types", []))
    if actual_source_types != expected_source_types:
        failures.append(
            "source_types: "
            f"expected {expected_source_types}, got {actual_source_types}"
        )

    if "retrieval_modes" in expected:
        actual_modes = _retrieval_modes(payload)
        expected_modes = sorted(str(item) for item in expected["retrieval_modes"])
        if actual_modes != expected_modes:
            failures.append(
                f"retrieval_modes: expected {expected_modes}, got {actual_modes}"
            )

    actual_case_ids = _evidence_case_ids(payload)
    if "evidence_case_ids" in expected:
        expected_case_ids = sorted(str(item) for item in expected["evidence_case_ids"])
        if actual_case_ids != expected_case_ids:
            failures.append(
                f"evidence_case_ids: expected {expected_case_ids}, got {actual_case_ids}"
            )

    evidence_count = len(payload.get("evidence", []))
    minimum = int(expected.get("min_evidence", 0))
    if evidence_count < minimum:
        failures.append(
            f"evidence_count: expected at least {minimum}, got {evidence_count}"
        )

    answer = str(payload.get("answer") or "").strip()
    if "answer_nonempty" in expected:
        expected_nonempty = bool(expected["answer_nonempty"])
        if bool(answer) != expected_nonempty:
            failures.append(
                f"answer_nonempty: expected {expected_nonempty}, got {bool(answer)}"
            )
    for keyword in expected.get("answer_contains", []):
        if str(keyword) not in answer:
            failures.append(f"answer missing keyword: {keyword}")

    expected_candidates = expected.get("candidate_case_ids")
    if expected_candidates is not None:
        actual_candidates = sorted(payload.get("candidate_case_ids") or [])
        if actual_candidates != sorted(expected_candidates):
            failures.append(
                "candidate_case_ids: "
                f"expected {sorted(expected_candidates)}, got {actual_candidates}"
            )
    return failures


def evaluate_case(client: HTTPClient, case: dict[str, Any]) -> dict[str, Any]:
    """在独立 Session 中执行单条样本，避免文档在样本间串扰。"""

    started = perf_counter()
    session_response = client.post("/sessions")
    if session_response.status_code != 201:
        return {
            "id": case["id"],
            "passed": False,
            "failures": [f"create_session returned {session_response.status_code}"],
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        }

    session_id = session_response.json()["session_id"]
    upload_status: int | None = None
    try:
        document = case.get("document")
        if document:
            upload_response = client.post(
                f"/sessions/{session_id}/documents",
                files={
                    "file": (
                        document["file_name"],
                        document["content"],
                        document.get("content_type", "text/plain"),
                    )
                },
            )
            upload_status = upload_response.status_code
            if upload_status != 201:
                return {
                    "id": case["id"],
                    "passed": False,
                    "failures": [f"upload_document returned {upload_status}"],
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                }

        request_payload: dict[str, Any] = {"question": case["question"]}
        if case.get("case_id") is not None:
            request_payload["case_id"] = case["case_id"]
        response = client.post(
            f"/sessions/{session_id}/questions",
            json=request_payload,
        )
        payload = response.json()
        failures = _evaluate_expectations(
            response.status_code,
            payload,
            case["expected"],
        )
        return {
            "id": case["id"],
            "description": case.get("description", ""),
            "passed": not failures,
            "failures": failures,
            "status_code": response.status_code,
            "route": payload.get("route"),
            "source_types": sorted(
                {item.get("source_type") for item in payload.get("evidence", [])}
                - {None}
            ),
            "evidence_case_ids": _evidence_case_ids(payload),
            "retrieval_modes": _retrieval_modes(payload),
            "evidence_count": len(payload.get("evidence", [])),
            "answer_nonempty": bool(str(payload.get("answer") or "").strip()),
            "upload_status": upload_status,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        }
    finally:
        client.delete(f"/sessions/{session_id}")


def run_evaluation(
    client: HTTPClient, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """执行全部样本并汇总通过率。"""

    results = [evaluate_case(client, case) for case in cases]
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行第二阶段问答评测集")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="JSONL 评测集路径",
    )
    parser.add_argument(
        "--base-url",
        help="远程 API 地址；不传时在进程内启动 Mock API",
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 报告输出路径")
    return parser


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        f"评测完成：{summary['passed']}/{summary['total']} 通过，"
        f"通过率 {summary['pass_rate']:.1%}"
    )
    for result in report["results"]:
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"[{marker}] {result['id']} ({result['duration_ms']} ms)")
        for failure in result["failures"]:
            print(f"  - {failure}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases = load_cases(args.dataset)
    if args.base_url:
        import httpx

        with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120) as client:
            report = run_evaluation(client, cases)
    else:
        container = build_container(
            QAConfig(api_mode="mock"),
            llm_client=MockLLMClient("模拟回答：已根据给定证据完成作答。"),
        )
        with TestClient(create_app(container)) as client:
            report = run_evaluation(client, cases)

    _print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"报告已写入：{args.output}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

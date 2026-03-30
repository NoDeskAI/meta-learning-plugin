#!/usr/bin/env python3
"""Analyze MCP activation and usage from bridge logs and bench outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean


REGISTER_RE = re.compile(r"registered tool '(mcp_meta_learning_[a-z0-9_]+)'")
CALL_RE = re.compile(r"Tool call:\s*(mcp_meta_learning_[a-z0-9_]+)\(")
TOOL_CALL_ANY_RE = re.compile(r"Tool call:\s*([a-zA-Z0-9_]+)\(")
TASK_ID_RE = re.compile(r"TASK_ID:\s*([a-z0-9\-]+)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_bridge_log(path: Path) -> dict:
    text = _read(path)
    registered = REGISTER_RE.findall(text)
    mcp_calls = CALL_RE.findall(text)
    all_tool_calls = TOOL_CALL_ANY_RE.findall(text)
    task_ids = TASK_ID_RE.findall(text)

    task_counter = Counter(task_ids)
    repeated = {task_id: count for task_id, count in task_counter.items() if count > 1}

    return {
        "bridge_log": str(path),
        "registered_tools": dict(Counter(registered)),
        "registered_total": len(registered),
        "mcp_tool_calls": dict(Counter(mcp_calls)),
        "mcp_tool_call_total": len(mcp_calls),
        "all_tool_call_total": len(all_tool_calls),
        "mcp_tool_call_rate_in_all_calls": (
            len(mcp_calls) / len(all_tool_calls) if all_tool_calls else 0.0
        ),
        "task_dispatch_total": len(task_ids),
        "unique_task_count": len(task_counter),
        "repeated_task_dispatches": repeated,
        "retry_density": (
            len(repeated) / len(task_counter) if task_counter else 0.0
        ),
    }


def _parse_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "exists": False}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    task_results = data.get("task_results", [])
    durations = [float(item.get("duration_s", 0.0)) for item in task_results]
    passed = sum(1 for item in task_results if bool(item.get("passed")))
    total = len(task_results)
    return {
        "summary_path": str(summary_path),
        "exists": True,
        "tasks_total": total,
        "tasks_passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "avg_duration_s": mean(durations) if durations else 0.0,
    }


def _group_report(group: str, bridge_log: Path, result_dir: Path) -> dict:
    bridge = _parse_bridge_log(bridge_log)
    summary = _parse_summary(result_dir / "summary.json")
    connected = bridge["registered_total"] > 0
    activated = bridge["mcp_tool_call_total"] > 0
    return {
        "group": group,
        "bridge": bridge,
        "result": summary,
        "mcp_connected": connected,
        "mcp_activated": activated,
    }


def _activation_gate(report_a: dict, report_b: dict, min_calls: int) -> dict:
    a_calls = int(report_a["bridge"]["mcp_tool_call_total"])
    b_calls = int(report_b["bridge"]["mcp_tool_call_total"])
    a_connected = bool(report_a["mcp_connected"])
    b_connected = bool(report_b["mcp_connected"])

    if not a_connected or not b_connected:
        status = "not_connected"
        reason = "At least one group did not connect to meta-learning MCP."
    elif a_calls + b_calls == 0:
        status = "not_activated"
        reason = "No mcp_meta_learning_* tool call found in either group."
    elif min(a_calls, b_calls) < min_calls:
        status = "low_activation"
        reason = (
            f"At least one group has fewer than {min_calls} MCP calls "
            "and cannot support reliable optimization attribution."
        )
    else:
        status = "activated"
        reason = "Both groups show sufficient MCP call activity for comparison."

    return {
        "status": status,
        "reason": reason,
        "min_calls_required_per_group": min_calls,
        "observed_calls": {"A": a_calls, "B": b_calls},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MCP usage evidence for AB runs")
    parser.add_argument("--bridge-log-a", required=True)
    parser.add_argument("--bridge-log-b", required=True)
    parser.add_argument("--result-dir-a", required=True)
    parser.add_argument("--result-dir-b", required=True)
    parser.add_argument("--min-mcp-calls", type=int, default=3)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    report_a = _group_report(
        "A",
        Path(args.bridge_log_a).expanduser().resolve(),
        Path(args.result_dir_a).expanduser().resolve(),
    )
    report_b = _group_report(
        "B",
        Path(args.bridge_log_b).expanduser().resolve(),
        Path(args.result_dir_b).expanduser().resolve(),
    )
    gate = _activation_gate(report_a, report_b, args.min_mcp_calls)

    final = {
        "schema_version": "1.0",
        "group_reports": {"A": report_a, "B": report_b},
        "activation_gate": gate,
    }
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote MCP usage report: {out_path}")
    print(f"Activation gate: {gate['status']} - {gate['reason']}")


if __name__ == "__main__":
    main()

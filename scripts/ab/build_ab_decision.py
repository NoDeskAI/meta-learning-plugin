#!/usr/bin/env python3
"""Build AB final decision JSON with MCP activation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("scores", {})
    return {
        "path": str(path),
        "pass_rate": float(scores.get("pass_rate", 0.0)) / 100.0,
        "tasks_total": int(scores.get("tasks_total", 0)),
        "tasks_passed": int(scores.get("tasks_passed", 0)),
    }


def _load_usage(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(summary_a: dict, summary_b: dict, usage: dict) -> dict:
    gate = usage.get("activation_gate", {})
    gate_status = gate.get("status", "unknown")
    pass_delta = summary_a["pass_rate"] - summary_b["pass_rate"]

    if gate_status != "activated":
        verdict = "insufficient_evidence"
        rationale = (
            "MCP activation gate is not satisfied; do not attribute A/B difference "
            "to meta-learning optimization."
        )
    elif pass_delta > 0:
        verdict = "prefer_A"
        rationale = "A has higher pass rate under activated MCP conditions."
    elif pass_delta < 0:
        verdict = "prefer_B"
        rationale = "B has higher pass rate under activated MCP conditions."
    else:
        verdict = "no_material_difference"
        rationale = "A and B have no pass-rate difference under activated conditions."

    return {
        "verdict": verdict,
        "rationale": rationale,
        "pass_rate_delta_A_minus_B": pass_delta,
        "activation_gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AB decision with MCP activation gate")
    parser.add_argument("--summary-a", required=True)
    parser.add_argument("--summary-b", required=True)
    parser.add_argument("--mcp-usage-report", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    summary_a = _load_summary(Path(args.summary_a).expanduser().resolve())
    summary_b = _load_summary(Path(args.summary_b).expanduser().resolve())
    usage = _load_usage(Path(args.mcp_usage_report).expanduser().resolve())
    decision = _decision(summary_a, summary_b, usage)

    out = {
        "schema_version": "1.0",
        "inputs": {
            "summary_a": summary_a,
            "summary_b": summary_b,
            "mcp_usage_report_path": args.mcp_usage_report,
        },
        "decision": decision,
    }
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote AB decision: {out_path}")
    print(f"Verdict: {decision['verdict']}")


if __name__ == "__main__":
    main()

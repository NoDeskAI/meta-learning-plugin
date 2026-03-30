#!/usr/bin/env python3
"""Analyze OpenClaw-Test A/B experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openclaw_ab_lib import (  # noqa: E402
    ActivationGate,
    RoundMetrics,
    StageAggregate,
    as_json,
    build_activation_gate,
    build_decision,
    build_effect_gate,
    dataclass_to_dict,
    parse_bridge_usage,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rounds(path: Path) -> list[RoundMetrics]:
    rows = _load_json(path)
    return [RoundMetrics(**row) for row in rows]


def _aggregate(rounds: list[RoundMetrics]) -> dict[str, dict[str, StageAggregate]]:
    by_stage: dict[str, dict[str, StageAggregate]] = {}
    stage_names = sorted({r.stage for r in rounds})
    for stage in stage_names:
        selected = [r for r in rounds if r.stage == stage]
        by_group: dict[str, StageAggregate] = {}
        for group in ["A", "B"]:
            rows = [r for r in selected if r.group == group]
            if not rows:
                by_group[group] = StageAggregate(
                    group=group,
                    stage=stage,
                    rounds=0,
                    mean_student_rate=0.0,
                    mean_teacher_rate=0.0,
                    mean_e2e_rate=0.0,
                    mean_duration_per_problem_s=0.0,
                    e2e_rate_slope=0.0,
                    duration_per_problem_s_slope=0.0,
                )
                continue
            by_group[group] = StageAggregate(
                group=group,
                stage=stage,
                rounds=len(rows),
                mean_student_rate=sum(x.student_rate for x in rows) / len(rows),
                mean_teacher_rate=sum(x.teacher_rate for x in rows) / len(rows),
                mean_e2e_rate=sum(x.e2e_rate for x in rows) / len(rows),
                mean_duration_per_problem_s=(
                    sum(x.duration_per_problem_s for x in rows) / len(rows)
                ),
                e2e_rate_slope=_linear_slope([x.e2e_rate for x in rows]),
                duration_per_problem_s_slope=_linear_slope(
                    [x.duration_per_problem_s for x in rows]
                ),
            )
        by_stage[stage] = by_group
    return by_stage


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    xs = [float(i + 1) for i in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


def _render_md(
    *,
    result_root: Path,
    stage_report: dict[str, dict[str, StageAggregate]],
    activation: ActivationGate,
    effect_gate: dict[str, Any],
    decision: dict[str, Any],
    effect_stage: str,
) -> str:
    lines = [
        "# OpenClaw-Test A/B Trend Report",
        "",
        f"- result_root: `{result_root}`",
        "",
        "## Stage Metrics",
    ]
    for stage_name in sorted(stage_report):
        a = stage_report[stage_name]["A"]
        b = stage_report[stage_name]["B"]
        lines.extend(
            [
                f"### {stage_name}",
                (
                    f"- A: e2e={a.mean_e2e_rate:.4f}, duration={a.mean_duration_per_problem_s:.2f}s, "
                    f"e2e_slope={a.e2e_rate_slope:.6f}, duration_slope={a.duration_per_problem_s_slope:.6f}"
                ),
                (
                    f"- B: e2e={b.mean_e2e_rate:.4f}, duration={b.mean_duration_per_problem_s:.2f}s, "
                    f"e2e_slope={b.e2e_rate_slope:.6f}, duration_slope={b.duration_per_problem_s_slope:.6f}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Gates",
            f"- activation: {activation.status} ({activation.reason})",
            (
                f"- effect ({effect_stage}): {effect_gate['status']}, "
                f"success_delta={effect_gate['observed']['success_delta_A_minus_B']:.4f}, "
                f"efficiency_gain_s={effect_gate['observed']['efficiency_gain_s_B_minus_A']:.2f}"
            ),
            "",
            "## Decision",
            f"- verdict: `{decision['verdict']}`",
            f"- rationale: {decision['rationale']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OpenClaw-Test A/B result root")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--bridge-log-a", default="")
    parser.add_argument("--bridge-log-b", default="")
    parser.add_argument("--min-mcp-calls", type=int, default=3)
    parser.add_argument("--min-success-delta", type=float, default=0.03)
    parser.add_argument("--min-efficiency-gain-s", type=float, default=5.0)
    parser.add_argument("--effect-stage", default="formal")
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    rounds_path = result_root / "rounds.json"
    manifest_path = result_root / "manifest.json"
    if not rounds_path.exists():
        raise RuntimeError(f"rounds.json not found: {rounds_path}")
    rounds = _rounds(rounds_path)
    stage_report = _aggregate(rounds)
    if not stage_report:
        raise RuntimeError("No round metrics found in rounds.json")

    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {}

    bridge_log_a = args.bridge_log_a.strip() or manifest.get("bridge_logs", {}).get("A", "")
    bridge_log_b = args.bridge_log_b.strip() or manifest.get("bridge_logs", {}).get("B", "")
    usage_a = parse_bridge_usage(bridge_log_a) if bridge_log_a else parse_bridge_usage("/dev/null.missing")
    usage_b = parse_bridge_usage(bridge_log_b) if bridge_log_b else parse_bridge_usage("/dev/null.missing")
    activation = build_activation_gate(
        usage_a=usage_a,
        usage_b=usage_b,
        min_calls_per_group=args.min_mcp_calls,
    )

    effect_stage = args.effect_stage.strip()
    if effect_stage not in stage_report:
        effect_stage = sorted(stage_report.keys())[-1]
    formal_a = stage_report[effect_stage]["A"]
    formal_b = stage_report[effect_stage]["B"]
    effect_gate = build_effect_gate(
        formal_a=formal_a,
        formal_b=formal_b,
        min_success_delta=args.min_success_delta,
        min_efficiency_gain_s=args.min_efficiency_gain_s,
    )
    decision = build_decision(
        activation_gate=activation,
        effect_gate=effect_gate,
    )

    out = {
        "manifest": manifest,
        "stage_report": {
            stage: {k: dataclass_to_dict(v) for k, v in groups.items()}
            for stage, groups in stage_report.items()
        },
        "usage": {"A": dataclass_to_dict(usage_a), "B": dataclass_to_dict(usage_b)},
        "activation_gate": dataclass_to_dict(activation),
        "effect_gate": effect_gate,
        "decision": decision,
    }
    (result_root / "report.json").write_text(as_json(out), encoding="utf-8")
    (result_root / "decision.json").write_text(as_json(decision), encoding="utf-8")
    (result_root / "report.md").write_text(
        _render_md(
            result_root=result_root,
            stage_report=stage_report,
            activation=activation,
            effect_gate=effect_gate,
            decision=decision,
            effect_stage=effect_stage,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {result_root / 'report.json'}")
    print(f"Wrote: {result_root / 'report.md'}")
    print(f"Wrote: {result_root / 'decision.json'}")


if __name__ == "__main__":
    main()

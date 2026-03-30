#!/usr/bin/env python3
"""Shared metrics and decision logic for OpenClaw-Test A/B evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STUDENT_SUMMARY_RE = re.compile(
    r"# Summary:\s*(\d+)\s*/\s*(\d+)\s*problems completed within turn limit",
    re.IGNORECASE,
)
TEACHER_SUMMARY_RE = re.compile(
    r"# Summary:\s*(\d+)\s*/\s*(\d+)\s*problems graded within turn limit",
    re.IGNORECASE,
)

REGISTER_RE = re.compile(r"registered tool '(mcp_meta_learning_[a-z0-9_]+)'")
MCP_CALL_RE = re.compile(r"Tool call:\s*(mcp_meta_learning_[a-zA-Z0-9_]+)\(")
ANY_CALL_RE = re.compile(r"Tool call:\s*([a-zA-Z0-9_]+)\(")


@dataclass(frozen=True)
class RoundMetrics:
    stage: str
    group: str
    round_index: int
    num_problems: int
    max_turns: int
    student_done: int
    student_total: int
    teacher_done: int
    teacher_total: int
    e2e_done: int
    e2e_total: int
    student_rate: float
    teacher_rate: float
    e2e_rate: float
    student_duration_s: float
    teacher_duration_s: float
    total_duration_s: float
    duration_per_problem_s: float


@dataclass(frozen=True)
class StageAggregate:
    group: str
    stage: str
    rounds: int
    mean_student_rate: float
    mean_teacher_rate: float
    mean_e2e_rate: float
    mean_duration_per_problem_s: float
    e2e_rate_slope: float
    duration_per_problem_s_slope: float


@dataclass(frozen=True)
class BridgeUsage:
    bridge_log: str
    exists: bool
    registered_total: int
    mcp_tool_call_total: int
    all_tool_call_total: int
    mcp_tool_call_rate_in_all_calls: float
    registered_tools: dict[str, int]
    mcp_tool_calls: dict[str, int]


@dataclass(frozen=True)
class ActivationGate:
    status: str
    reason: str
    min_calls_required_per_group: int
    observed_calls: dict[str, int]


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    xs = [float(i + 1) for i in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    return (num / den) if den else 0.0


def _extract_summary(text: str, *, is_student: bool) -> tuple[int, int]:
    pattern = STUDENT_SUMMARY_RE if is_student else TEACHER_SUMMARY_RE
    matches = pattern.findall(text)
    if not matches:
        phase = "student" if is_student else "teacher"
        raise RuntimeError(f"Cannot parse {phase} summary from log output.")
    done, total = matches[-1]
    return int(done), int(total)


def parse_round_metrics(
    *,
    stage: str,
    group: str,
    round_index: int,
    num_problems: int,
    max_turns: int,
    student_log_text: str,
    teacher_log_text: str,
    student_duration_s: float,
    teacher_duration_s: float,
) -> RoundMetrics:
    student_done, student_total = _extract_summary(student_log_text, is_student=True)
    teacher_done, teacher_total = _extract_summary(teacher_log_text, is_student=False)
    e2e_total = min(student_total, teacher_total)
    e2e_done = min(student_done, teacher_done)
    e2e_rate = (e2e_done / e2e_total) if e2e_total else 0.0
    student_rate = (student_done / student_total) if student_total else 0.0
    teacher_rate = (teacher_done / teacher_total) if teacher_total else 0.0
    total_duration_s = student_duration_s + teacher_duration_s
    duration_per_problem_s = total_duration_s / max(e2e_total, 1)
    return RoundMetrics(
        stage=stage,
        group=group,
        round_index=round_index,
        num_problems=num_problems,
        max_turns=max_turns,
        student_done=student_done,
        student_total=student_total,
        teacher_done=teacher_done,
        teacher_total=teacher_total,
        e2e_done=e2e_done,
        e2e_total=e2e_total,
        student_rate=student_rate,
        teacher_rate=teacher_rate,
        e2e_rate=e2e_rate,
        student_duration_s=student_duration_s,
        teacher_duration_s=teacher_duration_s,
        total_duration_s=total_duration_s,
        duration_per_problem_s=duration_per_problem_s,
    )


def aggregate_stage(
    rounds: list[RoundMetrics],
    *,
    group: str,
    stage: str,
) -> StageAggregate:
    selected = [r for r in rounds if r.group == group and r.stage == stage]
    if not selected:
        return StageAggregate(
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
    student_rates = [r.student_rate for r in selected]
    teacher_rates = [r.teacher_rate for r in selected]
    e2e_rates = [r.e2e_rate for r in selected]
    durations = [r.duration_per_problem_s for r in selected]
    return StageAggregate(
        group=group,
        stage=stage,
        rounds=len(selected),
        mean_student_rate=sum(student_rates) / len(student_rates),
        mean_teacher_rate=sum(teacher_rates) / len(teacher_rates),
        mean_e2e_rate=sum(e2e_rates) / len(e2e_rates),
        mean_duration_per_problem_s=sum(durations) / len(durations),
        e2e_rate_slope=_linear_slope(e2e_rates),
        duration_per_problem_s_slope=_linear_slope(durations),
    )


def parse_bridge_usage(bridge_log_path: str) -> BridgeUsage:
    path = Path(bridge_log_path).expanduser().resolve()
    if not path.exists():
        return BridgeUsage(
            bridge_log=str(path),
            exists=False,
            registered_total=0,
            mcp_tool_call_total=0,
            all_tool_call_total=0,
            mcp_tool_call_rate_in_all_calls=0.0,
            registered_tools={},
            mcp_tool_calls={},
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    registered = REGISTER_RE.findall(text)
    mcp_calls = MCP_CALL_RE.findall(text)
    all_calls = ANY_CALL_RE.findall(text)
    return BridgeUsage(
        bridge_log=str(path),
        exists=True,
        registered_total=len(registered),
        mcp_tool_call_total=len(mcp_calls),
        all_tool_call_total=len(all_calls),
        mcp_tool_call_rate_in_all_calls=(
            (len(mcp_calls) / len(all_calls)) if all_calls else 0.0
        ),
        registered_tools=dict(Counter(registered)),
        mcp_tool_calls=dict(Counter(mcp_calls)),
    )


def build_activation_gate(
    *,
    usage_a: BridgeUsage,
    usage_b: BridgeUsage,
    min_calls_per_group: int,
) -> ActivationGate:
    a_calls = usage_a.mcp_tool_call_total
    b_calls = usage_b.mcp_tool_call_total
    if not usage_a.exists or not usage_b.exists:
        return ActivationGate(
            status="not_connected",
            reason="At least one group has no bridge log evidence.",
            min_calls_required_per_group=min_calls_per_group,
            observed_calls={"A": a_calls, "B": b_calls},
        )
    if a_calls + b_calls == 0:
        return ActivationGate(
            status="not_activated",
            reason="No mcp_meta_learning_* tool call found in either group.",
            min_calls_required_per_group=min_calls_per_group,
            observed_calls={"A": a_calls, "B": b_calls},
        )
    if min(a_calls, b_calls) < min_calls_per_group:
        return ActivationGate(
            status="low_activation",
            reason=(
                f"At least one group has fewer than {min_calls_per_group} MCP calls, "
                "cannot support reliable attribution."
            ),
            min_calls_required_per_group=min_calls_per_group,
            observed_calls={"A": a_calls, "B": b_calls},
        )
    return ActivationGate(
        status="activated",
        reason="Both groups show sufficient MCP call activity.",
        min_calls_required_per_group=min_calls_per_group,
        observed_calls={"A": a_calls, "B": b_calls},
    )


def build_effect_gate(
    *,
    formal_a: StageAggregate,
    formal_b: StageAggregate,
    min_success_delta: float,
    min_efficiency_gain_s: float,
) -> dict[str, Any]:
    success_delta = formal_a.mean_e2e_rate - formal_b.mean_e2e_rate
    efficiency_gain = (
        formal_b.mean_duration_per_problem_s - formal_a.mean_duration_per_problem_s
    )
    success_ok = success_delta >= min_success_delta
    efficiency_ok = efficiency_gain >= min_efficiency_gain_s
    return {
        "status": "pass" if (success_ok or efficiency_ok) else "fail",
        "reason": (
            "A reaches predefined gain threshold in success rate or efficiency."
            if (success_ok or efficiency_ok)
            else "A does not reach predefined gain threshold."
        ),
        "thresholds": {
            "min_success_delta": min_success_delta,
            "min_efficiency_gain_s": min_efficiency_gain_s,
        },
        "observed": {
            "success_delta_A_minus_B": success_delta,
            "efficiency_gain_s_B_minus_A": efficiency_gain,
            "success_ok": success_ok,
            "efficiency_ok": efficiency_ok,
        },
    }


def build_decision(
    *,
    activation_gate: ActivationGate,
    effect_gate: dict[str, Any],
) -> dict[str, Any]:
    if activation_gate.status != "activated":
        verdict = "insufficient_evidence"
        rationale = (
            "Activation gate is not satisfied; do not attribute differences to "
            "meta-learning."
        )
    elif effect_gate.get("status") == "pass":
        verdict = "prefer_A"
        rationale = "A passes effect gate under activated conditions."
    else:
        verdict = "no_material_difference"
        rationale = "Activation is sufficient but A does not meet effect threshold."
    return {
        "verdict": verdict,
        "rationale": rationale,
        "activation_gate": asdict(activation_gate),
        "effect_gate": effect_gate,
    }


def as_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def dataclass_to_dict(item: Any) -> dict[str, Any]:
    return asdict(item)

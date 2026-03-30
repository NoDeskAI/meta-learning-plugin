#!/usr/bin/env python3
"""Run OpenClaw-Test A/B experiment with smoke + formal stages."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openclaw_ab_lib import (  # noqa: E402
    ActivationGate,
    BridgeUsage,
    RoundMetrics,
    StageAggregate,
    aggregate_stage,
    as_json,
    build_activation_gate,
    build_decision,
    build_effect_gate,
    dataclass_to_dict,
    parse_bridge_usage,
    parse_round_metrics,
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    num_problems: int
    rounds: int
    max_turns: int


def _load_secret(preferred: str | None, env_names: list[str], *, name: str) -> str:
    if preferred and preferred.strip():
        return preferred.strip()
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    joined = ", ".join(env_names)
    raise RuntimeError(f"Missing {name}. Set argument or env: {joined}")


def _run_and_log(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: int,
) -> float:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.perf_counter() - started
    combined = (proc.stdout or "") + ("\n" if proc.stdout else "") + (proc.stderr or "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}; log={log_path}"
        )
    return elapsed


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _stage_sequence(args: argparse.Namespace) -> list[StageSpec]:
    stages: list[StageSpec] = []
    if args.phase in {"smoke", "all"}:
        stages.append(
            StageSpec(
                name="smoke",
                num_problems=args.smoke_problems,
                rounds=args.smoke_rounds,
                max_turns=args.max_turns,
            )
        )
    if args.phase in {"formal", "all"}:
        stages.append(
            StageSpec(
                name="formal",
                num_problems=args.formal_problems,
                rounds=args.formal_rounds,
                max_turns=args.max_turns,
            )
        )
    return stages


def _group_env(
    *,
    base_env: dict[str, str],
    gateway_url: str,
    gateway_token: str,
    workspace: Path,
    openai_api_key: str,
    openai_base_url: str,
    external_model: str,
) -> dict[str, str]:
    env = dict(base_env)
    env["OPENCLAW_GATEWAY_URL"] = gateway_url
    env["OPENCLAW_GATEWAY_TOKEN"] = gateway_token
    env["OPENCLAW_WORKSPACE"] = str(workspace)
    env["OPENAI_API_KEY"] = openai_api_key
    if openai_base_url:
        env["OPENAI_BASE_URL"] = openai_base_url
    env["EXTERNAL_MODEL"] = external_model
    return env


def _usage_or_empty(path: str | None) -> BridgeUsage:
    if not path or not path.strip():
        return BridgeUsage(
            bridge_log="",
            exists=False,
            registered_total=0,
            mcp_tool_call_total=0,
            all_tool_call_total=0,
            mcp_tool_call_rate_in_all_calls=0.0,
            registered_tools={},
            mcp_tool_calls={},
        )
    return parse_bridge_usage(path.strip())


def _aggregate_all(rounds: list[RoundMetrics], stages: list[StageSpec]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage in stages:
        agg_a = aggregate_stage(rounds, group="A", stage=stage.name)
        agg_b = aggregate_stage(rounds, group="B", stage=stage.name)
        out[stage.name] = {
            "A": dataclass_to_dict(agg_a),
            "B": dataclass_to_dict(agg_b),
        }
    return out


def _pick_effect_stage(stages: list[StageSpec]) -> str:
    names = [s.name for s in stages]
    return "formal" if "formal" in names else names[-1]


def _render_md(
    *,
    run_id: str,
    result_root: Path,
    stage_report: dict[str, Any],
    activation_gate: ActivationGate,
    effect_gate: dict[str, Any],
    decision: dict[str, Any],
    effect_stage: str,
) -> str:
    lines = [
        "# OpenClaw-Test A/B Report",
        "",
        f"- run_id: `{run_id}`",
        f"- result_root: `{result_root}`",
        "",
        "## Stage Metrics",
    ]
    for stage_name, stage_data in stage_report.items():
        a = stage_data["A"]
        b = stage_data["B"]
        lines.extend(
            [
                f"### {stage_name}",
                (
                    f"- A: mean_e2e_rate={a['mean_e2e_rate']:.4f}, "
                    f"mean_duration_per_problem_s={a['mean_duration_per_problem_s']:.2f}, "
                    f"e2e_rate_slope={a['e2e_rate_slope']:.6f}, "
                    f"duration_slope={a['duration_per_problem_s_slope']:.6f}"
                ),
                (
                    f"- B: mean_e2e_rate={b['mean_e2e_rate']:.4f}, "
                    f"mean_duration_per_problem_s={b['mean_duration_per_problem_s']:.2f}, "
                    f"e2e_rate_slope={b['e2e_rate_slope']:.6f}, "
                    f"duration_slope={b['duration_per_problem_s_slope']:.6f}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Gates",
            (
                f"- activation_gate: {activation_gate.status} "
                f"(A calls={activation_gate.observed_calls.get('A', 0)}, "
                f"B calls={activation_gate.observed_calls.get('B', 0)})"
            ),
            f"- activation_reason: {activation_gate.reason}",
            (
                f"- effect_stage: `{effect_stage}`; effect_gate={effect_gate['status']} "
                f"(success_delta={effect_gate['observed']['success_delta_A_minus_B']:.4f}, "
                f"efficiency_gain_s={effect_gate['observed']['efficiency_gain_s_B_minus_A']:.2f})"
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
    parser = argparse.ArgumentParser(description="Run OpenClaw-Test A/B evaluation")
    parser.add_argument("--openclaw-test-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--result-root", default="")
    parser.add_argument("--phase", choices=["smoke", "formal", "all"], default="all")
    parser.add_argument("--smoke-problems", type=int, default=10)
    parser.add_argument("--formal-problems", type=int, default=80)
    parser.add_argument("--smoke-rounds", type=int, default=1)
    parser.add_argument("--formal-rounds", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--gateway-url-a", default=os.environ.get("OPENCLAW_GATEWAY_URL_A", ""))
    parser.add_argument("--gateway-url-b", default=os.environ.get("OPENCLAW_GATEWAY_URL_B", ""))
    parser.add_argument("--gateway-token-a", default="")
    parser.add_argument("--gateway-token-b", default="")
    parser.add_argument("--workspace-a", default="")
    parser.add_argument("--workspace-b", default="")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--external-model", default=os.environ.get("EXTERNAL_MODEL", "gpt-4o"))
    parser.add_argument("--bridge-log-a", default="")
    parser.add_argument("--bridge-log-b", default="")
    parser.add_argument("--min-mcp-calls", type=int, default=3)
    parser.add_argument("--min-success-delta", type=float, default=0.03)
    parser.add_argument("--min-efficiency-gain-s", type=float, default=5.0)
    parser.add_argument("--student-timeout-s", type=int, default=7200)
    parser.add_argument("--teacher-timeout-s", type=int, default=7200)
    parser.add_argument("--sleep-between-rounds-s", type=float, default=0.0)
    args = parser.parse_args()

    openclaw_root = Path(args.openclaw_test_root).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not openclaw_root.exists():
        raise RuntimeError(f"openclaw-test root not found: {openclaw_root}")
    if not dataset_path.exists():
        raise RuntimeError(f"dataset not found: {dataset_path}")

    run_id = args.run_id.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.result_root.strip():
        result_root = Path(args.result_root).expanduser().resolve()
    else:
        result_root = (
            Path("/Users/yumeng/Documents/Projects/lingmin-meta-learning/abtest/results")
            / f"openclaw_ab_{run_id}"
        )
    if result_root.exists():
        raise RuntimeError(f"Result root already exists: {result_root}")
    result_root.mkdir(parents=True, exist_ok=False)

    stages = _stage_sequence(args)
    if not stages:
        raise RuntimeError("No stage selected.")

    base_workspace = Path(f"/tmp/openclaw_ab/{run_id}").resolve()
    workspace_a = (
        Path(args.workspace_a).expanduser().resolve()
        if args.workspace_a.strip()
        else (base_workspace / "A" / "workspace")
    )
    workspace_b = (
        Path(args.workspace_b).expanduser().resolve()
        if args.workspace_b.strip()
        else (base_workspace / "B" / "workspace")
    )
    workspace_a.mkdir(parents=True, exist_ok=True)
    workspace_b.mkdir(parents=True, exist_ok=True)

    gateway_url_a = _load_secret(
        args.gateway_url_a,
        ["OPENCLAW_GATEWAY_URL_A", "OPENCLAW_GATEWAY_URL"],
        name="gateway_url_a",
    )
    gateway_url_b = _load_secret(
        args.gateway_url_b,
        ["OPENCLAW_GATEWAY_URL_B", "OPENCLAW_GATEWAY_URL"],
        name="gateway_url_b",
    )
    gateway_token_a = _load_secret(
        args.gateway_token_a,
        ["OPENCLAW_GATEWAY_TOKEN_A", "OPENCLAW_GATEWAY_TOKEN"],
        name="gateway_token_a",
    )
    gateway_token_b = _load_secret(
        args.gateway_token_b,
        ["OPENCLAW_GATEWAY_TOKEN_B", "OPENCLAW_GATEWAY_TOKEN"],
        name="gateway_token_b",
    )
    openai_api_key = _load_secret(
        args.openai_api_key,
        ["OPENAI_API_KEY"],
        name="openai_api_key",
    )

    base_env = dict(os.environ)
    env_a = _group_env(
        base_env=base_env,
        gateway_url=gateway_url_a,
        gateway_token=gateway_token_a,
        workspace=workspace_a,
        openai_api_key=openai_api_key,
        openai_base_url=args.openai_base_url.strip(),
        external_model=args.external_model.strip(),
    )
    env_b = _group_env(
        base_env=base_env,
        gateway_url=gateway_url_b,
        gateway_token=gateway_token_b,
        workspace=workspace_b,
        openai_api_key=openai_api_key,
        openai_base_url=args.openai_base_url.strip(),
        external_model=args.external_model.strip(),
    )

    rounds: list[RoundMetrics] = []
    for stage in stages:
        for round_index in range(1, stage.rounds + 1):
            for group, env in [("A", env_a), ("B", env_b)]:
                round_dir = result_root / stage.name / group / f"round_{round_index:02d}"
                student_log = round_dir / "student.log"
                teacher_log = round_dir / "teacher.log"
                round_env = dict(env)
                round_env["OPENCLAW_AB_ROUND_INDEX"] = str(round_index)
                round_env["MOCK_ROUND_INDEX"] = str(round_index)

                student_cmd = [
                    sys.executable,
                    str(openclaw_root / "student_chat.py"),
                    "--dataset",
                    str(dataset_path),
                    "--num-problems",
                    str(stage.num_problems),
                    "--max-turns",
                    str(stage.max_turns),
                ]
                teacher_cmd = [
                    sys.executable,
                    str(openclaw_root / "teacher_chat.py"),
                    "--dataset",
                    str(dataset_path),
                    "--num-problems",
                    str(stage.num_problems),
                    "--max-turns",
                    str(stage.max_turns),
                ]

                print(
                    f"[run] stage={stage.name} group={group} round={round_index} "
                    f"num_problems={stage.num_problems}"
                )
                student_duration_s = _run_and_log(
                    cmd=student_cmd,
                    cwd=openclaw_root,
                    env=round_env,
                    log_path=student_log,
                    timeout_s=args.student_timeout_s,
                )
                teacher_duration_s = _run_and_log(
                    cmd=teacher_cmd,
                    cwd=openclaw_root,
                    env=round_env,
                    log_path=teacher_log,
                    timeout_s=args.teacher_timeout_s,
                )
                row = parse_round_metrics(
                    stage=stage.name,
                    group=group,
                    round_index=round_index,
                    num_problems=stage.num_problems,
                    max_turns=stage.max_turns,
                    student_log_text=_read_text(student_log),
                    teacher_log_text=_read_text(teacher_log),
                    student_duration_s=student_duration_s,
                    teacher_duration_s=teacher_duration_s,
                )
                rounds.append(row)
            if args.sleep_between_rounds_s > 0:
                time.sleep(args.sleep_between_rounds_s)

    stage_report = _aggregate_all(rounds, stages)
    usage_a = _usage_or_empty(args.bridge_log_a)
    usage_b = _usage_or_empty(args.bridge_log_b)
    activation_gate = build_activation_gate(
        usage_a=usage_a,
        usage_b=usage_b,
        min_calls_per_group=args.min_mcp_calls,
    )
    effect_stage = _pick_effect_stage(stages)
    formal_a = StageAggregate(**stage_report[effect_stage]["A"])
    formal_b = StageAggregate(**stage_report[effect_stage]["B"])
    effect_gate = build_effect_gate(
        formal_a=formal_a,
        formal_b=formal_b,
        min_success_delta=args.min_success_delta,
        min_efficiency_gain_s=args.min_efficiency_gain_s,
    )
    decision = build_decision(
        activation_gate=activation_gate,
        effect_gate=effect_gate,
    )

    manifest = {
        "run_id": run_id,
        "result_root": str(result_root),
        "openclaw_test_root": str(openclaw_root),
        "dataset": str(dataset_path),
        "stages": [asdict(s) for s in stages],
        "workspaces": {"A": str(workspace_a), "B": str(workspace_b)},
        "bridge_logs": {"A": args.bridge_log_a.strip(), "B": args.bridge_log_b.strip()},
        "thresholds": {
            "min_mcp_calls": args.min_mcp_calls,
            "min_success_delta": args.min_success_delta,
            "min_efficiency_gain_s": args.min_efficiency_gain_s,
            "effect_stage": effect_stage,
        },
    }
    report = {
        "manifest": manifest,
        "stage_report": stage_report,
        "usage": {"A": dataclass_to_dict(usage_a), "B": dataclass_to_dict(usage_b)},
        "activation_gate": dataclass_to_dict(activation_gate),
        "effect_gate": effect_gate,
        "decision": decision,
    }

    (result_root / "manifest.json").write_text(as_json(manifest), encoding="utf-8")
    (result_root / "rounds.json").write_text(
        json.dumps([dataclass_to_dict(r) for r in rounds], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_root / "report.json").write_text(as_json(report), encoding="utf-8")
    (result_root / "decision.json").write_text(as_json(decision), encoding="utf-8")
    (result_root / "report.md").write_text(
        _render_md(
            run_id=run_id,
            result_root=result_root,
            stage_report=stage_report,
            activation_gate=activation_gate,
            effect_gate=effect_gate,
            decision=decision,
            effect_stage=effect_stage,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote result to: {result_root}")


if __name__ == "__main__":
    main()

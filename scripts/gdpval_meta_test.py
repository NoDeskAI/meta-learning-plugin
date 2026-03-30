#!/usr/bin/env python3
"""GDPVal Meta-Learning Validation Runner.

Runs two experiment modes:
  1. within-task: Same task repeated N rounds; meta-learning accumulates between rounds.
  2. cross-task: Train on tasks from one occupation, inject taxonomy into tasks from
     a different (but related) occupation.

Uses a minimal tool-calling agent (AsyncOpenAI + shell) instead of OpenSpace,
so the ONLY variable between control/treatment is meta-learning taxonomy injection.

Usage:
    python scripts/gdpval_meta_test.py --mode within-task --rounds 3
    python scripts/gdpval_meta_test.py --mode cross-task --pairs "data_analyst:financial_analyst"

Requires:
    - DASHSCOPE_API_KEY env var (Alibaba DashScope / 百炼)
    - ClawWork repo at expected path (for task loading + evaluation)
"""

from __future__ import annotations

import os as _os
for _pv in ("http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    _os.environ.pop(_pv, None)

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OPENSPACE_ROOT = Path(os.environ.get(
    "OPENSPACE_ROOT",
    str(PROJECT_ROOT.parent / "OpenSpace"),
))
CLAWWORK_ROOT = Path(os.environ.get(
    "CLAWWORK_ROOT",
    str(PROJECT_ROOT.parent / "Benchmarks" / "ClawWork"),
))

for p in [str(OPENSPACE_ROOT), str(CLAWWORK_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gdpval_meta_test")

RESULTS_BASE = PROJECT_ROOT / "abtest" / "results" / "gdpval_meta"
CONFIG_PATH = PROJECT_ROOT / "abtest" / "config.gdpval.yaml"

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
NODESK_GATEWAY = "https://llm-gateway-api.nodesk.tech/default/v1"
AGENT_MODEL = "qwen3.5-plus"
EVAL_MODEL = "openai/gpt-4o"

AGENT_SYSTEM_PROMPT = """You are an expert professional completing a work task.
You have access to a shell tool to execute commands in the workspace directory.
Create all required deliverable files in the current working directory.
Work systematically: understand requirements, create files, verify output.
When finished, stop calling tools."""

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a shell command in the workspace directory. Use this to create files, run scripts, inspect the filesystem, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def _ensure_env():
    """Set up environment variables for LLM calls.

    Agent execution: DashScope (DASHSCOPE_API_KEY)
    Evaluation: nodesk gateway (NODESK_API_KEY) with gpt-4o
    Meta-learning Layer2: DashScope (same as agent)
    """
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        sys.exit("ERROR: Set DASHSCOPE_API_KEY")

    nodesk_key = os.environ.get("NODESK_API_KEY", "")
    if not nodesk_key:
        sys.exit("ERROR: Set NODESK_API_KEY (for evaluation via gpt-4o)")

    os.environ.setdefault("META_LEARNING_LLM_API_KEY", dashscope_key)
    os.environ.setdefault("META_LEARNING_LLM_BASE_URL", DASHSCOPE_BASE)
    os.environ.setdefault("OPENAI_API_KEY", dashscope_key)
    os.environ.setdefault("OPENAI_API_BASE", DASHSCOPE_BASE)

    os.environ["EVALUATION_API_KEY"] = nodesk_key
    os.environ["EVALUATION_API_BASE"] = NODESK_GATEWAY
    os.environ["EVALUATION_MODEL"] = EVAL_MODEL

    os.environ.setdefault("META_LEARNING_CONFIG", str(CONFIG_PATH))


def _get_agent_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=DASHSCOPE_BASE,
    )


def _run_shell(command: str, cwd: str, timeout: int = 60) -> str:
    """Execute a shell command and return stdout+stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output[:8000] if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as e:
        return f"[shell error: {e}]"


async def _execute_task(
    task: dict[str, Any],
    workspace_dir: str,
    taxonomy_text: str | None = None,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Execute a single GDPVal task via minimal tool-calling agent."""
    augmented_prompt = _prepare_workspace(task, workspace_dir)

    system_prompt = AGENT_SYSTEM_PROMPT
    if taxonomy_text:
        system_prompt += f"\n\n{taxonomy_text}"

    client = _get_agent_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_prompt},
    ]

    iterations = 0
    tool_executions = 0
    tools_used: list[str] = []

    for _ in range(max_iterations):
        iterations += 1
        try:
            response = await client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=[SHELL_TOOL],
                extra_body={"enable_thinking": False},
            )
        except Exception as e:
            logger.error("LLM call failed at iteration %d: %s", iterations, e)
            break

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            logger.info("    Agent finished (no tool calls) at iteration %d", iterations)
            break

        for tc in msg.tool_calls:
            tool_executions += 1
            args = json.loads(tc.function.arguments)
            command = args.get("command", "")
            tools_used.append("shell")

            logger.info("    [iter %d] shell: %s", iterations, command[:120])
            output = _run_shell(command, cwd=workspace_dir)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

    return {
        "status": "completed" if iterations < max_iterations else "max_iterations",
        "iterations": iterations,
        "tool_executions": tool_executions,
        "messages": messages,
        "tools_used": tools_used,
    }


# ---------------------------------------------------------------------------
# Task loading, evaluation, meta-learning (unchanged from original)
# ---------------------------------------------------------------------------

def _load_tasks(
    occupations: list[str] | None = None,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    from gdpval_bench.task_loader import load_tasks
    return load_tasks(
        clawwork_root=str(CLAWWORK_ROOT),
        occupations=occupations,
        task_ids=task_ids,
        max_tasks=max_tasks,
    )


def _prepare_workspace(task: dict[str, Any], workspace_dir: str) -> str:
    from gdpval_bench.task_loader import prepare_task_workspace
    return prepare_task_workspace(task, workspace_dir)


def _evaluate(task: dict[str, Any], workspace_dir: str) -> dict[str, Any]:
    from gdpval_bench.run_benchmark import _evaluate_task
    return _evaluate_task(task, workspace_dir, {"clawwork_root": str(CLAWWORK_ROOT)})


def _capture_signal(
    task: dict[str, Any],
    evaluation: dict[str, Any],
    execution: dict[str, Any],
    tool_calls: list[str] | None = None,
):
    """Capture a meta-learning signal from GDPVal evaluation results."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from gdpval_signal_adapter import build_signal_from_gdpval, build_session_from_execution

    from meta_learning.shared.io import load_config, ensure_directories
    config = load_config(CONFIG_PATH)
    ensure_directories(config)

    signal_kwargs = build_signal_from_gdpval(
        task=task,
        evaluation=evaluation,
        execution=execution,
        tool_calls=tool_calls,
    )

    build_session_from_execution(
        workspace=config.workspace_root,
        session_id=signal_kwargs["session_id"],
        task=task,
        feedback=evaluation.get("feedback", ""),
        execution=execution,
    )

    from meta_learning.mcp_server import capture_signal
    result = capture_signal(**signal_kwargs)
    logger.info("Signal capture: %s", result)
    return result


async def _run_layer2():
    """Force-run Layer 2 pipeline to generate taxonomy."""
    from meta_learning.mcp_server import run_layer2
    result = await run_layer2(force=True)
    logger.info("Layer 2 result: %s", result)
    return result


def _load_taxonomy() -> str | None:
    """Load current taxonomy as injection text."""
    from meta_learning.shared.io import load_config, load_error_taxonomy
    config = load_config(CONFIG_PATH)
    taxonomy = load_error_taxonomy(config)
    entries = taxonomy.all_entries()

    if not entries:
        return None

    lines = ["# Pre-Completion Checklist (from meta-learning)\n"]
    lines.append("Before finishing the task, you MUST verify each item below.\n")
    for entry in entries:
        lines.append(f"## {entry.name}")
        lines.append(f"Applies when: {entry.trigger}")
        lines.append(f"Steps:\n{entry.fix_sop}")
        lines.append(f"Rule: {entry.prevention}")
        lines.append("")

    return "\n".join(lines)


def _reset_workspace():
    """Clear meta-learning workspace for a fresh experiment."""
    from meta_learning.shared.io import load_config
    config = load_config(CONFIG_PATH)
    ws = Path(config.workspace_root)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    for sub in ["signal_buffer", "experience_pool", "sessions", "skills"]:
        (ws / sub).mkdir(parents=True, exist_ok=True)


def _append_result(result_file: Path, record: dict[str, Any]):
    result_file.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in record.items() if not k.startswith("_")}
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Experiment runners (unchanged logic)
# ---------------------------------------------------------------------------

async def _run_one_round(
    task: dict[str, Any],
    ws_dir: str,
    rnd: int,
    group: str,
    taxonomy_text: str | None,
    max_iterations: int,
) -> dict[str, Any]:
    """Execute one round: run agent, evaluate, return record."""
    t0 = time.time()
    try:
        exec_result = await _execute_task(
            task, ws_dir,
            taxonomy_text=taxonomy_text,
            max_iterations=max_iterations,
        )
        wall_s = time.time() - t0
        status = exec_result.get("status", "unknown")
        iterations = exec_result.get("iterations", 0)
        tool_calls_count = exec_result.get("tool_executions", 0)
    except Exception as e:
        wall_s = time.time() - t0
        logger.error("  Execution error: %s", e)
        exec_result = {"status": "error", "error": str(e)}
        status = "error"
        iterations = 0
        tool_calls_count = 0

    eval_result = _evaluate(task, ws_dir)

    execution_info = {
        "iterations": iterations,
        "tool_calls": tool_calls_count,
        "time_sec": wall_s,
    }
    tools_used = exec_result.get("tools_used", [])

    tid = task["task_id"]
    occupation = task.get("occupation", "unknown")

    return {
        "task_id": tid,
        "occupation": occupation,
        "round": rnd,
        "group": group,
        "has_taxonomy": taxonomy_text is not None,
        "taxonomy_length": len(taxonomy_text) if taxonomy_text else 0,
        "status": status,
        "iterations": iterations,
        "tool_calls": tool_calls_count,
        "wall_sec": round(wall_s, 1),
        "score": eval_result.get("evaluation_score", 0.0),
        "score_10": eval_result.get("score_10", 0),
        "has_evaluation": eval_result.get("has_evaluation", False),
        "feedback": eval_result.get("feedback", ""),
        "cliff_applied": eval_result.get("cliff_applied", False),
        "artifact_count": eval_result.get("artifact_count", 0),
        "_eval_result": eval_result,
        "_execution_info": execution_info,
        "_tools_used": tools_used,
    }


async def run_within_task(
    task_ids: list[str] | None,
    occupations: list[str] | None,
    baseline_rounds: int = 3,
    treatment_rounds: int = 3,
    max_tasks: int = 2,
    max_iterations: int = 20,
):
    """Same task repeated: baseline phase (no taxonomy) then treatment phase (with taxonomy)."""
    total_rounds = baseline_rounds + treatment_rounds
    run_id = f"within_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir = RESULTS_BASE / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / "results.jsonl"
    summary_file = result_dir / "summary.json"

    tasks = _load_tasks(occupations=occupations, task_ids=task_ids, max_tasks=max_tasks)
    if not tasks:
        logger.error("No tasks loaded")
        return

    logger.info(
        "Within-task experiment: %d tasks x (%d baseline + %d treatment) rounds",
        len(tasks), baseline_rounds, treatment_rounds,
    )

    all_results: list[dict] = []

    for task in tasks:
        tid = task["task_id"]
        occupation = task.get("occupation", "unknown")
        logger.info("=== Task: %s [%s] ===", tid, occupation)

        _reset_workspace()

        # --- Phase A: Baseline (no taxonomy) ---
        logger.info("  --- Phase A: Baseline (%d rounds, no taxonomy) ---", baseline_rounds)
        for rnd in range(1, baseline_rounds + 1):
            ws_dir = str(result_dir / "workspaces" / tid / f"round_{rnd}")
            logger.info("  Round %d/%d [baseline]", rnd, total_rounds)

            record = await _run_one_round(
                task, ws_dir, rnd, "baseline", None, max_iterations,
            )
            all_results.append(record)
            _append_result(result_file, record)
            logger.info("  → score=%.1f/10, status=%s, wall=%.0fs",
                       record["score_10"], record["status"], record["wall_sec"])

            if record["has_evaluation"]:
                _capture_signal(
                    task, record["_eval_result"],
                    record["_execution_info"], record["_tools_used"],
                )

        # --- Run Layer2 once after all baseline rounds ---
        logger.info("  --- Running Layer2 (consolidate baseline experiences) ---")
        await _run_layer2()

        # --- Phase B: Treatment (with taxonomy) ---
        logger.info("  --- Phase B: Treatment (%d rounds, with taxonomy) ---", treatment_rounds)
        for rnd_offset in range(1, treatment_rounds + 1):
            rnd = baseline_rounds + rnd_offset
            taxonomy_text = _load_taxonomy()
            ws_dir = str(result_dir / "workspaces" / tid / f"round_{rnd}")
            logger.info("  Round %d/%d [treatment] taxonomy=%s",
                       rnd, total_rounds,
                       f"{len(taxonomy_text)} chars" if taxonomy_text else "none")

            record = await _run_one_round(
                task, ws_dir, rnd, "treatment", taxonomy_text, max_iterations,
            )
            all_results.append(record)
            _append_result(result_file, record)
            logger.info("  → score=%.1f/10, status=%s, wall=%.0fs",
                       record["score_10"], record["status"], record["wall_sec"])

            if record["has_evaluation"]:
                _capture_signal(
                    task, record["_eval_result"],
                    record["_execution_info"], record["_tools_used"],
                )
                await _run_layer2()

    _write_summary(summary_file, all_results, "within-task", total_rounds)
    logger.info("Results written to %s", result_dir)


async def run_cross_task(
    pairs: list[str],
    rounds_per_train: int = 2,
    max_tasks: int = 3,
    max_iterations: int = 20,
):
    """Train on occupation A, test on occupation B."""
    run_id = f"cross_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir = RESULTS_BASE / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / "results.jsonl"
    summary_file = result_dir / "summary.json"

    all_results: list[dict] = []

    for pair_str in pairs:
        parts = pair_str.split(":")
        if len(parts) != 2:
            logger.error("Invalid pair format: %s (expected 'occupation_a:occupation_b')", pair_str)
            continue
        train_occ, test_occ = parts[0].strip(), parts[1].strip()

        logger.info("=== Cross-task pair: train=%s → test=%s ===", train_occ, test_occ)

        _reset_workspace()

        train_tasks = _load_tasks(occupations=[train_occ], max_tasks=max_tasks)
        test_tasks = _load_tasks(occupations=[test_occ], max_tasks=max_tasks)

        if not train_tasks:
            logger.error("No training tasks for occupation: %s", train_occ)
            continue
        if not test_tasks:
            logger.error("No test tasks for occupation: %s", test_occ)
            continue

        logger.info("  Training on %d tasks from '%s'", len(train_tasks), train_occ)
        for i, task in enumerate(train_tasks):
            tid = task["task_id"]
            for rnd in range(1, rounds_per_train + 1):
                ws_dir = str(result_dir / "workspaces" / "train" / tid / f"round_{rnd}")
                taxonomy_text = _load_taxonomy() if (i > 0 or rnd > 1) else None

                logger.info("    Train task %s round %d/%d", tid, rnd, rounds_per_train)
                t0 = time.time()
                try:
                    exec_result = await _execute_task(
                        task, ws_dir,
                        taxonomy_text=taxonomy_text,
                        max_iterations=max_iterations,
                    )
                    wall_s = time.time() - t0
                    iterations = exec_result.get("iterations", 0)
                    tool_calls_count = exec_result.get("tool_executions", 0)
                except Exception as e:
                    wall_s = time.time() - t0
                    logger.error("    Train execution error: %s", e)
                    iterations = 0
                    tool_calls_count = 0

                eval_result = _evaluate(task, ws_dir)

                if eval_result.get("has_evaluation"):
                    _capture_signal(
                        task, eval_result,
                        {"iterations": iterations, "tool_calls": tool_calls_count, "time_sec": wall_s},
                    )

                record = {
                    "pair": pair_str,
                    "phase": "train",
                    "task_id": tid,
                    "occupation": train_occ,
                    "round": rnd,
                    "score": eval_result.get("evaluation_score", 0.0),
                    "score_10": eval_result.get("score_10", 0),
                    "has_evaluation": eval_result.get("has_evaluation", False),
                    "feedback": eval_result.get("feedback", ""),
                    "wall_sec": round(wall_s, 1),
                }
                all_results.append(record)
                _append_result(result_file, record)

            await _run_layer2()

        taxonomy_text = _load_taxonomy()
        logger.info("  Testing %d tasks from '%s' (taxonomy=%s)",
                    len(test_tasks), test_occ,
                    f"{len(taxonomy_text)} chars" if taxonomy_text else "none")

        for task in test_tasks:
            tid = task["task_id"]

            for group_label, inject in [("control", None), ("treatment", taxonomy_text)]:
                ws_dir = str(result_dir / "workspaces" / "test" / tid / group_label)

                logger.info("    Test task %s [%s]", tid, group_label)
                t0 = time.time()
                try:
                    exec_result = await _execute_task(
                        task, ws_dir,
                        taxonomy_text=inject,
                        max_iterations=max_iterations,
                    )
                    wall_s = time.time() - t0
                    iterations = exec_result.get("iterations", 0)
                    tool_calls_count = exec_result.get("tool_executions", 0)
                except Exception as e:
                    wall_s = time.time() - t0
                    logger.error("    Test execution error: %s", e)
                    iterations = 0
                    tool_calls_count = 0

                eval_result = _evaluate(task, ws_dir)

                record = {
                    "pair": pair_str,
                    "phase": "test",
                    "group": group_label,
                    "task_id": tid,
                    "occupation": test_occ,
                    "has_taxonomy": inject is not None,
                    "score": eval_result.get("evaluation_score", 0.0),
                    "score_10": eval_result.get("score_10", 0),
                    "has_evaluation": eval_result.get("has_evaluation", False),
                    "feedback": eval_result.get("feedback", ""),
                    "wall_sec": round(wall_s, 1),
                    "iterations": iterations,
                    "tool_calls": tool_calls_count,
                }
                all_results.append(record)
                _append_result(result_file, record)

                logger.info("    → score=%.1f/10, wall=%.0fs", record["score_10"], wall_s)

    _write_summary(summary_file, all_results, "cross-task", rounds_per_train)
    logger.info("Results written to %s", result_dir)


def _write_summary(path: Path, results: list[dict], mode: str, rounds: int):
    """Write a summary JSON with aggregated statistics."""
    if mode == "within-task":
        baseline_scores = [
            r["score"] for r in results
            if r.get("group") == "baseline" and r.get("has_evaluation")
        ]
        treatment_scores = [
            r["score"] for r in results
            if r.get("group") == "treatment" and r.get("has_evaluation")
        ]
        by_round: dict[int, list[float]] = {}
        for r in results:
            if r.get("has_evaluation"):
                by_round.setdefault(r["round"], []).append(r["score"])

        baseline_mean = round(sum(baseline_scores) / len(baseline_scores), 4) if baseline_scores else 0
        treatment_mean = round(sum(treatment_scores) / len(treatment_scores), 4) if treatment_scores else 0

        summary = {
            "mode": mode,
            "total_records": len(results),
            "rounds": rounds,
            "baseline": {
                "count": len(baseline_scores),
                "mean_score": baseline_mean,
                "scores": baseline_scores,
            },
            "treatment": {
                "count": len(treatment_scores),
                "mean_score": treatment_mean,
                "scores": treatment_scores,
            },
            "delta": round(treatment_mean - baseline_mean, 4),
            "by_round": {
                k: {
                    "count": len(v),
                    "mean_score": round(sum(v) / len(v), 4) if v else 0,
                    "scores": v,
                }
                for k, v in sorted(by_round.items())
            },
        }
    else:
        test_results = [r for r in results if r.get("phase") == "test" and r.get("has_evaluation")]
        control = [r["score"] for r in test_results if r.get("group") == "control"]
        treatment = [r["score"] for r in test_results if r.get("group") == "treatment"]
        summary = {
            "mode": mode,
            "total_records": len(results),
            "test_control": {
                "count": len(control),
                "mean_score": round(sum(control) / len(control), 4) if control else 0,
            },
            "test_treatment": {
                "count": len(treatment),
                "mean_score": round(sum(treatment) / len(treatment), 4) if treatment else 0,
            },
            "delta": round(
                (sum(treatment) / len(treatment) if treatment else 0)
                - (sum(control) / len(control) if control else 0),
                4,
            ),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Summary: %s", json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    _ensure_env()

    parser = argparse.ArgumentParser(description="GDPVal Meta-Learning Validation")
    parser.add_argument("--mode", choices=["within-task", "cross-task"], required=True)
    parser.add_argument("--baseline-rounds", type=int, default=3, help="Baseline rounds (no taxonomy)")
    parser.add_argument("--treatment-rounds", type=int, default=3, help="Treatment rounds (with taxonomy)")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--occupations", nargs="+", help="Occupations for within-task")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs")
    parser.add_argument(
        "--pairs", nargs="+",
        help="Occupation pairs for cross-task (format: 'train_occ:test_occ')",
    )
    parser.add_argument("--rounds-per-train", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "within-task":
        asyncio.run(run_within_task(
            task_ids=args.task_ids,
            occupations=args.occupations,
            baseline_rounds=args.baseline_rounds,
            treatment_rounds=args.treatment_rounds,
            max_tasks=args.max_tasks,
            max_iterations=args.max_iterations,
        ))
    else:
        if not args.pairs:
            parser.error("--pairs required for cross-task mode")
        asyncio.run(run_cross_task(
            pairs=args.pairs,
            rounds_per_train=args.rounds_per_train,
            max_tasks=args.max_tasks,
            max_iterations=args.max_iterations,
        ))


if __name__ == "__main__":
    main()

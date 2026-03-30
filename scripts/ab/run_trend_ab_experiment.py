#!/usr/bin/env python3
"""Run isolated A/B trend experiment for wfl-004 and tau3 airline tasks."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from claw_bench.cli.run import run_cmd
from meta_learning.mcp_server import capture_signal, quick_think, run_layer2


ROOT = Path("/Users/yumeng/Documents/Projects/lingmin-meta-learning")
DESKCLAW_ROOT = Path("/Users/yumeng/Documents/Projects/DeskClaw-Arena")
TAU_VENV_PY = Path("/Users/yumeng/Documents/Projects/Benchmarks/tau2-bench/.venv/bin/python3")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def _spawn(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_port(port: int, timeout_s: int = 20) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            capture_output=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Port {port} not ready within {timeout_s}s")


def _terminate_all(processes: list[subprocess.Popen[str]]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    time.sleep(1.0)
    for p in processes:
        if p.poll() is None:
            p.kill()


def _remove_stale_result(path: Path) -> bool:
    """Delete result file before each run to prevent stale reads. Returns True if file was removed."""
    if path.exists():
        path.unlink()
        return True
    return False


def _load_tau_single_result(path: Path, expected_task_id: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not tasks:
        raise RuntimeError(f"No task result in {path}")
    row = tasks[0]
    file_tid = str(row.get("task_id", ""))
    if expected_task_id and file_tid != expected_task_id:
        raise RuntimeError(
            f"task_id mismatch in result file: expected '{expected_task_id}', "
            f"got '{file_tid}' — likely stale data from a previous run"
        )
    return {
        "task_id": file_tid,
        "reward": float(row.get("reward", 0.0) or 0.0),
        "duration_s": float(row.get("duration_s", 0.0) or 0.0),
        "tool_calls_count": int(row.get("tool_calls_count", 0) or 0),
        "termination": str(row.get("termination", "")),
    }


def _load_tau_single_task_row(path: Path, expected_task_id: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not tasks:
        raise RuntimeError(f"No task result in {path}")
    row = tasks[0]
    if not isinstance(row, dict):
        raise RuntimeError(f"Invalid task row in {path}")
    file_tid = str(row.get("task_id", ""))
    if expected_task_id and file_tid != expected_task_id:
        raise RuntimeError(
            f"task_id mismatch in task_row: expected '{expected_task_id}', "
            f"got '{file_tid}' — likely stale data from a previous run"
        )
    return row


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_session_from_conversation(
    workspace: Path,
    session_id: str,
    conversation: list[dict[str, Any]],
    agent_tool_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Write session file with conversation turns and agent tool call log.

    ``agent_tool_calls`` comes from ``task_row["tool_calls"]`` (MCP server
    state) and contains the agent's actual tool invocations with arguments and
    results — the most valuable data for meta-learning diagnosis.
    """
    sessions_dir = workspace / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    tc_idx = 0
    tc_list = agent_tool_calls or []

    with open(sessions_dir / f"{session_id}.jsonl", "w", encoding="utf-8") as f:
        for turn in conversation:
            role = str(turn.get("role", "")).strip().lower()

            if role == "user_tool_calls":
                for tc in turn.get("tool_calls", []):
                    name = tc.get("name", "?")
                    args_s = json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:300]
                    result_s = str(tc.get("result", ""))[:500]
                    f.write(json.dumps({
                        "role": "user_tool",
                        "content": f"[user_tool:{name}] args={args_s} result={result_s}",
                    }, ensure_ascii=False) + "\n")
                continue

            if role == "assistant":
                tc_count = turn.get("tc_count", 0)
                content = turn.get("content")
                if isinstance(content, str) and content.strip():
                    f.write(json.dumps({"role": "assistant", "content": content.strip()}, ensure_ascii=False) + "\n")
                if tc_count > 0 and tc_list:
                    for _ in range(tc_count):
                        if tc_idx >= len(tc_list):
                            break
                        tc = tc_list[tc_idx]
                        tc_idx += 1
                        name = tc.get("name", "?")
                        args_s = json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:300]
                        result_s = str(tc.get("result", ""))[:500]
                        f.write(json.dumps({
                            "role": "agent_tool",
                            "content": f"[agent_tool:{name}] args={args_s} result={result_s}",
                        }, ensure_ascii=False) + "\n")
                continue

            if role not in {"user", "system", "tool"}:
                continue
            content = turn.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            content = content.strip()
            if not content:
                continue
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

        if tc_idx < len(tc_list):
            for tc in tc_list[tc_idx:]:
                name = tc.get("name", "?")
                args_s = json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:300]
                result_s = str(tc.get("result", ""))[:500]
                f.write(json.dumps({
                    "role": "agent_tool",
                    "content": f"[agent_tool:{name}] args={args_s} result={result_s}",
                }, ensure_ascii=False) + "\n")


def _build_meta_context(workspace: Path) -> str:
    """Read accumulated taxonomy & skills from workspace, format as agent-readable guidance."""
    sections: list[str] = []

    tax_path = workspace / "error_taxonomy.yaml"
    if tax_path.exists():
        raw = yaml.safe_load(tax_path.read_text(encoding="utf-8")) or {}
        taxonomy = raw.get("taxonomy", {})
        entries: list[dict[str, Any]] = []
        for _domain, subdomains in taxonomy.items():
            if not isinstance(subdomains, dict):
                continue
            for _sub, items in subdomains.items():
                if isinstance(items, list):
                    entries.extend(items)
        if entries:
            lines = [
                "<meta-learning-experience>",
                "The following patterns were identified from your previous attempts at this task.",
                "Use them to avoid repeating the same mistakes:\n",
            ]
            for entry in entries[:5]:
                name = entry.get("name", "")
                trigger = entry.get("trigger", "")
                prevention = entry.get("prevention", "")
                fix_sop = entry.get("fix_sop", "")
                lines.append(f"### {name}")
                if trigger:
                    lines.append(f"**When it happens**: {trigger}")
                if prevention:
                    lines.append(f"**How to prevent**: {prevention}")
                if fix_sop:
                    lines.append(f"**Fix procedure**: {fix_sop}")
                lines.append("")
            lines.append("</meta-learning-experience>")
            sections.append("\n".join(lines))

    skills_dir = workspace / "skills"
    if skills_dir.is_dir():
        for skill_md in skills_dir.rglob("SKILL.md"):
            content = skill_md.read_text(encoding="utf-8").strip()
            if content:
                sections.append(
                    f"<meta-learning-skill source=\"{skill_md.parent.name}\">\n"
                    f"{content}\n"
                    f"</meta-learning-skill>"
                )

    return "\n\n".join(sections)


def _tool_call_summary(tool_calls: list[dict[str, Any]], max_tools: int = 8) -> str:
    """Compact summary of agent tool calls: name(key_args) -> result_head."""
    parts: list[str] = []
    for tc in tool_calls[:max_tools]:
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        args_brief = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3]) if isinstance(args, dict) else ""
        result_head = str(tc.get("result", ""))[:120].replace("\n", " ")
        parts.append(f"{name}({args_brief}) -> {result_head}")
    if len(tool_calls) > max_tools:
        parts.append(f"... +{len(tool_calls) - max_tools} more calls")
    return "\n".join(parts)


_PARAM_PATTERN = re.compile(
    r"(HAT\d{3}|flight\s+\w+\d|reservation\s+\w+|\$[\d,]+\.\d+|\$[\d,]+|credit_card_\w+|gift_card_\w+|certificate_\w+)",
    re.IGNORECASE,
)


def _is_strategy_level_assertion(assertion_text: str) -> bool:
    """Filter NL assertions: keep strategy-level, reject parameter-level.

    Strategy-level: "should not offer", "does not cancel", "should check"
    Parameter-level: contains specific flight numbers, amounts, card IDs
    """
    param_matches = _PARAM_PATTERN.findall(assertion_text)
    if len(param_matches) >= 2:
        return False
    strategy_keywords = [
        "should not", "does not", "should check", "should detect",
        "should verify", "should realize", "should not offer",
        "cannot be", "is not allowed", "not modified", "not cancel",
    ]
    return any(kw in assertion_text.lower() for kw in strategy_keywords)


def _extract_nl_corrections(task_row: dict[str, Any]) -> list[str]:
    """Extract strategy-level NL assertion failures as supervisor corrections."""
    nl_assertions = task_row.get("nl_assertions")
    if not nl_assertions or not isinstance(nl_assertions, list):
        return []
    corrections: list[str] = []
    for a in nl_assertions:
        if not isinstance(a, dict):
            continue
        if a.get("met", True):
            continue
        text = a.get("assertion", "")
        if not text:
            continue
        if _is_strategy_level_assertion(text):
            corrections.append(f"QA feedback: {text} — but agent did not follow this.")
    return corrections


def _extract_signal_info(
    task_row: dict[str, Any],
    row: dict[str, Any],
    prev_a_rows: list[dict[str, Any]],
    known_tools: set[str],
) -> dict[str, Any]:
    """Extract meaningful signal info from a tau3 task result for capture_signal.

    Focuses on data the agent naturally observes: conversation + tool calls + outcome.
    NL assertion failures (strategy-level only) are included as ``user_corrections``
    to represent QA supervisor feedback.
    """
    reward = row.get("reward", 0.0)
    conv = task_row.get("conversation", [])

    tool_calls = task_row.get("tool_calls", [])
    tool_names = sorted({tc.get("name", "") for tc in tool_calls if isinstance(tc, dict)} - {""})
    new_tools_detected = [tn for tn in tool_names if tn not in known_tools]

    tc_summary = _tool_call_summary(tool_calls)

    errors: list[str] = []
    resolution = ""
    if reward == 0.0:
        asst_msgs = [m for m in conv if isinstance(m, dict) and m.get("role") == "assistant"]
        if asst_msgs:
            last_msg = str(asst_msgs[-1].get("content", ""))[:300]
            errors.append(f"Task failed (reward=0). Agent's last response: {last_msg}")
        else:
            errors.append("Task failed (reward=0) with no captured agent dialogue.")

        if tc_summary:
            errors.append(f"Agent tool calls during failed task:\n{tc_summary}")
        elif tool_calls:
            errors.append(f"Agent made {len(tool_calls)} tool calls but details unavailable.")
        else:
            errors.append("Agent made 0 tool calls.")
    else:
        if tc_summary:
            resolution = f"Task succeeded (reward={reward}). Agent tool call chain:\n{tc_summary}"
        else:
            asst_msgs = [m for m in conv if isinstance(m, dict) and m.get("role") == "assistant"]
            if asst_msgs:
                resolution = f"Task succeeded. Agent's final response: {str(asst_msgs[-1].get('content', ''))[:300]}"
            else:
                resolution = f"Task succeeded (reward={reward}), {len(tool_calls)} tool calls."

    user_corrections = _extract_nl_corrections(task_row)

    errors_fixed = False
    if reward == 1.0 and prev_a_rows:
        errors_fixed = any(r.get("reward", 0.0) == 0.0 for r in prev_a_rows[-3:])
    if reward == 0.0 and errors and prev_a_rows:
        errors_fixed = True

    return {
        "errors_encountered": errors,
        "errors_fixed": errors_fixed,
        "user_corrections": user_corrections,
        "tools_used": tool_names or ["mcp_bench"],
        "new_tools": new_tools_detected,
        "resolution_snapshot": resolution or f"reward={reward}, duration={row.get('duration_s', 0):.1f}s, tools={len(tool_calls)}",
    }


def _summarize_conversation_for_session(conversation: list[dict[str, Any]]) -> str:
    """Build a concise session summary from conversation turns."""
    lines: list[str] = []
    for turn in conversation:
        role = str(turn.get("role", "")).strip().lower()
        if role == "user_tool_calls":
            for tc in turn.get("tool_calls", []):
                name = tc.get("name", "?")
                result = str(tc.get("result", ""))[:200]
                lines.append(f"[tool:{name}] {result}")
            continue
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        content = turn.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        content = content.strip()
        if not content:
            continue
        lines.append(f"[{role}] {content[:500]}")
    return "\n".join(lines)


def _prepare_isolated_configs(
    *,
    tmp_root: Path,
    bridge_port_a: int,
    bridge_port_b: int,
    mcp_port_a: int,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/ab/prepare_nobot_ab_configs.py"),
        "--out-dir",
        str(tmp_root),
        "--mcp-port-a",
        str(mcp_port_a),
        "--mcp-port-b",
        "19999",
        "--bridge-port-a",
        str(bridge_port_a),
        "--bridge-port-b",
        str(bridge_port_b),
        "--disable-meta-learning-for",
        "B",
    ]
    ret = _run(cmd, cwd=ROOT)
    if ret.stdout:
        print(ret.stdout.strip())


def _run_claw_wfl004_iterations(
    *,
    result_root: Path,
    agent_url: str,
    agent_name: str,
    mcp_servers: str,
    runs: int,
) -> list[dict[str, Any]]:
    from claw_bench.core.cache import result_cache

    rows: list[dict[str, Any]] = []
    for i in range(1, runs + 1):
        # Disable cache carry-over so each repetition is real execution.
        result_cache.invalidate(task_id="wfl-004")
        out_dir = result_root / f"iter_{i:02d}"
        run_cmd(
            framework="nanobot",
            model="minimax-m2.7",
            tasks="wfl-004",
            skills="vanilla",
            model_tier=None,
            runs=1,
            parallel=1,
            timeout=600,
            output=out_dir,
            dry_run=False,
            sandbox=False,
            tier=None,
            mcp_servers=mcp_servers or None,
            memory_modules=None,
            claw_id=None,
            agent_url=agent_url,
            agent_name=agent_name,
            resume=False,
        )
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        item = summary["task_results"][0]
        rows.append(
            {
                "iteration": i,
                "task_id": item.get("task_id"),
                "passed": bool(item.get("passed")),
                "score": float(item.get("score", 0.0)),
                "duration_s": float(item.get("duration_s", 0.0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated A/B trend experiment")
    parser.add_argument("--runs", type=int, default=10, help="Repetitions per group per task")
    parser.add_argument("--layer2-every", type=int, default=5, help="A group run_layer2 frequency")
    parser.add_argument("--bridge-port-a", type=int, default=5263)
    parser.add_argument("--bridge-port-b", type=int, default=5262)
    parser.add_argument("--meta-port-a", type=int, default=19811)
    parser.add_argument("--tau-port", type=int, default=18795)
    parser.add_argument("--tau-timeout-s", type=int, default=420)
    parser.add_argument("--skip-claw", action="store_true", help="Run only tau3 airline tasks")
    parser.add_argument(
        "--tasks",
        default="35",
        help="Comma-separated airline task IDs (default: 35). Example: 2,9,18",
    )
    parser.add_argument(
        "--llm-base-url",
        default="https://llm-gateway-api.nodesk.tech/default/v1",
        help="OpenAI-compatible base URL for meta-learning LLM",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("META_LEARNING_LLM_API_KEY", ""),
        help="API key for meta-learning LLM (prefer env META_LEARNING_LLM_API_KEY)",
    )
    parser.add_argument("--run-id", default="", help="Optional custom run id")
    parser.add_argument(
        "--warmstart-taxonomy",
        default="",
        help="Path to a taxonomy YAML to pre-seed into A workspace before the experiment",
    )
    args = parser.parse_args()
    if not args.llm_api_key.strip():
        raise RuntimeError(
            "Missing --llm-api-key (or env META_LEARNING_LLM_API_KEY) for provider=openai"
        )

    run_id = args.run_id.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    result_root = ROOT / "abtest/results" / f"trend_ab_{run_id}"
    if result_root.exists():
        raise RuntimeError(f"Result root already exists: {result_root}")
    result_root.mkdir(parents=True, exist_ok=False)

    tmp_root = Path(f"/tmp/nobot_ab_trend/{run_id}").resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[run] run_id={run_id}")
    print(f"[run] result_root={result_root}")
    print(f"[run] tmp_root={tmp_root}")

    _prepare_isolated_configs(
        tmp_root=tmp_root,
        bridge_port_a=args.bridge_port_a,
        bridge_port_b=args.bridge_port_b,
        mcp_port_a=args.meta_port_a,
    )

    processes: list[subprocess.Popen[str]] = []
    try:
        meta_env = {
            "META_LEARNING_LLM_BASE_URL": args.llm_base_url.strip(),
            "META_LEARNING_LLM_API_KEY": args.llm_api_key.strip(),
        }

        # A meta-learning MCP server (HTTP).
        p_meta_a = _spawn(
            [
                sys.executable,
                str(ROOT / "scripts/ab/launch_meta_mcp_http.py"),
                "--config",
                str(ROOT / "abtest/config.meta-learning.A.yaml"),
                "--workspace",
                str(tmp_root / "A/workspace"),
                "--port",
                str(args.meta_port_a),
            ],
            cwd=ROOT,
            env=meta_env,
        )
        processes.append(p_meta_a)
        _wait_port(args.meta_port_a)

        if not args.skip_claw:
            # A bridge for claw-bench A group (meta capability via MCP config).
            p_bridge_a = _spawn(
                [
                    sys.executable,
                    str(DESKCLAW_ROOT / "src/eval/nobot_openai_bridge.py"),
                    "--port",
                    str(args.bridge_port_a),
                    "--api-key",
                    "nobot-bridge-key",
                    "--config",
                    str(tmp_root / "nobot.config.A.json"),
                    "--workspace",
                    str(tmp_root / "A/workspace"),
                ],
                cwd=DESKCLAW_ROOT,
                env=meta_env,
            )
            processes.append(p_bridge_a)
            _wait_port(args.bridge_port_a)

            # B bridge: plain path, isolated workspace/config.
            p_bridge_b = _spawn(
                [
                    sys.executable,
                    str(DESKCLAW_ROOT / "src/eval/nobot_openai_bridge.py"),
                    "--port",
                    str(args.bridge_port_b),
                    "--api-key",
                    "nobot-bridge-key",
                    "--config",
                    str(tmp_root / "nobot.config.B.json"),
                    "--workspace",
                    str(tmp_root / "B/workspace"),
                ],
                cwd=DESKCLAW_ROOT,
            )
            processes.append(p_bridge_b)
            _wait_port(args.bridge_port_b)

        task_ids = [t.strip() for t in args.tasks.split(",")]
        print(f"[run] task_ids={task_ids}")

        # tau3 domain MCP server (shared benchmark env, non-meta).
        p_tau = _spawn(
            [
                str(TAU_VENV_PY),
                str(DESKCLAW_ROOT / "src/mcp_tau_bench_server.py"),
                "--domain",
                "airline",
                "--port",
                str(args.tau_port),
                "--task-id",
                task_ids[0],
            ],
            cwd=DESKCLAW_ROOT,
        )
        processes.append(p_tau)
        _wait_port(args.tau_port)

        # Ensure direct meta calls in this process use A-group config/workspace.
        os.environ["META_LEARNING_CONFIG"] = str(ROOT / "abtest/config.meta-learning.A.yaml")
        os.environ["META_LEARNING_WORKSPACE"] = str(tmp_root / "A/workspace")
        os.environ["META_LEARNING_LLM_BASE_URL"] = args.llm_base_url.strip()
        os.environ["META_LEARNING_LLM_API_KEY"] = args.llm_api_key.strip()

        # claw-bench wfl-004 repeated runs.
        claw_a_rows: list[dict[str, Any]] = []
        claw_b_rows: list[dict[str, Any]] = []
        if not args.skip_claw:
            claw_a_rows = _run_claw_wfl004_iterations(
                result_root=result_root / "claw_A",
                agent_url=f"http://127.0.0.1:{args.bridge_port_a}",
                agent_name="NobotBridgeA_Trend",
                mcp_servers="meta_learning",
                runs=args.runs,
            )
            claw_b_rows = _run_claw_wfl004_iterations(
                result_root=result_root / "claw_B",
                agent_url=f"http://127.0.0.1:{args.bridge_port_b}",
                agent_name="NobotBridgeB_Trend",
                mcp_servers="",
                runs=args.runs,
            )
        _save_json(result_root / "claw_A_runs.json", claw_a_rows)
        _save_json(result_root / "claw_B_runs.json", claw_b_rows)

        # tau3 airline multi-task repeated runs.
        tau_a_rows: list[dict[str, Any]] = []
        tau_b_rows: list[dict[str, Any]] = []
        trace_path = result_root / "meta_trace.jsonl"
        tau_tmp_result = DESKCLAW_ROOT / "results_nobot_real" / "airline_results.json"

        a_workspace = tmp_root / "A/workspace"
        a_workspace.mkdir(parents=True, exist_ok=True)

        if args.warmstart_taxonomy:
            ws_tax = Path(args.warmstart_taxonomy).expanduser().resolve()
            if not ws_tax.exists():
                raise FileNotFoundError(f"Warmstart taxonomy not found: {ws_tax}")
            import shutil
            shutil.copy2(ws_tax, a_workspace / "error_taxonomy.yaml")
            print(f"[warmstart] Copied taxonomy seed to {a_workspace / 'error_taxonomy.yaml'}")

        extra_ctx_path = tmp_root / "meta_context_for_agent.md"
        known_tools_a: set[str] = set()
        global_iter = 0

        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"[task] Starting task {task_id} ({args.runs} runs per group)")
            print(f"{'='*60}")

            tau_cmd_for_task = [
                str(TAU_VENV_PY),
                str(DESKCLAW_ROOT / "src/agents/run_nobot_real.py"),
                "--domain", "airline",
                "--task-ids", task_id,
                "--mcp-port", str(args.tau_port),
                "--max-turns", "20",
            ]

            task_a_rows = [r for r in tau_a_rows if str(r.get("task_id")) == task_id]

            for i in range(1, args.runs + 1):
                global_iter += 1

                # ── A group: meta-learning enhanced ──────────────────────
                meta_ctx = _build_meta_context(a_workspace)

                prev_success = sum(1 for r in task_a_rows if r.get("reward", 0) == 1.0)
                pre = quick_think(
                    user_message=(
                        f"[tau3-airline-task{task_id}][iter={i}] Customer service task. "
                        f"Previous {len(task_a_rows)} attempts on this task: "
                        f"{prev_success} succeeded."
                    ),
                    tools_used=["mcp_bench_get_user_details", "mcp_bench_get_reservation_details",
                                 "mcp_bench_cancel_reservation", "mcp_bench_update_reservation",
                                 "mcp_bench_book_reservation", "mcp_bench_search_flights"],
                )
                _append_jsonl(trace_path, {
                    "global_iter": global_iter, "task_id": task_id, "iter": i,
                    "group": "A", "event": "pre_quick_think",
                    "content": pre,
                })

                if pre != "no risk detected":
                    meta_ctx += f"\n\n<meta-learning-risk-warning>\n{pre}\n</meta-learning-risk-warning>"

                extra_ctx_path.write_text(meta_ctx, encoding="utf-8")
                _append_jsonl(trace_path, {
                    "global_iter": global_iter, "task_id": task_id, "iter": i,
                    "group": "A", "event": "meta_context_injected",
                    "context_length": len(meta_ctx),
                    "has_taxonomy": "meta-learning-experience" in meta_ctx,
                    "has_risk_warning": "risk-warning" in meta_ctx,
                })

                a_cmd = tau_cmd_for_task + [
                    "--config", str(tmp_root / "nobot.config.A.json"),
                    "--extra-context-file", str(extra_ctx_path),
                ]

                had_stale = _remove_stale_result(tau_tmp_result)
                print(f"[A][task={task_id}][iter={i}] start (stale_removed={had_stale})")
                print(f"[A][task={task_id}][iter={i}] cmd: {' '.join(a_cmd[-6:])}")
                iter_t0 = time.time()
                try:
                    a_run = _run(a_cmd, cwd=DESKCLAW_ROOT, timeout=args.tau_timeout_s, check=False)
                    wall_s = round(time.time() - iter_t0, 1)
                    result_exists = tau_tmp_result.exists()
                    result_size = tau_tmp_result.stat().st_size if result_exists else 0
                    print(
                        f"[A][task={task_id}][iter={i}] subprocess done: "
                        f"exit={a_run.returncode}, wall={wall_s}s, "
                        f"result_exists={result_exists}, result_bytes={result_size}"
                    )
                    if a_run.returncode != 0:
                        stderr_tail = (a_run.stderr or "").strip()[-300:]
                        print(f"[A][task={task_id}][iter={i}] stderr tail: {stderr_tail}")
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "A", "event": "subprocess_completed",
                        "exit_code": a_run.returncode,
                        "wall_time_s": wall_s,
                        "result_file_exists": result_exists,
                        "result_file_bytes": result_size,
                    })
                    try:
                        task_row = _load_tau_single_task_row(tau_tmp_result, expected_task_id=task_id)
                        row = _load_tau_single_result(tau_tmp_result, expected_task_id=task_id)
                        print(
                            f"[A][task={task_id}][iter={i}] result loaded: "
                            f"file_task_id={row['task_id']}, reward={row['reward']}, "
                            f"dur={row['duration_s']}s, tools={row['tool_calls_count']}"
                        )
                    except Exception as load_exc:
                        raise RuntimeError(f"result load failed: {load_exc}")
                except subprocess.TimeoutExpired:
                    wall_s = round(time.time() - iter_t0, 1)
                    print(f"[A][task={task_id}][iter={i}] TIMEOUT after {wall_s}s")
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "A", "event": "subprocess_timeout",
                        "wall_time_s": wall_s,
                    })
                    task_row = {}
                    row = {
                        "task_id": task_id,
                        "reward": 0.0,
                        "duration_s": wall_s,
                        "tool_calls_count": 0,
                        "termination": "timeout",
                    }
                except Exception as exc:
                    wall_s = round(time.time() - iter_t0, 1)
                    print(f"[A][task={task_id}][iter={i}] ERROR: {exc}")
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "A", "event": "subprocess_error",
                        "wall_time_s": wall_s,
                        "error": str(exc),
                    })
                    task_row = {}
                    row = {
                        "task_id": task_id,
                        "reward": 0.0,
                        "duration_s": wall_s,
                        "tool_calls_count": 0,
                        "termination": "error",
                    }
                row.update({"iteration": i, "global_iter": global_iter, "group": "A", "quick_think": pre})
                tau_a_rows.append(row)
                task_a_rows.append(row)

                session_id_a = f"tau3-airline-{task_id}-A-{i}"
                conv = task_row.get("conversation")
                agent_tcs = task_row.get("tool_calls", [])
                if isinstance(conv, list) and conv:
                    valid_turns = [c for c in conv if isinstance(c, dict)]
                    _write_session_from_conversation(
                        a_workspace, session_id_a, valid_turns,
                        agent_tool_calls=agent_tcs if isinstance(agent_tcs, list) else [],
                    )
                else:
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "A", "event": "session_missing_real_conversation",
                        "session_id": session_id_a,
                    })
                    session_id_a = "unknown"

                sig_info = _extract_signal_info(task_row, row, task_a_rows[:-1], known_tools_a)
                known_tools_a.update(sig_info["tools_used"])
                capture_res = capture_signal(
                    task_description=(
                        f"[tau3-airline-task{task_id}][iter={i}] "
                        f"Customer service. reward={row['reward']}, "
                        f"duration={row['duration_s']:.1f}s, "
                        f"tools={row['tool_calls_count']}"
                    ),
                    session_id=session_id_a,
                    errors_encountered=sig_info["errors_encountered"],
                    errors_fixed=sig_info["errors_fixed"],
                    user_corrections=sig_info.get("user_corrections", []),
                    tools_used=sig_info["tools_used"],
                    new_tools=sig_info["new_tools"],
                    resolution_snapshot=sig_info["resolution_snapshot"],
                    step_count=max(row["tool_calls_count"], 1),
                )
                _append_jsonl(trace_path, {
                    "global_iter": global_iter, "task_id": task_id, "iter": i,
                    "group": "A", "event": "post_capture_signal",
                    "content": capture_res,
                    "reward": row["reward"], "duration_s": row["duration_s"],
                    "signal_info": sig_info,
                })

                if args.layer2_every > 0 and global_iter % args.layer2_every == 0:
                    layer2_res = asyncio.run(run_layer2(force=True))
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "A", "event": "layer2_forced",
                        "content": layer2_res,
                    })

                # ── B group: plain run, no meta hooks ────────────────────
                b_cmd = tau_cmd_for_task + ["--config", str(tmp_root / "nobot.config.B.json")]
                had_stale_b = _remove_stale_result(tau_tmp_result)
                print(f"[B][task={task_id}][iter={i}] start (stale_removed={had_stale_b})")
                iter_t0_b = time.time()
                try:
                    b_run = _run(b_cmd, cwd=DESKCLAW_ROOT, timeout=args.tau_timeout_s, check=False)
                    wall_s_b = round(time.time() - iter_t0_b, 1)
                    result_exists_b = tau_tmp_result.exists()
                    result_size_b = tau_tmp_result.stat().st_size if result_exists_b else 0
                    print(
                        f"[B][task={task_id}][iter={i}] subprocess done: "
                        f"exit={b_run.returncode}, wall={wall_s_b}s, "
                        f"result_exists={result_exists_b}, result_bytes={result_size_b}"
                    )
                    if b_run.returncode != 0:
                        stderr_tail_b = (b_run.stderr or "").strip()[-300:]
                        print(f"[B][task={task_id}][iter={i}] stderr tail: {stderr_tail_b}")
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "B", "event": "subprocess_completed",
                        "exit_code": b_run.returncode,
                        "wall_time_s": wall_s_b,
                        "result_file_exists": result_exists_b,
                        "result_file_bytes": result_size_b,
                    })
                    try:
                        row_b = _load_tau_single_result(tau_tmp_result, expected_task_id=task_id)
                        print(
                            f"[B][task={task_id}][iter={i}] result loaded: "
                            f"file_task_id={row_b['task_id']}, reward={row_b['reward']}, "
                            f"dur={row_b['duration_s']}s, tools={row_b['tool_calls_count']}"
                        )
                    except Exception as load_exc:
                        raise RuntimeError(f"result load failed: {load_exc}")
                except subprocess.TimeoutExpired:
                    wall_s_b = round(time.time() - iter_t0_b, 1)
                    print(f"[B][task={task_id}][iter={i}] TIMEOUT after {wall_s_b}s")
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "B", "event": "subprocess_timeout",
                        "wall_time_s": wall_s_b,
                    })
                    row_b = {
                        "task_id": task_id,
                        "reward": 0.0,
                        "duration_s": wall_s_b,
                        "tool_calls_count": 0,
                        "termination": "timeout",
                    }
                except Exception as exc:
                    wall_s_b = round(time.time() - iter_t0_b, 1)
                    print(f"[B][task={task_id}][iter={i}] ERROR: {exc}")
                    _append_jsonl(trace_path, {
                        "global_iter": global_iter, "task_id": task_id, "iter": i,
                        "group": "B", "event": "subprocess_error",
                        "wall_time_s": wall_s_b,
                        "error": str(exc),
                    })
                    row_b = {
                        "task_id": task_id,
                        "reward": 0.0,
                        "duration_s": wall_s_b,
                        "tool_calls_count": 0,
                        "termination": "error",
                    }
                row_b.update({"iteration": i, "global_iter": global_iter, "group": "B"})
                tau_b_rows.append(row_b)
                _append_jsonl(trace_path, {
                    "global_iter": global_iter, "task_id": task_id, "iter": i,
                    "group": "B", "event": "plain_run",
                    "reward": row_b["reward"],
                    "duration_s": row_b["duration_s"],
                    "termination": row_b["termination"],
                })

            # Force Layer2 at end of each task block
            if args.layer2_every > 0:
                layer2_res = asyncio.run(run_layer2(force=True))
                _append_jsonl(trace_path, {
                    "global_iter": global_iter, "task_id": task_id,
                    "group": "A", "event": "layer2_end_of_task_block",
                    "content": layer2_res,
                })

        _save_json(result_root / "tau3_A_runs.json", tau_a_rows)
        _save_json(result_root / "tau3_B_runs.json", tau_b_rows)

        manifest = {
            "run_id": run_id,
            "result_root": str(result_root),
            "tmp_root": str(tmp_root),
            "runs_per_group": args.runs,
            "layer2_every": args.layer2_every,
            "warmstart_taxonomy": args.warmstart_taxonomy or None,
            "claw_task": None if args.skip_claw else "wfl-004",
            "tau3_tasks": [f"airline:{tid}" for tid in task_ids],
            "llm_base_url": args.llm_base_url.strip(),
            "llm_model": "minimax-m2.7",
            "ports": {
                "bridge_a": args.bridge_port_a,
                "bridge_b": args.bridge_port_b,
                "meta_a": args.meta_port_a,
                "tau_mcp": args.tau_port,
            },
        }
        _save_json(result_root / "manifest.json", manifest)
        print(f"[done] trend experiment finished: {result_root}")
    finally:
        _terminate_all(processes)


if __name__ == "__main__":
    main()

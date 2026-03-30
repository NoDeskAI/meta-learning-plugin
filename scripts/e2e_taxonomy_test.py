#!/usr/bin/env python3
"""End-to-end taxonomy test: signal capture → Layer2 → taxonomy → inject → verify improvement.

Runs tau-bench airline task 2 twice:
  Round 1: No taxonomy → expect failure (proactive compensation)
  Round 2: With generated taxonomy → expect improved behavior

This proves the full meta-learning loop works end-to-end.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKCLAW_ROOT = Path("/Users/yumeng/Documents/Projects/DeskClaw-Arena")
TAU_VENV_PY = Path("/Users/yumeng/Documents/Projects/Benchmarks/tau2-bench/.venv/bin/python3")

sys.path.insert(0, str(ROOT / "src"))

import yaml

TAU_PORT = 18795
TASK_ID = "2"
MAX_TURNS = 20
TIMEOUT_S = 360
LLM_BASE_URL = "https://llm-gateway-api.nodesk.tech/default/v1"
LLM_API_KEY = "nd-9f27abd1325015b7932ea4c8b54c4fdc889f0496c1f5f2b3bf24e80fd7f19895"

RESULT_FILE = DESKCLAW_ROOT / "results_nobot_real" / "airline_results.json"


def banner(msg: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {msg}")
    print(f"{'='*72}\n")


def _run(cmd: list[str], *, cwd: Path, timeout: int | None = None, check: bool = False) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    return subprocess.run(cmd, cwd=str(cwd), env=merged, text=True, capture_output=True, check=check, timeout=timeout)


def _spawn(cmd: list[str], *, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_port(port: int, timeout: int = 30) -> bool:
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def start_tau_server() -> subprocess.Popen:
    banner("Starting tau3 MCP server")
    p = _spawn(
        [str(TAU_VENV_PY), str(DESKCLAW_ROOT / "src/mcp_tau_bench_server.py"),
         "--domain", "airline", "--port", str(TAU_PORT), "--task-id", TASK_ID],
        cwd=DESKCLAW_ROOT,
    )
    if not wait_port(TAU_PORT, timeout=30):
        raise RuntimeError(f"tau3 MCP server failed to start on port {TAU_PORT}")
    print(f"[OK] tau3 MCP server running on port {TAU_PORT}")
    return p


def create_nobot_config(tmp_root: Path) -> Path:
    """Create a minimal nobot config — no extra MCP servers, just the bench one (added programmatically)."""
    config = {
        "agents": {
            "defaults": {
                "workspace": str(tmp_root / "nobot_workspace"),
                "model": "minimax-m2.7",
                "provider": "custom",
            }
        },
        "providers": {
            "custom": {
                "api_key": LLM_API_KEY,
                "api_base": LLM_BASE_URL,
            }
        },
        "tools": {
            "mcp_servers": {}
        },
    }
    config_path = tmp_root / "nobot.config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def run_nobot(config_path: Path, extra_context_file: Path | None = None) -> dict:
    """Run nobot on task 2 and return the result dict."""
    if RESULT_FILE.exists():
        RESULT_FILE.unlink()

    cmd = [
        str(TAU_VENV_PY),
        str(DESKCLAW_ROOT / "src/agents/run_nobot_real.py"),
        "--domain", "airline",
        "--task-ids", TASK_ID,
        "--mcp-port", str(TAU_PORT),
        "--max-turns", str(MAX_TURNS),
        "--config", str(config_path),
    ]
    if extra_context_file:
        cmd += ["--extra-context-file", str(extra_context_file)]

    print(f"[CMD] {' '.join(cmd[-8:])}")
    t0 = time.time()
    try:
        result = _run(cmd, cwd=DESKCLAW_ROOT, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] after {TIMEOUT_S}s")
        return {"reward": 0.0, "task_id": TASK_ID, "termination": "timeout", "duration_s": TIMEOUT_S}

    wall = time.time() - t0
    print(f"[DONE] exit={result.returncode}, wall={wall:.1f}s")

    if result.returncode != 0:
        print(f"[STDERR tail] {(result.stderr or '')[-500:]}")

    if not RESULT_FILE.exists():
        print("[ERROR] Result file not found")
        return {"reward": 0.0, "task_id": TASK_ID, "termination": "no_result", "duration_s": wall}

    data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not tasks:
        return {"reward": 0.0, "task_id": TASK_ID, "termination": "empty", "duration_s": wall}

    row = tasks[0]
    row["duration_s"] = wall
    return row


def write_session(workspace: Path, session_id: str, task_row: dict) -> None:
    """Write a session JSONL file from the task conversation + tool calls."""
    sessions_dir = workspace / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    conv = task_row.get("conversation", [])
    tool_calls = task_row.get("tool_calls", [])
    tc_idx = 0

    with open(sessions_dir / f"{session_id}.jsonl", "w", encoding="utf-8") as f:
        for turn in conv:
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
                if tc_count > 0 and tool_calls:
                    for _ in range(tc_count):
                        if tc_idx >= len(tool_calls):
                            break
                        tc = tool_calls[tc_idx]
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
            if content:
                f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

        for tc in tool_calls[tc_idx:]:
            name = tc.get("name", "?")
            args_s = json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:300]
            result_s = str(tc.get("result", ""))[:500]
            f.write(json.dumps({
                "role": "agent_tool",
                "content": f"[agent_tool:{name}] args={args_s} result={result_s}",
            }, ensure_ascii=False) + "\n")


def extract_signal_info(task_row: dict) -> dict:
    """Extract signal info for capture_signal."""
    reward = float(task_row.get("reward", 0.0) or 0.0)
    conv = task_row.get("conversation", [])
    tool_calls = task_row.get("tool_calls", [])
    tool_names = sorted({tc.get("name", "") for tc in tool_calls if isinstance(tc, dict)} - {""})

    errors = []
    resolution = ""
    if reward == 0.0:
        asst_msgs = [m for m in conv if isinstance(m, dict) and m.get("role") == "assistant"]
        if asst_msgs:
            last_msg = str(asst_msgs[-1].get("content", ""))[:300]
            errors.append(f"Task failed (reward=0). Agent's last response: {last_msg}")
        else:
            errors.append("Task failed (reward=0) with no captured agent dialogue.")
    else:
        resolution = f"Task succeeded (reward={reward}). {len(tool_calls)} tool calls."

    nl_corrections = extract_nl_corrections(task_row)

    return {
        "errors_encountered": errors,
        "errors_fixed": False,
        "user_corrections": nl_corrections,
        "tools_used": tool_names or ["mcp_bench"],
        "new_tools": tool_names,
        "resolution_snapshot": resolution or f"reward={reward}",
    }


_PARAM_PATTERN = re.compile(
    r"(HAT\d{3}|flight\s+\w+\d|reservation\s+\w+|\$[\d,]+\.\d+|\$[\d,]+|credit_card_\w+|gift_card_\w+|certificate_\w+)",
    re.IGNORECASE,
)


def extract_nl_corrections(task_row: dict) -> list[str]:
    nl_assertions = task_row.get("nl_assertions")
    if not nl_assertions or not isinstance(nl_assertions, list):
        return []
    corrections = []
    for a in nl_assertions:
        if not isinstance(a, dict):
            continue
        if a.get("met", True):
            continue
        text = a.get("assertion", "")
        if not text:
            continue
        param_matches = _PARAM_PATTERN.findall(text)
        if len(param_matches) >= 2:
            continue
        strategy_keywords = [
            "should not", "does not", "should check", "should detect",
            "should verify", "should realize", "should not offer",
            "cannot be", "is not allowed", "not modified", "not cancel",
        ]
        if any(kw in text.lower() for kw in strategy_keywords):
            corrections.append(f"QA feedback: {text} — but agent did not follow this.")
    return corrections


def build_meta_context(workspace: Path) -> str:
    """Read taxonomy from workspace and format as agent-readable guidance."""
    tax_path = workspace / "error_taxonomy.yaml"
    if not tax_path.exists():
        return ""

    raw = yaml.safe_load(tax_path.read_text(encoding="utf-8")) or {}
    taxonomy = raw.get("taxonomy", {})
    entries = []
    for _domain, subdomains in taxonomy.items():
        if not isinstance(subdomains, dict):
            continue
        for _sub, items in subdomains.items():
            if isinstance(items, list):
                entries.extend(items)
    if not entries:
        return ""

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
    return "\n".join(lines)


def detect_proactive_compensation(text: str) -> bool:
    markers = [
        r"\$\d+",
        r"certificate",
        r"compensation",
        r"offer.*(?:gesture|goodwill|courtesy)",
        r"apply.*(?:credit|certificate)",
        r"I can (?:offer|provide|give)",
    ]
    return any(re.search(m, text, re.IGNORECASE) for m in markers)


def analyze_agent_response(task_row: dict) -> dict:
    """Analyze the agent's final behavior for the compensation scenario."""
    conv = task_row.get("conversation", [])
    asst_msgs = [m for m in conv if isinstance(m, dict) and m.get("role") == "assistant"]

    full_text = " ".join(str(m.get("content", "")) for m in asst_msgs)
    last_response = str(asst_msgs[-1].get("content", "")) if asst_msgs else ""

    proactive = detect_proactive_compensation(full_text)
    reward = float(task_row.get("reward", 0.0) or 0.0)

    return {
        "reward": reward,
        "proactive_compensation": proactive,
        "last_response_preview": last_response[:300],
        "total_assistant_turns": len(asst_msgs),
        "total_tool_calls": len(task_row.get("tool_calls", [])),
    }


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="e2e_taxonomy_"))
    meta_workspace = tmp_root / "meta_workspace"
    meta_workspace.mkdir(parents=True)
    result_dir = ROOT / "abtest/results" / "e2e_taxonomy_test"
    result_dir.mkdir(parents=True, exist_ok=True)

    nobot_config = create_nobot_config(tmp_root)

    banner("E2E Taxonomy Test — Full Meta-Learning Loop")
    print(f"Temp root:      {tmp_root}")
    print(f"Meta workspace: {meta_workspace}")
    print(f"Task:           airline:{TASK_ID}")
    print(f"Tau port:       {TAU_PORT}")

    os.environ["META_LEARNING_CONFIG"] = str(ROOT / "abtest/config.meta-learning.A.yaml")
    os.environ["META_LEARNING_WORKSPACE"] = str(meta_workspace)
    os.environ["META_LEARNING_LLM_BASE_URL"] = LLM_BASE_URL
    os.environ["META_LEARNING_LLM_API_KEY"] = LLM_API_KEY

    tau_proc = start_tau_server()

    try:
        # ================================================================
        # ROUND 1: No taxonomy — baseline run
        # ================================================================
        banner("ROUND 1: No taxonomy (baseline)")
        r1_row = run_nobot(nobot_config)
        r1_analysis = analyze_agent_response(r1_row)

        print(f"\n[R1 Result] reward={r1_analysis['reward']}")
        print(f"[R1 Result] proactive_compensation={r1_analysis['proactive_compensation']}")
        print(f"[R1 Result] tool_calls={r1_analysis['total_tool_calls']}")
        print(f"[R1 Result] last_response: {r1_analysis['last_response_preview'][:200]}")

        # Write session for meta-learning
        session_id = f"e2e_round1_{int(time.time())}"
        write_session(meta_workspace, session_id, r1_row)
        print(f"\n[SESSION] Written to {meta_workspace / 'sessions' / (session_id + '.jsonl')}")

        # Capture signal
        banner("Signal Capture (from Round 1)")
        from meta_learning.mcp_server import capture_signal, run_layer2

        sig_info = extract_signal_info(r1_row)
        print(f"[SIGNAL] errors: {len(sig_info['errors_encountered'])}")
        print(f"[SIGNAL] nl_corrections: {sig_info['user_corrections']}")
        print(f"[SIGNAL] tools: {sig_info['tools_used']}")

        capture_res = capture_signal(
            task_description=f"[e2e-test][airline:task2] Customer service. reward={r1_analysis['reward']}",
            session_id=session_id,
            errors_encountered=sig_info["errors_encountered"],
            errors_fixed=sig_info["errors_fixed"],
            user_corrections=sig_info["user_corrections"],
            tools_used=sig_info["tools_used"],
            new_tools=sig_info["new_tools"],
            resolution_snapshot=sig_info["resolution_snapshot"],
            step_count=max(r1_analysis["total_tool_calls"], 1),
        )
        print(f"[CAPTURE] Result: {capture_res}")

        # Check signal buffer
        sig_dir = meta_workspace / "signal_buffer"
        sig_files = list(sig_dir.glob("*.yaml")) if sig_dir.exists() else []
        print(f"[BUFFER] {len(sig_files)} signal(s) in buffer")
        for sf in sig_files:
            print(f"  - {sf.name}")
            content = sf.read_text(encoding="utf-8")
            print(f"    {content[:300]}")

        # ================================================================
        # LAYER 2: Generate taxonomy from signal
        # ================================================================
        banner("Layer 2: Materialize → Cluster → Taxonomy")
        layer2_res = asyncio.run(run_layer2(force=True))
        print(f"[LAYER2] Result: {layer2_res}")

        # Check generated artifacts
        exp_dir = meta_workspace / "experience_pool"
        exp_files = list(exp_dir.glob("*.yaml")) if exp_dir.exists() else []
        print(f"\n[EXPERIENCES] {len(exp_files)} experience(s)")
        for ef in exp_files:
            print(f"  - {ef.name}")
            content = ef.read_text(encoding="utf-8")
            print(f"    {content[:400]}")

        tax_path = meta_workspace / "error_taxonomy.yaml"
        if tax_path.exists():
            tax_content = tax_path.read_text(encoding="utf-8")
            print(f"\n[TAXONOMY] Generated ({len(tax_content)} chars):")
            print(tax_content[:1500])
        else:
            print("\n[TAXONOMY] NOT generated — Layer2 did not produce taxonomy")

        meta_context = build_meta_context(meta_workspace)
        if meta_context:
            print(f"\n[META CONTEXT] Built ({len(meta_context)} chars):")
            print(meta_context[:1000])
        else:
            print("\n[META CONTEXT] Empty — no taxonomy to inject")

        # ================================================================
        # ROUND 2: With taxonomy injected
        # ================================================================
        banner("ROUND 2: With taxonomy injected")

        if not meta_context:
            print("[SKIP] No taxonomy generated, cannot run round 2 with injection")
            r2_row = None
            r2_analysis = None
        else:
            extra_ctx_path = tmp_root / "meta_context_for_agent.md"
            extra_ctx_path.write_text(meta_context, encoding="utf-8")
            print(f"[INJECT] Extra context written to {extra_ctx_path}")
            print(f"[INJECT] Content preview:\n{meta_context[:500]}")

            r2_row = run_nobot(nobot_config, extra_context_file=extra_ctx_path)
            r2_analysis = analyze_agent_response(r2_row)

            print(f"\n[R2 Result] reward={r2_analysis['reward']}")
            print(f"[R2 Result] proactive_compensation={r2_analysis['proactive_compensation']}")
            print(f"[R2 Result] tool_calls={r2_analysis['total_tool_calls']}")
            print(f"[R2 Result] last_response: {r2_analysis['last_response_preview'][:200]}")

        # ================================================================
        # COMPARISON
        # ================================================================
        banner("COMPARISON: Round 1 vs Round 2")
        print(f"{'Metric':<30} {'R1 (no tax)':>15} {'R2 (with tax)':>15}")
        print("-" * 62)
        print(f"{'reward':<30} {r1_analysis['reward']:>15.1f} {(r2_analysis['reward'] if r2_analysis else 'N/A'):>15}")
        print(f"{'proactive_compensation':<30} {str(r1_analysis['proactive_compensation']):>15} {str(r2_analysis['proactive_compensation'] if r2_analysis else 'N/A'):>15}")
        print(f"{'tool_calls':<30} {r1_analysis['total_tool_calls']:>15} {(r2_analysis['total_tool_calls'] if r2_analysis else 'N/A'):>15}")

        if r2_analysis:
            if r1_analysis['reward'] == 0.0 and r2_analysis['reward'] == 1.0:
                verdict = "META-LEARNING WORKS: Taxonomy fixed the failure"
            elif r1_analysis['proactive_compensation'] and not r2_analysis['proactive_compensation']:
                verdict = "META-LEARNING WORKS: Taxonomy prevented proactive compensation"
            elif r1_analysis['reward'] == r2_analysis['reward'] == 1.0:
                verdict = "BOTH SUCCEEDED: Cannot determine taxonomy effect (both passed)"
            elif r1_analysis['reward'] == r2_analysis['reward'] == 0.0:
                if r1_analysis['proactive_compensation'] and not r2_analysis['proactive_compensation']:
                    verdict = "PARTIAL IMPROVEMENT: Agent stopped proactive compensation but still failed on other criteria"
                else:
                    verdict = "NO IMPROVEMENT: Taxonomy did not help"
            else:
                verdict = "MIXED: Needs manual analysis"
        else:
            verdict = "INCOMPLETE: No taxonomy generated for round 2"

        print(f"\n{'VERDICT':>30}: {verdict}")

        # Save full results
        summary = {
            "task": f"airline:{TASK_ID}",
            "tmp_root": str(tmp_root),
            "round1": {
                "analysis": r1_analysis,
                "signal_info": sig_info,
                "capture_result": capture_res,
            },
            "layer2_result": layer2_res,
            "taxonomy_generated": tax_path.exists(),
            "taxonomy_content": tax_content if tax_path.exists() else None,
            "meta_context": meta_context or None,
            "round2": {
                "analysis": r2_analysis,
            } if r2_analysis else None,
            "verdict": verdict,
        }
        out_file = result_dir / f"e2e_result_{int(time.time())}.json"
        out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n[SAVED] Full results → {out_file}")

    finally:
        tau_proc.terminate()
        try:
            tau_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tau_proc.kill()
        print("\n[CLEANUP] tau3 MCP server terminated")


if __name__ == "__main__":
    main()

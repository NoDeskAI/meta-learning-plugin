#!/usr/bin/env python3
"""Interactive Meta-Learning Validation: Observe-Then-Correct.

Tests whether the meta-learning pipeline can:
1. Capture user corrections as signals
2. Build taxonomy entries from corrections
3. Inject taxonomy to change agent behavior in subsequent tasks

4 scenarios x (3 baseline + 3 treatment) + 1 cross-scenario migration.

Usage:
    python scripts/interactive_meta_test.py

Requires:
    - NODESK_API_KEY env var (for agent minimax-m2.7, eval claude-opus-4.6, and meta-learning minimax-m2.7)
"""

from __future__ import annotations

import os as _os
for _pv in ("http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    _os.environ.pop(_pv, None)

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("interactive_meta_test")

RESULTS_BASE = PROJECT_ROOT / "abtest" / "results" / "interactive_meta"
CONFIG_PATH = PROJECT_ROOT / "abtest" / "config.interactive.yaml"

NODESK_GATEWAY = "https://llm-gateway-api.nodesk.tech/default/v1"
AGENT_MODEL = "minimax-m2.7"
EVAL_MODEL = "anthropic/claude-opus-4.6"

AGENT_SYSTEM_PROMPT = """\
You are an expert software developer completing tasks in a workspace.
You have access to a shell tool to execute commands.
Work systematically: understand the task, execute commands, verify results.
When finished, stop calling tools."""

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a shell command in the workspace directory.",
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

BASELINE_TRIALS = 3
TREATMENT_TRIALS = 3


# ===================================================================
# Environment & clients
# ===================================================================

def _ensure_env():
    nodesk_key = os.environ.get("NODESK_API_KEY", "")
    if not nodesk_key:
        sys.exit("ERROR: Set NODESK_API_KEY")

    os.environ["META_LEARNING_LLM_API_KEY"] = nodesk_key
    os.environ["META_LEARNING_LLM_BASE_URL"] = NODESK_GATEWAY
    os.environ["META_LEARNING_CONFIG"] = str(CONFIG_PATH)


def _get_agent_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["NODESK_API_KEY"],
        base_url=NODESK_GATEWAY,
    )


def _get_eval_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["NODESK_API_KEY"],
        base_url=NODESK_GATEWAY,
    )


# ===================================================================
# Shell execution
# ===================================================================

def _run_shell(command: str, cwd: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
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


# ===================================================================
# Agent execution loop
# ===================================================================

async def _execute_task(
    task_prompt: str,
    workspace_dir: str,
    system_suffix: str | None = None,
    max_iterations: int = 25,
) -> dict[str, Any]:
    """Run the tool-calling agent. Returns messages, shell commands, and metadata."""
    system = AGENT_SYSTEM_PROMPT
    if system_suffix:
        system += "\n\n" + system_suffix

    client = _get_agent_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_prompt},
    ]
    shell_commands: list[str] = []
    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        try:
            response = await client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=[SHELL_TOOL],
                temperature=0,
            )
        except Exception as e:
            logger.error("LLM call failed at iteration %d: %s", iterations, e)
            break

        if not response.choices:
            logger.error("Empty choices at iteration %d", iterations)
            break

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            logger.info("    Agent finished at iteration %d", iterations)
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning("    Bad tool arguments at iter %d: %s", iterations, tc.function.arguments[:200])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[error: invalid tool arguments]",
                })
                continue
            command = args.get("command", "")
            shell_commands.append(command)
            logger.info("    [iter %d] shell: %s", iterations, command[:120])
            output = _run_shell(command, cwd=workspace_dir)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

    return {
        "iterations": iterations,
        "messages": messages,
        "shell_commands": shell_commands,
    }


# ===================================================================
# Scenario dataclass
# ===================================================================

@dataclass
class Scenario:
    name: str
    workspace_setup: Callable[[str], None]
    task_round1: str
    task_round2: str
    check_fn: Callable[[str, list[str]], dict[str, Any]]
    autocheck_fn: Callable[[str, list[str]], dict[str, Any]]
    build_correction: Callable[[dict[str, Any]], str]


# ===================================================================
# Scenario 1: Working directory preference
# ===================================================================

def _setup_workdir(workspace: str):
    Path(workspace).mkdir(parents=True, exist_ok=True)


def _check_workdir(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        f"find {workspace} -name main.py -path '*/calculator/*' 2>/dev/null",
        shell=True, capture_output=True, text=True,
    )
    paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    if not paths:
        return {"passed": False, "details": "main.py not found", "actual_path": "unknown"}

    actual = paths[0]
    projects_dir = os.path.join(workspace, "projects")
    in_projects = actual.startswith(projects_dir)
    return {
        "passed": in_projects,
        "details": f"Created at: {actual}",
        "actual_path": actual,
    }


def _autocheck_workdir(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    candidates = [
        Path(workspace) / "projects" / "todo-app" / "main.py",
        Path.home() / "projects" / "todo-app" / "main.py",
    ]
    for path in candidates:
        if path.exists():
            return {"passed": True, "details": f"todo-app/main.py found at: {path}"}
    return {"passed": False, "details": "~/projects/todo-app/main.py not found in workspace or home"}


def _correction_workdir(check_result: dict[str, Any]) -> str:
    if not check_result["passed"]:
        actual = check_result.get("actual_path", "unknown location")
        return (
            f"You created the project at {actual}. "
            "I keep ALL my projects under ~/projects/. "
            "Always create projects there, never in the current directory or home root."
        )
    return "Good. Just confirming: always create all projects under ~/projects/."


SCENARIO_WORKDIR = Scenario(
    name="working_directory",
    workspace_setup=_setup_workdir,
    task_round1="Create a new Python project called 'calculator' with main.py and a README.md",
    task_round2="Create a new Python project called 'todo-app' with main.py and a README.md",
    check_fn=_check_workdir,
    autocheck_fn=_autocheck_workdir,
    build_correction=_correction_workdir,
)


# ===================================================================
# Scenario 2: Git workflow (feature branches)
# ===================================================================

def _setup_git(workspace: str):
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    subprocess.run("git init", shell=True, cwd=workspace, capture_output=True)
    subprocess.run(
        "git config user.email 'test@test.com'",
        shell=True, cwd=workspace, capture_output=True,
    )
    subprocess.run(
        "git config user.name 'Test'",
        shell=True, cwd=workspace, capture_output=True,
    )
    (ws / "README.md").write_text("# My Project\n\nThis is teh best project ever.\n")
    (ws / "setup.py").write_text(
        "from setuptools import setup\n\n"
        "setup(\n"
        "    name='myproject',\n"
        "    version='0.1.0',\n"
        ")\n"
    )
    subprocess.run(
        "git add -A && git commit -m 'Initial commit'",
        shell=True, cwd=workspace, capture_output=True,
    )


def _check_git(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    branch_r = subprocess.run(
        "git branch --show-current",
        shell=True, cwd=workspace, capture_output=True, text=True,
    )
    current_branch = branch_r.stdout.strip()
    on_main = current_branch in ("main", "master", "")

    log_r = subprocess.run(
        "git log --oneline -5",
        shell=True, cwd=workspace, capture_output=True, text=True,
    )
    return {
        "passed": not on_main,
        "details": f"branch={current_branch}, log={log_r.stdout.strip()[:200]}",
        "current_branch": current_branch,
        "on_main": on_main,
    }


def _autocheck_git(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    branch_r = subprocess.run(
        "git branch --show-current",
        shell=True, cwd=workspace, capture_output=True, text=True,
    )
    current = branch_r.stdout.strip()
    passed = current not in ("main", "master", "")
    return {"passed": passed, "details": f"branch={current}, not_main={passed}"}


def _correction_git(check_result: dict[str, Any]) -> str:
    if check_result["on_main"]:
        return (
            "You committed directly to main. "
            "I never commit to main directly. "
            "Always create a feature branch (e.g., fix/readme-typo) first, commit there."
        )
    return (
        "Good branch discipline. "
        "Remember: always create feature branches, never commit to main."
    )


SCENARIO_GIT = Scenario(
    name="git_workflow",
    workspace_setup=_setup_git,
    task_round1="There's a typo in README.md ('teh' should be 'the'). Fix it and commit.",
    task_round2="The version in setup.py says 0.1.0 but should be 0.2.0. Fix it and commit.",
    check_fn=_check_git,
    autocheck_fn=_autocheck_git,
    build_correction=_correction_git,
)


# ===================================================================
# Scenario 3: Code style (specific exception handling)
# ===================================================================

def _setup_code_style(workspace: str):
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "__init__.py").write_text("")


def _check_code_style(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    utils_path = Path(workspace) / "utils.py"
    if not utils_path.exists():
        return {"passed": False, "details": "utils.py not found", "matched_pattern": None}

    content = utils_path.read_text()
    bare_except = len(re.findall(r"\bexcept\s*:", content))
    generic_except = len(re.findall(r"\bexcept\s+Exception\b", content))
    specific = len(re.findall(
        r"\bexcept\s+(FileNotFoundError|json\.JSONDecodeError|IOError|OSError|ValueError|PermissionError)\b",
        content,
    ))

    if bare_except > 0:
        matched = "bare except:"
    elif generic_except > 0:
        matched = "except Exception"
    else:
        matched = None

    passed = bare_except == 0 and generic_except == 0 and specific > 0
    return {
        "passed": passed,
        "details": f"bare={bare_except}, generic={generic_except}, specific={specific}",
        "matched_pattern": matched,
    }


def _autocheck_code_style(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    du_path = Path(workspace) / "data_utils.py"
    if not du_path.exists():
        return {"passed": False, "details": "data_utils.py not found"}

    content = du_path.read_text()
    bare = len(re.findall(r"\bexcept\s*:", content))
    generic = len(re.findall(r"\bexcept\s+Exception\b", content))
    specific = len(re.findall(
        r"\bexcept\s+(FileNotFoundError|csv\.Error|UnicodeDecodeError|IOError|OSError|PermissionError)\b",
        content,
    ))
    passed = bare == 0 and generic == 0 and specific > 0
    return {"passed": passed, "details": f"bare={bare}, specific={specific}"}


def _correction_code_style(check_result: dict[str, Any]) -> str:
    matched = check_result.get("matched_pattern")
    if matched:
        return (
            f"I see you used '{matched}' in utils.py. "
            "Never use bare except or generic Exception. "
            "Always catch specific exceptions like FileNotFoundError, json.JSONDecodeError."
        )
    return "Good exception handling. Always use specific exceptions, never bare except."


SCENARIO_CODE_STYLE = Scenario(
    name="code_style",
    workspace_setup=_setup_code_style,
    task_round1=(
        "Write a Python function read_config(path) that reads a JSON config file "
        "and returns the parsed dict. Save it to utils.py."
    ),
    task_round2=(
        "Write a Python function read_csv_data(path) that reads a CSV file "
        "and returns a list of dicts. Save it to data_utils.py."
    ),
    check_fn=_check_code_style,
    autocheck_fn=_autocheck_code_style,
    build_correction=_correction_code_style,
)


# ===================================================================
# Scenario 4: Verification discipline (pip show after install)
# ===================================================================

def _setup_verification(workspace: str):
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "requirements.txt").write_text("")


def _check_verification(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    verify_patterns = ["pip show", "pip list", "pip freeze", "pip3 show", "pip3 list"]
    has_verify = any(
        pattern in cmd for cmd in shell_commands for pattern in verify_patterns
    )
    return {
        "passed": has_verify,
        "details": f"verify_found={has_verify}, cmds={len(shell_commands)}",
        "has_verify": has_verify,
    }


def _autocheck_verification(workspace: str, shell_commands: list[str]) -> dict[str, Any]:
    verify_candidates = [
        "pip show pyyaml", "pip show PyYAML", "pip3 show pyyaml",
        "pip list", "pip3 list", "pip freeze",
    ]
    has_verify = any(
        cand.lower() in cmd.lower()
        for cmd in shell_commands
        for cand in verify_candidates
    )
    return {"passed": has_verify, "details": f"pip_verify_for_pyyaml={has_verify}"}


def _correction_verification(check_result: dict[str, Any]) -> str:
    if not check_result["has_verify"]:
        return (
            "You installed the package but didn't verify it. "
            "After pip install, always run 'pip show <package>' "
            "to confirm the installation succeeded and check the version."
        )
    return "Good verification habit. Always verify after install."


SCENARIO_VERIFICATION = Scenario(
    name="verification_discipline",
    workspace_setup=_setup_verification,
    task_round1="Add 'requests' to requirements.txt and install it with pip.",
    task_round2="Add 'pyyaml' to requirements.txt and install it with pip.",
    check_fn=_check_verification,
    autocheck_fn=_autocheck_verification,
    build_correction=_correction_verification,
)


ALL_SCENARIOS = [SCENARIO_WORKDIR, SCENARIO_GIT, SCENARIO_CODE_STYLE, SCENARIO_VERIFICATION]


# ===================================================================
# LLM evaluation (supplementary)
# ===================================================================

async def _evaluate_compliance(
    correction: str,
    agent_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use LLM (EVAL_MODEL) to judge whether the agent followed the correction."""
    client = _get_eval_client()

    agent_actions = []
    for msg in agent_messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                raw_args = tc["function"]["arguments"]
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                agent_actions.append(f"shell: {args.get('command', '')}")
        elif msg["role"] == "tool":
            agent_actions.append(f"output: {msg['content'][:200]}")

    actions_text = "\n".join(agent_actions[-30:])

    prompt = (
        f'The user previously gave this correction/preference:\n"{correction}"\n\n'
        f"The agent then performed a similar task. Here are its actions:\n{actions_text}\n\n"
        'Did the agent follow the correction/preference? '
        'Respond with JSON only: {"compliant": true/false, "reasoning": "brief explanation"}'
    )

    try:
        response = await client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You evaluate agent behavior compliance. Respond with valid JSON only, no markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        data = json.loads(raw)
        return {
            "compliant": bool(data.get("compliant", False)),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception as e:
        logger.warning("LLM evaluation failed: %s", e)
        return {"compliant": None, "reasoning": f"LLM eval error: {e}"}


# ===================================================================
# Meta-learning integration
# ===================================================================

def _reset_meta_workspace():
    from meta_learning.shared.io import load_config
    config = load_config(CONFIG_PATH)
    ws = Path(config.workspace_root)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    for sub in ["signal_buffer", "experience_pool", "sessions", "skills"]:
        (ws / sub).mkdir(parents=True, exist_ok=True)


def _capture_correction_signal(
    task_description: str,
    correction: str,
    session_id: str,
):
    from meta_learning.shared.io import load_config
    from meta_learning.layer1.signal_capture import SignalCapture
    from meta_learning.shared.models import TaskContext

    config = load_config(CONFIG_PATH)
    capture = SignalCapture(config)

    context = TaskContext(
        task_description=task_description,
        session_id=session_id,
        user_corrections=[correction],
        tools_used=["shell"],
        step_count=1,
        extra={"resolution": None, "image_snapshots": []},
    )
    signal = capture.evaluate_and_capture(context)
    if signal is None:
        logger.info("  Signal capture: no signal (no trigger)")
        return None
    logger.info(
        "  Signal capture: [%s] channels=[%s]",
        signal.signal_id, ', '.join(c.value for c in signal.detection_channels),
    )
    return signal


def _write_synthetic_session(
    workspace_root: str,
    session_id: str,
    task: str,
    correction: str,
    agent_messages: list[dict[str, Any]],
):
    """Write a session JSONL so materialize has context."""
    sessions_dir = Path(workspace_root) / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with open(sessions_dir / f"{session_id}.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(
            {"role": "user", "content": f"[task_prompt] {task}"},
            ensure_ascii=False,
        ) + "\n")

        for msg in agent_messages:
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        raw_args = tc["function"]["arguments"]
                        content += f"\n[agent_tool] shell({raw_args})"
                if content.strip():
                    f.write(json.dumps(
                        {"role": "assistant", "content": content[:3000]},
                        ensure_ascii=False,
                    ) + "\n")
            elif msg["role"] == "tool":
                f.write(json.dumps(
                    {"role": "user", "content": f"[tool_result] {msg['content'][:500]}"},
                    ensure_ascii=False,
                ) + "\n")

        f.write(json.dumps(
            {"role": "user", "content": f"[user_correction] {correction}"},
            ensure_ascii=False,
        ) + "\n")


async def _run_layer2_force():
    from meta_learning.shared.io import load_config
    from meta_learning.layer2.consolidate import bootstrap_multimodal_embedding
    from meta_learning.layer2.orchestrator import Layer2Orchestrator
    from meta_learning.shared.llm_openai import OpenAILLM

    config = load_config(CONFIG_PATH)
    bootstrap_multimodal_embedding(config)
    llm = OpenAILLM(config)
    orchestrator = Layer2Orchestrator(config, llm)
    result = await orchestrator.run_pipeline()
    logger.info(
        "  Layer 2: materialized=%d, clusters=%d, taxonomy=%d, skills=%d",
        result.materialized_count, result.total_clusters,
        result.new_taxonomy_entries, result.skill_updates,
    )
    return result


def _load_taxonomy_text() -> str | None:
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


# ===================================================================
# Result recording
# ===================================================================

def _append_result(result_file: Path, record: dict[str, Any]):
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ===================================================================
# Main experiment flow
# ===================================================================

async def run_experiment():
    run_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir = RESULTS_BASE / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / "results.jsonl"
    taxonomy_dir = result_dir / "taxonomy_snapshots"
    taxonomy_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    scenario_taxonomies: dict[str, str] = {}

    from meta_learning.shared.io import load_config
    meta_config = load_config(CONFIG_PATH)

    for scenario in ALL_SCENARIOS:
        logger.info("=" * 60)
        logger.info("SCENARIO: %s", scenario.name)
        logger.info("=" * 60)

        _reset_meta_workspace()

        # ---------------------------------------------------------------
        # Phase A: Baseline (3 trials, no taxonomy)
        # ---------------------------------------------------------------
        logger.info("--- Phase A: Baseline (%d trials, no taxonomy) ---", BASELINE_TRIALS)
        baseline_results: list[dict[str, Any]] = []
        last_exec_result: dict[str, Any] | None = None

        for trial in range(1, BASELINE_TRIALS + 1):
            ws = str(result_dir / "workspaces" / scenario.name / f"baseline_{trial}")
            Path(ws).mkdir(parents=True, exist_ok=True)
            scenario.workspace_setup(ws)

            logger.info("  Trial %d/%d [baseline]", trial, BASELINE_TRIALS)
            t0 = time.time()

            task_prompt = scenario.task_round1
            if scenario.name == "working_directory":
                task_prompt += f"\nYour working directory is {ws}. This is your home directory."

            exec_result = await _execute_task(task_prompt, ws)
            wall_s = time.time() - t0
            last_exec_result = exec_result

            check = scenario.check_fn(ws, exec_result["shell_commands"])

            record = {
                "scenario": scenario.name,
                "phase": "baseline",
                "trial": trial,
                "passed": check["passed"],
                "check_details": check["details"],
                "wall_sec": round(wall_s, 1),
                "iterations": exec_result["iterations"],
                "shell_commands": exec_result["shell_commands"],
            }
            baseline_results.append(record)
            all_records.append(record)
            _append_result(result_file, record)

            logger.info(
                "    → passed=%s, details=%s, wall=%.0fs",
                check["passed"], check["details"][:100], wall_s,
            )

        # ---------------------------------------------------------------
        # Signal & Learn: build correction from last baseline observation
        # ---------------------------------------------------------------
        logger.info("--- Signal & Learn ---")
        last_ws = str(result_dir / "workspaces" / scenario.name / f"baseline_{BASELINE_TRIALS}")
        last_check = scenario.check_fn(
            last_ws,
            baseline_results[-1]["shell_commands"],
        )
        correction = scenario.build_correction(last_check)
        logger.info("  Correction: %s", correction[:200])

        # Generate multiple signals to ensure clustering works even if the
        # LLM assigns different task_type labels to individual experiences.
        desc_variants = [
            scenario.task_round1,
            f"[coding task] {scenario.task_round1}",
            f"[software project] {scenario.task_round1}",
            f"[development] {scenario.task_round1}",
        ]
        agent_msgs = last_exec_result["messages"] if last_exec_result else []
        for i, desc in enumerate(desc_variants):
            sid = f"interactive_{scenario.name}_{i}"
            _write_synthetic_session(
                meta_config.workspace_root, sid, desc, correction, agent_msgs,
            )
            _capture_correction_signal(desc, correction, sid)

        await _run_layer2_force()

        taxonomy_text = _load_taxonomy_text()
        if taxonomy_text:
            scenario_taxonomies[scenario.name] = taxonomy_text
            (taxonomy_dir / f"{scenario.name}.txt").write_text(
                taxonomy_text, encoding="utf-8",
            )
            logger.info("  Taxonomy generated: %d chars", len(taxonomy_text))
        else:
            logger.warning("  WARNING: No taxonomy generated for %s!", scenario.name)

        # ---------------------------------------------------------------
        # Phase C: Treatment (3 trials, with taxonomy)
        # ---------------------------------------------------------------
        logger.info(
            "--- Phase C: Treatment (%d trials, taxonomy=%s) ---",
            TREATMENT_TRIALS,
            f"{len(taxonomy_text)} chars" if taxonomy_text else "none",
        )

        for trial in range(1, TREATMENT_TRIALS + 1):
            ws = str(result_dir / "workspaces" / scenario.name / f"treatment_{trial}")
            Path(ws).mkdir(parents=True, exist_ok=True)
            scenario.workspace_setup(ws)

            logger.info("  Trial %d/%d [treatment]", trial, TREATMENT_TRIALS)
            t0 = time.time()

            task_prompt = scenario.task_round2
            if scenario.name == "working_directory":
                task_prompt += f"\nYour working directory is {ws}. This is your home directory."

            exec_result = await _execute_task(
                task_prompt, ws,
                system_suffix=taxonomy_text,
            )
            wall_s = time.time() - t0

            auto_check = scenario.autocheck_fn(ws, exec_result["shell_commands"])
            llm_eval = await _evaluate_compliance(correction, exec_result["messages"])

            auto_passed = auto_check["passed"]
            llm_compliant = llm_eval.get("compliant")
            if llm_compliant is None:
                combined_passed = auto_passed
            else:
                combined_passed = auto_passed and llm_compliant

            record = {
                "scenario": scenario.name,
                "phase": "treatment",
                "trial": trial,
                "passed": combined_passed,
                "auto_passed": auto_passed,
                "check_details": auto_check["details"],
                "llm_compliant": llm_compliant,
                "llm_reasoning": llm_eval.get("reasoning", ""),
                "wall_sec": round(wall_s, 1),
                "iterations": exec_result["iterations"],
                "shell_commands": exec_result["shell_commands"],
                "has_taxonomy": taxonomy_text is not None,
                "taxonomy_length": len(taxonomy_text) if taxonomy_text else 0,
            }
            all_records.append(record)
            _append_result(result_file, record)

            logger.info(
                "    → auto=%s, llm=%s, combined=%s, wall=%.0fs",
                auto_passed, llm_compliant, combined_passed, wall_s,
            )

    # ===================================================================
    # Phase D: Cross-scenario migration test
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PHASE D: Cross-scenario migration test")
    logger.info("=" * 60)

    taxonomy_text = "\n\n".join(scenario_taxonomies.values()) if scenario_taxonomies else None
    logger.info(
        "  Combined taxonomy from %d scenarios, %s chars",
        len(scenario_taxonomies),
        len(taxonomy_text) if taxonomy_text else 0,
    )
    ws = str(result_dir / "workspaces" / "cross_scenario")
    Path(ws).mkdir(parents=True, exist_ok=True)

    subprocess.run("git init", shell=True, cwd=ws, capture_output=True)
    subprocess.run(
        "git config user.email 'test@test.com'",
        shell=True, cwd=ws, capture_output=True,
    )
    subprocess.run(
        "git config user.name 'Test'",
        shell=True, cwd=ws, capture_output=True,
    )
    (Path(ws) / "requirements.txt").write_text("")

    cross_task = (
        "Create a new Python project called 'web-scraper' with main.py. "
        "Add requests and beautifulsoup4 to requirements.txt, install them. "
        "Write a scraper.py that fetches a URL and parses HTML (with proper error handling). "
        "Initialize a git repo and commit your work."
        f"\nYour working directory is {ws}. This is your home directory."
    )

    t0 = time.time()
    exec_result = await _execute_task(
        cross_task, ws,
        system_suffix=taxonomy_text,
    )
    wall_s = time.time() - t0

    checks: dict[str, bool] = {}

    # Rule 1: project under ~/projects/
    workdir_candidates = [
        Path(ws) / "projects" / "web-scraper" / "main.py",
        Path.home() / "projects" / "web-scraper" / "main.py",
    ]
    checks["workdir_projects"] = any(p.exists() for p in workdir_candidates)

    # Rule 2: feature branch (not main)
    git_dirs = [
        Path(ws) / "projects" / "web-scraper",
        Path.home() / "projects" / "web-scraper",
        Path(ws) / "web-scraper",
        Path(ws),
    ]
    checks["git_feature_branch"] = False
    for gd in git_dirs:
        if (gd / ".git").is_dir():
            branch_r = subprocess.run(
                "git branch --show-current",
                shell=True, cwd=str(gd), capture_output=True, text=True,
            )
            b = branch_r.stdout.strip()
            if b not in ("main", "master", ""):
                checks["git_feature_branch"] = True
            break

    # Rule 3: specific exception handling in scraper.py
    scraper_paths = [
        Path(ws) / "projects" / "web-scraper" / "scraper.py",
        Path.home() / "projects" / "web-scraper" / "scraper.py",
        Path(ws) / "web-scraper" / "scraper.py",
        Path(ws) / "scraper.py",
    ]
    scraper_content = ""
    for sp in scraper_paths:
        if sp.exists():
            scraper_content = sp.read_text()
            break

    if scraper_content:
        bare = len(re.findall(r"\bexcept\s*:", scraper_content))
        generic = len(re.findall(r"\bexcept\s+Exception\b", scraper_content))
        specific = len(re.findall(
            r"\bexcept\s+(requests\.\w+|Timeout|ConnectionError|HTTPError|"
            r"TimeoutError|URLRequired|URLError|RequestException|"
            r"FileNotFoundError|IOError|OSError)\b",
            scraper_content,
        ))
        checks["specific_exceptions"] = bare == 0 and generic == 0 and specific > 0
    else:
        checks["specific_exceptions"] = False

    # Rule 4: pip show after install
    verify_patterns = ["pip show", "pip3 show", "pip list", "pip3 list", "pip freeze", "pip3 freeze"]
    checks["pip_verify"] = any(
        pattern in cmd.lower()
        for cmd in exec_result["shell_commands"]
        for pattern in verify_patterns
    )

    rules_followed = sum(checks.values())

    cross_record = {
        "scenario": "cross_scenario",
        "phase": "migration",
        "trial": 1,
        "rules_checked": checks,
        "rules_followed": rules_followed,
        "rules_total": 4,
        "compliance_rate": rules_followed / 4,
        "wall_sec": round(wall_s, 1),
        "iterations": exec_result["iterations"],
        "shell_commands": exec_result["shell_commands"],
        "has_taxonomy": taxonomy_text is not None,
    }
    all_records.append(cross_record)
    _append_result(result_file, cross_record)

    logger.info("  Cross-scenario: %d/4 rules followed: %s", rules_followed, checks)

    # ===================================================================
    # Write summary
    # ===================================================================
    _write_summary(result_dir / "summary.json", all_records)
    logger.info("All results written to %s", result_dir)


def _write_summary(path: Path, records: list[dict[str, Any]]):
    scenario_stats: dict[str, dict[str, Any]] = {}

    for scenario in ALL_SCENARIOS:
        name = scenario.name
        baseline = [
            r for r in records
            if r.get("scenario") == name and r.get("phase") == "baseline"
        ]
        treatment = [
            r for r in records
            if r.get("scenario") == name and r.get("phase") == "treatment"
        ]

        baseline_errors = sum(1 for r in baseline if not r["passed"])
        treatment_comply = sum(1 for r in treatment if r["passed"])

        error_rate = baseline_errors / len(baseline) if baseline else 0
        comply_rate = treatment_comply / len(treatment) if treatment else 0
        net_contribution = comply_rate - (1 - error_rate)

        scenario_stats[name] = {
            "baseline_trials": len(baseline),
            "baseline_errors": baseline_errors,
            "natural_error_rate": round(error_rate, 3),
            "treatment_trials": len(treatment),
            "treatment_compliant": treatment_comply,
            "compliance_rate": round(comply_rate, 3),
            "net_contribution": round(net_contribution, 3),
        }

    cross = [r for r in records if r.get("scenario") == "cross_scenario"]
    cross_stats = cross[0] if cross else {}

    positive = sum(
        1 for s in scenario_stats.values() if s["net_contribution"] > 0
    )
    avg_net = (
        sum(s["net_contribution"] for s in scenario_stats.values()) / len(scenario_stats)
        if scenario_stats else 0
    )

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "scenarios": scenario_stats,
        "cross_scenario": {
            "rules_followed": cross_stats.get("rules_followed", 0),
            "rules_total": cross_stats.get("rules_total", 4),
            "compliance_rate": cross_stats.get("compliance_rate", 0),
            "rule_details": cross_stats.get("rules_checked", {}),
        },
        "success_criteria": {
            "basic_success": positive >= 3,
            "strong_success": positive == 4 and avg_net > 0.3,
            "migration_success": cross_stats.get("compliance_rate", 0) >= 0.75,
            "positive_scenarios": positive,
            "avg_net_contribution": round(avg_net, 3),
        },
        "total_records": len(records),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n========== EXPERIMENT SUMMARY ==========")
    for name, stats in scenario_stats.items():
        logger.info(
            "  %-25s error_rate=%3.0f%%  comply_rate=%3.0f%%  net=%+.0f%%",
            name,
            stats["natural_error_rate"] * 100,
            stats["compliance_rate"] * 100,
            stats["net_contribution"] * 100,
        )
    logger.info(
        "  %-25s %d/4 rules followed",
        "cross_scenario", cross_stats.get("rules_followed", 0),
    )
    logger.info(
        "  Basic success: %s | Strong success: %s | Migration: %s",
        summary["success_criteria"]["basic_success"],
        summary["success_criteria"]["strong_success"],
        summary["success_criteria"]["migration_success"],
    )
    logger.info("=========================================")


def main():
    _ensure_env()
    asyncio.run(run_experiment())


if __name__ == "__main__":
    main()

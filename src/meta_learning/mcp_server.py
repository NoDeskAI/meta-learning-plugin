"""MCP Server for the meta-learning system.

Exposes Layer 1 Quick Think / Signal Capture, Layer 2/3 pipeline triggers,
and system status as MCP tools, resources, and prompts.

Run:  python -m meta_learning.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from meta_learning.layer1.quick_think import QuickThinkIndex
from meta_learning.layer1.signal_capture import SignalCapture
from meta_learning.shared.io import (
    boost_taxonomy_confidence,
    enrich_from_session,
    list_all_experiences,
    list_pending_signals,
    load_config,
    load_error_taxonomy,
    penalize_taxonomy_confidence,
    save_error_taxonomy,
)
from meta_learning.shared.models import (
    ErrorTaxonomy,
    MetaLearningConfig,
    QuickThinkResult,
    TaskContext,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("meta_learning.mcp")


def _configure_windows_stdio() -> None:
    """Force UTF-8 stdio on Windows for MCP stdio transport.

    Some Windows environments still default to GBK/ANSI for inherited stdio.
    The MCP client/server exchange is UTF-8 text over stdio, so we reconfigure
    the process streams early to avoid decode failures such as:
    `'gbk' codec can't decode byte ...`.
    """
    if os.name != "nt":
        return

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            logger.debug("Failed to reconfigure %s to UTF-8", stream_name, exc_info=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_server_config() -> MetaLearningConfig:
    config_path = os.environ.get("META_LEARNING_CONFIG")
    if config_path:
        config_path = str(Path(config_path).expanduser())
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = MetaLearningConfig()

    workspace = os.environ.get("META_LEARNING_WORKSPACE")
    if workspace:
        config.workspace_root = str(Path(workspace).expanduser())

    sessions_root = os.environ.get("META_LEARNING_SESSIONS_ROOT")
    if sessions_root:
        config.sessions_root = str(Path(sessions_root).expanduser())

    return config


_config: MetaLearningConfig | None = None


def _get_config() -> MetaLearningConfig:
    global _config
    if _config is None:
        _config = _load_server_config()
        logger.info("Config loaded: workspace=%s", _config.workspace_root)
    return _config


# ---------------------------------------------------------------------------
# QuickThinkIndex singleton with auto-reload
# ---------------------------------------------------------------------------

_qt_index: QuickThinkIndex | None = None
_taxonomy_mtime: float = 0.0
_layer2_task: asyncio.Task | None = None
_layer2_thread: threading.Thread | None = None


def _get_quick_think_index() -> QuickThinkIndex:
    global _qt_index, _taxonomy_mtime

    config = _get_config()
    tax_path = Path(config.error_taxonomy_full_path)

    current_mtime = tax_path.stat().st_mtime if tax_path.exists() else 0.0

    if _qt_index is None or current_mtime != _taxonomy_mtime:
        taxonomy = load_error_taxonomy(config)
        if _qt_index is None:
            _qt_index = QuickThinkIndex(taxonomy, config)
        else:
            _qt_index.update_taxonomy(taxonomy)
        _taxonomy_mtime = current_mtime
        logger.info("QuickThinkIndex (re)loaded, taxonomy mtime=%.0f", current_mtime)

    return _qt_index


# ---------------------------------------------------------------------------
# LLM factory (for Layer 2/3)
# ---------------------------------------------------------------------------

def _create_llm(config: MetaLearningConfig):
    if config.llm.provider == "openai":
        from meta_learning.shared.llm_openai import OpenAILLM
        return OpenAILLM(config)
    from meta_learning.shared.llm import StubLLM
    logger.warning("Using StubLLM — set llm.provider='openai' for real LLM calls")
    return StubLLM()


def _session_id_from_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("agent_"):
        parts = stem.split("_", 2)
        if len(parts) == 3:
            return f"agent:{parts[1]}:{parts[2]}"
    return stem.replace("_", ":")


def _infer_recent_session_id(config: MetaLearningConfig) -> str | None:
    session_dirs = [
        Path(config.sessions_full_path).expanduser(),
        Path(config.workspace_root).expanduser() / "sessions",
    ]
    candidates: list[Path] = []
    for session_dir in session_dirs:
        if not session_dir.exists():
            continue
        candidates.extend(
            path for path in session_dir.glob("*.jsonl")
            if path.is_file() and "capture-tmp-" not in path.name
        )
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return _session_id_from_file(latest)


def _coerce_str_list(value: list[str] | str | None) -> list[str]:
    """Normalize MCP list arguments.

    LLMs sometimes send a JSON-encoded array string for list parameters. Accept
    it here so learning capture is not lost to a recoverable argument-shape
    error.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    text = str(value).strip()
    return [text] if text else []


async def _run_layer2_background(config: MetaLearningConfig) -> None:
    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    bg_orchestrator = Layer2Orchestrator(config, _create_llm(config))
    bg_orchestrator.mark_running()
    try:
        result = await bg_orchestrator.run_pipeline()
        bg_orchestrator.mark_completed(result)
        logger.info(
            "Background Layer 2 complete: materialized=%d, clusters=%d, "
            "new_taxonomy=%d, skill_updates=%d",
            result.materialized_count,
            result.total_clusters,
            result.new_taxonomy_entries,
            result.skill_updates,
        )
    except Exception as exc:
        bg_orchestrator.mark_failed(str(exc))
        logger.exception("Background Layer 2 pipeline failed")


def _schedule_layer2_recovery(config: MetaLearningConfig) -> tuple[bool, int, str]:
    """Resume pending Layer 2 work in this MCP process when possible.

    Background tasks are process-local. If the MCP server or gateway restarts
    after signals are captured, the old asyncio task disappears while the
    signal files remain pending. The next status check should therefore recover
    the backlog instead of leaving it stuck forever.
    """
    global _layer2_task

    pending = list_pending_signals(config)
    if not pending:
        return False, 0, "no pending signals"

    if _layer2_task is not None and not _layer2_task.done():
        return False, len(pending), "already running in this process"

    if _layer2_thread is not None and _layer2_thread.is_alive():
        return False, len(pending), "already running in startup thread"

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    orchestrator = Layer2Orchestrator(config, _create_llm(config))
    if not orchestrator.should_trigger():
        return False, len(pending), "trigger conditions not met"

    _layer2_task = asyncio.create_task(_run_layer2_background(config))
    return True, len(pending), "scheduled"


def _start_layer2_recovery_thread_if_needed() -> None:
    """Start backlog recovery when the MCP server process starts.

    This covers gateway/MCP restarts where pending signal files survive but the
    previous in-process asyncio background task is gone.
    """
    global _layer2_thread

    config = _get_config()
    pending = list_pending_signals(config)
    if not pending:
        return

    if _layer2_thread is not None and _layer2_thread.is_alive():
        return

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    orchestrator = Layer2Orchestrator(config, _create_llm(config))
    if not orchestrator.should_trigger():
        return

    def _runner() -> None:
        asyncio.run(_run_layer2_background(config))

    _layer2_thread = threading.Thread(
        target=_runner,
        name="meta-learning-layer2-recovery",
        daemon=True,
    )
    _layer2_thread.start()
    logger.info(
        "Started Layer 2 recovery thread for %d pending signal(s)",
        len(pending),
    )


# ---------------------------------------------------------------------------
# Risk-warning formatter (port from TS quick-think.ts formatRiskWarning)
# ---------------------------------------------------------------------------

def _format_risk_warning(result: QuickThinkResult, taxonomy: ErrorTaxonomy) -> str:
    lines: list[str] = [f'<meta-learning-risk-assessment level="{result.risk_level}">']

    if "irreversible_operation" in result.matched_signals:
        lines.append(
            "WARNING: This task involves irreversible operations. "
            "Generate a rollback plan before executing."
        )

    if "keyword_taxonomy_hit" in result.matched_signals:
        lines.append("Known error patterns detected from past experience:")
        entries = taxonomy.all_entries()
        entry_map = {e.id: e for e in entries}
        for entry_id in result.matched_taxonomy_entries[:3]:
            entry = entry_map.get(entry_id)
            if entry:
                lines.append(f"  - [{entry.id}] {entry.name}: {entry.prevention}")

    if "recent_failure_pattern" in result.matched_signals:
        lines.append("WARNING: Similar pattern to a recent failure detected.")

    if "new_tool_usage" in result.matched_signals:
        lines.append(
            "NOTE: New/unfamiliar tools detected — proceed with extra caution."
        )

    lines.append("</meta-learning-risk-assessment>")
    return "\n".join(lines)


# ===================================================================
# FastMCP Server
# ===================================================================

mcp = FastMCP(
    "meta-learning",
    instructions=(
        "Meta-learning system for agent self-improvement. "
        "Use `quick_think` before executing risky tasks to get risk assessments. "
        "Use `capture_signal` after completing a task to record learning signals — "
        "Layer 2 consolidation runs automatically in the background when needed. "
        "If you need to wait for or verify learning completion, prefer spawning a "
        "separate learning worker and let the main user conversation continue. "
        "Use `status` to check the learning system state."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def quick_think(
    user_message: str,
    tools_used: list[str] | str | None = None,
) -> str:
    """Assess risk of a task before execution.

    Checks the user message against known error patterns (taxonomy) and
    irreversible-operation keywords. Returns a risk warning if any match,
    otherwise returns "no risk detected".

    Call this BEFORE executing a task that might involve destructive operations
    or areas where you have previously made mistakes.

    Args:
        user_message: The user's task description / latest message.
        tools_used: Names of tools you plan to use (optional). Prefer a real
            array of strings, e.g. ["read_file"]; a single string is accepted.
    """
    config = _get_config()
    index = _get_quick_think_index()

    context = TaskContext(
        task_description=user_message,
        tools_used=_coerce_str_list(tools_used),
    )
    result = index.evaluate(context)

    if not result.hit:
        return "no risk detected"

    taxonomy = load_error_taxonomy(config)
    return _format_risk_warning(result, taxonomy)


@mcp.tool()
async def capture_signal(
    task_description: str,
    session_id: str = "unknown",
    errors_encountered: list[str] | str | None = None,
    errors_fixed: bool = False,
    user_corrections: list[str] | str | None = None,
    tools_used: list[str] | str | None = None,
    new_tools: list[str] | str | None = None,
    resolution_snapshot: str | None = None,
    image_snapshots: list[str] | str | None = None,
    step_count: int = 0,
) -> str:
    """Record a learning signal after completing a task.

    Analyzes the task outcome and, if meaningful learning triggers are found
    (self-recovery, unresolved error, user correction, new tool, efficiency
    anomaly), writes a YAML signal file to signal_buffer/ for later Layer 2
    processing.

    If Layer 2 trigger conditions are met, the consolidation pipeline runs
    automatically in the background — no need to call run_layer2 separately.
    If you need to monitor completion or verify generated rules, spawn a
    separate learning worker to call layer2_status; do not keep the main user
    conversation blocked on learning progress.

    Call this AFTER completing a task, especially when:
    - You encountered and fixed errors
    - The user corrected your approach
    - You used a new/unfamiliar tool
    - The task took many steps

    Args:
        task_description: Brief summary of the task.
        session_id: Current session identifier.
        errors_encountered: Error messages or tracebacks seen during the task.
            Prefer a real array of strings, not a JSON string.
        errors_fixed: Whether the errors were successfully resolved.
        user_corrections: User feedback that corrected your approach. Must be
            a real array of strings when possible, e.g. ["use spawn for
            background learning"]; do not pass a JSON-encoded array string.
        tools_used: All tools invoked during the task. Prefer a real array.
        new_tools: Tools used for the first time. Prefer a real array.
        resolution_snapshot: Short summary of how the issue was resolved.
        image_snapshots: Paths to screenshots captured during this task.
            Prefer a real array.
        step_count: Total number of tool calls / steps taken.
    """
    config = _get_config()
    capture = SignalCapture(config)
    errors_encountered_list = _coerce_str_list(errors_encountered)
    user_corrections_list = _coerce_str_list(user_corrections)
    tools_used_list = _coerce_str_list(tools_used)
    new_tools_list = _coerce_str_list(new_tools)
    image_snapshots_list = _coerce_str_list(image_snapshots)

    if not session_id or session_id == "unknown":
        session_id = _infer_recent_session_id(config) or "unknown"

    if session_id and session_id != "unknown":
        enrichment = enrich_from_session(session_id, config)
        if not tools_used_list and enrichment.tools_used:
            tools_used_list = enrichment.tools_used
        if step_count == 0 and enrichment.step_count > 0:
            step_count = enrichment.step_count
    else:
        enrichment = None

    context = TaskContext(
        task_description=task_description,
        session_id=session_id,
        errors_encountered=errors_encountered_list,
        errors_fixed=errors_fixed,
        user_corrections=user_corrections_list,
        tools_used=tools_used_list,
        new_tools=new_tools_list,
        step_count=step_count,
        extra={
            "resolution": resolution_snapshot,
            "image_snapshots": image_snapshots_list,
            "action_trace": enrichment.action_trace if enrichment else None,
        },
    )

    signal = capture.evaluate_and_capture(context)
    if signal is None:
        return "no signal captured — task did not trigger any learning condition"

    if signal.error_snapshot and _qt_index is not None:
        _qt_index.register_failure_signature(signal.error_snapshot)

    result = (
        f"Signal captured: [{signal.signal_id}] "
        f"trigger={signal.trigger_reason}, "
        f"file={config.signal_buffer_path}/{signal.signal_id}.yaml"
    )

    pending = list_pending_signals(config)
    scheduled, _, reason = _schedule_layer2_recovery(config)
    if scheduled:
        result += (
            f"\n\nLayer 2 triggered in background "
            f"({len(pending)} pending signal(s)). "
            "Learning takes ~1-2 minutes. If completion must be verified, spawn "
            "a separate learning worker to poll `layer2_status` and notify the "
            "original conversation; keep the main response unblocked."
        )
    elif pending and reason == "already running in this process":
        result += "\n\nLayer 2 is already running in the background."

    return result


@mcp.tool()
async def run_layer2(force: bool = False) -> str:
    """Trigger the Layer 2 near-line consolidation pipeline.

    Processes pending signals into structured experiences, clusters them,
    builds/updates the error taxonomy, and evolves skills.

    Trigger conditions: USER_CORRECTION signals trigger immediately (1 signal);
    other signals trigger at >= 2 pending or >= 8h since last run.
    Use force=True to override.

    Args:
        force: Run even if trigger conditions are not met.
    """
    config = _get_config()
    llm = _create_llm(config)

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    orchestrator = Layer2Orchestrator(config, llm)

    if not force and not orchestrator.should_trigger():
        pending = list_pending_signals(config)
        return (
            f"Layer 2 trigger conditions not met "
            f"(pending signals: {len(pending)}). Use force=True to override."
        )

    orchestrator.mark_running()
    try:
        result = await orchestrator.run_pipeline()
        orchestrator.mark_completed(result)
    except Exception as exc:
        orchestrator.mark_failed(str(exc))
        raise

    return (
        f"Layer 2 complete: materialized={result.materialized_count}, "
        f"clusters={result.total_clusters}, "
        f"new_taxonomy={result.new_taxonomy_entries}, "
        f"skill_updates={result.skill_updates}"
    )


@mcp.tool()
async def layer2_status() -> str:
    """Check the current Layer 2 pipeline status.

    Returns whether the pipeline is idle, running, completed, or failed,
    along with timing and result details. When waiting for background learning
    after capture_signal, prefer calling this from a spawned learning worker so
    the main user conversation is not blocked by polling.
    """
    config = _get_config()

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    scheduled, pending_count, reason = _schedule_layer2_recovery(config)
    state = Layer2Orchestrator.load_state(config)
    status = state.get("status", "idle")

    if scheduled:
        return (
            f"Layer 2: RECOVERING ({pending_count} pending signal(s)). "
            "A background consolidation task was scheduled after restart. "
            "SKILL.md has NOT been updated yet — check again later."
        )

    if status == "idle":
        return "Layer 2: idle (no recent activity)"

    if status == "running":
        started = state.get("started_at", "unknown")
        try:
            elapsed = datetime.now() - datetime.fromisoformat(started)
            elapsed_s = int(elapsed.total_seconds())
            return (
                f"Layer 2: RUNNING (started {elapsed_s}s ago at {started}). "
                f"SKILL.md has NOT been updated yet — wait for completion."
            )
        except (ValueError, TypeError):
            return "Layer 2: RUNNING. SKILL.md has NOT been updated yet."

    if status == "completed":
        completed_at = state.get("completed_at", "unknown")
        res = state.get("result", {})
        pending_note = (
            f" Pending signals remain: {pending_count} ({reason})."
            if pending_count
            else ""
        )
        return (
            f"Layer 2: completed at {completed_at}. "
            f"Materialized {res.get('materialized_count', 0)} experiences, "
            f"created {res.get('new_taxonomy_entries', 0)} taxonomy entries, "
            f"{res.get('skill_updates', 0)} skill updates. "
            f"SKILL.md is up to date."
            f"{pending_note}"
        )

    if status == "failed":
        failed_at = state.get("failed_at", "unknown")
        error = state.get("error", "unknown error")
        return (
            f"Layer 2: FAILED at {failed_at}. Error: {error}. "
            f"SKILL.md may be stale. Run `run_layer2(force=True)` to retry."
        )

    return f"Layer 2: unknown status '{status}'"


@mcp.tool()
async def run_layer3() -> str:
    """Trigger the Layer 3 offline deep-learning pipeline.

    Performs cross-task pattern mining, capability gap detection, and
    memory architecture optimization. Should be run infrequently (e.g. weekly)
    when the experience pool has grown sufficiently.
    """
    config = _get_config()
    llm = _create_llm(config)

    from meta_learning.layer3.orchestrator import Layer3Orchestrator

    orchestrator = Layer3Orchestrator(config, llm)
    result = await orchestrator.run_pipeline()
    return (
        f"Layer 3 complete: patterns={len(result.cross_task_patterns)}, "
        f"gaps={len(result.capability_gaps)}, "
        f"recommendations={len(result.memory_recommendations)}"
    )


@mcp.tool()
def sync_taxonomy_to_nobot(
    nobot_workspace: str | None = None,
) -> str:
    """Sync the error taxonomy to nanobot workspace files.

    Generates/updates:
    - skills/meta-learning/SKILL.md (top-10 rules, always:true)
    - skills/meta-learning/rules/*.md (categorized details for human review)

    Args:
        nobot_workspace: Path to the nanobot workspace root.
            Defaults to ~/.deskclaw/nanobot/workspace.
    """
    config = _get_config()
    workspace = nobot_workspace or os.path.expanduser("~/.deskclaw/nanobot/workspace")
    skills_path = str(Path(workspace) / "skills")

    from meta_learning.sync_nobot import sync_taxonomy_to_nobot_workspace

    result = sync_taxonomy_to_nobot_workspace(config, skills_path)
    if result.total_entries == 0:
        return "No taxonomy entries to sync."
    return (
        f"Synced {result.total_entries} entries → "
        f"SKILL.md (top-{result.top_n_in_skill}), "
        f"{len(result.rules_written)} category files"
    )


@mcp.tool()
def status() -> str:
    """Show current meta-learning system status.

    Returns counts of pending signals, total experiences, taxonomy entries,
    and recent signal summaries.
    """
    config = _get_config()

    pending_signals = list_pending_signals(config)
    all_experiences = list_all_experiences(config)
    taxonomy = load_error_taxonomy(config)
    taxonomy_entries = taxonomy.all_entries()

    lines = [
        f"Workspace: {config.workspace_root}",
        f"Pending signals: {len(pending_signals)}",
        f"Total experiences: {len(all_experiences)}",
        f"Taxonomy entries: {len(taxonomy_entries)}",
    ]

    if pending_signals:
        lines.append("\nRecent pending signals:")
        for sig in pending_signals[-5:]:
            lines.append(
                f"  [{sig.signal_id}] [trigger={sig.trigger_reason}]: "
                f"{sig.task_summary[:60]}"
            )

    if taxonomy_entries:
        lines.append("\nTaxonomy entries:")
        for entry in taxonomy_entries[:10]:
            lines.append(
                f"  [{entry.id}] {entry.name} "
                f"(confidence={entry.confidence:.2f})"
            )

    return "\n".join(lines)


@mcp.tool()
def confirm_taxonomy_entry(entry_id: str) -> str:
    """Report that a taxonomy entry's advice was helpful during a task.

    Increases the entry's confidence adjustment (capped at +0.4).
    Call this when following a risk-assessment suggestion led to a
    successful outcome.

    Args:
        entry_id: The taxonomy entry ID (e.g. "tax-cod-typ-001").
    """
    config = _get_config()
    entry = boost_taxonomy_confidence(entry_id, config)
    if entry is None:
        return f"Taxonomy entry '{entry_id}' not found."
    return (
        f"Boosted [{entry.id}] {entry.name}: "
        f"confidence={entry.confidence:.2f} "
        f"(adjustment={entry.confidence_adjustment:+.2f})"
    )


@mcp.tool()
def contradict_taxonomy_entry(entry_id: str) -> str:
    """Report that a taxonomy entry's advice was wrong or harmful.

    Decreases the entry's confidence adjustment (floored at -0.5).
    Call this when following a risk-assessment suggestion led to a
    worse outcome or was irrelevant.

    Args:
        entry_id: The taxonomy entry ID (e.g. "tax-cod-typ-001").
    """
    config = _get_config()
    entry = penalize_taxonomy_confidence(entry_id, config)
    if entry is None:
        return f"Taxonomy entry '{entry_id}' not found."
    return (
        f"Penalized [{entry.id}] {entry.name}: "
        f"confidence={entry.confidence:.2f} "
        f"(adjustment={entry.confidence_adjustment:+.2f})"
    )


@mcp.tool()
def delete_taxonomy_entry(entry_id: str, sync_to_nobot: bool = True) -> str:
    """Delete a bad taxonomy entry by ID.

    Use this for polluted or invalid learned rules, such as placeholder/test
    signals accidentally materialized into the taxonomy.

    Args:
        entry_id: The taxonomy entry ID to remove.
        sync_to_nobot: Whether to sync SKILL.md/rules files after deletion.
    """
    global _qt_index, _taxonomy_mtime

    config = _get_config()
    taxonomy = load_error_taxonomy(config)
    removed = taxonomy.remove_entry(entry_id)
    if not removed:
        return f"Taxonomy entry '{entry_id}' not found."

    save_error_taxonomy(taxonomy, config)
    _qt_index = None
    _taxonomy_mtime = 0.0

    sync_msg = ""
    if sync_to_nobot:
        sync_msg = f"\n{sync_taxonomy_to_nobot()}"
    return f"Deleted taxonomy entry '{entry_id}'.{sync_msg}"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("meta-learning://taxonomy")
def get_taxonomy() -> str:
    """Current error taxonomy (YAML). Empty string if no taxonomy exists yet."""
    config = _get_config()
    tax_path = Path(config.error_taxonomy_full_path)
    if not tax_path.exists():
        return ""
    return tax_path.read_text(encoding="utf-8")


@mcp.resource("meta-learning://config")
def get_config_resource() -> str:
    """Current meta-learning configuration (JSON)."""
    config = _get_config()
    return config.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def risk_assessment(task_description: str) -> str:
    """Generate a risk-assessment prompt for a given task.

    Returns a system-level instruction that includes quick_think results
    and guidance on how to proceed safely.

    Args:
        task_description: What the agent is about to do.
    """
    config = _get_config()
    index = _get_quick_think_index()

    context = TaskContext(task_description=task_description)
    result = index.evaluate(context)

    parts = [
        "Before executing this task, review the following risk assessment:\n"
    ]

    if result.hit:
        taxonomy = load_error_taxonomy(config)
        parts.append(_format_risk_warning(result, taxonomy))
        parts.append(
            "\nProceed with caution. If the risk level is 'high', "
            "create a rollback plan before making changes."
        )
    else:
        parts.append("No known risk patterns detected for this task.")
        parts.append("Proceed normally, but remain vigilant for unexpected errors.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_windows_stdio()
    _start_layer2_recovery_thread_if_needed()
    mcp.run()


if __name__ == "__main__":
    main()

"""MCP Server for the meta-learning system.

Exposes Layer 1 Quick Think / Signal Capture, Layer 2/3 pipeline triggers,
and system status as MCP tools, resources, and prompts.

Run:  python -m meta_learning.mcp_server
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from meta_learning.layer1.quick_think import QuickThinkIndex
from meta_learning.layer1.signal_capture import SignalCapture
from meta_learning.layer2.consolidate import bootstrap_multimodal_embedding
from meta_learning.shared.io import (
    list_all_experiences,
    list_pending_signals,
    load_config,
    load_error_taxonomy,
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


def _make_embedding_fn(config: MetaLearningConfig):
    """Create an embedding function if DashScope + vector fallback are enabled."""
    if not config.dashscope.enabled:
        return None
    if not config.layer1.quick_think.vector_fallback_enabled:
        return None
    try:
        from meta_learning.shared.embedding_dashscope import MultimodalEmbedding
        emb = MultimodalEmbedding(config.dashscope)
        logger.info("Vector fallback enabled (DashScope %s)", config.dashscope.model)
        return emb.embed_text_only
    except Exception:
        logger.warning("Failed to create embedding function, vector fallback disabled", exc_info=True)
        return None


def _get_quick_think_index() -> QuickThinkIndex:
    global _qt_index, _taxonomy_mtime

    config = _get_config()
    tax_path = Path(config.error_taxonomy_full_path)

    current_mtime = tax_path.stat().st_mtime if tax_path.exists() else 0.0

    if _qt_index is None or current_mtime != _taxonomy_mtime:
        taxonomy = load_error_taxonomy(config)
        if _qt_index is None:
            embedding_fn = _make_embedding_fn(config)
            _qt_index = QuickThinkIndex(taxonomy, config, embedding_fn=embedding_fn)
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
        "Use `capture_signal` after completing a task to record learning signals. "
        "Use `status` to check the learning system state."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def quick_think(
    user_message: str,
    tools_used: list[str] | None = None,
) -> str:
    """Assess risk of a task before execution.

    Checks the user message against known error patterns (taxonomy) and
    irreversible-operation keywords. Returns a risk warning if any match,
    otherwise returns "no risk detected".

    Call this BEFORE executing a task that might involve destructive operations
    or areas where you have previously made mistakes.

    Args:
        user_message: The user's task description / latest message.
        tools_used: Names of tools you plan to use (optional).
    """
    config = _get_config()
    index = _get_quick_think_index()

    context = TaskContext(
        task_description=user_message,
        tools_used=tools_used or [],
    )
    result = index.evaluate(context)

    if not result.hit:
        return "no risk detected"

    taxonomy = load_error_taxonomy(config)
    return _format_risk_warning(result, taxonomy)


@mcp.tool()
def capture_signal(
    task_description: str,
    session_id: str = "unknown",
    errors_encountered: list[str] | None = None,
    errors_fixed: bool = False,
    user_corrections: list[str] | None = None,
    tools_used: list[str] | None = None,
    new_tools: list[str] | None = None,
    resolution_snapshot: str | None = None,
    image_snapshots: list[str] | None = None,
    step_count: int = 0,
) -> str:
    """Record a learning signal after completing a task.

    Analyzes the task outcome and, if a meaningful learning trigger is found
    (error recovery, user correction, new tool, efficiency anomaly), writes a
    YAML signal file to signal_buffer/ for later Layer 2 processing.

    Call this AFTER completing a task, especially when:
    - You encountered and fixed errors
    - The user corrected your approach
    - You used a new/unfamiliar tool
    - The task took many steps

    Args:
        task_description: Brief summary of the task.
        session_id: Current session identifier.
        errors_encountered: Error messages or tracebacks seen during the task.
        errors_fixed: Whether the errors were successfully resolved.
        user_corrections: User feedback that corrected your approach.
        tools_used: All tools invoked during the task.
        new_tools: Tools used for the first time.
        resolution_snapshot: Short summary of how the issue was resolved.
        image_snapshots: Paths to screenshots captured during this task.
        step_count: Total number of tool calls / steps taken.
    """
    config = _get_config()
    capture = SignalCapture(config)

    context = TaskContext(
        task_description=task_description,
        session_id=session_id,
        errors_encountered=errors_encountered or [],
        errors_fixed=errors_fixed,
        user_corrections=user_corrections or [],
        tools_used=tools_used or [],
        new_tools=new_tools or [],
        step_count=step_count,
        extra={
            "resolution": resolution_snapshot,
            "image_snapshots": image_snapshots or [],
        },
    )

    signal = capture.evaluate_and_capture(context)
    if signal is None:
        return "no signal captured — task did not trigger any learning condition"

    # Cross-link: register failure signature for QuickThink
    if signal.error_snapshot and _qt_index is not None:
        _qt_index.register_failure_signature(signal.error_snapshot)

    result = (
        f"Signal captured: [{signal.signal_id}] "
        f"trigger={signal.trigger_reason.value}, "
        f"file={config.signal_buffer_path}/{signal.signal_id}.yaml"
    )

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    orchestrator = Layer2Orchestrator(config, _create_llm(config))
    pending = list_pending_signals(config)
    if orchestrator.should_trigger():
        result += (
            f"\n\n[Action Required] Layer 2 trigger conditions met "
            f"({len(pending)} pending signal(s)). "
            f"Call `run_layer2` now to consolidate learnings into skills."
        )

    return result


@mcp.tool()
async def run_layer2(force: bool = False) -> str:
    """Trigger the Layer 2 near-line consolidation pipeline.

    Processes pending signals into structured experiences, clusters them,
    builds/updates the error taxonomy, and evolves skills.

    Normally only runs when trigger conditions are met (>= 5 pending signals
    or >= 24h since last run). Use force=True to override.

    Args:
        force: Run even if trigger conditions are not met.
    """
    config = _get_config()
    bootstrap_multimodal_embedding(config)
    llm = _create_llm(config)

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    orchestrator = Layer2Orchestrator(config, llm)

    if not force and not orchestrator.should_trigger():
        pending = list_pending_signals(config)
        return (
            f"Layer 2 trigger conditions not met "
            f"(pending signals: {len(pending)}). Use force=True to override."
        )

    result = await orchestrator.run_pipeline()
    return (
        f"Layer 2 complete: materialized={result.materialized_count}, "
        f"clusters={result.total_clusters}, "
        f"new_taxonomy={result.new_taxonomy_entries}, "
        f"skill_updates={result.skill_updates}"
    )


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
                f"  [{sig.signal_id}] {sig.trigger_reason}: "
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

if __name__ == "__main__":
    mcp.run()

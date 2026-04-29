from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from meta_learning.shared.models import (
    ErrorTaxonomy,
    Experience,
    ExperienceIndex,
    Layer3Result,
    MetaLearningConfig,
    Signal,
    TaxonomyEntry,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> MetaLearningConfig:
    path = Path(config_path)
    if not path.exists():
        return MetaLearningConfig()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MetaLearningConfig(**raw)


def ensure_directories(config: MetaLearningConfig) -> None:
    for dir_path in [
        config.signal_buffer_path,
        config.experience_pool_path,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def write_signal(signal: Signal, config: MetaLearningConfig) -> Path:
    ensure_directories(config)
    buf_dir = Path(config.signal_buffer_path)
    file_path = buf_dir / f"{signal.signal_id}.yaml"
    data = signal.model_dump(mode="json")
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return file_path


def read_signal(file_path: str | Path) -> Signal:
    with open(file_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Signal(**raw)


def list_pending_signals(config: MetaLearningConfig) -> list[Signal]:
    buf_dir = Path(config.signal_buffer_path)
    if not buf_dir.exists():
        return []
    signals = []
    for p in sorted(buf_dir.glob("sig-*.yaml")):
        sig = read_signal(p)
        if not sig.processed:
            signals.append(sig)
    return signals


def mark_signal_processed(signal_id: str, config: MetaLearningConfig) -> None:
    buf_dir = Path(config.signal_buffer_path)
    file_path = buf_dir / f"{signal_id}.yaml"
    if not file_path.exists():
        return
    sig = read_signal(file_path)
    sig.processed = True
    write_signal(sig, config)


def write_experience(experience: Experience, config: MetaLearningConfig) -> Path:
    pool_dir = Path(config.experience_pool_path)
    type_dir = pool_dir / experience.task_type.value
    type_dir.mkdir(parents=True, exist_ok=True)
    file_path = type_dir / f"{experience.id}.yaml"
    data = experience.model_dump(mode="json")
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return file_path


def read_experience(file_path: str | Path) -> Experience:
    with open(file_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Experience(**raw)


def list_all_experiences(config: MetaLearningConfig) -> list[Experience]:
    pool_dir = Path(config.experience_pool_path)
    if not pool_dir.exists():
        return []
    experiences = []
    for p in pool_dir.rglob("exp-*.yaml"):
        experiences.append(read_experience(p))
    return experiences


def load_experience_index(config: MetaLearningConfig) -> ExperienceIndex:
    pool_dir = Path(config.experience_pool_path)
    index_path = pool_dir / "index.yaml"
    if not index_path.exists():
        return ExperienceIndex(last_updated=datetime.now())
    with open(index_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ExperienceIndex(**raw)


def save_experience_index(index: ExperienceIndex, config: MetaLearningConfig) -> Path:
    pool_dir = Path(config.experience_pool_path)
    pool_dir.mkdir(parents=True, exist_ok=True)
    index_path = pool_dir / "index.yaml"
    data = index.model_dump(mode="json")
    with open(index_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return index_path


def load_error_taxonomy(config: MetaLearningConfig) -> ErrorTaxonomy:
    tax_path = Path(config.error_taxonomy_full_path)
    if not tax_path.exists():
        return ErrorTaxonomy()
    with open(tax_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "taxonomy" not in raw:
        return ErrorTaxonomy()
    taxonomy_data: dict = raw["taxonomy"]
    result: dict[str, dict[str, list[TaxonomyEntry]]] = {}
    for domain, subdomains in taxonomy_data.items():
        result[domain] = {}
        for subdomain, entries in subdomains.items():
            result[domain][subdomain] = [TaxonomyEntry(**e) for e in entries]
    return ErrorTaxonomy(taxonomy=result)


def save_error_taxonomy(taxonomy: ErrorTaxonomy, config: MetaLearningConfig) -> Path:
    tax_path = Path(config.error_taxonomy_full_path)
    tax_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"taxonomy": {}}
    for domain, subdomains in taxonomy.taxonomy.items():
        data["taxonomy"][domain] = {}
        for subdomain, entries in subdomains.items():
            data["taxonomy"][domain][subdomain] = [
                e.model_dump(mode="json") for e in entries
            ]
    with open(tax_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return tax_path


def read_session_context(
    session_id: str,
    config: MetaLearningConfig,
    max_chars: int = 6000,
) -> str:
    """Read session and return a head + tail summary that fits within *max_chars*.

    Tool-call entries are condensed to one line each to maximise conversational
    coverage within the budget.
    """
    session_file = resolve_session_file(session_id, config)
    if not session_file.exists():
        return f"Session {session_id} not found"

    all_lines: list[str] = []
    with open(session_file, encoding="utf-8") as f:
        for raw in f:
            try:
                record = json.loads(raw.strip())
            except json.JSONDecodeError:
                continue
            role = record.get("role", "unknown")
            content = record.get("content", "")
            if not (isinstance(content, str) and content.strip()):
                continue
            if role in ("agent_tool", "user_tool"):
                all_lines.append(f"[{role}] {content[:500]}")
            else:
                all_lines.append(f"[{role}] {content[:500]}")

    if not all_lines:
        return f"Session {session_id}: no readable content"

    full = "\n".join(all_lines)
    if len(full) <= max_chars:
        return full

    head_budget = int(max_chars * 0.45)
    tail_budget = int(max_chars * 0.45)
    sep = "\n\n... [middle of conversation omitted] ...\n\n"

    head_lines: list[str] = []
    head_len = 0
    for ln in all_lines:
        if head_len + len(ln) + 1 > head_budget:
            break
        head_lines.append(ln)
        head_len += len(ln) + 1

    tail_lines: list[str] = []
    tail_len = 0
    for ln in reversed(all_lines):
        if tail_len + len(ln) + 1 > tail_budget:
            break
        tail_lines.insert(0, ln)
        tail_len += len(ln) + 1

    return "\n".join(head_lines) + sep + "\n".join(tail_lines)


def resolve_session_file(session_id: str, config: MetaLearningConfig) -> Path:
    """
    Resolve the real session jsonl path across known layouts.

    A/B runs may store sessions under `<workspace_root>/sessions` while
    `sessions_root` can point to another directory.

    DeskClaw (nanobot) 通常把会话放在 ``~/.deskclaw/nanobot/workspace/sessions/``，
    文件名为 ``{channel}_{chat_id}.jsonl``，其中 ``chat_id`` 内可能含 ``:``（如
    ``main:desk-xxx``），落盘时常写作 ``agent_main_desk-xxx.jsonl``。若 Signal 里的
    ``session_id`` 仍带 ``:``，本函数会同时尝试 ``{session_id}.jsonl`` 与将 ``:``
    替换为 ``_`` 后的文件名。
    """
    sessions_dir = Path(config.sessions_full_path).expanduser()
    workspace_sessions_dir = Path(config.workspace_root).expanduser() / "sessions"
    candidate_dirs: list[Path] = []
    for d in [sessions_dir, workspace_sessions_dir]:
        if d not in candidate_dirs:
            candidate_dirs.append(d)

    normalized = session_id.replace(":", "_")
    candidate_names: list[str] = [f"{session_id}.jsonl"]
    if normalized != session_id:
        candidate_names.append(f"{normalized}.jsonl")
    # nanobot convention: agent_{channel}_{chat_id}.jsonl
    candidate_names.append(f"agent_{session_id}.jsonl")
    if normalized != session_id:
        candidate_names.append(f"agent_{normalized}.jsonl")
    # session_id without channel prefix → default channel "main"
    if ":" not in session_id and not session_id.startswith("agent_"):
        candidate_names.append(f"agent_main_{session_id}.jsonl")

    for base in candidate_dirs:
        for name in candidate_names:
            p = base / name
            if p.exists():
                return p

    return sessions_dir / f"{session_id}.jsonl"


_META_LEARNING_TOOLS = frozenset({
    "mcp_meta-learning_capture_signal",
    "mcp_meta-learning_quick_think",
    "mcp_meta-learning_run_layer2",
    "mcp_meta-learning_run_layer3",
    "mcp_meta-learning_status",
    "mcp_meta-learning_layer2_status",
    "mcp_meta-learning_sync_taxonomy_to_nobot",
    "mcp_meta-learning_confirm_taxonomy_entry",
    "mcp_meta-learning_contradict_taxonomy_entry",
    "mcp_meta-learning_delete_taxonomy_entry",
    "capture_signal",
    "quick_think",
    "run_layer2",
    "run_layer3",
    "status",
    "layer2_status",
    "sync_taxonomy_to_nobot",
    "confirm_taxonomy_entry",
    "contradict_taxonomy_entry",
    "delete_taxonomy_entry",
})


@dataclass
class SessionEnrichment:
    tools_used: list[str] = field(default_factory=list)
    step_count: int = 0
    action_trace: str | None = None


def enrich_from_session(
    session_id: str,
    config: MetaLearningConfig,
) -> SessionEnrichment:
    """Parse session JSONL to extract structural data for signal enrichment.

    Extracts tool names, step count, and an action trace string from the
    session transcript.  Pure structural JSON parsing — no LLM, no keyword
    heuristics.
    """
    session_file = resolve_session_file(session_id, config)
    if not session_file.exists():
        logger.debug("enrich_from_session: session file not found for %s", session_id)
        return SessionEnrichment()

    seen_tools: list[str] = []
    step_count = 0
    trace_parts: list[str] = []

    try:
        with open(session_file, encoding="utf-8") as f:
            for raw_line in f:
                try:
                    record = json.loads(raw_line.strip())
                except json.JSONDecodeError:
                    continue

                if record.get("role") != "assistant":
                    continue
                tool_calls = record.get("tool_calls")
                if not isinstance(tool_calls, list) or not tool_calls:
                    continue

                has_business_call = False
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    if not name or name in _META_LEARNING_TOOLS:
                        continue
                    has_business_call = True
                    if name not in seen_tools:
                        seen_tools.append(name)
                    trace_parts.append(_format_trace_entry(name, func.get("arguments")))

                if has_business_call:
                    step_count += 1
    except OSError:
        logger.warning("enrich_from_session: failed to read %s", session_file)
        return SessionEnrichment()

    action_trace: str | None = None
    if trace_parts:
        action_trace = " → ".join(trace_parts)
        if len(action_trace) > 2000:
            action_trace = action_trace[:1997] + "..."

    return SessionEnrichment(
        tools_used=seen_tools,
        step_count=step_count,
        action_trace=action_trace,
    )


def _format_trace_entry(name: str, arguments_raw: str | dict | None) -> str:
    """Format a single tool call into a readable trace entry.

    Extracts the ``path`` argument when present (the most common key param
    across file-operation tools).  Falls back to just the tool name.
    """
    if arguments_raw is None:
        return name

    args: dict
    if isinstance(arguments_raw, str):
        try:
            args = json.loads(arguments_raw)
        except (json.JSONDecodeError, TypeError):
            return name
    elif isinstance(arguments_raw, dict):
        args = arguments_raw
    else:
        return name

    path = args.get("path")
    if isinstance(path, str):
        return f"{name}({path})"

    return name


class _IdCounter:
    """Thread-safe in-memory ID counter with filesystem bootstrap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signal_counts: dict[str, int] = {}
        self._experience_counts: dict[str, int] = {}

    def next_signal_id(self, config: MetaLearningConfig) -> str:
        buf_dir = Path(config.signal_buffer_path)
        today = datetime.now().strftime("%Y%m%d")
        cache_key = f"{buf_dir}:{today}"

        with self._lock:
            if cache_key not in self._signal_counts:
                existing = (
                    list(buf_dir.glob(f"sig-{today}-*.yaml"))
                    if buf_dir.exists()
                    else []
                )
                self._signal_counts[cache_key] = len(existing)
            self._signal_counts[cache_key] += 1
            num = self._signal_counts[cache_key]
        return f"sig-{today}-{num:03d}"

    def next_experience_id(self, config: MetaLearningConfig) -> str:
        pool_dir = Path(config.experience_pool_path)
        cache_key = str(pool_dir)

        with self._lock:
            if cache_key not in self._experience_counts:
                existing = (
                    list(pool_dir.rglob("exp-*.yaml")) if pool_dir.exists() else []
                )
                self._experience_counts[cache_key] = len(existing)
            self._experience_counts[cache_key] += 1
            num = self._experience_counts[cache_key]
        return f"exp-{num:03d}"

    def reset(self) -> None:
        with self._lock:
            self._signal_counts.clear()
            self._experience_counts.clear()


_id_counter = _IdCounter()


def next_signal_id(config: MetaLearningConfig) -> str:
    return _id_counter.next_signal_id(config)


def next_experience_id(config: MetaLearningConfig) -> str:
    return _id_counter.next_experience_id(config)


def reset_id_counters() -> None:
    _id_counter.reset()


def next_cluster_id(index: ExperienceIndex) -> str:
    next_num = len(index.clusters) + 1
    return f"clust-{next_num:03d}"


def next_taxonomy_id(taxonomy: ErrorTaxonomy, domain_prefix: str) -> str:
    existing = [
        e for e in taxonomy.all_entries() if e.id.startswith(f"tax-{domain_prefix}")
    ]
    next_num = len(existing) + 1
    return f"tax-{domain_prefix}-{next_num:03d}"


def boost_taxonomy_confidence(
    entry_id: str,
    config: MetaLearningConfig,
    max_adjustment: float = 0.4,
) -> TaxonomyEntry | None:
    """Increase a taxonomy entry's confidence adjustment after positive validation."""
    from datetime import date as _date

    taxonomy = load_error_taxonomy(config)
    entry = taxonomy.find_entry(entry_id)
    if entry is None:
        return None

    boost = config.confidence.hit_success_boost
    new_adj = min(entry.confidence_adjustment + boost, max_adjustment)
    delta = new_adj - entry.confidence_adjustment
    entry.confidence_adjustment = new_adj
    entry.confidence = max(0.0, min(entry.confidence + delta, 1.0))
    entry.last_verified = _date.today()

    save_error_taxonomy(taxonomy, config)
    return entry


def penalize_taxonomy_confidence(
    entry_id: str,
    config: MetaLearningConfig,
    min_adjustment: float = -0.5,
) -> TaxonomyEntry | None:
    """Decrease a taxonomy entry's confidence adjustment after contradiction."""
    from datetime import date as _date

    taxonomy = load_error_taxonomy(config)
    entry = taxonomy.find_entry(entry_id)
    if entry is None:
        return None

    penalty = config.confidence.contradiction_penalty
    new_adj = max(entry.confidence_adjustment - penalty, min_adjustment)
    delta = new_adj - entry.confidence_adjustment
    entry.confidence_adjustment = new_adj
    entry.confidence = max(0.0, min(entry.confidence + delta, 1.0))
    entry.last_verified = _date.today()

    save_error_taxonomy(taxonomy, config)
    return entry


def save_layer3_result(result: Layer3Result, config: MetaLearningConfig) -> Path:
    workspace = Path(config.resolve_workspace_path("layer3_results"))
    workspace.mkdir(parents=True, exist_ok=True)
    ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
    file_path = workspace / f"l3-result-{ts}.yaml"
    data = result.model_dump(mode="json")
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return file_path


def load_latest_layer3_result(
    config: MetaLearningConfig,
) -> Layer3Result | None:
    workspace = Path(config.resolve_workspace_path("layer3_results"))
    if not workspace.exists():
        return None
    files = sorted(workspace.glob("l3-result-*.yaml"), reverse=True)
    if not files:
        return None
    with open(files[0], encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Layer3Result(**raw)

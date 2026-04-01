"""Sync error taxonomy to nanobot workspace (SKILL.md + rules/*.md).

SKILL.md is injected into every nanobot conversation (always:true) as a
concise natural-language summary (~200-300 tokens).  rules/*.md files are
generated for human review / debugging only — agents use `quick_think` MCP
tool to retrieve detailed guidance at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from meta_learning.shared.io import load_error_taxonomy
from meta_learning.shared.models import MetaLearningConfig, TaxonomyEntry

logger = logging.getLogger(__name__)

CATEGORY_KEYWORD_MAP: dict[str, set[str]] = {
    "preferences": {"path", "directory", "folder", "workspace", "dir", "home"},
    "git-workflow": {"git", "branch", "commit", "push", "merge", "rebase", "pull"},
    "code-style": {"style", "indent", "format", "naming", "semicolon", "tab", "space", "lint"},
    "verification": {"verify", "check", "test", "backup", "validate", "assert", "confirm"},
}

_RENDER_STOPWORDS = frozenset({
    "the", "a", "an", "is", "of", "to", "in", "for", "and", "or", "not",
    "it", "this", "that", "with", "from", "are", "was", "were", "been",
    "be", "has", "have", "had", "but", "if", "its", "can", "does", "do",
    "did", "will", "would", "should", "could", "may", "might", "on", "no",
    "always", "never", "before", "after", "using", "use", "any", "all",
})
_STRIP_CHARS = ".:,;()[]{}\"'`<>!?/\\#@$%^&*+=~|"

RENDER_DEDUP_THRESHOLD = 0.6


def _tokenize_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.lower().split():
        cleaned = raw.strip(_STRIP_CHARS)
        if cleaned and len(cleaned) > 1 and cleaned not in _RENDER_STOPWORDS:
            tokens.add(cleaned)
    return tokens


def _entry_topic_tokens(entry: TaxonomyEntry) -> set[str]:
    kw_text = " ".join(entry.keywords)
    combined = f"{entry.name} {kw_text}"
    combined = combined.replace("_", " ").replace("-", " ")
    return _tokenize_text(combined)


def _select_diverse_top_n(
    entries: list[TaxonomyEntry], max_rules: int,
) -> list[TaxonomyEntry]:
    """Greedy selection: pick highest-confidence entries, skipping near-duplicates.

    Uses keyword + name overlap (topic-level) to detect duplicates rather than
    full prevention text, since keywords are curated topic indicators.
    """
    sorted_entries = sorted(entries, key=lambda e: e.confidence, reverse=True)
    selected: list[TaxonomyEntry] = []
    selected_tokens: list[set[str]] = []

    for entry in sorted_entries:
        if len(selected) >= max_rules:
            break
        tokens = _entry_topic_tokens(entry)
        if not tokens:
            continue
        is_dup = False
        for prev_tokens in selected_tokens:
            overlap = len(tokens & prev_tokens)
            smaller = min(len(tokens), len(prev_tokens))
            if smaller > 0 and overlap / smaller >= RENDER_DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            selected.append(entry)
            selected_tokens.append(tokens)

    return selected


def _classify_entry(entry: TaxonomyEntry) -> str:
    kw_set = {k.lower() for k in entry.keywords}
    full_text = f"{entry.name} {entry.trigger} {entry.prevention}".lower()
    all_tokens = kw_set | set(full_text.split())

    best_cat = "general"
    best_overlap = 0
    for cat, cat_kws in CATEGORY_KEYWORD_MAP.items():
        overlap = len(all_tokens & cat_kws)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cat = cat
    return best_cat


def _render_skill_md(entries: list[TaxonomyEntry], max_rules: int = 10) -> str:
    top = _select_diverse_top_n(entries, max_rules)
    return _render_skill_md_from_selected(top)


def _render_skill_md_from_selected(selected: list[TaxonomyEntry]) -> str:
    lines = [
        "---",
        "name: meta-learning",
        "description: Learned rules from past mistakes and user corrections.",
        "always: true",
        "---",
        "# Meta-Learning Rules",
        "",
        "You have learned the following rules from past interactions:",
    ]
    for entry in selected:
        lines.append(f"- {entry.prevention}")

    lines.append("")
    lines.append("## When to call meta-learning tools")
    lines.append("")
    lines.append("- **SESSION START**: Call `layer2_status` once at the beginning of each session. If status is \"running\", wait briefly and re-check before proceeding — SKILL.md may be stale.")
    lines.append("- **MUST**: When the user corrects, disagrees with, or redirects your approach, IMMEDIATELY call `capture_signal` with `user_corrections` set to the user's exact feedback. Just saying \"understood\" is NOT enough — you must also call the tool. Learning consolidation runs automatically in the background.")
    lines.append("- Before risky or repetitive actions, call `quick_think` to get detailed guidance.")
    lines.append("")
    lines.append("Example: User says \"不对，应该用 X\" → call capture_signal(user_corrections=[\"不对，应该用 X\"]) → reply to user.")
    return "\n".join(lines) + "\n"


def _render_category_md(category: str, entries: list[TaxonomyEntry]) -> str:
    sorted_entries = sorted(entries, key=lambda e: e.confidence, reverse=True)
    lines = [f"# {category.replace('-', ' ').title()}", ""]
    for entry in sorted_entries:
        lines.append(f"## {entry.name}")
        lines.append(f"**Trigger:** {entry.trigger}")
        lines.append(f"**Prevention:** {entry.prevention}")
        lines.append(f"**Fix SOP:** {entry.fix_sop}")
        lines.append(f"**Confidence:** {entry.confidence:.2f}")
        lines.append(f"**Keywords:** {', '.join(entry.keywords)}")
        lines.append("")
    return "\n".join(lines)


@dataclass
class SyncResult:
    skill_md_path: str = ""
    rules_written: list[str] = field(default_factory=list)
    total_entries: int = 0
    top_n_in_skill: int = 0


def sync_taxonomy_to_nobot_workspace(
    config: MetaLearningConfig,
    nobot_skills_path: str,
    max_always_rules: int = 10,
) -> SyncResult:
    """Read error_taxonomy.yaml and sync to nanobot skill files.

    Args:
        config: meta-learning config (used to locate taxonomy file).
        nobot_skills_path: absolute path to the nanobot skills directory
            (e.g. ``~/.deskclaw/nanobot/workspace/skills``).
        max_always_rules: max entries to include in SKILL.md summary.

    Returns:
        SyncResult with paths of files written.
    """
    taxonomy = load_error_taxonomy(config)
    entries = taxonomy.all_entries()
    result = SyncResult(total_entries=len(entries))

    if not entries:
        logger.info("No taxonomy entries to sync")
        return result

    skills_root = Path(nobot_skills_path).expanduser()
    meta_dir = skills_root / "meta-learning"
    rules_dir = meta_dir / "rules"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rules_dir.mkdir(parents=True, exist_ok=True)

    diverse_top = _select_diverse_top_n(entries, max_always_rules)
    skill_md = _render_skill_md_from_selected(diverse_top)
    skill_path = meta_dir / "SKILL.md"
    skill_path.write_text(skill_md, encoding="utf-8")
    result.skill_md_path = str(skill_path)
    result.top_n_in_skill = len(diverse_top)
    logger.info("Wrote SKILL.md with %d diverse rules (from %d total) at %s",
                result.top_n_in_skill, len(entries), skill_path)

    categorized: dict[str, list[TaxonomyEntry]] = {}
    for entry in entries:
        cat = _classify_entry(entry)
        categorized.setdefault(cat, []).append(entry)

    for cat, cat_entries in categorized.items():
        md = _render_category_md(cat, cat_entries)
        cat_path = rules_dir / f"{cat}.md"
        cat_path.write_text(md, encoding="utf-8")
        result.rules_written.append(str(cat_path))
        logger.info("Wrote rules/%s.md (%d entries)", cat, len(cat_entries))

    return result

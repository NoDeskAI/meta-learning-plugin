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
    sorted_entries = sorted(entries, key=lambda e: e.confidence, reverse=True)
    top = sorted_entries[:max_rules]

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
    for entry in top:
        lines.append(f"- {entry.prevention}")

    lines.append("")
    lines.append("Before risky or repetitive actions, call `quick_think` to get detailed guidance.")
    lines.append("After the user corrects your approach, call `capture_signal` to record the lesson.")
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

    skill_md = _render_skill_md(entries, max_rules=max_always_rules)
    skill_path = meta_dir / "SKILL.md"
    skill_path.write_text(skill_md, encoding="utf-8")
    result.skill_md_path = str(skill_path)
    result.top_n_in_skill = min(len(entries), max_always_rules)
    logger.info("Wrote SKILL.md with top-%d rules at %s", result.top_n_in_skill, skill_path)

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

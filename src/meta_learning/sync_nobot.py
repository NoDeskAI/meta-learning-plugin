"""Sync error taxonomy to nanobot workspace (SKILL.md + rules/*.md + AGENTS.md).

SKILL.md is injected into every nanobot conversation (always:true) as a
concise natural-language summary (~200-300 tokens).  rules/*.md files are
generated for human review / debugging only — agents use `quick_think` MCP
tool to retrieve detailed guidance at runtime.

AGENTS.md is the agent's primary instruction file.  ``install.sh`` injects a
delimited section so the agent knows *when* and *how* to call meta-learning
tools.  The section is replaced on upgrade and removed on uninstall.
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
        if not _entry_rule_text(entry):
            continue
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


def render_bootstrap_skill_md() -> str:
    """Render a SKILL.md with no learned rules — only tool-calling instructions.

    Deploy this during installation or cleanup to bootstrap the learning loop:
    without it, the agent never calls ``capture_signal``, so the first SKILL.md
    is never generated (chicken-and-egg problem).
    """
    return _render_skill_md_from_selected([])


def _render_skill_md(entries: list[TaxonomyEntry], max_rules: int = 10) -> str:
    top = _select_diverse_top_n(entries, max_rules)
    return _render_skill_md_from_selected(top)


def _entry_rule_text(entry: TaxonomyEntry) -> str:
    """Best available one-liner for an entry: prevention > fix_sop > trigger."""
    return entry.prevention or entry.fix_sop or entry.trigger


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
        text = _entry_rule_text(entry)
        if text:
            lines.append(f"- {text}")

    lines.append("")
    lines.append("## When to call meta-learning tools")
    lines.append("")
    lines.append("- **SESSION START**: Call `layer2_status` once at the beginning of each session. If status is \"running\", wait briefly and re-check before proceeding — SKILL.md may be stale.")
    lines.append("- **MUST**: When the user corrects, disagrees with, or redirects your approach, IMMEDIATELY call `capture_signal` with `user_corrections` set to the user's exact feedback. Just saying \"understood\" is NOT enough — you must also call the tool. Learning consolidation runs automatically in the background.")
    lines.append("- If you need to wait for Layer 2 or verify generated rules after `capture_signal`, spawn a separate learning worker to call `layer2_status` and notify the original conversation. Do not block the main user conversation on learning progress.")
    lines.append("- Before risky or repetitive actions, call `quick_think` to get detailed guidance.")
    lines.append("- Treat the meta-learning data directory as MCP-owned state. You may inspect files for diagnosis, but do not create, delete, or hand-edit `meta-learning-data` YAML/JSON to record or repair learning; use the meta-learning MCP tools instead.")
    lines.append("")
    lines.append("Example: User says \"不对，应该用 X\" → call capture_signal(user_corrections=[\"不对，应该用 X\"]) → reply to user.")
    return "\n".join(lines) + "\n"


def _render_category_md(category: str, entries: list[TaxonomyEntry]) -> str:
    deduped = _select_diverse_top_n(entries, max_rules=len(entries))
    lines = [f"# {category.replace('-', ' ').title()}", ""]
    for entry in deduped:
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


# ---------------------------------------------------------------------------
# AGENTS.md injection
# ---------------------------------------------------------------------------

AGENTS_MD_BEGIN = "<!-- meta-learning:begin -->"
AGENTS_MD_END = "<!-- meta-learning:end -->"


def render_agents_md_section() -> str:
    """Return the delimited AGENTS.md section for meta-learning.

    The block is wrapped in HTML-comment delimiters so ``inject_agents_md``
    can replace it on upgrade and ``remove_agents_md_section`` can strip it
    on uninstall.
    """
    return f"""{AGENTS_MD_BEGIN}
## Meta-Learning 自进化学习

你已安装 Meta-Learning 插件，它能从过去的错误中学习并持续改进。严格遵循以下规则：

1. **会话开始时**：调用 `layer2_status` 检查学习系统状态。如果返回 "running"，等待片刻后重新检查再继续——此时 SKILL.md 可能尚未更新。
2. **用户纠正/反驳你时**（最高优先级）：必须**立即**调用 `capture_signal`，将 `user_corrections` 设为用户的原话。仅回复"好的/明白了"是**不够的**，必须同时调用工具记录信号。
3. **执行危险/不可逆操作前**：调用 `quick_think` 检查任务是否命中已知错误模式，获取风险评估和防范建议。
4. **任务完成后**，如果遇到了以下任一情况，调用 `capture_signal` 记录学习信号：
   - 执行过程中遇到错误（无论是否自行修复）
   - 首次使用某个工具
   - 步骤数明显偏多
5. `capture_signal` 后 Layer 2 整合流水线会在后台自动运行，**无需**手动调用 `run_layer2`。
如果需要等待 Layer 2 完成或验证生成规则，开启一个 spawn 学习 worker 去调用
`layer2_status` 并在完成后通知原会话；不要让主用户会话阻塞在学习进度轮询上。
6. `meta-learning-data` 是 Meta-Learning MCP 的内部状态。可以读取这些文件用于诊断，但**不要**通过文件工具创建、删除或手写修改其中的 YAML/JSON 来记录、修复或回滚学习结果；学习数据只能通过 Meta-Learning MCP 工具维护。

示例：用户说"不对，应该用 X" → 调用 `capture_signal(user_corrections=["不对，应该用 X"])` → 回复用户。
{AGENTS_MD_END}
"""


def inject_agents_md(agents_md_path: str | Path) -> bool:
    """Inject or replace the meta-learning section in AGENTS.md.

    Returns True if the file was modified, False if unchanged (content
    identical to what would be written).
    """
    path = Path(agents_md_path).expanduser()
    section = render_agents_md_section()

    if path.exists():
        content = path.read_text(encoding="utf-8")
    else:
        content = ""

    if AGENTS_MD_BEGIN in content:
        begin = content.index(AGENTS_MD_BEGIN)
        end = content.index(AGENTS_MD_END) + len(AGENTS_MD_END)
        while end < len(content) and content[end] in ("\n", "\r"):
            end += 1
        new_content = content[:begin] + section + content[end:]
    else:
        separator = "\n" if content and not content.endswith("\n") else ""
        new_content = content + separator + section

    if new_content == content:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return True


def remove_agents_md_section(agents_md_path: str | Path) -> bool:
    """Remove the meta-learning section from AGENTS.md.

    Returns True if the section was found and removed.
    """
    path = Path(agents_md_path).expanduser()
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    if AGENTS_MD_BEGIN not in content:
        return False

    begin = content.index(AGENTS_MD_BEGIN)
    end = content.index(AGENTS_MD_END) + len(AGENTS_MD_END)
    while end < len(content) and content[end] in ("\n", "\r"):
        end += 1
    new_content = content[:begin] + content[end:]

    path.write_text(new_content, encoding="utf-8")
    return True

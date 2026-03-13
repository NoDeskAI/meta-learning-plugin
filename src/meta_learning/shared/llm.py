from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meta_learning.shared.models import (
        CapabilityAnalysis,
        ConsolidateJudgment,
        CrossTaskAnalysis,
        Experience,
        MaterializeResult,
        MemoryAnalysis,
        Signal,
        SkillEvolveResult,
        TaxonomyEntry,
        TaxonomyExtraction,
    )


class LLMInterface(ABC):
    @abstractmethod
    async def materialize_signal(
        self,
        signal: Signal,
        session_context: str,
    ) -> MaterializeResult: ...

    @abstractmethod
    async def judge_same_class(
        self,
        exp_a: Experience,
        exp_b: Experience,
    ) -> ConsolidateJudgment: ...

    @abstractmethod
    async def extract_taxonomy(
        self,
        experiences: list[Experience],
    ) -> TaxonomyExtraction: ...

    @abstractmethod
    async def evaluate_skill_update(
        self,
        taxonomy_entry: TaxonomyEntry,
        existing_skill_content: str | None,
    ) -> SkillEvolveResult: ...

    @abstractmethod
    async def analyze_cross_task_patterns(
        self,
        experience_groups: list[list[Experience]],
    ) -> list[CrossTaskAnalysis]: ...

    @abstractmethod
    async def analyze_capability_gaps(
        self,
        ungrouped_experiences: list[Experience],
        existing_taxonomy_keywords: list[str],
    ) -> list[CapabilityAnalysis]: ...

    @abstractmethod
    async def analyze_memory(
        self,
        high_confidence_experiences: list[Experience],
        low_confidence_experiences: list[Experience],
    ) -> MemoryAnalysis: ...


def _extract_representative_token(signatures: list[str]) -> str:
    if not signatures:
        return "general"
    for word in signatures[0].split():
        cleaned = _clean_token(word)
        if cleaned and any(c.isdigit() for c in cleaned):
            return cleaned.lower()
        if cleaned and cleaned == cleaned.upper() and "_" in cleaned:
            return cleaned.lower()
    for word in signatures[0].split():
        cleaned = _clean_token(word).lower()
        if cleaned and len(cleaned) > 3 and cleaned not in _KEYWORD_STOPWORDS:
            return cleaned
    return "general"


class StubLLM(LLMInterface):
    async def materialize_signal(
        self,
        signal: Signal,
        session_context: str,
    ) -> MaterializeResult:
        from meta_learning.shared.models import MaterializeResult, TaskType

        task_type = TaskType.UNCLASSIFIED
        for tt in TaskType:
            if tt.value != "_unclassified" and tt.value in signal.task_summary.lower():
                task_type = tt
                break

        snapshot = signal.resolution_snapshot or "unknown"
        kw_str = ", ".join(signal.keywords)
        return MaterializeResult(
            scene=signal.task_summary,
            failure_signature=signal.error_snapshot,
            root_cause=f"Root cause derived from: {snapshot}",
            resolution=signal.resolution_snapshot or "No resolution captured",
            meta_insight=f"Insight from signal {signal.signal_id}: {kw_str}",
            task_type=task_type,
        )

    async def judge_same_class(
        self,
        exp_a: Experience,
        exp_b: Experience,
    ) -> ConsolidateJudgment:
        from meta_learning.shared.models import ConsolidateJudgment

        same = (
            exp_a.task_type == exp_b.task_type
            and exp_a.failure_signature is not None
            and exp_b.failure_signature is not None
            and _has_keyword_overlap(exp_a.failure_signature, exp_b.failure_signature)
        )
        return ConsolidateJudgment(
            same_class=same,
            reason="Stub: keyword overlap check on failure signatures",
        )

    async def extract_taxonomy(
        self,
        experiences: list[Experience],
    ) -> TaxonomyExtraction:
        from meta_learning.shared.models import TaxonomyExtraction

        signatures = [e.failure_signature for e in experiences if e.failure_signature]
        resolutions = [e.resolution for e in experiences]

        first_sig = signatures[0] if signatures else "unknown"
        task_type = experiences[0].task_type.value if experiences else "unknown"
        representative = _extract_representative_token(signatures)
        name = f"{task_type}: {representative} ({len(experiences)} experiences)"

        return TaxonomyExtraction(
            name=name,
            trigger=first_sig if signatures else "unknown trigger",
            fix_sop="\n".join(f"- {r}" for r in resolutions[:3]),
            prevention=f"Avoid conditions leading to: {first_sig}",
            keywords=_extract_common_keywords(experiences),
        )

    async def evaluate_skill_update(
        self,
        taxonomy_entry: TaxonomyEntry,
        existing_skill_content: str | None,
    ) -> SkillEvolveResult:
        from meta_learning.shared.models import SkillEvolveResult, SkillUpdateAction

        if existing_skill_content is None:
            if (
                taxonomy_entry.confidence >= 0.8
                and len(taxonomy_entry.source_exps) >= 5
            ):
                tid = taxonomy_entry.id
                return SkillEvolveResult(
                    action=SkillUpdateAction.CREATE,
                    target_skill=taxonomy_entry.name.lower().replace(" ", "-"),
                    changes_description=f"Create new skill from {tid}",
                    new_content=_generate_skill_content(taxonomy_entry),
                    version_bump="1.0.0",
                )
            return SkillEvolveResult(
                action=SkillUpdateAction.NONE,
                changes_description="Insufficient confidence or evidence for new skill",
            )

        tid = taxonomy_entry.id
        sop = taxonomy_entry.fix_sop
        return SkillEvolveResult(
            action=SkillUpdateAction.APPEND,
            target_skill=taxonomy_entry.name.lower().replace(" ", "-"),
            changes_description=f"Append knowledge from {tid}",
            new_content=f"\n\n## Update from {tid}\n{sop}",
            version_bump=None,
        )

    async def analyze_cross_task_patterns(
        self,
        experience_groups: list[list[Experience]],
    ) -> list[CrossTaskAnalysis]:
        from meta_learning.shared.models import CrossTaskAnalysis

        results: list[CrossTaskAnalysis] = []
        for group in experience_groups:
            if len(group) < 2:
                continue
            task_types = {e.task_type.value for e in group}
            if len(task_types) < 2:
                continue
            root_causes = [e.root_cause for e in group]
            first_cause = root_causes[0] if root_causes else "unknown"
            results.append(
                CrossTaskAnalysis(
                    description=(
                        f"Cross-task pattern across {', '.join(sorted(task_types))}: "
                        f"{len(group)} experiences share similar root cause"
                    ),
                    shared_root_cause=first_cause,
                    meta_strategy=(
                        f"Before tasks involving {', '.join(sorted(task_types))}, "
                        f"check for: {first_cause}"
                    ),
                    confidence=min(len(group) * 0.15, 0.9),
                )
            )
        return results

    async def analyze_capability_gaps(
        self,
        ungrouped_experiences: list[Experience],
        existing_taxonomy_keywords: list[str],
    ) -> list[CapabilityAnalysis]:
        from collections import Counter

        from meta_learning.shared.models import CapabilityAnalysis

        type_counter: Counter[str] = Counter()
        for exp in ungrouped_experiences:
            type_counter[exp.task_type.value] += 1

        results: list[CapabilityAnalysis] = []
        for task_type, count in type_counter.most_common():
            if count < 3:
                continue
            results.append(
                CapabilityAnalysis(
                    description=(
                        f"Skill gap in '{task_type}': {count} experiences "
                        f"lack taxonomy coverage"
                    ),
                    suggested_action=(
                        f"Create skill for handling recurring '{task_type}' patterns"
                    ),
                    priority=min(count * 0.2, 1.0),
                )
            )
        return results

    async def analyze_memory(
        self,
        high_confidence_experiences: list[Experience],
        low_confidence_experiences: list[Experience],
    ) -> MemoryAnalysis:
        from meta_learning.shared.models import (
            MemoryAction,
            MemoryAnalysis,
            MemoryRecommendation,
        )

        recommendations: list[MemoryRecommendation] = []

        for exp in high_confidence_experiences:
            recommendations.append(
                MemoryRecommendation(
                    action=MemoryAction.EXTRACT,
                    target=exp.id,
                    reason=(
                        f"High confidence ({exp.confidence:.2f}) experience "
                        f"should be promoted to long-term memory"
                    ),
                    content=exp.meta_insight,
                )
            )

        for exp in low_confidence_experiences:
            recommendations.append(
                MemoryRecommendation(
                    action=MemoryAction.PRUNE,
                    target=exp.id,
                    reason=(
                        f"Low confidence ({exp.confidence:.2f}) experience "
                        f"candidate for removal"
                    ),
                )
            )

        return MemoryAnalysis(recommendations=recommendations)


def _has_keyword_overlap(sig_a: str, sig_b: str) -> bool:
    words_a = {_clean_token(w) for w in sig_a.lower().split()}
    words_b = {_clean_token(w) for w in sig_b.lower().split()}
    words_a = {w for w in words_a if w and len(w) > 1 and w not in _KEYWORD_STOPWORDS}
    words_b = {w for w in words_b if w and len(w) > 1 and w not in _KEYWORD_STOPWORDS}
    if not words_a or not words_b:
        return False
    overlap = words_a & words_b
    if len(overlap) / min(len(words_a), len(words_b)) >= 0.3:
        return True
    return _shared_error_code_prefix(words_a, words_b)


def _shared_error_code_prefix(words_a: set[str], words_b: set[str]) -> bool:
    """Check if both sets contain error codes sharing a prefix (e.g. TS2345 and TS2322)."""
    import re

    code_pattern = re.compile(r"^([a-zA-Z]+)(\d{3,})$")
    prefixes_a: set[str] = set()
    prefixes_b: set[str] = set()
    for w in words_a:
        m = code_pattern.match(w)
        if m:
            prefixes_a.add(m.group(1))
    for w in words_b:
        m = code_pattern.match(w)
        if m:
            prefixes_b.add(m.group(1))
    return bool(prefixes_a & prefixes_b)


_KEYWORD_STOPWORDS = frozenset(
    {
        # English stopwords
        "the",
        "a",
        "an",
        "is",
        "of",
        "to",
        "in",
        "for",
        "and",
        "or",
        "not",
        "it",
        "this",
        "that",
        "with",
        "from",
        "are",
        "was",
        "were",
        "been",
        "be",
        "has",
        "have",
        "had",
        "but",
        "if",
        "its",
        "can",
        "does",
        "do",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "on",
        "no",
        # Programming-generic (appear in nearly all error messages)
        "error",
        "type",
        "cannot",
        "read",
        "properties",
        "undefined",
        "null",
        "variable",
        "function",
        "argument",
        "parameter",
        "value",
        "missing",
        "expected",
        "return",
        "module",
        "name",
        "call",
    }
)


def _clean_token(token: str) -> str:
    """Strip surrounding punctuation, quotes, parens from a token."""
    return token.strip(".:,;()[]{}\"'`<>!?/\\#@$%^&*+=~|")


def _extract_common_keywords(experiences: list[Experience]) -> list[str]:
    from collections import Counter

    word_counter: Counter[str] = Counter()
    for exp in experiences:
        if exp.failure_signature:
            for raw_word in exp.failure_signature.lower().split():
                word = _clean_token(raw_word)
                if len(word) > 2 and word not in _KEYWORD_STOPWORDS:
                    word_counter[word] += 1
    # Require frequency >= 2 (appears in multiple experiences)
    candidates = [(w, c) for w, c in word_counter.most_common(20) if c >= 2]

    # Sort: error-code-like first (contains digits or ALL_CAPS identifiers),
    # then by frequency descending
    def _score(item: tuple[str, int]) -> tuple[int, int]:
        w, c = item
        has_digits = any(ch.isdigit() for ch in w)
        is_upper_ident = w == w.upper() and "_" in w  # e.g., DATABASE_URL
        priority = 0 if (has_digits or is_upper_ident) else 1
        return (priority, -c)

    candidates.sort(key=_score)
    return [w for w, _ in candidates[:10]]


def _generate_skill_content(entry: TaxonomyEntry) -> str:
    return f"""# {entry.name}

## When to Apply
{entry.trigger}

## Standard Fix Procedure
{entry.fix_sop}

## Prevention
{entry.prevention}

## Source
- Taxonomy ID: {entry.id}
- Confidence: {entry.confidence}
- Based on {len(entry.source_exps)} experiences
"""

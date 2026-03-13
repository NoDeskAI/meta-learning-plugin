from __future__ import annotations

from meta_learning.shared.models import (
    ErrorTaxonomy,
    MetaLearningConfig,
    QuickThinkResult,
    TaskContext,
    TaxonomyEntry,
)


class QuickThinkIndex:
    def __init__(self, taxonomy: ErrorTaxonomy, config: MetaLearningConfig) -> None:
        self._config = config
        self._taxonomy = taxonomy
        self._keyword_index: dict[str, list[TaxonomyEntry]] = taxonomy.all_keywords()
        self._irreversible_patterns: list[str] = [
            kw.lower() for kw in config.layer1.quick_think.irreversible_keywords
        ]
        self._recent_failure_signatures: set[str] = set()
        self._known_tools: set[str] = set()

    def update_taxonomy(self, taxonomy: ErrorTaxonomy) -> None:
        self._taxonomy = taxonomy
        self._keyword_index = taxonomy.all_keywords()

    def register_failure_signature(self, signature: str) -> None:
        self._recent_failure_signatures.add(signature.lower())

    def register_known_tool(self, tool: str) -> None:
        self._known_tools.add(tool.lower())

    def evaluate(self, context: TaskContext) -> QuickThinkResult:
        matched_signals: list[str] = []
        matched_taxonomy_ids: list[str] = []

        taxonomy_hits = self._check_keyword_match(context)
        if taxonomy_hits:
            matched_signals.append("keyword_taxonomy_hit")
            matched_taxonomy_ids.extend(e.id for e in taxonomy_hits)

        if self._check_irreversible_ops(context):
            matched_signals.append("irreversible_operation")

        if self._check_recent_failures(context):
            matched_signals.append("recent_failure_pattern")

        if self._check_new_tools(context):
            matched_signals.append("new_tool_usage")

        hit = len(matched_signals) > 0
        risk_level = self._assess_risk_level(matched_signals)

        return QuickThinkResult(
            hit=hit,
            matched_signals=matched_signals,
            matched_taxonomy_entries=matched_taxonomy_ids,
            risk_level=risk_level,
        )

    def _check_keyword_match(self, context: TaskContext) -> list[TaxonomyEntry]:
        hits: list[TaxonomyEntry] = []
        seen_ids: set[str] = set()

        text_corpus = _build_searchable_text(context)

        for keyword, entries in self._keyword_index.items():
            if keyword in text_corpus:
                for entry in entries:
                    if entry.id not in seen_ids:
                        hits.append(entry)
                        seen_ids.add(entry.id)

        return hits

    def _check_irreversible_ops(self, context: TaskContext) -> bool:
        text = context.task_description.lower()
        for tool in context.tools_used:
            text += " " + tool.lower()

        return any(pattern in text for pattern in self._irreversible_patterns)

    def _check_recent_failures(self, context: TaskContext) -> bool:
        if not self._recent_failure_signatures:
            return False
        text = _build_searchable_text(context)
        return any(sig in text for sig in self._recent_failure_signatures)

    def _check_new_tools(self, context: TaskContext) -> bool:
        if not self._known_tools:
            return bool(context.new_tools)
        return any(tool.lower() not in self._known_tools for tool in context.new_tools)

    def _assess_risk_level(self, signals: list[str]) -> str:
        if "irreversible_operation" in signals:
            return "high"
        if len(signals) >= 2:
            return "medium"
        if signals:
            return "low"
        return "none"


def _build_searchable_text(context: TaskContext) -> str:
    parts = [
        context.task_description.lower(),
        " ".join(e.lower() for e in context.errors_encountered),
        " ".join(t.lower() for t in context.tools_used),
    ]
    return " ".join(parts)

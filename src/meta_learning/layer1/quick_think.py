from __future__ import annotations

import logging
import math
from typing import Callable

from meta_learning.shared.models import (
    ErrorTaxonomy,
    MetaLearningConfig,
    QuickThinkResult,
    TaskContext,
    TaxonomyEntry,
)

logger = logging.getLogger(__name__)

EmbeddingFn = Callable[[str], list[float]]


class QuickThinkIndex:
    def __init__(
        self,
        taxonomy: ErrorTaxonomy,
        config: MetaLearningConfig,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self._config = config
        self._taxonomy = taxonomy
        self._keyword_index: dict[str, list[TaxonomyEntry]] = taxonomy.all_keywords()
        self._irreversible_patterns: list[str] = [
            kw.lower() for kw in config.layer1.quick_think.irreversible_keywords
        ]
        self._recent_failure_signatures: set[str] = set()
        self._known_tools: set[str] = set()

        self._embedding_fn = embedding_fn
        self._taxonomy_embeddings: dict[str, list[float]] = {}
        if embedding_fn is not None:
            self._precompute_embeddings()

    def _embedding_text_for_entry(self, entry: TaxonomyEntry) -> str:
        return f"{entry.name} {entry.trigger} {entry.prevention} {' '.join(entry.keywords)}"

    def _precompute_embeddings(self) -> None:
        if self._embedding_fn is None:
            return
        self._taxonomy_embeddings.clear()
        entries = self._taxonomy.all_entries()
        for entry in entries:
            text = self._embedding_text_for_entry(entry)
            try:
                vec = self._embedding_fn(text)
                self._taxonomy_embeddings[entry.id] = vec
            except Exception:
                logger.warning("Failed to compute embedding for %s, skipping", entry.id, exc_info=True)

    def update_taxonomy(self, taxonomy: ErrorTaxonomy) -> None:
        self._taxonomy = taxonomy
        self._keyword_index = taxonomy.all_keywords()
        if self._embedding_fn is not None:
            self._precompute_embeddings()

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
        else:
            vector_hits = self._check_vector_match(context)
            if vector_hits:
                matched_signals.append("keyword_taxonomy_hit")
                matched_taxonomy_ids.extend(e.id for e in vector_hits)

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

    def _check_vector_match(self, context: TaskContext) -> list[TaxonomyEntry]:
        qt_cfg = self._config.layer1.quick_think
        if not qt_cfg.vector_fallback_enabled:
            return []
        if self._embedding_fn is None or not self._taxonomy_embeddings:
            return []

        query_text = _build_searchable_text(context)
        try:
            query_vec = self._embedding_fn(query_text)
        except Exception:
            logger.warning("Vector fallback: failed to embed query, degrading to keyword-only", exc_info=True)
            return []

        scored: list[tuple[str, float]] = []
        for entry_id, entry_vec in self._taxonomy_embeddings.items():
            sim = _cosine_similarity(query_vec, entry_vec)
            if sim >= qt_cfg.vector_similarity_threshold:
                scored.append((entry_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [eid for eid, _ in scored[: qt_cfg.vector_top_k]]

        entry_map = {e.id: e for e in self._taxonomy.all_entries()}
        return [entry_map[eid] for eid in top_ids if eid in entry_map]

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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from meta_learning.shared.models import (
    ErrorTaxonomy,
    MetaLearningConfig,
    QuickThinkResult,
    TaskContext,
    TaxonomyEntry,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LOCAL_MATCH_THRESHOLD = 0.24
_LOCAL_MATCH_TOP_K = 3
_FIELD_WEIGHTS = {
    "name": 2.0,
    "trigger": 2.2,
    "keywords": 3.0,
    "prevention": 1.8,
    "fix_sop": 1.2,
}
_LOCAL_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "when",
        "is",
        "are",
        "be",
        "it",
        "this",
        "that",
        "user",
        "task",
    }
)


@dataclass(frozen=True)
class _RuleDoc:
    entry: TaxonomyEntry
    weighted_terms: Counter[str]
    unique_terms: set[str]
    char_ngrams: set[str]
    length: float


class QuickThinkIndex:
    def __init__(
        self,
        taxonomy: ErrorTaxonomy,
        config: MetaLearningConfig,
    ) -> None:
        self._config = config
        self._taxonomy = taxonomy
        self._keyword_index: dict[str, list[TaxonomyEntry]] = taxonomy.all_keywords()
        self._rule_matcher = LocalRuleMatcher(taxonomy)
        self._irreversible_patterns: list[str] = [
            kw.lower() for kw in config.layer1.quick_think.irreversible_keywords
        ]
        self._recent_failure_signatures: set[str] = set()
        self._known_tools: set[str] = set()

    def update_taxonomy(self, taxonomy: ErrorTaxonomy) -> None:
        self._taxonomy = taxonomy
        self._keyword_index = taxonomy.all_keywords()
        self._rule_matcher = LocalRuleMatcher(taxonomy)

    def register_failure_signature(self, signature: str) -> None:
        self._recent_failure_signatures.add(signature.lower())

    def register_known_tool(self, tool: str) -> None:
        self._known_tools.add(tool.lower())

    def evaluate(self, context: TaskContext) -> QuickThinkResult:
        matched_signals: list[str] = []
        matched_taxonomy_ids: list[str] = []

        taxonomy_hits = self._check_taxonomy_match(context)
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

    def _check_taxonomy_match(self, context: TaskContext) -> list[TaxonomyEntry]:
        hits = self._check_keyword_match(context)
        seen_ids = {entry.id for entry in hits}
        for entry in self._rule_matcher.match(context):
            if entry.id not in seen_ids:
                hits.append(entry)
                seen_ids.add(entry.id)
        return hits[:_LOCAL_MATCH_TOP_K]

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


class LocalRuleMatcher:
    """Small local lexical matcher for taxonomy rules.

    It combines exact keywords, weighted BM25-style token overlap, and character
    n-gram overlap. This gives useful fuzzy matching for English and Chinese
    without external embedding services.
    """

    def __init__(self, taxonomy: ErrorTaxonomy) -> None:
        self._docs = [_build_rule_doc(entry) for entry in taxonomy.all_entries()]
        self._avg_len = (
            sum(doc.length for doc in self._docs) / len(self._docs)
            if self._docs
            else 0.0
        )
        self._idf = self._build_idf(self._docs)

    def match(self, context: TaskContext) -> list[TaxonomyEntry]:
        if not self._docs:
            return []

        query = _build_searchable_text(context)
        query_terms = set(_tokenize_for_local_match(query))
        query_ngrams = _char_ngrams(query)
        if not query_terms and not query_ngrams:
            return []

        scored: list[tuple[float, TaxonomyEntry]] = []
        for doc in self._docs:
            score = self._score_doc(doc, query_terms, query_ngrams)
            if score >= _LOCAL_MATCH_THRESHOLD:
                scored.append((score, doc.entry))

        scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        return [entry for _, entry in scored[:_LOCAL_MATCH_TOP_K]]

    def _build_idf(self, docs: list[_RuleDoc]) -> dict[str, float]:
        doc_count = len(docs)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.unique_terms)
        return {
            term: math.log(1 + (doc_count - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def _score_doc(
        self,
        doc: _RuleDoc,
        query_terms: set[str],
        query_ngrams: set[str],
    ) -> float:
        bm25 = self._bm25_score(doc, query_terms)
        ngram_overlap = _overlap_ratio(query_ngrams, doc.char_ngrams)
        keyword_boost = _keyword_overlap_boost(doc.entry, query_terms, query_ngrams)
        confidence_weight = 0.8 + min(max(doc.entry.confidence, 0.0), 1.0) * 0.2
        return (bm25 + ngram_overlap * 0.6 + keyword_boost) * confidence_weight

    def _bm25_score(self, doc: _RuleDoc, query_terms: set[str]) -> float:
        if not query_terms or not self._avg_len:
            return 0.0

        k1 = 1.2
        b = 0.75
        raw_score = 0.0
        for term in query_terms:
            freq = doc.weighted_terms.get(term, 0.0)
            if freq <= 0:
                continue
            denom = freq + k1 * (1 - b + b * doc.length / self._avg_len)
            raw_score += self._idf.get(term, 0.0) * (freq * (k1 + 1)) / denom

        return raw_score / max(len(query_terms) ** 0.5, 1.0)


def _build_rule_doc(entry: TaxonomyEntry) -> _RuleDoc:
    weighted_terms: Counter[str] = Counter()
    weighted_terms.update(
        _weighted_terms(entry.name, _FIELD_WEIGHTS["name"])
    )
    weighted_terms.update(
        _weighted_terms(entry.trigger, _FIELD_WEIGHTS["trigger"])
    )
    weighted_terms.update(
        _weighted_terms(" ".join(entry.keywords), _FIELD_WEIGHTS["keywords"])
    )
    weighted_terms.update(
        _weighted_terms(entry.prevention, _FIELD_WEIGHTS["prevention"])
    )
    weighted_terms.update(
        _weighted_terms(entry.fix_sop, _FIELD_WEIGHTS["fix_sop"])
    )
    text = " ".join(
        [
            entry.name,
            entry.trigger,
            entry.prevention,
            entry.fix_sop,
            " ".join(entry.keywords),
        ]
    )
    return _RuleDoc(
        entry=entry,
        weighted_terms=weighted_terms,
        unique_terms=set(weighted_terms),
        char_ngrams=_char_ngrams(text),
        length=sum(weighted_terms.values()) or 1.0,
    )


def _weighted_terms(text: str, weight: float) -> Counter[str]:
    terms = Counter(_tokenize_for_local_match(text))
    return Counter({term: count * weight for term, count in terms.items()})


def _tokenize_for_local_match(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [
        token
        for token in _WORD_RE.findall(lowered)
        if token not in _LOCAL_STOPWORDS
    ]
    for cjk_run in _CJK_RE.findall(lowered):
        tokens.extend(cjk_run)
        tokens.extend(_sliding_ngrams(cjk_run, 2))
        tokens.extend(_sliding_ngrams(cjk_run, 3))
    return [token for token in tokens if len(token) > 1]


def _char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < 3:
        return {compact} if compact else set()
    grams = set(_sliding_ngrams(compact, 3))
    grams.update(_sliding_ngrams(compact, 4))
    return grams


def _sliding_ngrams(text: str, size: int) -> list[str]:
    if len(text) < size:
        return []
    return [text[i : i + size] for i in range(len(text) - size + 1)]


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _keyword_overlap_boost(
    entry: TaxonomyEntry,
    query_terms: set[str],
    query_ngrams: set[str],
) -> float:
    boost = 0.0
    for keyword in entry.keywords:
        keyword_text = keyword.lower()
        keyword_terms = set(_tokenize_for_local_match(keyword_text))
        keyword_grams = _char_ngrams(keyword_text)
        if keyword_terms and keyword_terms <= query_terms:
            boost += 0.4
        elif keyword_grams and _overlap_ratio(keyword_grams, query_ngrams) >= 0.8:
            boost += 0.25
    return min(boost, 0.8)

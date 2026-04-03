from __future__ import annotations

import logging
from datetime import date

from meta_learning.shared.io import (
    list_all_experiences,
    load_error_taxonomy,
    load_experience_index,
    next_taxonomy_id,
    read_experience,
    save_error_taxonomy,
    save_experience_index,
    write_experience,
)
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    Experience,
    ExperienceCluster,
    MetaLearningConfig,
    TaxonomyEntry,
    TaxonomyExtraction,
)

logger = logging.getLogger(__name__)

_STRIP_CHARS = ".:,;()[]{}\"'`<>!?/\\#@$%^&*+=~|"

_SIMILARITY_STOPWORDS = frozenset({
    "the", "a", "an", "is", "of", "to", "in", "for", "and", "or", "not",
    "it", "this", "that", "with", "from", "are", "was", "were", "been",
    "be", "has", "have", "had", "but", "if", "its", "can", "does", "do",
    "did", "will", "would", "should", "could", "may", "might", "on", "no",
})


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.lower().split():
        cleaned = raw.strip(_STRIP_CHARS)
        if cleaned and len(cleaned) > 1 and cleaned not in _SIMILARITY_STOPWORDS:
            tokens.add(cleaned)
    return tokens


def _entry_text_similarity(
    new_name: str, new_prevention: str, new_trigger: str,
    existing: TaxonomyEntry,
) -> float:
    text_new = f"{new_name} {new_prevention} {new_trigger}"
    text_old = f"{existing.name} {existing.prevention} {existing.trigger}"
    tokens_new = _tokenize(text_new)
    tokens_old = _tokenize(text_old)
    if not tokens_new or not tokens_old:
        return 0.0
    return len(tokens_new & tokens_old) / min(len(tokens_new), len(tokens_old))


MERGE_SIMILARITY_THRESHOLD = 0.7


def _merge_into_existing(
    existing: TaxonomyEntry,
    extraction: TaxonomyExtraction,
    new_experiences: list[Experience],
) -> None:
    new_exp_ids = [e.id for e in new_experiences]
    for eid in new_exp_ids:
        if eid not in existing.source_exps:
            existing.source_exps.append(eid)
    for kw in extraction.keywords:
        if kw not in existing.keywords:
            existing.keywords.append(kw)
    existing.confidence = min(
        existing.confidence + 0.05 * len(new_experiences), 1.0
    )
    existing.last_verified = date.today()


class TaxonomyBuilder:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def build_from_clusters(
        self, clusters: list[ExperienceCluster]
    ) -> list[TaxonomyEntry]:
        if not clusters:
            return []

        taxonomy = load_error_taxonomy(self._config)
        existing_entry_ids = {e.id for e in taxonomy.all_entries()}
        all_existing = taxonomy.all_entries()
        new_entries: list[TaxonomyEntry] = []

        for cluster in clusters:
            experiences = self._load_cluster_experiences(cluster)
            if not experiences:
                continue

            promoted_groups: dict[str, list[Experience]] = {}
            unpromoted: list[Experience] = []
            for exp in experiences:
                if exp.promoted_to and exp.promoted_to in existing_entry_ids:
                    promoted_groups.setdefault(exp.promoted_to, []).append(exp)
                else:
                    unpromoted.append(exp)

            if not unpromoted and promoted_groups:
                logger.info(
                    "Cluster %s: all %d experiences already covered, skipping",
                    cluster.cluster_id, len(experiences),
                )
                continue

            if unpromoted and promoted_groups:
                dominant_tax_id = max(
                    promoted_groups, key=lambda k: len(promoted_groups[k])
                )
                dominant_entry = next(
                    (e for e in all_existing if e.id == dominant_tax_id), None
                )
                if dominant_entry is not None:
                    for exp in unpromoted:
                        if exp.id not in dominant_entry.source_exps:
                            dominant_entry.source_exps.append(exp.id)
                    dominant_entry.confidence = min(
                        dominant_entry.confidence + 0.05 * len(unpromoted), 1.0
                    )
                    dominant_entry.last_verified = date.today()
                    self._mark_experiences_promoted(unpromoted, dominant_tax_id)
                    self._mark_cluster_promoted(cluster, dominant_tax_id)
                    logger.info(
                        "Incremental update: added %d experiences to %s",
                        len(unpromoted), dominant_tax_id,
                    )
                    continue

            extraction = await self._llm.extract_taxonomy(experiences)

            best_match: TaxonomyEntry | None = None
            best_sim = 0.0
            for entry in all_existing:
                sim = _entry_text_similarity(
                    extraction.name, extraction.prevention, extraction.trigger,
                    entry,
                )
                if sim > best_sim:
                    best_sim = sim
                    best_match = entry

            if best_match is not None and best_sim >= MERGE_SIMILARITY_THRESHOLD:
                logger.info(
                    "Merging into existing taxonomy %s (similarity=%.2f)",
                    best_match.id, best_sim,
                )
                _merge_into_existing(best_match, extraction, experiences)
                self._mark_experiences_promoted(experiences, best_match.id)
                self._mark_cluster_promoted(cluster, best_match.id)
                continue

            domain = cluster.task_type.value
            subdomain = _infer_subdomain(experiences)
            entry_id = next_taxonomy_id(
                taxonomy, _make_prefix(domain, subdomain)
            )

            entry = TaxonomyEntry(
                id=entry_id,
                name=extraction.name,
                trigger=extraction.trigger,
                fix_sop=extraction.fix_sop,
                prevention=extraction.prevention,
                confidence=_compute_cluster_confidence(experiences),
                source_exps=[e.id for e in experiences],
                keywords=extraction.keywords,
                created_at=date.today(),
                last_verified=date.today(),
            )

            taxonomy.add_entry(domain, subdomain, entry)
            all_existing.append(entry)
            existing_entry_ids.add(entry_id)
            new_entries.append(entry)

            self._mark_experiences_promoted(experiences, entry_id)
            self._mark_cluster_promoted(cluster, entry_id)

        gc_count = _gc_orphan_entries(taxonomy, self._config)
        if gc_count:
            logger.info("Orphan GC removed %d stale taxonomy entries", gc_count)

        save_error_taxonomy(taxonomy, self._config)
        return new_entries

    def _load_cluster_experiences(self, cluster: ExperienceCluster) -> list[Experience]:
        from pathlib import Path

        pool_dir = Path(self._config.experience_pool_path)
        experiences: list[Experience] = []
        for exp_id in cluster.experience_ids:
            for p in pool_dir.rglob(f"{exp_id}.yaml"):
                try:
                    experiences.append(read_experience(p))
                except Exception:
                    continue
        return experiences

    def _mark_experiences_promoted(
        self, experiences: list[Experience], taxonomy_id: str
    ) -> None:
        for exp in experiences:
            exp.promoted_to = taxonomy_id
            write_experience(exp, self._config)

    def _mark_cluster_promoted(
        self, cluster: ExperienceCluster, taxonomy_id: str
    ) -> None:
        index = load_experience_index(self._config)
        for c in index.clusters:
            if c.cluster_id == cluster.cluster_id:
                c.promoted_to_taxonomy = taxonomy_id
                break
        save_experience_index(index, self._config)


def _infer_subdomain(experiences: list[Experience]) -> str:
    signatures = [e.failure_signature for e in experiences if e.failure_signature]
    if not signatures:
        return "general"

    first_sig = signatures[0].lower()
    if any(kw in first_sig for kw in ["typescript", "ts2", "type"]):
        return "typescript"
    if any(kw in first_sig for kw in ["python", "import", "module"]):
        return "python"
    if any(kw in first_sig for kw in ["docker", "container", "image"]):
        return "docker"
    if any(kw in first_sig for kw in ["git", "merge", "branch"]):
        return "git"
    return "general"


def _make_prefix(domain: str, subdomain: str) -> str:
    d = domain[:3] if len(domain) >= 3 else domain
    s = subdomain[:3] if len(subdomain) >= 3 else subdomain
    return f"{d}-{s}"


def _compute_cluster_confidence(experiences: list[Experience]) -> float:
    if not experiences:
        return 0.6
    avg_conf = sum(e.confidence for e in experiences) / len(experiences)
    size_bonus = min(len(experiences) * 0.05, 0.2)
    return min(avg_conf + size_bonus, 1.0)


def _gc_orphan_entries(
    taxonomy: "ErrorTaxonomy", config: MetaLearningConfig
) -> int:
    """Remove taxonomy entries that no experience's ``promoted_to`` points to."""
    all_exps = list_all_experiences(config)
    live_tax_ids = {e.promoted_to for e in all_exps if e.promoted_to}

    all_entries = taxonomy.all_entries()
    orphan_ids = [e.id for e in all_entries if e.id not in live_tax_ids]

    for oid in orphan_ids:
        taxonomy.remove_entry(oid)
        logger.info("GC: removed orphan taxonomy entry %s", oid)

    return len(orphan_ids)

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from meta_learning.shared.io import (
    list_all_experiences,
    load_error_taxonomy,
    load_experience_index,
    next_cluster_id,
    save_experience_index,
    write_experience,
)
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    Experience,
    ExperienceCluster,
    ExperienceIndex,
    MetaLearningConfig,
    TaskType,
)

logger = logging.getLogger(__name__)

_STRIP_CHARS = ".:,;()[]{}\"'`<>!?/\\#@$%^&*+=~|"

_SIMILARITY_STOPWORDS = frozenset(
    {
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


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.lower().split():
        cleaned = raw.strip(_STRIP_CHARS)
        if cleaned and len(cleaned) > 1 and cleaned not in _SIMILARITY_STOPWORDS:
            tokens.add(cleaned)
    return tokens


def _experience_text(exp: Experience) -> str:
    parts = [exp.scene, exp.root_cause, exp.resolution]
    if exp.failure_signature:
        parts.append(exp.failure_signature)
    return " ".join(parts)


def _keyword_similarity(exp_a: Experience, exp_b: Experience) -> float:
    """Jaccard-like overlap on meaningful tokens.  O(len(text)) per pair."""
    words_a = _tokenize(_experience_text(exp_a))
    words_b = _tokenize(_experience_text(exp_b))
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    return len(overlap) / min(len(words_a), len(words_b))


def _compute_similarity(exp_a: Experience, exp_b: Experience) -> float:
    return _keyword_similarity(exp_a, exp_b)


class Consolidator:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def consolidate(self) -> ExperienceIndex:
        all_exps = list_all_experiences(self._config)

        if self._config.confidence.decay_enabled:
            all_exps = self._apply_confidence_decay(all_exps)

        valid_exps = [
            e
            for e in all_exps
            if e.confidence >= self._config.confidence.prune_threshold
        ]

        index = load_experience_index(self._config)
        old_promoted_map = _build_promoted_map(index)
        grouped = _group_by_task_type(valid_exps)

        stale_taxonomy_ids: set[str] = set()

        for task_type, exps in grouped.items():
            new_clusters = await self._cluster_within_group(
                task_type, exps, index
            )
            stale_taxonomy_ids.update(
                _detect_stale_taxonomies(old_promoted_map, new_clusters)
            )
            index.clusters = _replace_task_clusters(
                index=index,
                task_type=task_type,
                clusters_for_task=new_clusters,
            )

        self._stale_taxonomy_ids = stale_taxonomy_ids
        index.last_updated = datetime.now()
        save_experience_index(index, self._config)
        return index

    def _apply_confidence_decay(
        self, experiences: list[Experience]
    ) -> list[Experience]:
        now = datetime.now()
        decay_base = self._config.confidence.decay_base
        for exp in experiences:
            days_old = (now - exp.created_at).total_seconds() / 86400.0
            if days_old <= 0:
                continue
            decay_factor = decay_base**days_old
            decayed = exp.initial_confidence * decay_factor
            if decayed != exp.confidence:
                exp.confidence = max(decayed, 0.0)
                write_experience(exp, self._config)
        return experiences

    async def _cluster_within_group(
        self,
        task_type: TaskType,
        experiences: list[Experience],
        index: ExperienceIndex,
    ) -> list[ExperienceCluster]:
        if len(experiences) < 2:
            return []

        consolidate_cfg = self._config.layer2.consolidate
        distance_threshold = 1.0 - consolidate_cfg.similarity_threshold
        n = len(experiences)
        exp_map = {e.id: e for e in experiences}

        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = _compute_similarity(experiences[i], experiences[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        condensed = squareform(dist_matrix, checks=False)

        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=distance_threshold, criterion="distance")

        label_groups: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_groups[label].append(idx)

        clusters: list[ExperienceCluster] = []
        for members in label_groups.values():
            if len(members) < 2:
                continue
            member_ids = sorted(experiences[i].id for i in members)
            representative = exp_map[member_ids[0]]
            cluster = ExperienceCluster(
                cluster_id=next_cluster_id(index),
                task_type=task_type,
                failure_signature_pattern=representative.failure_signature or "unknown",
                experience_ids=member_ids,
            )
            clusters.append(cluster)

        return clusters

    def get_clusters_ready_for_taxonomy(self) -> list[ExperienceCluster]:
        index = load_experience_index(self._config)
        min_size = self._config.layer2.consolidate.min_cluster_size_for_taxonomy

        taxonomy = load_error_taxonomy(self._config)
        existing_tax_ids = {e.id for e in taxonomy.all_entries()}

        all_exp_map = {e.id: e for e in list_all_experiences(self._config)}

        ready: list[ExperienceCluster] = []
        for c in index.clusters:
            if len(c.experience_ids) < min_size:
                continue

            members = [all_exp_map.get(eid) for eid in c.experience_ids]
            members = [m for m in members if m is not None]
            if not members:
                continue

            all_covered = all(
                m.promoted_to is not None and m.promoted_to in existing_tax_ids
                for m in members
            )
            if not all_covered:
                ready.append(c)

        return ready


def _group_by_task_type(
    experiences: list[Experience],
) -> dict[TaskType, list[Experience]]:
    groups: dict[TaskType, list[Experience]] = {}
    for exp in experiences:
        groups.setdefault(exp.task_type, []).append(exp)
    return groups


def _build_promoted_map(
    index: ExperienceIndex,
) -> dict[str, set[str]]:
    """Map taxonomy_id -> set of experience_ids that were in its cluster."""
    result: dict[str, set[str]] = {}
    for cluster in index.clusters:
        if cluster.promoted_to_taxonomy:
            result[cluster.promoted_to_taxonomy] = set(cluster.experience_ids)
    return result


def _detect_stale_taxonomies(
    old_promoted_map: dict[str, set[str]],
    new_clusters: list[ExperienceCluster],
) -> set[str]:
    """Find taxonomy entries whose underlying cluster membership changed."""
    stale: set[str] = set()
    new_cluster_sets = [set(c.experience_ids) for c in new_clusters]
    for tax_id, old_members in old_promoted_map.items():
        still_together = any(old_members <= s for s in new_cluster_sets)
        if not still_together:
            stale.add(tax_id)
            logger.info("Taxonomy %s marked stale (cluster membership changed)", tax_id)
    return stale


def _replace_task_clusters(
    *,
    index: ExperienceIndex,
    task_type: TaskType,
    clusters_for_task: list[ExperienceCluster],
) -> list[ExperienceCluster]:
    keep_other = [c for c in index.clusters if c.task_type != task_type]
    return keep_other + clusters_for_task

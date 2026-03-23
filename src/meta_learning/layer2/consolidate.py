from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from meta_learning.shared.io import (
    list_all_experiences,
    load_experience_index,
    next_cluster_id,
    read_signal,
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

EmbeddingFn = Callable[[str], list[float]]
_get_embedding: EmbeddingFn | None = None


def set_embedding_fn(fn: EmbeddingFn | None) -> None:
    global _get_embedding  # noqa: PLW0603
    _get_embedding = fn


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


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_similarity(exp_a: Experience, exp_b: Experience) -> float:
    """Jaccard-like overlap on meaningful tokens.  O(len(text)) per pair."""
    words_a = _tokenize(_experience_text(exp_a))
    words_b = _tokenize(_experience_text(exp_b))
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    return len(overlap) / min(len(words_a), len(words_b))


def _compute_similarity(exp_a: Experience, exp_b: Experience) -> float:
    if _get_embedding is not None:
        vec_a = _get_embedding(_experience_text(exp_a))
        vec_b = _get_embedding(_experience_text(exp_b))
        return _cosine_similarity(vec_a, vec_b)
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

        grouped = _group_by_task_type(valid_exps)

        index = load_experience_index(self._config)
        already_clustered = _already_clustered_ids(index)

        for task_type, exps in grouped.items():
            unclustered = [e for e in exps if e.id not in already_clustered]
            if not unclustered:
                continue

            new_clusters = await self._cluster_within_group(
                task_type, unclustered, index
            )
            index.clusters.extend(new_clusters)

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
            decayed = exp.confidence * decay_factor
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
        similarity_threshold = consolidate_cfg.similarity_threshold
        use_llm = consolidate_cfg.use_llm_clustering
        max_llm_calls = consolidate_cfg.max_llm_calls_per_group

        adjacency: dict[str, set[str]] = {e.id: set() for e in experiences}
        exp_map = {e.id: e for e in experiences}

        # Phase 1 — lightweight pre-filter: O(N^2) text similarity (cheap)
        # splits pairs into auto-merge (high sim) vs LLM-candidates (moderate sim)
        candidate_pairs: list[tuple[Experience, Experience]] = []
        auto_merge_pairs: list[tuple[str, str]] = []

        for i, exp_a in enumerate(experiences):
            for exp_b in experiences[i + 1 :]:
                sim = _compute_similarity(exp_a, exp_b)
                if sim >= 0.7:
                    auto_merge_pairs.append((exp_a.id, exp_b.id))
                elif sim >= similarity_threshold:
                    candidate_pairs.append((exp_a, exp_b))

        for id_a, id_b in auto_merge_pairs:
            adjacency[id_a].add(id_b)
            adjacency[id_b].add(id_a)

        # Phase 2 — LLM confirmation on ambiguous pairs (capped at max_llm_calls)
        if use_llm and candidate_pairs:
            llm_calls = 0
            for exp_a, exp_b in candidate_pairs:
                if llm_calls >= max_llm_calls:
                    logger.info(
                        "LLM call cap reached (%d); skipping remaining %d candidate pairs",
                        max_llm_calls,
                        len(candidate_pairs) - llm_calls,
                    )
                    break
                judgment = await self._llm.judge_same_class(exp_a, exp_b)
                llm_calls += 1
                if judgment.same_class:
                    adjacency[exp_a.id].add(exp_b.id)
                    adjacency[exp_b.id].add(exp_a.id)

        # Phase 3 — BFS connected-component discovery
        clusters: list[ExperienceCluster] = []
        visited: set[str] = set()

        for exp_id in adjacency:
            if exp_id in visited:
                continue
            component = _bfs_component(exp_id, adjacency)
            visited.update(component)

            if len(component) < 2:
                continue

            representative_exp = exp_map[sorted(component)[0]]
            cluster = ExperienceCluster(
                cluster_id=next_cluster_id(index),
                task_type=task_type,
                failure_signature_pattern=representative_exp.failure_signature
                or "unknown",
                experience_ids=sorted(component),
            )
            clusters.append(cluster)

        return clusters

    def get_clusters_ready_for_taxonomy(self) -> list[ExperienceCluster]:
        index = load_experience_index(self._config)
        min_size = self._config.layer2.consolidate.min_cluster_size_for_taxonomy
        return [
            c
            for c in index.clusters
            if len(c.experience_ids) >= min_size and c.promoted_to_taxonomy is None
        ]


def _group_by_task_type(
    experiences: list[Experience],
) -> dict[TaskType, list[Experience]]:
    groups: dict[TaskType, list[Experience]] = {}
    for exp in experiences:
        groups.setdefault(exp.task_type, []).append(exp)
    return groups


def _already_clustered_ids(index: ExperienceIndex) -> set[str]:
    ids: set[str] = set()
    for cluster in index.clusters:
        ids.update(cluster.experience_ids)
    return ids


def _bfs_component(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


def _resolve_experience_image_paths(
    experiences: list[Experience],
    config: MetaLearningConfig,
) -> dict[str, list[str]]:
    text_to_images: dict[str, list[str]] = {}
    signal_dir = Path(config.signal_buffer_path)
    if not signal_dir.exists():
        return text_to_images

    for exp in experiences:
        sig_path = signal_dir / f"{exp.source_signal}.yaml"
        if not sig_path.exists():
            continue
        try:
            signal = read_signal(sig_path)
        except Exception:
            continue
        if not signal.image_snapshots:
            continue
        text_key = _experience_text(exp)
        text_to_images[text_key] = signal.image_snapshots

    return text_to_images


def bootstrap_multimodal_embedding(config: MetaLearningConfig) -> None:
    if not config.dashscope.enabled:
        return

    from meta_learning.shared.embedding_dashscope import MultimodalEmbedding

    all_exps = list_all_experiences(config)
    image_lookup = _resolve_experience_image_paths(all_exps, config)

    embedding = MultimodalEmbedding(config.dashscope)
    set_embedding_fn(embedding.make_embedding_fn(image_lookup))

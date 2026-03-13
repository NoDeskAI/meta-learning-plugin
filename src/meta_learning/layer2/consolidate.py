from __future__ import annotations

from datetime import datetime
from itertools import combinations

from meta_learning.shared.io import (
    list_all_experiences,
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

        adjacency: dict[str, set[str]] = {e.id: set() for e in experiences}
        exp_map = {e.id: e for e in experiences}

        for exp_a, exp_b in combinations(experiences, 2):
            judgment = await self._llm.judge_same_class(exp_a, exp_b)
            if judgment.same_class:
                adjacency[exp_a.id].add(exp_b.id)
                adjacency[exp_b.id].add(exp_a.id)

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
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited

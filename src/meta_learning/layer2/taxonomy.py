from __future__ import annotations

from datetime import date

from meta_learning.shared.io import (
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
)


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
        new_entries: list[TaxonomyEntry] = []

        for cluster in clusters:
            experiences = self._load_cluster_experiences(cluster)
            if not experiences:
                continue

            extraction = await self._llm.extract_taxonomy(experiences)

            domain = cluster.task_type.value
            subdomain = _infer_subdomain(experiences)
            entry_id = next_taxonomy_id(taxonomy, _make_prefix(domain, subdomain))

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
            new_entries.append(entry)

            self._mark_experiences_promoted(experiences, entry_id)
            self._mark_cluster_promoted(cluster, entry_id)

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

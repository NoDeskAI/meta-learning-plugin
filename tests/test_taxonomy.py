from datetime import datetime
from pathlib import Path

import pytest

from meta_learning.layer2.taxonomy import TaxonomyBuilder
from meta_learning.shared.io import (
    load_error_taxonomy,
    read_experience,
    write_experience,
)
from meta_learning.shared.models import (
    Experience,
    ExperienceCluster,
    TaskType,
)


def _make_cluster_with_experiences(tmp_config, count: int = 3):
    experiences = []
    for i in range(count):
        exp = Experience(
            id=f"exp-{i + 1:03d}",
            task_type=TaskType.CODING,
            created_at=datetime.now(),
            source_signal=f"sig-{i + 1:03d}",
            confidence=0.7,
            scene=f"Fix TypeScript error #{i}",
            failure_signature="TS2345 generic type inference failure",
            root_cause="Generic constraint issue",
            resolution="Add explicit type annotation",
            meta_insight="Always annotate complex generics",
        )
        write_experience(exp, tmp_config)
        experiences.append(exp)

    cluster = ExperienceCluster(
        cluster_id="clust-001",
        task_type=TaskType.CODING,
        failure_signature_pattern="TS2345 generic type inference failure",
        experience_ids=[e.id for e in experiences],
    )
    return cluster, experiences


@pytest.mark.asyncio
class TestTaxonomyBuilder:
    async def test_empty_clusters(self, tmp_config, stub_llm):
        builder = TaxonomyBuilder(tmp_config, stub_llm)
        entries = await builder.build_from_clusters([])
        assert entries == []

    async def test_builds_taxonomy_entry(self, tmp_config, stub_llm):
        cluster, _ = _make_cluster_with_experiences(tmp_config, 3)
        builder = TaxonomyBuilder(tmp_config, stub_llm)
        entries = await builder.build_from_clusters([cluster])
        assert len(entries) == 1
        entry = entries[0]
        assert entry.id.startswith("tax-")
        assert len(entry.source_exps) == 3
        assert entry.confidence > 0.6

    async def test_saves_to_taxonomy_file(self, tmp_config, stub_llm):
        cluster, _ = _make_cluster_with_experiences(tmp_config, 3)
        builder = TaxonomyBuilder(tmp_config, stub_llm)
        await builder.build_from_clusters([cluster])
        tax = load_error_taxonomy(tmp_config)
        assert len(tax.all_entries()) == 1

    async def test_marks_experiences_promoted(self, tmp_config, stub_llm):
        cluster, experiences = _make_cluster_with_experiences(tmp_config, 3)
        builder = TaxonomyBuilder(tmp_config, stub_llm)
        entries = await builder.build_from_clusters([cluster])
        pool_dir = Path(tmp_config.experience_pool_path)
        for exp_id in cluster.experience_ids:
            for p in pool_dir.rglob(f"{exp_id}.yaml"):
                loaded = read_experience(p)
                assert loaded.promoted_to == entries[0].id

    async def test_confidence_increases_with_cluster_size(self, tmp_config, stub_llm):
        cluster_small, _ = _make_cluster_with_experiences(tmp_config, 3)
        builder = TaxonomyBuilder(tmp_config, stub_llm)
        entries_small = await builder.build_from_clusters([cluster_small])
        original_confidence = entries_small[0].confidence

        pool_dir = Path(tmp_config.experience_pool_path)
        for p in pool_dir.rglob("exp-*.yaml"):
            p.unlink()

        cluster_large, _ = _make_cluster_with_experiences(tmp_config, 6)
        cluster_large.cluster_id = "clust-002"
        await builder.build_from_clusters([cluster_large])

        tax = load_error_taxonomy(tmp_config)
        merged_entry = tax.all_entries()[0]
        assert merged_entry.confidence >= original_confidence

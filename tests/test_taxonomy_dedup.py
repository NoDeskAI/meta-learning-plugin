"""Tests for taxonomy deduplication and merge logic."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from meta_learning.layer2.taxonomy import (
    MERGE_SIMILARITY_THRESHOLD,
    TaxonomyBuilder,
    _entry_text_similarity,
    _gc_orphan_entries,
    _merge_into_existing,
    _tokenize,
)
from meta_learning.shared.io import (
    load_error_taxonomy,
    read_experience,
    save_error_taxonomy,
    write_experience,
)
from meta_learning.shared.models import (
    ErrorTaxonomy,
    Experience,
    ExperienceCluster,
    TaskType,
    TaxonomyEntry,
    TaxonomyExtraction,
)


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Always create projects under ~/workspace/")
        assert "always" in tokens
        assert "create" in tokens
        assert "projects" in tokens
        assert "~/workspace/" not in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize("the error is in the function")
        assert "the" not in tokens
        assert "is" not in tokens

    def test_short_tokens_removed(self):
        tokens = _tokenize("a b cd efg")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens
        assert "efg" in tokens


class TestEntryTextSimilarity:
    def test_identical_entries_score_1(self):
        entry = TaxonomyEntry(
            id="tax-001", name="Always create under workspace",
            trigger="creating projects", fix_sop="mkdir workspace",
            prevention="Always create under ~/workspace/",
            confidence=0.8, source_exps=["exp-001"],
            created_at=date.today(), last_verified=date.today(),
        )
        sim = _entry_text_similarity(
            entry.name, entry.prevention, entry.trigger, entry,
        )
        assert sim == 1.0

    def test_similar_entries_high_score(self):
        entry = TaxonomyEntry(
            id="tax-001", name="Create projects under workspace",
            trigger="when creating new projects",
            fix_sop="use mkdir", prevention="Always create under ~/workspace/",
            confidence=0.8, source_exps=["exp-001"],
            created_at=date.today(), last_verified=date.today(),
        )
        sim = _entry_text_similarity(
            "Always create projects under workspace directory",
            "Create new projects under ~/workspace/, never under ~/projects/",
            "when user asks to create a project",
            entry,
        )
        assert sim >= 0.5

    def test_unrelated_entries_low_score(self):
        entry = TaxonomyEntry(
            id="tax-001", name="Backup env files",
            trigger="editing configuration", fix_sop="cp .env .env.bak",
            prevention="Always backup .env before modifying",
            confidence=0.7, source_exps=["exp-001"],
            created_at=date.today(), last_verified=date.today(),
        )
        sim = _entry_text_similarity(
            "Use 4 spaces for Python indentation",
            "Always use 4 spaces indent in Python files",
            "writing Python code",
            entry,
        )
        assert sim < MERGE_SIMILARITY_THRESHOLD


class TestMergeIntoExisting:
    def test_merge_boosts_confidence(self):
        entry = TaxonomyEntry(
            id="tax-001", name="Test", trigger="test",
            fix_sop="test", prevention="test",
            confidence=0.7, source_exps=["exp-001"],
            keywords=["workspace"],
            created_at=date.today(), last_verified=date(2026, 1, 1),
        )
        extraction = TaxonomyExtraction(
            name="Test2", trigger="test2", fix_sop="test2",
            prevention="test2", keywords=["projects", "workspace"],
        )
        new_exps = [
            Experience(
                id="exp-002", task_type=TaskType.CODING,
                created_at=datetime.now(), source_signal="sig-002",
                confidence=0.6, scene="test", root_cause="test",
                resolution="test", meta_insight="test",
            ),
        ]
        _merge_into_existing(entry, extraction, new_exps)

        assert entry.confidence == 0.75
        assert "exp-002" in entry.source_exps
        assert "projects" in entry.keywords
        assert entry.last_verified == date.today()

    def test_merge_deduplicates_exp_ids(self):
        entry = TaxonomyEntry(
            id="tax-001", name="Test", trigger="test",
            fix_sop="test", prevention="test",
            confidence=0.7, source_exps=["exp-001"],
            keywords=[], created_at=date.today(), last_verified=date.today(),
        )
        extraction = TaxonomyExtraction(
            name="Test", trigger="test", fix_sop="test",
            prevention="test", keywords=[],
        )
        existing_exp = Experience(
            id="exp-001", task_type=TaskType.CODING,
            created_at=datetime.now(), source_signal="sig-001",
            confidence=0.6, scene="test", root_cause="test",
            resolution="test", meta_insight="test",
        )
        _merge_into_existing(entry, extraction, [existing_exp])
        assert entry.source_exps.count("exp-001") == 1

    def test_merge_caps_confidence_at_1(self):
        entry = TaxonomyEntry(
            id="tax-001", name="Test", trigger="test",
            fix_sop="test", prevention="test",
            confidence=0.98, source_exps=["exp-001"],
            keywords=[], created_at=date.today(), last_verified=date.today(),
        )
        extraction = TaxonomyExtraction(
            name="Test", trigger="test", fix_sop="test",
            prevention="test", keywords=[],
        )
        new_exps = [
            Experience(
                id=f"exp-{i:03d}", task_type=TaskType.CODING,
                created_at=datetime.now(), source_signal=f"sig-{i:03d}",
                confidence=0.6, scene="test", root_cause="test",
                resolution="test", meta_insight="test",
            )
            for i in range(10, 20)
        ]
        _merge_into_existing(entry, extraction, new_exps)
        assert entry.confidence == 1.0


@pytest.mark.asyncio
class TestBuildFromClustersDedup:

    async def test_duplicate_cluster_merges_into_existing(self, tmp_config, stub_llm):
        taxonomy = ErrorTaxonomy()
        existing_entry = TaxonomyEntry(
            id="tax-cod-gen-001",
            name="coding: ts2345 (3 experiences)",
            trigger="TS2345 generic type inference failure",
            fix_sop="- Add explicit type annotation",
            prevention="Avoid conditions leading to: TS2345 generic type inference failure",
            confidence=0.8,
            source_exps=["exp-001", "exp-002", "exp-003"],
            keywords=["ts2345", "generic", "inference"],
            created_at=date.today(),
            last_verified=date(2026, 1, 1),
        )
        taxonomy.add_entry("coding", "typescript", existing_entry)
        save_error_taxonomy(taxonomy, tmp_config)

        for i in range(4, 7):
            exp = Experience(
                id=f"exp-{i:03d}", task_type=TaskType.CODING,
                created_at=datetime.now(), source_signal=f"sig-{i:03d}",
                confidence=0.7,
                scene=f"Fix TypeScript error #{i}",
                failure_signature="TS2345 generic type inference failure",
                root_cause="Generic constraint issue",
                resolution="Add explicit type annotation",
                meta_insight="Always annotate complex generics",
            )
            write_experience(exp, tmp_config)

        cluster = ExperienceCluster(
            cluster_id="clust-002", task_type=TaskType.CODING,
            failure_signature_pattern="TS2345 generic type inference failure",
            experience_ids=["exp-004", "exp-005", "exp-006"],
        )

        builder = TaxonomyBuilder(tmp_config, stub_llm)
        new_entries = await builder.build_from_clusters([cluster])

        assert len(new_entries) == 0, "Should merge, not create new entry"

        tax = load_error_taxonomy(tmp_config)
        all_entries = tax.all_entries()
        assert len(all_entries) == 1
        merged = all_entries[0]
        assert merged.id == "tax-cod-gen-001"
        assert merged.confidence > 0.8
        assert "exp-004" in merged.source_exps
        assert merged.last_verified == date.today()

    async def test_novel_cluster_creates_new_entry(self, tmp_config, stub_llm):
        taxonomy = ErrorTaxonomy()
        existing_entry = TaxonomyEntry(
            id="tax-cod-gen-001",
            name="coding: ts2345 (3 experiences)",
            trigger="TS2345 generic type inference failure",
            fix_sop="- Add explicit type annotation",
            prevention="Avoid conditions leading to: TS2345 generic type inference failure",
            confidence=0.8,
            source_exps=["exp-001"],
            keywords=["ts2345", "generic"],
            created_at=date.today(),
            last_verified=date.today(),
        )
        taxonomy.add_entry("coding", "typescript", existing_entry)
        save_error_taxonomy(taxonomy, tmp_config)

        anchor_exp = Experience(
            id="exp-001", task_type=TaskType.CODING,
            created_at=datetime.now(), source_signal="sig-001",
            confidence=0.7,
            scene="Fix TypeScript error",
            failure_signature="TS2345 generic type inference failure",
            root_cause="Generic constraint issue",
            resolution="Add explicit type annotation",
            meta_insight="Always annotate complex generics",
            promoted_to="tax-cod-gen-001",
        )
        write_experience(anchor_exp, tmp_config)

        exp = Experience(
            id="exp-010", task_type=TaskType.CONFIGURATION,
            created_at=datetime.now(), source_signal="sig-010",
            confidence=0.6,
            scene="Modify .env configuration",
            failure_signature="direct config modification without backup",
            root_cause="No backup before editing",
            resolution="Always backup .env before modifying",
            meta_insight="Backup config files before editing",
        )
        write_experience(exp, tmp_config)

        cluster = ExperienceCluster(
            cluster_id="clust-003", task_type=TaskType.CONFIGURATION,
            failure_signature_pattern="direct config modification without backup",
            experience_ids=["exp-010"],
        )

        builder = TaxonomyBuilder(tmp_config, stub_llm)
        new_entries = await builder.build_from_clusters([cluster])

        assert len(new_entries) == 1, "Should create new entry for novel pattern"

        tax = load_error_taxonomy(tmp_config)
        assert len(tax.all_entries()) == 2


def _seed_taxonomy_and_experiences(tmp_config, stub_llm):
    """Create 3 promoted experiences and their taxonomy entry."""
    experiences = []
    for i in range(1, 4):
        exp = Experience(
            id=f"exp-{i:03d}", task_type=TaskType.CODING,
            created_at=datetime.now(), source_signal=f"sig-{i:03d}",
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
        cluster_id="clust-001", task_type=TaskType.CODING,
        failure_signature_pattern="TS2345 generic type inference failure",
        experience_ids=[e.id for e in experiences],
    )
    return cluster, experiences


@pytest.mark.asyncio
class TestMatchFirstIdempotency:
    """Verify the core fix: repeated runs on the same cluster do not inflate taxonomy."""

    async def test_repeated_runs_no_inflation(self, tmp_config, stub_llm):
        cluster, _ = _seed_taxonomy_and_experiences(tmp_config, stub_llm)
        builder = TaxonomyBuilder(tmp_config, stub_llm)

        entries1 = await builder.build_from_clusters([cluster])
        count_after_first = len(load_error_taxonomy(tmp_config).all_entries())

        entries2 = await builder.build_from_clusters([cluster])
        count_after_second = len(load_error_taxonomy(tmp_config).all_entries())

        entries3 = await builder.build_from_clusters([cluster])
        count_after_third = len(load_error_taxonomy(tmp_config).all_entries())

        assert len(entries1) == 1, "First run should create one entry"
        assert len(entries2) == 0, "Second run should create nothing (all promoted)"
        assert len(entries3) == 0, "Third run should create nothing"
        assert count_after_first == count_after_second == count_after_third

    async def test_incremental_update_for_new_experience(self, tmp_config, stub_llm):
        cluster, _ = _seed_taxonomy_and_experiences(tmp_config, stub_llm)
        builder = TaxonomyBuilder(tmp_config, stub_llm)

        entries1 = await builder.build_from_clusters([cluster])
        tax_id = entries1[0].id
        original_confidence = entries1[0].confidence

        new_exp = Experience(
            id="exp-004", task_type=TaskType.CODING,
            created_at=datetime.now(), source_signal="sig-004",
            confidence=0.7,
            scene="Fix TypeScript error #4",
            failure_signature="TS2345 generic type inference failure",
            root_cause="Generic constraint issue",
            resolution="Add explicit type annotation",
            meta_insight="Always annotate complex generics",
        )
        write_experience(new_exp, tmp_config)

        mixed_cluster = ExperienceCluster(
            cluster_id="clust-002", task_type=TaskType.CODING,
            failure_signature_pattern="TS2345 generic type inference failure",
            experience_ids=["exp-001", "exp-002", "exp-003", "exp-004"],
        )

        entries2 = await builder.build_from_clusters([mixed_cluster])
        assert len(entries2) == 0, "Incremental update should not count as new entry"

        tax = load_error_taxonomy(tmp_config)
        all_entries = tax.all_entries()
        assert len(all_entries) == 1, "Still only one taxonomy entry"

        updated = all_entries[0]
        assert updated.id == tax_id
        assert "exp-004" in updated.source_exps
        assert updated.confidence > original_confidence

        pool_dir = Path(tmp_config.experience_pool_path)
        for p in pool_dir.rglob("exp-004.yaml"):
            loaded = read_experience(p)
            assert loaded.promoted_to == tax_id


@pytest.mark.asyncio
class TestOrphanGC:

    async def test_gc_removes_orphaned_entries(self, tmp_config, stub_llm):
        taxonomy = ErrorTaxonomy()
        orphan = TaxonomyEntry(
            id="tax-orphan-001", name="Orphan",
            trigger="never", fix_sop="n/a", prevention="n/a",
            confidence=0.5, source_exps=["exp-999"],
            created_at=date.today(), last_verified=date.today(),
        )
        taxonomy.add_entry("coding", "general", orphan)
        save_error_taxonomy(taxonomy, tmp_config)

        taxonomy = load_error_taxonomy(tmp_config)
        removed = _gc_orphan_entries(taxonomy, tmp_config)
        save_error_taxonomy(taxonomy, tmp_config)

        assert removed == 1
        assert len(load_error_taxonomy(tmp_config).all_entries()) == 0

    async def test_gc_preserves_live_entries(self, tmp_config, stub_llm):
        cluster, _ = _seed_taxonomy_and_experiences(tmp_config, stub_llm)
        builder = TaxonomyBuilder(tmp_config, stub_llm)
        entries = await builder.build_from_clusters([cluster])
        tax_id = entries[0].id

        taxonomy = ErrorTaxonomy()
        orphan = TaxonomyEntry(
            id="tax-orphan-002", name="Another orphan",
            trigger="never", fix_sop="n/a", prevention="n/a",
            confidence=0.3, source_exps=["exp-888"],
            created_at=date.today(), last_verified=date.today(),
        )
        live = load_error_taxonomy(tmp_config).all_entries()[0]
        taxonomy.add_entry("coding", "general", live)
        taxonomy.add_entry("coding", "general", orphan)
        save_error_taxonomy(taxonomy, tmp_config)

        taxonomy = load_error_taxonomy(tmp_config)
        removed = _gc_orphan_entries(taxonomy, tmp_config)
        save_error_taxonomy(taxonomy, tmp_config)

        assert removed == 1
        remaining = load_error_taxonomy(tmp_config).all_entries()
        assert len(remaining) == 1
        assert remaining[0].id == tax_id

"""Tests for taxonomy deduplication and merge logic."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from meta_learning.layer2.taxonomy import (
    MERGE_SIMILARITY_THRESHOLD,
    TaxonomyBuilder,
    _entry_text_similarity,
    _merge_into_existing,
    _tokenize,
)
from meta_learning.shared.io import (
    load_error_taxonomy,
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

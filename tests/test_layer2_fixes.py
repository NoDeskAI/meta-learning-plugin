"""Targeted verification tests for the Layer 2 fixes.

Each test class verifies a specific fix introduced in the
"Layer2 修复与置信度统一设计" plan:
  P0-1: Confidence decay idempotency + dual-factor taxonomy confidence
  P0-2: Materializer exception isolation
  P1-3: MERGE/SPLIT removal
  P1-4: Skill gating enforcement
  P2-6: SKILL.md prevention fallback
  P2-7: Category rules dedup
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from meta_learning.layer2.consolidate import Consolidator
from meta_learning.layer2.materialize import Materializer
from meta_learning.layer2.skill_evolve import SkillEvolver
from meta_learning.layer2.taxonomy import TaxonomyBuilder
from meta_learning.shared.llm import StubLLM
from meta_learning.shared.io import (
    boost_taxonomy_confidence,
    list_all_experiences,
    load_error_taxonomy,
    penalize_taxonomy_confidence,
    save_error_taxonomy,
    write_experience,
)
from meta_learning.shared.models import (
    ErrorTaxonomy,
    Experience,
    ExperienceCluster,
    TaxonomyExtraction,
    SkillUpdateAction,
    TaxonomyEntry,
    TaskType,
)
from meta_learning.sync_nobot import (
    _entry_rule_text,
    _render_category_md,
    _render_skill_md,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _exp(
    exp_id: str,
    days_old: int = 0,
    confidence: float = 0.6,
    initial_confidence: float = 0.6,
) -> Experience:
    return Experience(
        id=exp_id,
        task_type=TaskType.CODING,
        created_at=datetime.now() - timedelta(days=days_old),
        source_signal=f"sig-for-{exp_id}",
        initial_confidence=initial_confidence,
        confidence=confidence,
        scene="test scene",
        failure_signature="TS2345 type error",
        root_cause="root cause",
        resolution="resolution",
        meta_insight="insight",
    )


def _tax(
    entry_id: str = "tax-cod-gen-001",
    confidence: float = 0.8,
    adjustment: float = 0.0,
    source_count: int = 5,
    prevention: str = "Avoid X",
    fix_sop: str = "Do Y",
    trigger: str = "When Z",
    name: str = "Test Entry",
    keywords: list[str] | None = None,
) -> TaxonomyEntry:
    return TaxonomyEntry(
        id=entry_id,
        name=name,
        trigger=trigger,
        fix_sop=fix_sop,
        prevention=prevention,
        confidence=confidence,
        confidence_adjustment=adjustment,
        source_exps=[f"exp-{i:03d}" for i in range(source_count)],
        keywords=keywords or ["test"],
        created_at=date.today(),
        last_verified=date.today(),
    )


def _save_taxonomy(config, entries: list[TaxonomyEntry]) -> None:
    taxonomy = ErrorTaxonomy()
    for entry in entries:
        taxonomy.add_entry("coding", "general", entry)
    save_error_taxonomy(taxonomy, config)


# =======================================================================
# P0-1: Confidence decay idempotency
# =======================================================================

@pytest.mark.asyncio
class TestDecayIdempotency:
    """Running decay twice on the same data must produce the same result."""

    async def test_double_run_produces_same_confidence(self, tmp_config, stub_llm):
        exp = _exp("exp-001", days_old=30, initial_confidence=0.6, confidence=0.6)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)

        await cons.consolidate()
        after_first = list_all_experiences(tmp_config)[0].confidence

        await cons.consolidate()
        after_second = list_all_experiences(tmp_config)[0].confidence

        assert after_first == pytest.approx(after_second, abs=1e-9)

    async def test_decay_uses_initial_not_current(self, tmp_config, stub_llm):
        """Even if current confidence was manually lowered, decay recomputes
        from initial_confidence."""
        exp = _exp("exp-001", days_old=10, initial_confidence=0.8, confidence=0.3)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()

        updated = list_all_experiences(tmp_config)[0]
        expected = 0.8 * (0.95 ** 10)
        assert updated.confidence == pytest.approx(expected, abs=0.01)
        assert updated.confidence > 0.3


# =======================================================================
# P0-1: Taxonomy dual-factor confidence (boost / penalize)
# =======================================================================

class TestTaxonomyDualFactor:
    def test_boost_increases_adjustment_and_confidence(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax(confidence=0.7, adjustment=0.0)])

        entry = boost_taxonomy_confidence("tax-cod-gen-001", tmp_config)
        assert entry is not None
        assert entry.confidence_adjustment == pytest.approx(0.1)
        assert entry.confidence == pytest.approx(0.8)

    def test_penalize_decreases_adjustment_and_confidence(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax(confidence=0.8, adjustment=0.0)])

        entry = penalize_taxonomy_confidence("tax-cod-gen-001", tmp_config)
        assert entry is not None
        assert entry.confidence_adjustment == pytest.approx(-0.2)
        assert entry.confidence == pytest.approx(0.6)

    def test_boost_caps_at_max_adjustment(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax(confidence=0.9, adjustment=0.35)])

        entry = boost_taxonomy_confidence("tax-cod-gen-001", tmp_config)
        assert entry is not None
        assert entry.confidence_adjustment == pytest.approx(0.4)

    def test_penalize_floors_at_min_adjustment(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax(confidence=0.3, adjustment=-0.45)])

        entry = penalize_taxonomy_confidence("tax-cod-gen-001", tmp_config)
        assert entry is not None
        assert entry.confidence_adjustment == pytest.approx(-0.5)

    def test_confidence_clamps_to_zero(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax(confidence=0.1, adjustment=-0.3)])

        entry = penalize_taxonomy_confidence("tax-cod-gen-001", tmp_config)
        assert entry is not None
        assert entry.confidence >= 0.0

    def test_nonexistent_entry_returns_none(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax()])
        assert boost_taxonomy_confidence("tax-nonexistent", tmp_config) is None

    def test_adjustment_persists_across_reload(self, tmp_config):
        _save_taxonomy(tmp_config, [_tax(confidence=0.7)])

        boost_taxonomy_confidence("tax-cod-gen-001", tmp_config)
        reloaded = load_error_taxonomy(tmp_config)
        entry = reloaded.find_entry("tax-cod-gen-001")
        assert entry is not None
        assert entry.confidence_adjustment == pytest.approx(0.1)
        assert entry.confidence == pytest.approx(0.8)


# =======================================================================
# P0-2: Materializer exception isolation
# =======================================================================

@pytest.mark.asyncio
class TestMaterializerIsolation:
    async def test_one_failure_does_not_abort_batch(self, tmp_config, stub_llm):
        from meta_learning.shared.io import write_signal
        from meta_learning.shared.models import Signal, TriggerReason

        sig1 = Signal(
            signal_id="sig-20260401-001",
            timestamp=datetime.now(),
            session_id="unknown",
            trigger_reason=TriggerReason.SELF_RECOVERY,
            keywords=["test"],
            task_summary="task 1",
            step_count=3,
        )
        sig2 = Signal(
            signal_id="sig-20260401-002",
            timestamp=datetime.now(),
            session_id="unknown",
            trigger_reason=TriggerReason.SELF_RECOVERY,
            keywords=["test"],
            task_summary="task 2",
            step_count=3,
        )
        write_signal(sig1, tmp_config)
        write_signal(sig2, tmp_config)

        call_count = 0
        original = stub_llm.materialize_signal

        async def _fail_first(signal, context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM timeout on first signal")
            return await original(signal, context)

        stub_llm.materialize_signal = _fail_first

        mat = Materializer(tmp_config, stub_llm)
        results = await mat.materialize_all_pending()

        assert len(results) == 1
        assert call_count == 2


# =======================================================================
# P1-3: MERGE/SPLIT removed from enum
# =======================================================================

class TestMergeSplitRemoved:
    def test_enum_has_no_merge_or_split(self):
        members = [m.value for m in SkillUpdateAction]
        assert "merge" not in members
        assert "split" not in members
        assert set(members) == {"append", "replace", "create", "none"}


# =======================================================================
# P1-4: Skill gating enforcement
# =======================================================================

@pytest.mark.asyncio
class TestSkillGating:
    async def test_below_confidence_threshold_skipped(self, tmp_config, stub_llm):
        entry = _tax(confidence=0.5, source_count=10)
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry])

        assert len(results) == 1
        assert results[0].action == SkillUpdateAction.NONE
        assert "gated" in results[0].changes_description

    async def test_below_source_exps_threshold_skipped(self, tmp_config, stub_llm):
        entry = _tax(confidence=0.95, source_count=2)
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry])

        assert len(results) == 1
        assert results[0].action == SkillUpdateAction.NONE
        assert "gated" in results[0].changes_description

    async def test_above_both_thresholds_proceeds(self, tmp_config, stub_llm):
        entry = _tax(confidence=0.9, source_count=6)
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry])

        assert len(results) == 1
        assert results[0].action != SkillUpdateAction.NONE

    async def test_mixed_entries_partial_gating(self, tmp_config, stub_llm):
        entries = [
            _tax("tax-a", confidence=0.5, source_count=1),
            _tax("tax-b", confidence=0.9, source_count=6),
            _tax("tax-c", confidence=0.85, source_count=3),
        ]
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy(entries)

        assert results[0].action == SkillUpdateAction.NONE  # gated
        assert results[1].action != SkillUpdateAction.NONE  # passes
        assert results[2].action == SkillUpdateAction.NONE  # gated (source_exps < 5)


# =======================================================================
# P2-6: SKILL.md prevention fallback
# =======================================================================

class TestPreventionFallback:
    def test_fallback_to_fix_sop(self):
        entry = _tax(prevention="", fix_sop="Run tests first", trigger="on deploy")
        assert _entry_rule_text(entry) == "Run tests first"

    def test_fallback_to_trigger(self):
        entry = _tax(prevention="", fix_sop="", trigger="When database is locked")
        assert _entry_rule_text(entry) == "When database is locked"

    def test_prevention_preferred(self):
        entry = _tax(prevention="Check types", fix_sop="Fix generics", trigger="On TS error")
        assert _entry_rule_text(entry) == "Check types"

    def test_unknown_prevention_excluded(self):
        entry = _tax(
            prevention="Avoid conditions leading to: unknown",
            fix_sop="",
            trigger="unknown trigger",
        )
        assert _entry_rule_text(entry) == ""

    def test_empty_entries_excluded_from_skill_md(self):
        entries = [
            _tax("a", prevention="Avoid X", confidence=0.9,
                 name="TypeScript Error", keywords=["typescript", "generic"]),
            _tax("b", prevention="", fix_sop="", trigger="", confidence=0.95,
                 name="Empty Entry", keywords=["empty", "nothing"]),
            _tax("c", prevention="", fix_sop="Do Y", confidence=0.85,
                 name="Database Migration", keywords=["database", "migration"]),
        ]
        md = _render_skill_md(entries, max_rules=10)
        assert "Avoid X" in md
        assert "Do Y" in md
        lines = [l for l in md.split("\n") if l.startswith("- ")]
        for line in lines:
            assert line.strip() != "-"

    def test_fallback_renders_in_skill_md(self):
        entries = [
            _tax("a", prevention="", fix_sop="Always run lint", confidence=0.9,
                 name="Lint Rule", keywords=["lint_check"]),
        ]
        md = _render_skill_md(entries, max_rules=10)
        assert "Always run lint" in md


@pytest.mark.asyncio
class TestTaxonomyQualityGuards:
    async def test_stub_llm_uses_meta_insight_not_unknown_signature(self):
        exp = Experience(
            id="exp-001",
            task_type=TaskType.UNCLASSIFIED,
            created_at=datetime.now(),
            source_signal="sig-001",
            scene="用户要求后台学习",
            failure_signature=None,
            root_cause="用户纠正了学习流程",
            resolution="后台学习应该通过 spawn 执行",
            meta_insight="meta-learning 后台学习应该通过 spawn 执行，避免阻塞主会话",
        )

        result = await StubLLM().extract_taxonomy([exp])

        assert result.prevention == "meta-learning 后台学习应该通过 spawn 执行，避免阻塞主会话"
        assert "unknown" not in result.name.lower()

    async def test_builder_skips_low_quality_unknown_extraction(self, tmp_config):
        class BadLLM(StubLLM):
            async def extract_taxonomy(self, experiences):
                return TaxonomyExtraction(
                    name="Unknown",
                    trigger="unknown trigger",
                    fix_sop="",
                    prevention="Avoid conditions leading to: unknown",
                    keywords=[],
                )

        exp = _exp("exp-001")
        write_experience(exp, tmp_config)
        cluster = ExperienceCluster(
            cluster_id="cluster-001",
            task_type=TaskType.UNCLASSIFIED,
            failure_signature_pattern="unknown",
            experience_ids=[exp.id],
        )

        entries = await TaxonomyBuilder(tmp_config, BadLLM()).build_from_clusters([cluster])

        assert entries == []
        taxonomy = load_error_taxonomy(tmp_config)
        assert taxonomy.all_entries() == []


# =======================================================================
# P2-7: Category rules dedup
# =======================================================================

class TestCategoryDedup:
    def test_similar_entries_deduped_in_category(self):
        entries = [
            _tax("a", name="Workspace Setup", prevention="Use ~/workspace",
                 keywords=["workspace", "project", "mkdir", "create"], confidence=0.9),
            _tax("b", name="Workspace Init", prevention="Init ~/workspace",
                 keywords=["workspace", "project", "mkdir", "create", "init"], confidence=0.85),
            _tax("c", name="Backup Config", prevention="Backup .env",
                 keywords=["backup", "env", "config"], confidence=0.7),
        ]
        md = _render_category_md("general", entries)
        assert "Workspace Setup" in md
        assert "Backup Config" in md
        assert "Workspace Init" not in md

    def test_unique_entries_all_kept(self):
        entries = [
            _tax("a", name="Topic Alpha", keywords=["alpha", "first"], confidence=0.9),
            _tax("b", name="Topic Beta", keywords=["beta", "second"], confidence=0.8),
            _tax("c", name="Topic Gamma", keywords=["gamma", "third"], confidence=0.7),
        ]
        md = _render_category_md("general", entries)
        assert "Topic Alpha" in md
        assert "Topic Beta" in md
        assert "Topic Gamma" in md

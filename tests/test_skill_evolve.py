from datetime import date
from pathlib import Path

import pytest

from meta_learning.layer2.skill_evolve import SkillEvolver
from meta_learning.shared.models import (
    SkillUpdateAction,
    TaxonomyEntry,
)


def _make_taxonomy_entry(
    entry_id: str = "tax-cod-001",
    confidence: float = 0.9,
    source_count: int = 6,
) -> TaxonomyEntry:
    return TaxonomyEntry(
        id=entry_id,
        name="Test Pattern",
        trigger="When X happens",
        fix_sop="1. Do A\n2. Do B",
        prevention="Avoid X",
        confidence=confidence,
        source_exps=[f"exp-{i:03d}" for i in range(source_count)],
        keywords=["test", "pattern"],
        created_at=date.today(),
        last_verified=date.today(),
    )


@pytest.mark.asyncio
class TestSkillEvolver:
    async def test_empty_entries(self, tmp_config, stub_llm):
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([])
        assert results == []

    async def test_creates_new_skill(self, tmp_config, stub_llm):
        entry = _make_taxonomy_entry(confidence=0.9, source_count=6)
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry])
        assert len(results) == 1
        assert results[0].action == SkillUpdateAction.CREATE

        skills_dir = Path(tmp_config.skills_path)
        assert any(skills_dir.rglob("SKILL.md"))

    async def test_skips_low_confidence(self, tmp_config, stub_llm):
        entry = _make_taxonomy_entry(confidence=0.5, source_count=2)
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry])
        assert results[0].action == SkillUpdateAction.NONE

    async def test_appends_to_existing_skill(self, tmp_config, stub_llm):
        skills_dir = Path(tmp_config.skills_path)
        skill_dir = skills_dir / "test-pattern"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Existing Skill\nSome content")

        entry = _make_taxonomy_entry()
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry])
        assert results[0].action == SkillUpdateAction.APPEND

        content = (skill_dir / "SKILL.md").read_text()
        assert "Existing Skill" in content
        assert "Update from" in content

    async def test_different_taxonomy_entries_create_separate_skills(
        self, tmp_config, stub_llm
    ):
        entry_a = TaxonomyEntry(
            id="tax-cod-001",
            name="coding: ts2345 (5 experiences)",
            trigger="TS2345 type error",
            fix_sop="Fix generics",
            prevention="Annotate types",
            confidence=0.9,
            source_exps=[f"exp-{i:03d}" for i in range(6)],
            keywords=["ts2345", "generic"],
            created_at=date.today(),
            last_verified=date.today(),
        )
        entry_b = TaxonomyEntry(
            id="tax-con-001",
            name="configuration: database_url (5 experiences)",
            trigger="DATABASE_URL not set",
            fix_sop="Set env vars",
            prevention="Use .env.example",
            confidence=0.9,
            source_exps=[f"exp-{i:03d}" for i in range(10, 16)],
            keywords=["database_url", "configuration"],
            created_at=date.today(),
            last_verified=date.today(),
        )
        evolver = SkillEvolver(tmp_config, stub_llm)
        results = await evolver.evolve_from_taxonomy([entry_a, entry_b])

        creates = [r for r in results if r.action == SkillUpdateAction.CREATE]
        assert len(creates) == 2

        skills_dir = Path(tmp_config.skills_path)
        skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
        assert len(skill_dirs) == 2

    async def test_find_matching_skill_uses_taxonomy_id(self, tmp_config, stub_llm):
        skills_dir = Path(tmp_config.skills_path)
        skill_dir = skills_dir / "some-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "# Some Skill\n\n## Source\n- Taxonomy ID: tax-cod-001\n"
        )

        entry_same_id = TaxonomyEntry(
            id="tax-cod-001",
            name="totally-different-name",
            trigger="trigger",
            fix_sop="sop",
            prevention="prevention",
            confidence=0.5,
            source_exps=["exp-001"],
            keywords=["unrelated"],
            created_at=date.today(),
            last_verified=date.today(),
        )
        entry_diff_id = TaxonomyEntry(
            id="tax-con-001",
            name="totally-different-name",
            trigger="trigger",
            fix_sop="sop",
            prevention="prevention",
            confidence=0.5,
            source_exps=["exp-002"],
            keywords=["unrelated"],
            created_at=date.today(),
            last_verified=date.today(),
        )

        evolver = SkillEvolver(tmp_config, stub_llm)
        match_same = evolver._find_matching_skill(entry_same_id)
        match_diff = evolver._find_matching_skill(entry_diff_id)

        assert match_same is not None
        assert match_diff is None

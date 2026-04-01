"""Tests for SKILL.md rendering with diversity-aware Top-N selection."""

from __future__ import annotations

from datetime import date

import pytest

from meta_learning.shared.models import TaxonomyEntry
from meta_learning.sync_nobot import (
    _entry_topic_tokens,
    _render_skill_md,
    _select_diverse_top_n,
)


def _make_entry(
    entry_id: str,
    name: str,
    prevention: str,
    keywords: list[str],
    confidence: float = 0.65,
) -> TaxonomyEntry:
    return TaxonomyEntry(
        id=entry_id,
        name=name,
        trigger="test trigger",
        fix_sop="1. test",
        prevention=prevention,
        confidence=confidence,
        source_exps=["exp-001"],
        keywords=keywords,
        created_at=date(2026, 4, 1),
        last_verified=date(2026, 4, 1),
    )


class TestEntryTopicTokens:

    def test_combines_name_and_keywords(self):
        entry = _make_entry(
            "t1", "Backup Config Files",
            "Always backup", ["backup", "config", "env"],
        )
        tokens = _entry_topic_tokens(entry)
        assert "backup" in tokens
        assert "config" in tokens
        assert "env" in tokens
        assert "files" in tokens

    def test_splits_compound_keywords(self):
        entry = _make_entry(
            "t1", "Verify Dir",
            "verify", ["directory_verification", "path_validation"],
        )
        tokens = _entry_topic_tokens(entry)
        assert "directory" in tokens
        assert "verification" in tokens
        assert "path" in tokens
        assert "validation" in tokens


class TestSelectDiverseTopN:

    def test_selects_unique_entries(self):
        entries = [
            _make_entry("a", "Workspace", "use workspace", ["workspace", "project"], 0.9),
            _make_entry("b", "Indentation", "4 spaces", ["python", "indent", "spaces"], 0.85),
            _make_entry("c", "Backup", "backup env", ["backup", "env", "config"], 0.7),
        ]
        result = _select_diverse_top_n(entries, 10)
        assert len(result) == 3
        assert [e.id for e in result] == ["a", "b", "c"]

    def test_skips_near_duplicate_entries(self):
        entries = [
            _make_entry("a", "Create Projects Workspace", "use ~/workspace",
                        ["workspace", "project", "mkdir", "create"], 0.9),
            _make_entry("b", "Create Projects Under Workspace", "create under ~/workspace",
                        ["workspace", "project", "mkdir", "create", "python"], 0.85),
            _make_entry("c", "Backup Env Files", "backup .env",
                        ["backup", "env", "config", "file"], 0.7),
        ]
        result = _select_diverse_top_n(entries, 10)
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "c"

    def test_respects_max_rules(self):
        entries = [
            _make_entry(f"e{i}", f"Topic {i}", f"rule {i}", [f"kw{i}"], 0.9 - i * 0.1)
            for i in range(20)
        ]
        result = _select_diverse_top_n(entries, 5)
        assert len(result) == 5

    def test_low_confidence_entry_survives_if_unique(self):
        entries = [
            _make_entry("ws1", "Workspace Rule", "use workspace",
                        ["workspace", "project", "create"], 0.95),
            _make_entry("ws2", "Workspace Alt", "workspace for projects",
                        ["workspace", "project", "create", "mkdir"], 0.90),
            _make_entry("ws3", "Workspace Python", "python projects workspace",
                        ["workspace", "project", "python", "create"], 0.85),
            _make_entry("backup", "Backup Config", "always backup .env",
                        ["backup", "env", "config"], 0.60),
        ]
        result = _select_diverse_top_n(entries, 3)
        ids = [e.id for e in result]
        assert "ws1" in ids
        assert "backup" in ids
        assert len(result) <= 3


class TestRenderSkillMdDiversity:

    def test_render_includes_diverse_rules(self):
        entries = [
            _make_entry("ws1", "Workspace", "Create under ~/workspace",
                        ["workspace", "project"], 0.9),
            _make_entry("ws2", "Workspace Alt", "Projects in ~/workspace",
                        ["workspace", "project", "python"], 0.85),
            _make_entry("indent", "Indentation", "Use 4 spaces",
                        ["python", "indent", "spaces", "pep8"], 0.8),
            _make_entry("backup", "Backup Config", "Backup .env before editing",
                        ["backup", "env", "config"], 0.6),
        ]
        md = _render_skill_md(entries, max_rules=10)
        assert "Backup .env before editing" in md
        assert "Create under ~/workspace" in md
        assert "Use 4 spaces" in md

    def test_render_deduplicates_similar_rules(self):
        entries = [
            _make_entry("ws1", "Workspace", "Create under ~/workspace",
                        ["workspace", "project"], 0.9),
            _make_entry("ws2", "Workspace Alt", "Projects in ~/workspace",
                        ["workspace", "project", "python"], 0.85),
        ]
        md = _render_skill_md(entries, max_rules=10)
        assert "Create under ~/workspace" in md
        assert "Projects in ~/workspace" not in md

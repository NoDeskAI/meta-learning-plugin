from datetime import date

from meta_learning.layer1.quick_think import QuickThinkIndex
from meta_learning.shared.models import (
    ErrorTaxonomy,
    TaskContext,
    TaxonomyEntry,
)


def _make_taxonomy_with_keywords(keywords: list[str]) -> ErrorTaxonomy:
    entry = TaxonomyEntry(
        id="tax-test-001",
        name="Test Entry",
        trigger="test",
        fix_sop="fix it",
        prevention="prevent it",
        confidence=0.9,
        source_exps=["exp-001"],
        keywords=keywords,
        created_at=date.today(),
        last_verified=date.today(),
    )
    tax = ErrorTaxonomy()
    tax.add_entry("coding", "test", entry)
    return tax


class TestQuickThinkKeywordMatch:
    def test_no_hit_on_empty_taxonomy(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        ctx = TaskContext(task_description="simple rename variable")
        result = idx.evaluate(ctx)
        assert result.hit is False
        assert result.risk_level == "none"

    def test_keyword_hit(self, tmp_config):
        tax = _make_taxonomy_with_keywords(["TS2345", "generic"])
        idx = QuickThinkIndex(tax, tmp_config)
        ctx = TaskContext(
            task_description="fix TS2345 generic error",
            errors_encountered=["TS2345: type mismatch"],
        )
        result = idx.evaluate(ctx)
        assert result.hit is True
        assert "keyword_taxonomy_hit" in result.matched_signals
        assert "tax-test-001" in result.matched_taxonomy_entries


class TestQuickThinkIrreversibleOps:
    def test_detects_rm_rf(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        ctx = TaskContext(task_description="clean up with rm -rf /tmp/old")
        result = idx.evaluate(ctx)
        assert result.hit is True
        assert "irreversible_operation" in result.matched_signals
        assert result.risk_level == "high"

    def test_detects_force_push(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        ctx = TaskContext(
            task_description="push changes",
            tools_used=["git push --force"],
        )
        result = idx.evaluate(ctx)
        assert result.hit is True
        assert "irreversible_operation" in result.matched_signals

    def test_safe_operations_pass(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        ctx = TaskContext(task_description="read the file and summarize")
        result = idx.evaluate(ctx)
        assert result.hit is False


class TestQuickThinkRecentFailures:
    def test_matches_registered_failure(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        idx.register_failure_signature("connection timeout")
        ctx = TaskContext(
            task_description="deploy to server",
            errors_encountered=["connection timeout on port 443"],
        )
        result = idx.evaluate(ctx)
        assert result.hit is True
        assert "recent_failure_pattern" in result.matched_signals


class TestQuickThinkNewTools:
    def test_detects_unknown_tool(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        idx.register_known_tool("read_file")
        idx.register_known_tool("edit_file")
        ctx = TaskContext(
            task_description="deploy app",
            new_tools=["docker_exec"],
        )
        result = idx.evaluate(ctx)
        assert result.hit is True
        assert "new_tool_usage" in result.matched_signals

    def test_known_tools_no_hit(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        idx.register_known_tool("read_file")
        ctx = TaskContext(
            task_description="read the file",
            new_tools=[],
        )
        result = idx.evaluate(ctx)
        assert result.hit is False


class TestQuickThinkRiskLevels:
    def test_high_for_irreversible(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        ctx = TaskContext(task_description="rm -rf everything")
        result = idx.evaluate(ctx)
        assert result.risk_level == "high"

    def test_medium_for_multiple_signals(self, tmp_config):
        tax = _make_taxonomy_with_keywords(["deploy"])
        idx = QuickThinkIndex(tax, tmp_config)
        idx.register_failure_signature("deploy")
        ctx = TaskContext(
            task_description="deploy application",
            errors_encountered=["deploy failed"],
        )
        result = idx.evaluate(ctx)
        assert result.risk_level == "medium"

    def test_update_taxonomy(self, tmp_config):
        idx = QuickThinkIndex(ErrorTaxonomy(), tmp_config)
        ctx = TaskContext(task_description="fix TS2345 error")
        assert idx.evaluate(ctx).hit is False

        new_tax = _make_taxonomy_with_keywords(["TS2345"])
        idx.update_taxonomy(new_tax)
        assert idx.evaluate(ctx).hit is True

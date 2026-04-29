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

    def test_local_fuzzy_match_without_exact_keyword(self, tmp_config):
        entry = TaxonomyEntry(
            id="tax-clarify-001",
            name="Ask User When Requirements Are Unclear",
            trigger="ambiguous requirements or missing decision context",
            fix_sop="pause and ask the user for clarification before proceeding",
            prevention="never guess when the requirement is unclear",
            confidence=0.9,
            source_exps=["exp-001"],
            keywords=["clarification", "ambiguous requirements"],
            created_at=date.today(),
            last_verified=date.today(),
        )
        tax = ErrorTaxonomy()
        tax.add_entry("configuration", "requirements", entry)
        idx = QuickThinkIndex(tax, tmp_config)

        ctx = TaskContext(
            task_description="the user request is unclear and missing context",
            tools_used=["ask_user"],
        )

        result = idx.evaluate(ctx)

        assert result.hit is True
        assert "keyword_taxonomy_hit" in result.matched_signals
        assert "tax-clarify-001" in result.matched_taxonomy_entries

    def test_local_fuzzy_match_for_chinese_rule(self, tmp_config):
        entry = TaxonomyEntry(
            id="tax-cn-clarify-001",
            name="不确定时先询问用户",
            trigger="需求不清楚、存在歧义或缺少必要信息时",
            fix_sop="暂停执行，使用 ask_user 向用户确认后再继续",
            prevention="遇到不确定点必须先问用户，不要猜测",
            confidence=0.9,
            source_exps=["exp-001"],
            keywords=["确认需求", "不确定"],
            created_at=date.today(),
            last_verified=date.today(),
        )
        tax = ErrorTaxonomy()
        tax.add_entry("configuration", "requirements", entry)
        idx = QuickThinkIndex(tax, tmp_config)

        ctx = TaskContext(
            task_description="用户的要求有些模糊，缺少必要上下文",
            tools_used=["ask_user"],
        )

        result = idx.evaluate(ctx)

        assert result.hit is True
        assert "tax-cn-clarify-001" in result.matched_taxonomy_entries


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

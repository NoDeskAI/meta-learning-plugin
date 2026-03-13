from __future__ import annotations

from datetime import date, datetime

import pytest

from meta_learning.shared.llm import (
    StubLLM,
    _clean_token,
    _extract_common_keywords,
    _extract_representative_token,
    _has_keyword_overlap,
    _shared_error_code_prefix,
)
from meta_learning.shared.models import (
    Experience,
    TaskType,
)


def _make_exp(
    failure_signature: str | None,
    task_type: TaskType = TaskType.CODING,
    exp_id: str = "exp-001",
) -> Experience:
    return Experience(
        id=exp_id,
        task_type=task_type,
        created_at=datetime.now(),
        source_signal="sig-test-001",
        confidence=0.6,
        scene="test scene",
        failure_signature=failure_signature,
        root_cause="test root cause",
        resolution="test resolution",
        meta_insight="test insight",
    )


class TestCleanToken:
    def test_strips_punctuation(self) -> None:
        assert _clean_token("(reading") == "reading"
        assert _clean_token("'string'") == "string"
        assert _clean_token("error:") == "error"
        assert _clean_token("[body]") == "body"

    def test_preserves_clean_words(self) -> None:
        assert _clean_token("TS2345") == "TS2345"
        assert _clean_token("DATABASE_URL") == "DATABASE_URL"

    def test_empty_after_strip(self) -> None:
        assert _clean_token("()") == ""
        assert _clean_token(".:;") == ""


class TestExtractCommonKeywords:
    def test_filters_stopwords(self) -> None:
        exps = [
            _make_exp(
                "configuration error: DATABASE_URL environment variable is not set",
                exp_id="exp-001",
            ),
            _make_exp(
                "configuration error: REDIS_URL environment variable is not set",
                exp_id="exp-002",
            ),
            _make_exp(
                "configuration error: API_SECRET environment variable is not set",
                exp_id="exp-003",
            ),
        ]
        keywords = _extract_common_keywords(exps)
        assert "not" not in keywords
        assert "error" not in keywords
        assert "variable" not in keywords
        assert "configuration" in keywords

    def test_strips_punctuation_from_keywords(self) -> None:
        exps = [
            _make_exp(
                "TypeError: Cannot read properties of undefined (reading 'map')",
                exp_id="exp-001",
            ),
            _make_exp(
                "TypeError: Cannot read properties of undefined (reading 'data')",
                exp_id="exp-002",
            ),
        ]
        keywords = _extract_common_keywords(exps)
        assert "(reading" not in keywords
        assert "error:" not in keywords

    def test_prefers_error_codes(self) -> None:
        exps = [
            _make_exp(
                "TS2345: Argument of type 'string' for generic param", exp_id="exp-001"
            ),
            _make_exp(
                "TS2345: Argument of type 'number' for generic param", exp_id="exp-002"
            ),
        ]
        keywords = _extract_common_keywords(exps)
        assert len(keywords) > 0
        assert keywords[0] == "ts2345"

    def test_empty_experiences(self) -> None:
        assert _extract_common_keywords([]) == []

    def test_no_failure_signatures(self) -> None:
        exps = [_make_exp(None), _make_exp(None)]
        assert _extract_common_keywords(exps) == []


class TestHasKeywordOverlap:
    def test_strips_punctuation_before_comparison(self) -> None:
        assert _has_keyword_overlap(
            "'string' assignable parameter",
            "parameter string mismatch",
        )

    def test_shared_error_code_prefix_ts(self) -> None:
        assert _has_keyword_overlap(
            "TS2345: Argument of type 'string | undefined' is not assignable",
            "TS2322: Type '{ onClick: () => void; }' is not assignable",
        )

    def test_different_domains_no_match(self) -> None:
        assert not _has_keyword_overlap(
            "configuration error: DATABASE_URL environment variable is not set",
            "TypeError: Cannot read properties of undefined (reading 'map')",
        )

    def test_empty_after_stopword_removal(self) -> None:
        assert not _has_keyword_overlap("is not a", "to be or")

    def test_high_overlap_direct_match(self) -> None:
        assert _has_keyword_overlap(
            "cannot find module 'react'",
            "cannot find module 'vue'",
        )


class TestSharedErrorCodePrefix:
    def test_ts_prefix_match(self) -> None:
        assert _shared_error_code_prefix({"ts2345"}, {"ts2322"})

    def test_different_prefix_no_match(self) -> None:
        assert not _shared_error_code_prefix({"ts2345"}, {"e0001"})

    def test_no_codes_no_match(self) -> None:
        assert not _shared_error_code_prefix({"hello", "world"}, {"foo", "bar"})


class TestExtractRepresentativeToken:
    def test_finds_error_code(self) -> None:
        result = _extract_representative_token(
            ["TS2345: Argument of type 'string' is not assignable"]
        )
        assert result == "ts2345"

    def test_finds_upper_ident(self) -> None:
        result = _extract_representative_token(
            ["configuration error: DATABASE_URL environment variable is not set"]
        )
        assert result == "database_url"

    def test_fallback_to_meaningful_word(self) -> None:
        result = _extract_representative_token(
            ["something went completely haywire in production"]
        )
        assert result == "something"

    def test_empty_signatures(self) -> None:
        assert _extract_representative_token([]) == "general"


class TestStubLLMExtractTaxonomy:
    @pytest.mark.asyncio
    async def test_includes_task_type_in_name(self) -> None:
        llm = StubLLM()
        exps = [
            _make_exp(
                "TS2345: type error", task_type=TaskType.CODING, exp_id=f"exp-{i:03d}"
            )
            for i in range(3)
        ]
        result = await llm.extract_taxonomy(exps)
        assert "coding" in result.name.lower()

    @pytest.mark.asyncio
    async def test_different_types_produce_different_names(self) -> None:
        llm = StubLLM()
        coding_exps = [
            _make_exp(
                "TS2345: type error", task_type=TaskType.CODING, exp_id=f"exp-c{i}"
            )
            for i in range(3)
        ]
        config_exps = [
            _make_exp(
                "configuration error: DATABASE_URL not set",
                task_type=TaskType.CONFIGURATION,
                exp_id=f"exp-d{i}",
            )
            for i in range(3)
        ]
        coding_result = await llm.extract_taxonomy(coding_exps)
        config_result = await llm.extract_taxonomy(config_exps)
        assert coding_result.name != config_result.name

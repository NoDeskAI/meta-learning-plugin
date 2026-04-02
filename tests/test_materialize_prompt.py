"""Tests for materialize_signal prompt differentiation."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pytest

from meta_learning.shared.llm_openai import OpenAILLM
from meta_learning.shared.models import (
    MetaLearningConfig,
    Signal,
    TriggerReason,
)


@pytest.fixture
def openai_llm(tmp_config: MetaLearningConfig) -> OpenAILLM:
    return OpenAILLM(tmp_config)


def _make_signal(trigger: TriggerReason, user_feedback=None):
    return Signal(
        signal_id="sig-test-001",
        timestamp=datetime.now(),
        session_id="session-test",
        memory_date=date(2026, 4, 1),
        trigger_reason=trigger,
        keywords=["env", "backup"],
        task_summary="Modify .env configuration file",
        error_snapshot=None,
        resolution_snapshot="Modified .env directly",
        user_feedback=user_feedback,
        step_count=3,
    )


class TestMaterializePromptDifferentiation:

    @pytest.mark.asyncio
    async def test_user_correction_appends_extraction_mode(self, openai_llm):
        signal = _make_signal(
            TriggerReason.USER_CORRECTION,
            user_feedback="should backup first",
        )
        captured = {}

        async def mock_chat_json(system, user):
            captured["system"] = system
            captured["user"] = user
            return {
                "scene": "modify env", "failure_signature": None,
                "root_cause": "no backup", "resolution": "backup before modify",
                "meta_insight": "always backup .env", "task_type": "configuration",
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.materialize_signal(signal, "session context here")

        assert "USER CORRECTION MODE" in captured["system"]
        assert "INSTRUCTION EXTRACTION" in captured["system"]

    @pytest.mark.asyncio
    async def test_user_correction_reorders_user_message(self, openai_llm):
        signal = _make_signal(
            TriggerReason.USER_CORRECTION,
            user_feedback="should backup first",
        )
        captured = {}

        async def mock_chat_json(system, user):
            captured["user"] = user
            return {
                "scene": "m", "root_cause": "r", "resolution": "r",
                "meta_insight": "m", "task_type": "configuration",
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.materialize_signal(signal, "ctx")

        user_msg = captured["user"]
        feedback_pos = user_msg.find("should backup first")
        signal_id_pos = user_msg.find("Signal ID:")
        assert feedback_pos < signal_id_pos

    @pytest.mark.asyncio
    async def test_non_correction_uses_standard_prompt(self, openai_llm):
        signal = _make_signal(TriggerReason.SELF_RECOVERY, user_feedback=None)
        captured = {}

        async def mock_chat_json(system, user):
            captured["system"] = system
            captured["user"] = user
            return {
                "scene": "fix", "root_cause": "bug", "resolution": "fix",
                "meta_insight": "test", "task_type": "coding",
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.materialize_signal(signal, "ctx")

        assert "USER CORRECTION MODE" not in captured["system"]
        assert "Trigger:" in captured["user"]

    @pytest.mark.asyncio
    async def test_user_correction_without_feedback_no_extraction_mode(self, openai_llm):
        signal = _make_signal(TriggerReason.USER_CORRECTION, user_feedback=None)
        captured = {}

        async def mock_chat_json(system, user):
            captured["system"] = system
            return {
                "scene": "t", "root_cause": "c", "resolution": "f",
                "meta_insight": "i", "task_type": "coding",
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.materialize_signal(signal, "ctx")

        assert "USER CORRECTION MODE" not in captured["system"]


class TestExtractTaxonomySingleExperience:

    @pytest.mark.asyncio
    async def test_single_experience_appends_preserve_specifics(self, openai_llm):
        from meta_learning.shared.models import Experience, TaskType
        single_exp = Experience(
            id="exp-001", task_type=TaskType.CONFIGURATION,
            created_at=datetime.now(), source_signal="sig-001", confidence=0.6,
            scene="Modify .env file", failure_signature=None,
            root_cause="Did not backup", resolution="Always backup .env",
            meta_insight="Backup config files",
        )
        captured = {}

        async def mock_chat_json(system, user):
            captured["system"] = system
            return {
                "name": "Backup", "trigger": "editing config",
                "fix_sop": "1. backup", "prevention": "Always backup",
                "keywords": ["backup"],
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.extract_taxonomy([single_exp])

        assert "SINGLE EXPERIENCE" in captured["system"]
        assert "PRESERVE SPECIFICS" in captured["system"]

    @pytest.mark.asyncio
    async def test_multiple_experiences_no_preserve_specifics(self, openai_llm):
        from meta_learning.shared.models import Experience, TaskType
        exps = [
            Experience(
                id=f"exp-{i:03d}", task_type=TaskType.CODING,
                created_at=datetime.now(), source_signal=f"sig-{i:03d}",
                confidence=0.6, scene=f"Task #{i}", failure_signature="type error",
                root_cause="type mismatch", resolution="add annotation",
                meta_insight="annotate types",
            )
            for i in range(3)
        ]
        captured = {}

        async def mock_chat_json(system, user):
            captured["system"] = system
            return {
                "name": "Type", "trigger": "type errors",
                "fix_sop": "annotate", "prevention": "always annotate",
                "keywords": ["type"],
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.extract_taxonomy(exps)

        assert "SINGLE EXPERIENCE" not in captured["system"]

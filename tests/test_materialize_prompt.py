"""Tests for materialize_signal prompt differentiation."""

from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import patch

import pytest

from meta_learning.shared.llm_openai import OpenAILLM, _load_current_deskclaw_llm_config
from meta_learning.shared.models import (
    MetaLearningConfig,
    Signal,
    TriggerReason,
)


@pytest.fixture
def openai_llm(
    tmp_config: MetaLearningConfig,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> OpenAILLM:
    settings_path = tmp_path / "deskclaw-settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "settings.configMode": "default",
                "settings.gatewayConfig": {
                    "apiUrl": "https://gateway.example.test/v1",
                    "apiKey": "test-key",
                    "model": "test-model",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DESKCLAW_SETTINGS_PATH", str(settings_path))
    monkeypatch.delenv("META_LEARNING_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("META_LEARNING_LLM_API_KEY", raising=False)
    monkeypatch.delenv("META_LEARNING_LLM_MODEL", raising=False)
    monkeypatch.delenv("META_LEARNING_LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("META_LEARNING_LLM_MAX_TOKENS", raising=False)
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


class TestDeskClawSettingsConfig:

    def test_default_gateway_config_used(
        self,
        tmp_config: MetaLearningConfig,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        settings_path = tmp_path / "deskclaw-settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "settings.configMode": "default",
                    "settings.gatewayConfig": {
                        "apiUrl": "https://gateway.example.test/v1/",
                        "apiKey": "gateway-key",
                        "model": "gateway-model",
                        "temperature": 0.2,
                        "maxTokens": 1234,
                    },
                    "env.version": "2.2.7",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("DESKCLAW_SETTINGS_PATH", str(settings_path))
        monkeypatch.delenv("META_LEARNING_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("META_LEARNING_LLM_API_KEY", raising=False)
        monkeypatch.delenv("META_LEARNING_LLM_MODEL", raising=False)
        monkeypatch.delenv("META_LEARNING_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("META_LEARNING_LLM_MAX_TOKENS", raising=False)

        loaded = _load_current_deskclaw_llm_config()
        llm = OpenAILLM(tmp_config)

        assert loaded["base_url"] == "https://gateway.example.test/v1"
        assert loaded["api_key"] == "gateway-key"
        assert llm._base_url == "https://gateway.example.test/v1"
        assert llm._api_key == "gateway-key"
        assert llm._model == "gateway-model"
        assert llm._temperature == 0.2
        assert llm._max_tokens == 1234
        assert llm._deskclaw_app_version == "2.2.7"

    def test_custom_config_selected_in_custom_mode(
        self,
        tmp_config: MetaLearningConfig,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        settings_path = tmp_path / "deskclaw-settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "settings.configMode": "custom",
                    "settings.gatewayConfig": {
                        "apiUrl": "https://gateway.example.test/v1",
                        "apiKey": "gateway-key",
                        "model": "gateway-model",
                    },
                    "settings.customConfig": {
                        "apiUrl": "https://custom.example.test/v1",
                        "apiKey": "custom-key",
                        "model": "custom-model",
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("DESKCLAW_SETTINGS_PATH", str(settings_path))
        monkeypatch.delenv("META_LEARNING_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("META_LEARNING_LLM_API_KEY", raising=False)
        monkeypatch.delenv("META_LEARNING_LLM_MODEL", raising=False)

        llm = OpenAILLM(tmp_config)

        assert llm._base_url == "https://custom.example.test/v1"
        assert llm._api_key == "custom-key"
        assert llm._model == "custom-model"

    def test_environment_overrides_deskclaw_settings(
        self,
        tmp_config: MetaLearningConfig,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        settings_path = tmp_path / "deskclaw-settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "settings.configMode": "default",
                    "settings.apiUrl": "https://settings.example.test/v1",
                    "settings.apiKey": "settings-key",
                    "settings.model": "settings-model",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("DESKCLAW_SETTINGS_PATH", str(settings_path))
        monkeypatch.setenv("META_LEARNING_LLM_BASE_URL", "https://env.example.test/v1")
        monkeypatch.setenv("META_LEARNING_LLM_API_KEY", "env-key")
        monkeypatch.setenv("META_LEARNING_LLM_MODEL", "env-model")
        monkeypatch.setenv("META_LEARNING_LLM_TEMPERATURE", "0.9")
        monkeypatch.setenv("META_LEARNING_LLM_MAX_TOKENS", "99")

        llm = OpenAILLM(tmp_config)

        assert llm._base_url == "https://env.example.test/v1"
        assert llm._api_key == "env-key"
        assert llm._model == "env-model"
        assert llm._temperature == 0.9
        assert llm._max_tokens == 99

    @pytest.mark.asyncio
    async def test_chat_sends_deskclaw_version_header(
        self,
        tmp_config: MetaLearningConfig,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        settings_path = tmp_path / "deskclaw-settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "settings.configMode": "default",
                    "settings.gatewayConfig": {
                        "apiUrl": "https://gateway.example.test/v1",
                        "apiKey": "gateway-key",
                        "model": "gateway-model",
                    },
                    "env.version": "2.2.7",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("DESKCLAW_SETTINGS_PATH", str(settings_path))
        monkeypatch.delenv("DESKCLAW_APP_VERSION", raising=False)
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, *, headers, json):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return _FakeResponse()

        llm = OpenAILLM(tmp_config)
        with patch("meta_learning.shared.llm_openai.httpx.AsyncClient", _FakeClient):
            result = await llm._chat("system", "user")

        assert result == "ok"
        assert captured["headers"]["X-DeskClaw-Version"] == "2.2.7"
        assert captured["headers"]["Authorization"] == "Bearer gateway-key"


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
    async def test_user_correction_uses_source_language(self, openai_llm):
        signal = _make_signal(
            TriggerReason.USER_CORRECTION,
            user_feedback="以后遇到不确定的需求时，必须先问我",
        )
        captured = {}

        async def mock_chat_json(system, user):
            captured["system"] = system
            captured["user"] = user
            return {
                "scene": "记录中文规则", "failure_signature": None,
                "root_cause": "未确认需求",
                "resolution": "不确定时先询问用户",
                "meta_insight": "遇到不确定需求必须先问用户",
                "task_type": "configuration",
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.materialize_signal(signal, "用户使用中文反馈")

        assert "TARGET OUTPUT LANGUAGE: Simplified Chinese" in captured["system"]

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
            captured["user"] = user
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
            captured["user"] = user
            return {
                "name": "Backup", "trigger": "editing config",
                "fix_sop": "1. backup", "prevention": "Always backup",
                "keywords": ["backup"],
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            await openai_llm.extract_taxonomy([single_exp])

        assert "SINGLE EXPERIENCE" in captured["system"]
        assert "PRESERVE SPECIFICS" in captured["system"]
        assert "Meta insight: Backup config files" in captured["user"]

    @pytest.mark.asyncio
    async def test_extract_taxonomy_fallback_uses_meta_insight(self, openai_llm):
        from meta_learning.shared.models import Experience, TaskType

        exp = Experience(
            id="exp-001",
            task_type=TaskType.UNCLASSIFIED,
            created_at=datetime.now(),
            source_signal="sig-001",
            confidence=0.6,
            scene="用户要求后台学习",
            failure_signature=None,
            root_cause="用户纠正了学习流程",
            resolution="后台学习应该通过 spawn 执行",
            meta_insight="meta-learning 后台学习应该通过 spawn 执行，避免阻塞主会话",
        )

        async def mock_chat_json(_system, _user):
            raise RuntimeError("upstream unavailable")

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            result = await openai_llm.extract_taxonomy([exp])

        assert result.prevention == "meta-learning 后台学习应该通过 spawn 执行，避免阻塞主会话"
        assert "unknown" not in result.name.lower()
        assert "Avoid conditions leading to" not in result.prevention

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

    @pytest.mark.asyncio
    async def test_extract_taxonomy_accepts_list_response(self, openai_llm):
        from meta_learning.shared.models import Experience, TaskType

        exp = Experience(
            id="exp-001",
            task_type=TaskType.CONFIGURATION,
            created_at=datetime.now(),
            source_signal="sig-001",
            confidence=0.6,
            scene="Need clarification",
            failure_signature=None,
            root_cause="Ambiguous input",
            resolution="Ask the user",
            meta_insight="Use ask_user when unclear",
        )

        async def mock_chat_json(_system, _user):
            return [{
                "name": "Ask User",
                "trigger": "unclear input",
                "fix_sop": "ask",
                "prevention": "Use ask_user",
                "keywords": ["ask_user"],
            }]

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            result = await openai_llm.extract_taxonomy([exp])

        assert result.name == "Ask User"
        assert result.keywords == ["ask_user"]

    @pytest.mark.asyncio
    async def test_extract_taxonomy_uses_experience_language(self, openai_llm):
        from meta_learning.shared.models import Experience, TaskType

        exp = Experience(
            id="exp-001",
            task_type=TaskType.CONFIGURATION,
            created_at=datetime.now(),
            source_signal="sig-001",
            confidence=0.6,
            scene="用户要求记录中文规则",
            failure_signature=None,
            root_cause="不确定需求时没有询问用户",
            resolution="遇到不确定点必须使用 ask_user 询问用户",
            meta_insight="不确定时先问用户，不要猜测",
        )
        captured = {}

        async def mock_chat_json(system, _user):
            captured["system"] = system
            return {
                "name": "不确定时先询问用户",
                "trigger": "需求不清楚或存在歧义时",
                "fix_sop": "1. 暂停执行\n2. 调用 ask_user 询问用户",
                "prevention": "遇到不确定点必须先问用户，不要猜测",
                "keywords": ["ask_user", "确认需求"],
            }

        with patch.object(openai_llm, "_chat_json", side_effect=mock_chat_json):
            result = await openai_llm.extract_taxonomy([exp])

        assert "TARGET OUTPUT LANGUAGE: Simplified Chinese" in captured["system"]
        assert result.name == "不确定时先询问用户"

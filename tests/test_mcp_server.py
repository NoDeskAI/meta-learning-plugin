"""Tests for the MCP server tool handlers.

These tests exercise the tool functions directly (not via MCP transport),
using the same tmp_config fixture as the rest of the test suite.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from meta_learning.shared.models import (
    ErrorTaxonomy,
    Signal,
    TaxonomyEntry,
    TriggerReason,
)


@pytest.fixture(autouse=True)
def _reset_mcp_globals():
    """Reset module-level singletons between tests."""
    import meta_learning.mcp_server as mod

    mod._config = None
    mod._qt_index = None
    mod._taxonomy_mtime = 0.0
    if mod._layer2_task is not None and not mod._layer2_task.done():
        mod._layer2_task.cancel()
    mod._layer2_task = None
    mod._layer2_thread = None
    yield
    if mod._layer2_task is not None and not mod._layer2_task.done():
        mod._layer2_task.cancel()
    mod._config = None
    mod._qt_index = None
    mod._taxonomy_mtime = 0.0
    mod._layer2_task = None
    mod._layer2_thread = None


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "signal_buffer").mkdir()
    (ws / "experience_pool").mkdir()
    return ws


@pytest.fixture
def _env(workspace: Path):
    """Set env vars so the MCP server picks up the temp workspace."""
    with patch.dict(os.environ, {
        "META_LEARNING_WORKSPACE": str(workspace),
        "META_LEARNING_CONFIG": "",
    }):
        yield


def _write_taxonomy(workspace: Path, taxonomy: ErrorTaxonomy) -> None:
    tax_path = workspace / "error_taxonomy.yaml"
    data: dict = {"taxonomy": {}}
    for domain, subdomains in taxonomy.taxonomy.items():
        data["taxonomy"][domain] = {}
        for subdomain, entries in subdomains.items():
            data["taxonomy"][domain][subdomain] = [
                e.model_dump(mode="json") for e in entries
            ]
    tax_path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")


def _sample_taxonomy() -> ErrorTaxonomy:
    entry = TaxonomyEntry(
        id="tax-cod-001",
        name="Generic Type Error",
        trigger="Nested generics",
        fix_sop="Add explicit type params",
        prevention="Always annotate generics",
        confidence=0.9,
        source_exps=["exp-001"],
        keywords=["TS2345", "generic"],
        created_at=date.today(),
        last_verified=date.today(),
    )
    tax = ErrorTaxonomy()
    tax.add_entry("coding", "typescript", entry)
    return tax


# -----------------------------------------------------------------------
# quick_think
# -----------------------------------------------------------------------


class TestQuickThink:
    @pytest.mark.usefixtures("_env")
    def test_no_risk(self):
        from meta_learning.mcp_server import quick_think

        result = quick_think("rename a variable")
        assert result == "no risk detected"

    @pytest.mark.usefixtures("_env")
    def test_irreversible_detected(self):
        from meta_learning.mcp_server import quick_think

        result = quick_think("rm -rf /tmp/old_data")
        assert "irreversible" in result.lower()
        assert "high" in result

    @pytest.mark.usefixtures("_env")
    def test_taxonomy_keyword_hit(self, workspace: Path):
        _write_taxonomy(workspace, _sample_taxonomy())
        from meta_learning.mcp_server import quick_think

        result = quick_think("fix TS2345 generic error")
        assert "tax-cod-001" in result
        assert "Generic Type Error" in result

    @pytest.mark.usefixtures("_env")
    def test_tools_used(self):
        from meta_learning.mcp_server import quick_think

        result = quick_think("push changes", tools_used=["git push --force"])
        assert "irreversible" in result.lower()


# -----------------------------------------------------------------------
# capture_signal
# -----------------------------------------------------------------------


@pytest.mark.asyncio
class TestCaptureSignal:
    @pytest.mark.usefixtures("_env")
    async def test_no_signal_for_clean_task(self):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(task_description="read a file")
        assert "no signal captured" in result

    @pytest.mark.usefixtures("_env")
    async def test_captures_error_recovery(self, workspace: Path):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(
            task_description="Fix TypeScript build error",
            errors_encountered=["TS2345: type mismatch"],
            errors_fixed=True,
            step_count=5,
        )
        assert "Signal captured" in result
        assert "self_recovery" in result

        sig_files = list((workspace / "signal_buffer").glob("sig-*.yaml"))
        assert len(sig_files) == 1

    @pytest.mark.usefixtures("_env")
    async def test_captures_user_correction(self, workspace: Path):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(
            task_description="Implement feature",
            user_corrections=["不对，应该用另一种方式"],
        )
        assert "Signal captured" in result
        assert "user_correction" in result

    @pytest.mark.usefixtures("_env")
    async def test_captures_efficiency_anomaly(self, workspace: Path):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(
            task_description="Complex refactoring",
            step_count=30,
        )
        assert "Signal captured" in result
        assert "efficiency_anomaly" in result

    @pytest.mark.usefixtures("_env")
    async def test_captures_new_tool(self, workspace: Path):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(
            task_description="Deploy with Docker",
            new_tools=["docker_exec"],
        )
        assert "Signal captured" in result
        assert "new_tool" in result

    @pytest.mark.usefixtures("_env")
    async def test_captures_resolution_and_images(self, workspace: Path):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(
            task_description="Fix flaky E2E test",
            errors_encountered=["Timeout waiting for selector"],
            errors_fixed=True,
            resolution_snapshot="added explicit wait and retry",
            image_snapshots=["screenshots/before.png", "screenshots/after.png"],
            step_count=8,
        )
        assert "Signal captured" in result

        sig_files = sorted((workspace / "signal_buffer").glob("sig-*.yaml"))
        assert len(sig_files) == 1
        signal_data = yaml.safe_load(sig_files[0].read_text(encoding="utf-8"))
        assert signal_data["resolution_snapshot"] == "added explicit wait and retry"
        assert signal_data["image_snapshots"] == [
            "screenshots/before.png",
            "screenshots/after.png",
        ]

    @pytest.mark.usefixtures("_env")
    async def test_user_correction_triggers_background_layer2(self, workspace: Path):
        from meta_learning.mcp_server import capture_signal

        result = await capture_signal(
            task_description="Implement feature incorrectly",
            user_corrections=["No, use the other API endpoint"],
        )
        assert "Signal captured" in result
        assert "user_correction" in result
        assert "Layer 2 triggered in background" in result


# -----------------------------------------------------------------------
# status
# -----------------------------------------------------------------------


class TestStatus:
    @pytest.mark.usefixtures("_env")
    def test_empty_workspace(self, workspace: Path):
        from meta_learning.mcp_server import status

        result = status()
        assert "Pending signals: 0" in result
        assert "Total experiences: 0" in result
        assert "Taxonomy entries: 0" in result
        assert str(workspace) in result


# -----------------------------------------------------------------------
# layer2_status
# -----------------------------------------------------------------------


class TestLayer2Status:
    @pytest.mark.usefixtures("_env")
    @pytest.mark.asyncio
    async def test_recovers_pending_backlog_after_restart(self, workspace: Path):
        from meta_learning.mcp_server import layer2_status
        from meta_learning.shared.io import write_signal
        from meta_learning.shared.models import MetaLearningConfig

        config = MetaLearningConfig(workspace_root=str(workspace))
        write_signal(
            Signal(
                signal_id="sig-20260429-001",
                timestamp=datetime(2026, 4, 29, 12, 0),
                session_id="unknown",
                trigger_reason=TriggerReason.USER_CORRECTION,
                keywords=["ask_user"],
                task_summary="remember clarification rule",
                user_feedback="ask when unclear",
                step_count=1,
            ),
            config,
        )

        async def _no_op(_config):
            return None

        with patch("meta_learning.mcp_server._run_layer2_background", new=_no_op):
            result = await layer2_status()

        assert "RECOVERING" in result
        assert "1 pending signal" in result


# -----------------------------------------------------------------------
# run_layer2
# -----------------------------------------------------------------------


class TestRunLayer2:
    @pytest.mark.usefixtures("_env")
    @pytest.mark.asyncio
    async def test_no_trigger(self):
        from meta_learning.mcp_server import run_layer2

        result = await run_layer2()
        assert "trigger conditions not met" in result

    @pytest.mark.usefixtures("_env")
    @pytest.mark.asyncio
    async def test_force_run(self):
        from meta_learning.mcp_server import run_layer2

        result = await run_layer2(force=True)
        assert "Layer 2 complete" in result

    @pytest.mark.usefixtures("_env")
    @pytest.mark.asyncio
    async def test_bootstrap_multimodal_embedding_called(self):
        from meta_learning.mcp_server import run_layer2

        with patch(
            "meta_learning.mcp_server.bootstrap_multimodal_embedding"
        ) as mock_bootstrap:
            await run_layer2(force=True)
            mock_bootstrap.assert_called_once()


# -----------------------------------------------------------------------
# sync_taxonomy_to_nobot
# -----------------------------------------------------------------------


class TestSyncTaxonomyToNobot:
    @pytest.mark.usefixtures("_env")
    def test_empty_taxonomy_message(self, tmp_path: Path):
        nobot_root = tmp_path / "nanobot_ws"
        (nobot_root / "skills").mkdir(parents=True)

        from meta_learning.mcp_server import sync_taxonomy_to_nobot

        msg = sync_taxonomy_to_nobot(nobot_workspace=str(nobot_root))
        assert "No taxonomy" in msg

    @pytest.mark.usefixtures("_env")
    def test_writes_skill_under_skills_dir(self, workspace: Path, tmp_path: Path):
        _write_taxonomy(workspace, _sample_taxonomy())
        nobot_root = tmp_path / "nanobot_ws"
        (nobot_root / "skills").mkdir(parents=True)

        from meta_learning.mcp_server import sync_taxonomy_to_nobot

        msg = sync_taxonomy_to_nobot(nobot_workspace=str(nobot_root))
        assert "Synced" in msg
        assert "SKILL.md" in msg

        skill = nobot_root / "skills" / "meta-learning" / "SKILL.md"
        assert skill.is_file()
        text = skill.read_text(encoding="utf-8")
        assert "Meta-Learning Rules" in text
        assert "quick_think" in text


# -----------------------------------------------------------------------
# taxonomy maintenance
# -----------------------------------------------------------------------


class TestTaxonomyMaintenance:
    @pytest.mark.usefixtures("_env")
    def test_delete_taxonomy_entry(self, workspace: Path):
        _write_taxonomy(workspace, _sample_taxonomy())

        from meta_learning.mcp_server import delete_taxonomy_entry, get_taxonomy

        msg = delete_taxonomy_entry("tax-cod-001", sync_to_nobot=False)

        assert "Deleted taxonomy entry" in msg
        assert "tax-cod-001" not in get_taxonomy()

    @pytest.mark.usefixtures("_env")
    def test_delete_taxonomy_entry_missing(self):
        from meta_learning.mcp_server import delete_taxonomy_entry

        msg = delete_taxonomy_entry("tax-missing", sync_to_nobot=False)

        assert "not found" in msg


# -----------------------------------------------------------------------
# run_layer3
# -----------------------------------------------------------------------


class TestRunLayer3:
    @pytest.mark.usefixtures("_env")
    @pytest.mark.asyncio
    async def test_runs(self):
        from meta_learning.mcp_server import run_layer3

        result = await run_layer3()
        assert "Layer 3 complete" in result


# -----------------------------------------------------------------------
# resources
# -----------------------------------------------------------------------


class TestServerConfigEnv:
    def test_sessions_root_env_override(self, workspace: Path, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("META_LEARNING_CONFIG", raising=False)
        desk_sessions = tmp_path / "deskclaw_sessions"
        desk_sessions.mkdir()
        monkeypatch.setenv("META_LEARNING_WORKSPACE", str(workspace))
        monkeypatch.setenv("META_LEARNING_SESSIONS_ROOT", str(desk_sessions))

        from meta_learning.mcp_server import _load_server_config

        cfg = _load_server_config()
        assert cfg.workspace_root == str(workspace)
        assert Path(cfg.sessions_root) == desk_sessions

    def test_config_path_tilde_expansion(self, workspace: Path, tmp_path: Path, monkeypatch):
        config_dir = tmp_path / "fake_home" / ".deskclaw" / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(
            "workspace_root: /tmp/test_ws\n"
            "llm:\n"
            "  provider: openai\n"
            "  model: test-model\n",
            encoding="utf-8",
        )

        tilde_path = str(config_file).replace(str(tmp_path / "fake_home"), "~")
        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        monkeypatch.setenv("META_LEARNING_CONFIG", tilde_path)
        monkeypatch.delenv("META_LEARNING_WORKSPACE", raising=False)

        from meta_learning.mcp_server import _load_server_config

        cfg = _load_server_config()
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "test-model"


class TestResources:
    @pytest.mark.usefixtures("_env")
    def test_taxonomy_empty(self):
        from meta_learning.mcp_server import get_taxonomy

        assert get_taxonomy() == ""

    @pytest.mark.usefixtures("_env")
    def test_taxonomy_with_data(self, workspace: Path):
        _write_taxonomy(workspace, _sample_taxonomy())
        from meta_learning.mcp_server import get_taxonomy

        result = get_taxonomy()
        assert "tax-cod-001" in result
        assert "TS2345" in result

    @pytest.mark.usefixtures("_env")
    def test_config_resource(self):
        from meta_learning.mcp_server import get_config_resource

        result = get_config_resource()
        assert "workspace_root" in result
        assert "signal_buffer_dir" in result


# -----------------------------------------------------------------------
# risk_assessment prompt
# -----------------------------------------------------------------------


class TestRiskAssessmentPrompt:
    @pytest.mark.usefixtures("_env")
    def test_safe_task(self):
        from meta_learning.mcp_server import risk_assessment

        result = risk_assessment("read a config file")
        assert "No known risk patterns" in result

    @pytest.mark.usefixtures("_env")
    def test_risky_task(self):
        from meta_learning.mcp_server import risk_assessment

        result = risk_assessment("rm -rf /data")
        assert "irreversible" in result.lower()
        assert "rollback" in result.lower()

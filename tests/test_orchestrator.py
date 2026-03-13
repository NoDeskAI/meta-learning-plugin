from datetime import date, datetime

import pytest

from meta_learning.layer2.orchestrator import Layer2Orchestrator
from meta_learning.shared.io import write_signal
from meta_learning.shared.models import (
    Signal,
    TriggerReason,
)


def _write_n_signals(n: int, tmp_config):
    for i in range(n):
        sig = Signal(
            signal_id=f"sig-20260309-{i + 1:03d}",
            timestamp=datetime.now(),
            session_id=f"session-{i}",
            memory_date=date(2026, 3, 9),
            trigger_reason=TriggerReason.ERROR_RECOVERY,
            keywords=["error", "test"],
            task_summary=f"Fix coding error #{i}",
            error_snapshot=f"Error: type mismatch #{i}",
            resolution_snapshot=f"Fix #{i}",
            step_count=5,
        )
        write_signal(sig, tmp_config)


class TestLayer2Trigger:
    def test_no_trigger_when_empty(self, tmp_config, stub_llm):
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        assert orch.should_trigger() is False

    def test_triggers_on_min_signals(self, tmp_config, stub_llm):
        _write_n_signals(5, tmp_config)
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        assert orch.should_trigger() is True

    def test_no_trigger_below_threshold_after_recent_run(self, tmp_config, stub_llm):
        _write_n_signals(3, tmp_config)
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        orch._save_last_run_time()
        assert orch.should_trigger() is False


@pytest.mark.asyncio
class TestLayer2Pipeline:
    async def test_full_pipeline_empty(self, tmp_config, stub_llm):
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        result = await orch.run_pipeline()
        assert result.materialized_count == 0
        assert result.total_clusters == 0
        assert result.new_taxonomy_entries == 0

    async def test_full_pipeline_with_signals(self, tmp_config, stub_llm):
        _write_n_signals(3, tmp_config)
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        result = await orch.run_pipeline()
        assert result.materialized_count == 3
        assert result.total_clusters >= 0

    async def test_pipeline_saves_state(self, tmp_config, stub_llm):
        _write_n_signals(2, tmp_config)
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        await orch.run_pipeline()
        last_run = orch._load_last_run_time()
        assert last_run is not None

    async def test_pipeline_idempotent(self, tmp_config, stub_llm):
        _write_n_signals(3, tmp_config)
        orch = Layer2Orchestrator(tmp_config, stub_llm)
        result1 = await orch.run_pipeline()
        result2 = await orch.run_pipeline()
        assert result1.materialized_count == 3
        assert result2.materialized_count == 0


@pytest.mark.asyncio
class TestEndToEnd:
    async def test_signals_to_taxonomy(self, tmp_config, stub_llm):
        for i in range(5):
            sig = Signal(
                signal_id=f"sig-20260309-{i + 1:03d}",
                timestamp=datetime.now(),
                session_id=f"session-{i}",
                trigger_reason=TriggerReason.ERROR_RECOVERY,
                keywords=["TS2345", "type", "error"],
                task_summary=f"Fix coding type error #{i}",
                error_snapshot="TS2345: type mismatch generic inference",
                resolution_snapshot="Add explicit generic annotation",
                step_count=5,
            )
            write_signal(sig, tmp_config)

        orch = Layer2Orchestrator(tmp_config, stub_llm)
        result = await orch.run_pipeline()

        assert result.materialized_count == 5
        assert result.total_clusters >= 0

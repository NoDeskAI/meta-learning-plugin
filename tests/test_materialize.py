from datetime import date, datetime
from pathlib import Path

import pytest

from meta_learning.layer2.materialize import Materializer
from meta_learning.shared.io import list_pending_signals, write_signal
from meta_learning.shared.models import Signal, TriggerReason

_MOCK_SESSION = '{"role":"user","content":"fix it"}\n{"role":"assistant","content":"done"}\n'


def _ensure_session_file(session_id: str, config) -> None:
    sessions_dir = Path(config.sessions_full_path)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.jsonl").write_text(_MOCK_SESSION)


@pytest.mark.asyncio
class TestMaterialize:
    async def test_no_pending_signals(self, tmp_config, stub_llm):
        mat = Materializer(tmp_config, stub_llm)
        results = await mat.materialize_all_pending()
        assert results == []

    async def test_materialize_single_signal(self, tmp_config, stub_llm, sample_signal):
        _ensure_session_file(sample_signal.session_id, tmp_config)
        write_signal(sample_signal, tmp_config)
        mat = Materializer(tmp_config, stub_llm)
        results = await mat.materialize_all_pending()
        assert len(results) == 1
        exp = results[0]
        assert exp.source_signal == sample_signal.signal_id
        assert exp.confidence == 0.6
        assert exp.scene == sample_signal.task_summary

    async def test_marks_signal_processed(self, tmp_config, stub_llm, sample_signal):
        _ensure_session_file(sample_signal.session_id, tmp_config)
        write_signal(sample_signal, tmp_config)
        mat = Materializer(tmp_config, stub_llm)
        await mat.materialize_all_pending()
        pending = list_pending_signals(tmp_config)
        assert len(pending) == 0

    async def test_materialize_multiple_signals(self, tmp_config, stub_llm):
        for i in range(3):
            session_id = f"session-{i}"
            _ensure_session_file(session_id, tmp_config)
            sig = Signal(
                signal_id=f"sig-20260309-{i + 1:03d}",
                timestamp=datetime.now(),
                session_id=session_id,
                memory_date=date(2026, 3, 9),
                trigger_reason=TriggerReason.SELF_RECOVERY,
                keywords=["error"],
                task_summary=f"Fix coding error #{i}",
                error_snapshot=f"Error #{i}",
                step_count=5,
            )
            write_signal(sig, tmp_config)
        mat = Materializer(tmp_config, stub_llm)
        results = await mat.materialize_all_pending()
        assert len(results) == 3

    async def test_experience_has_correct_fields(
        self, tmp_config, stub_llm, sample_signal
    ):
        _ensure_session_file(sample_signal.session_id, tmp_config)
        write_signal(sample_signal, tmp_config)
        mat = Materializer(tmp_config, stub_llm)
        results = await mat.materialize_all_pending()
        exp = results[0]
        assert exp.id.startswith("exp-")
        assert exp.source_session == sample_signal.session_id
        assert exp.verification_count == 1
        assert exp.root_cause is not None
        assert exp.meta_insight is not None

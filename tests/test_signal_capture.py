from pathlib import Path

from meta_learning.layer1.signal_capture import SignalCapture
from meta_learning.shared.models import (
    TaskContext,
    TriggerReason,
)


class TestSignalCaptureTriggering:
    def test_error_recovery_trigger(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Fix TypeScript error",
            errors_encountered=["TS2345: type mismatch"],
            errors_fixed=True,
            step_count=5,
            session_id="sess-001",
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert signal.trigger_reason == TriggerReason.ERROR_RECOVERY
        assert signal.error_snapshot is not None

    def test_user_correction_trigger(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Write function",
            user_corrections=["No, use async/await instead"],
            step_count=3,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert signal.trigger_reason == TriggerReason.USER_CORRECTION
        assert signal.user_feedback is not None

    def test_new_tool_trigger(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Deploy with Docker",
            new_tools=["docker_exec"],
            step_count=3,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert signal.trigger_reason == TriggerReason.NEW_TOOL

    def test_efficiency_anomaly_trigger(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Simple file edit",
            step_count=25,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert signal.trigger_reason == TriggerReason.EFFICIENCY_ANOMALY

    def test_no_trigger_on_smooth_task(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Rename variable x to y",
            step_count=2,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is None


class TestSignalCapturePersistence:
    def test_signal_written_to_disk(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Fix bug",
            errors_encountered=["Error: null pointer"],
            errors_fixed=True,
            step_count=5,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        buf_dir = Path(tmp_config.signal_buffer_path)
        yaml_files = list(buf_dir.glob("sig-*.yaml"))
        assert len(yaml_files) == 1


class TestSignalCaptureKeywordExtraction:
    def test_extracts_error_tokens(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Fix error",
            errors_encountered=["TS2345: Argument type mismatch in GenericComponent"],
            errors_fixed=True,
            step_count=5,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert len(signal.keywords) > 0
        assert any("TS2345" in kw for kw in signal.keywords)

    def test_includes_new_tools_as_keywords(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Deploy",
            new_tools=["kubectl"],
            step_count=3,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert "kubectl" in signal.keywords


class TestSignalCapturePriority:
    def test_error_recovery_prioritized_over_anomaly(self, tmp_config):
        capture = SignalCapture(tmp_config)
        ctx = TaskContext(
            task_description="Complex task",
            errors_encountered=["Error: something broke"],
            errors_fixed=True,
            step_count=50,
        )
        signal = capture.evaluate_and_capture(ctx)
        assert signal is not None
        assert signal.trigger_reason == TriggerReason.ERROR_RECOVERY

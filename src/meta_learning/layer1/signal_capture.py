from __future__ import annotations

from datetime import datetime

from meta_learning.shared.io import next_signal_id, write_signal
from meta_learning.shared.models import (
    MetaLearningConfig,
    Signal,
    TaskContext,
    TriggerReason,
)


class SignalCapture:
    def __init__(self, config: MetaLearningConfig) -> None:
        self._config = config

    def evaluate_and_capture(self, context: TaskContext) -> Signal | None:
        trigger = self._determine_trigger(context)
        if trigger is None:
            return None

        signal = self._build_signal(context, trigger)
        write_signal(signal, self._config)
        return signal

    def _determine_trigger(self, context: TaskContext) -> TriggerReason | None:
        if context.errors_fixed and context.errors_encountered:
            return TriggerReason.ERROR_RECOVERY

        if context.errors_encountered:
            return TriggerReason.ERROR_RECOVERY

        if context.user_corrections:
            return TriggerReason.USER_CORRECTION

        if context.new_tools:
            return TriggerReason.NEW_TOOL

        threshold = self._config.layer1.signal_capture.efficiency_anomaly_threshold
        avg = self._config.layer1.signal_capture.average_step_count
        if context.step_count > avg * threshold:
            return TriggerReason.EFFICIENCY_ANOMALY

        return None

    def _build_signal(self, context: TaskContext, trigger: TriggerReason) -> Signal:
        sig_id = next_signal_id(self._config)

        keywords = _extract_keywords(context)

        error_snapshot: str | None = None
        if context.errors_encountered:
            error_snapshot = "\n---\n".join(context.errors_encountered)[:2000]

        resolution_snapshot: str | None = None
        if context.extra.get("resolution"):
            resolution_snapshot = str(context.extra["resolution"])[:1000]

        user_feedback: str | None = None
        if context.user_corrections:
            user_feedback = context.user_corrections[0][:500]

        exp_cfg = self._config.experiment
        experiment_id: str | None = None
        experiment_group: str | None = None
        if exp_cfg.enabled and exp_cfg.experiment_id:
            experiment_id = exp_cfg.experiment_id
            experiment_group = exp_cfg.group.value

        raw_snapshots = context.extra.get("image_snapshots", [])
        image_snapshots: list[str] = (
            [str(p) for p in raw_snapshots] if isinstance(raw_snapshots, list) else []
        )

        return Signal(
            signal_id=sig_id,
            timestamp=datetime.now(),
            session_id=context.session_id or "unknown",
            memory_date=datetime.now().date(),
            trigger_reason=trigger,
            keywords=keywords,
            task_summary=context.task_description[:200],
            error_snapshot=error_snapshot,
            resolution_snapshot=resolution_snapshot,
            user_feedback=user_feedback,
            image_snapshots=image_snapshots,
            step_count=context.step_count,
            experiment_id=experiment_id,
            experiment_group=experiment_group,
        )


def _extract_keywords(context: TaskContext) -> list[str]:
    keywords: list[str] = []

    for tool in context.tools_used:
        if tool and tool not in keywords:
            keywords.append(tool)

    for tool in context.new_tools:
        if tool and tool not in keywords:
            keywords.append(tool)

    for error in context.errors_encountered[:3]:
        tokens = error.split()
        for token in tokens:
            cleaned = token.strip(".:,;()[]{}\"'")
            if (
                len(cleaned) > 2
                and (cleaned[0].isupper() or any(c.isdigit() for c in cleaned))
                and cleaned not in keywords
            ):
                keywords.append(cleaned)
                if len(keywords) >= 15:
                    break

    if not keywords:
        words = context.task_description.split()
        for w in words:
            cleaned = w.strip(".:,;()[]{}\"'").lower()
            if len(cleaned) > 3 and cleaned not in keywords:
                keywords.append(cleaned)
                if len(keywords) >= 5:
                    break

    return keywords[:15]

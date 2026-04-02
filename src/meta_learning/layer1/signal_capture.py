from __future__ import annotations

from datetime import datetime

from meta_learning.shared.io import next_signal_id, write_signal
from meta_learning.shared.models import (
    CHANNEL_PRIORITY,
    DetectionChannel,
    MetaLearningConfig,
    Signal,
    TaskContext,
)


class SignalCapture:
    def __init__(self, config: MetaLearningConfig) -> None:
        self._config = config

    def evaluate_and_capture(self, context: TaskContext) -> Signal | None:
        channels = self._detect_channels(context)
        if not channels:
            return None

        signal = self._build_signal(context, channels)
        write_signal(signal, self._config)
        return signal

    def _detect_channels(self, context: TaskContext) -> list[DetectionChannel]:
        channels: list[DetectionChannel] = []
        if context.user_corrections:
            channels.append(DetectionChannel.USER_CORRECTION)
        if context.errors_encountered:
            if context.errors_fixed:
                channels.append(DetectionChannel.SELF_RECOVERY)
            else:
                channels.append(DetectionChannel.UNRESOLVED_ERROR)
        if context.new_tools:
            channels.append(DetectionChannel.NEW_TOOL)
        threshold = self._config.layer1.signal_capture.efficiency_anomaly_threshold
        avg = self._config.layer1.signal_capture.average_step_count
        if context.step_count > avg * threshold:
            channels.append(DetectionChannel.EFFICIENCY_ANOMALY)
        return channels

    @staticmethod
    def _pick_primary(channels: list[DetectionChannel]) -> DetectionChannel:
        for ch in CHANNEL_PRIORITY:
            if ch in channels:
                return ch
        return channels[0]

    def _build_signal(self, context: TaskContext, channels: list[DetectionChannel]) -> Signal:
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
            detection_channels=channels,
            primary_channel=self._pick_primary(channels),
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

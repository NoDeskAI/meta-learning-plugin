from __future__ import annotations

import logging
from datetime import datetime

from meta_learning.shared.io import (
    list_pending_signals,
    mark_signal_processed,
    next_experience_id,
    read_session_context,
    resolve_session_file,
    write_experience,
)
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    Experience,
    MetaLearningConfig,
    Signal,
)

logger = logging.getLogger(__name__)


class Materializer:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def materialize_all_pending(self) -> list[Experience]:
        pending = list_pending_signals(self._config)
        if not pending:
            return []

        processable: list[Signal] = []
        for s in pending:
            if s.session_id and s.session_id != "unknown":
                if not resolve_session_file(s.session_id, self._config).exists():
                    logger.warning(
                        "Session file missing for %s (session=%s), "
                        "proceeding without session context",
                        s.signal_id, s.session_id,
                    )
            processable.append(s)

        experiences: list[Experience] = []
        failed_ids: list[str] = []
        for signal in processable:
            try:
                exp = await self._materialize_one(signal)
            except Exception:
                logger.exception(
                    "Failed to materialize signal %s, skipping",
                    signal.signal_id,
                )
                failed_ids.append(signal.signal_id)
                continue
            if exp is not None:
                experiences.append(exp)
                mark_signal_processed(signal.signal_id, self._config)

        if failed_ids:
            logger.warning(
                "Materialization completed with %d failure(s): %s",
                len(failed_ids),
                ", ".join(failed_ids),
            )
        return experiences

    async def _materialize_one(self, signal: Signal) -> Experience | None:
        session_context = ""
        if signal.session_id and signal.session_id != "unknown":
            resolved = resolve_session_file(signal.session_id, self._config)
            if resolved.exists():
                session_context = read_session_context(signal.session_id, self._config)
            else:
                logger.info(
                    "Materializing %s without session context (file not found)",
                    signal.signal_id,
                )

        result = await self._llm.materialize_signal(signal, session_context)

        exp_id = next_experience_id(self._config)
        init_conf = self._config.layer2.materialize.initial_confidence
        experience = Experience(
            id=exp_id,
            task_type=result.task_type,
            created_at=datetime.now(),
            source_signal=signal.signal_id,
            source_session=signal.session_id,
            source_memory=signal.memory_date,
            initial_confidence=init_conf,
            confidence=init_conf,
            verification_count=1,
            scene=result.scene,
            failure_signature=result.failure_signature,
            root_cause=result.root_cause,
            resolution=result.resolution,
            meta_insight=result.meta_insight,
        )

        write_experience(experience, self._config)
        return experience

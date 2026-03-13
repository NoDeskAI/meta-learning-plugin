from __future__ import annotations

from datetime import datetime

from meta_learning.shared.io import (
    list_pending_signals,
    mark_signal_processed,
    next_experience_id,
    read_session_context,
    write_experience,
)
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    Experience,
    MetaLearningConfig,
    Signal,
)


class Materializer:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def materialize_all_pending(self) -> list[Experience]:
        pending = list_pending_signals(self._config)
        if not pending:
            return []

        experiences: list[Experience] = []
        for signal in pending:
            exp = await self._materialize_one(signal)
            if exp is not None:
                experiences.append(exp)
                mark_signal_processed(signal.signal_id, self._config)

        return experiences

    async def _materialize_one(self, signal: Signal) -> Experience | None:
        session_context = ""
        if signal.session_id and signal.session_id != "unknown":
            session_context = read_session_context(signal.session_id, self._config)

        result = await self._llm.materialize_signal(signal, session_context)

        exp_id = next_experience_id(self._config)
        experience = Experience(
            id=exp_id,
            task_type=result.task_type,
            created_at=datetime.now(),
            source_signal=signal.signal_id,
            source_session=signal.session_id,
            source_memory=signal.memory_date,
            confidence=self._config.layer2.materialize.initial_confidence,
            verification_count=1,
            scene=result.scene,
            failure_signature=result.failure_signature,
            root_cause=result.root_cause,
            resolution=result.resolution,
            meta_insight=result.meta_insight,
        )

        write_experience(experience, self._config)
        return experience

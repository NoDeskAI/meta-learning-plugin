from __future__ import annotations

from meta_learning.shared.io import list_all_experiences
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    MemoryRecommendation,
    MetaLearningConfig,
)


class MemoryArchitect:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def optimize(self) -> list[MemoryRecommendation]:
        all_exps = list_all_experiences(self._config)
        if not all_exps:
            return []

        high_conf = [
            e
            for e in all_exps
            if e.confidence >= self._config.confidence.promote_threshold
        ]
        low_conf = [
            e
            for e in all_exps
            if e.confidence < self._config.layer3.prune_confidence_threshold
        ]

        analysis = await self._llm.analyze_memory(high_conf, low_conf)
        return analysis.recommendations

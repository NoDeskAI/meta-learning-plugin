from __future__ import annotations

from datetime import datetime

from meta_learning.shared.io import list_all_experiences, load_error_taxonomy
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    CapabilityGap,
    Experience,
    MetaLearningConfig,
)


class NewCapabilityDetector:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def detect_gaps(self) -> list[CapabilityGap]:
        all_exps = list_all_experiences(self._config)
        taxonomy = load_error_taxonomy(self._config)

        taxonomy_keywords = list(taxonomy.all_keywords().keys())
        promoted_ids = {e.id for e in all_exps if e.promoted_to is not None}
        ungrouped = [e for e in all_exps if e.id not in promoted_ids]

        if not ungrouped:
            return []

        analyses = await self._llm.analyze_capability_gaps(ungrouped, taxonomy_keywords)

        gaps: list[CapabilityGap] = []
        for analysis in analyses:
            matching_exps = _find_matching_experiences(ungrouped, analysis.description)
            gap = CapabilityGap(
                gap_id=f"gap-{len(gaps) + 1:03d}",
                gap_type=_infer_gap_type(matching_exps),
                description=analysis.description,
                evidence_ids=[e.id for e in matching_exps],
                suggested_action=analysis.suggested_action,
                priority=analysis.priority,
                created_at=datetime.now(),
            )
            gaps.append(gap)

        return gaps


def _find_matching_experiences(
    experiences: list[Experience], description: str
) -> list[Experience]:
    desc_lower = description.lower()
    return [e for e in experiences if e.task_type.value in desc_lower]


def _infer_gap_type(experiences: list[Experience]) -> str:
    has_failures = any(e.failure_signature for e in experiences)
    if has_failures:
        return "failure"
    avg_confidence = (
        sum(e.confidence for e in experiences) / len(experiences)
        if experiences
        else 0.0
    )
    if avg_confidence < 0.5:
        return "efficiency"
    return "frequency"

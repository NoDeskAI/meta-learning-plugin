from __future__ import annotations

from datetime import datetime
from itertools import combinations

from meta_learning.shared.io import list_all_experiences
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    CrossTaskPattern,
    Experience,
    MetaLearningConfig,
    TaskType,
)


class CrossTaskMiner:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def mine_patterns(self) -> list[CrossTaskPattern]:
        all_exps = list_all_experiences(self._config)
        if len(all_exps) < self._config.layer3.min_experiences_for_cross_task:
            return []

        cross_type_groups = self._build_cross_type_groups(all_exps)
        if not cross_type_groups:
            return []

        analyses = await self._llm.analyze_cross_task_patterns(cross_type_groups)

        patterns: list[CrossTaskPattern] = []
        for i, analysis in enumerate(analyses):
            if analysis.confidence < self._config.layer3.min_pattern_confidence:
                continue
            group = cross_type_groups[i] if i < len(cross_type_groups) else []
            pattern = CrossTaskPattern(
                pattern_id=f"ctp-{len(patterns) + 1:03d}",
                description=analysis.description,
                shared_root_cause=analysis.shared_root_cause,
                affected_task_types=list({e.task_type for e in group}),
                source_experience_ids=[e.id for e in group],
                confidence=analysis.confidence,
                meta_strategy=analysis.meta_strategy,
                created_at=datetime.now(),
            )
            patterns.append(pattern)

        return patterns

    def _build_cross_type_groups(
        self, experiences: list[Experience]
    ) -> list[list[Experience]]:
        by_type: dict[TaskType, list[Experience]] = {}
        for exp in experiences:
            by_type.setdefault(exp.task_type, []).append(exp)

        task_types = list(by_type.keys())
        if len(task_types) < 2:
            return []

        groups: list[list[Experience]] = []
        for type_a, type_b in combinations(task_types, 2):
            exps_a = by_type[type_a]
            exps_b = by_type[type_b]
            merged = _find_shared_root_cause_pairs(exps_a, exps_b)
            if len(merged) >= 2:
                groups.append(merged)

        return groups


def _find_shared_root_cause_pairs(
    exps_a: list[Experience], exps_b: list[Experience]
) -> list[Experience]:
    merged: list[Experience] = []
    seen_ids: set[str] = set()
    for a in exps_a:
        for b in exps_b:
            if _root_causes_overlap(a.root_cause, b.root_cause):
                if a.id not in seen_ids:
                    merged.append(a)
                    seen_ids.add(a.id)
                if b.id not in seen_ids:
                    merged.append(b)
                    seen_ids.add(b.id)
    return merged


def _root_causes_overlap(cause_a: str, cause_b: str) -> bool:
    words_a = {w.lower() for w in cause_a.split() if len(w) > 3}
    words_b = {w.lower() for w in cause_b.split() if len(w) > 3}
    if not words_a or not words_b:
        return False
    overlap = words_a & words_b
    return len(overlap) / min(len(words_a), len(words_b)) >= 0.3

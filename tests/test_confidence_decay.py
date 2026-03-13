from datetime import datetime, timedelta

import pytest

from meta_learning.layer2.consolidate import Consolidator
from meta_learning.shared.io import write_experience
from meta_learning.shared.models import (
    Experience,
    TaskType,
)


def _make_experience(
    exp_id: str,
    failure_sig: str,
    task_type: TaskType = TaskType.CODING,
    confidence: float = 0.6,
    days_old: int = 0,
) -> Experience:
    created = datetime.now() - timedelta(days=days_old)
    return Experience(
        id=exp_id,
        task_type=task_type,
        created_at=created,
        source_signal=f"sig-for-{exp_id}",
        confidence=confidence,
        scene=f"Scene for {exp_id}",
        failure_signature=failure_sig,
        root_cause="some root cause",
        resolution="some resolution",
        meta_insight="some insight",
    )


@pytest.mark.asyncio
class TestConfidenceDecay:
    async def test_decay_enabled_reduces_old_experience_confidence(
        self, tmp_config, stub_llm
    ):
        exp = _make_experience("exp-001", "TS2345 type error", days_old=30)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()

        from meta_learning.shared.io import list_all_experiences

        updated = list_all_experiences(tmp_config)
        assert len(updated) == 1
        assert updated[0].confidence < 0.6

    async def test_decay_disabled_preserves_confidence(self, tmp_config, stub_llm):
        tmp_config.confidence.decay_enabled = False
        exp = _make_experience("exp-001", "TS2345 type error", days_old=30)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()

        from meta_learning.shared.io import list_all_experiences

        updated = list_all_experiences(tmp_config)
        assert len(updated) == 1
        assert updated[0].confidence == pytest.approx(0.6)

    async def test_fresh_experience_no_decay(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", "TS2345 type error", days_old=0)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()

        from meta_learning.shared.io import list_all_experiences

        updated = list_all_experiences(tmp_config)
        assert len(updated) == 1
        assert updated[0].confidence == pytest.approx(0.6, abs=0.01)

    async def test_decay_prunes_very_old_low_confidence(self, tmp_config, stub_llm):
        exp = _make_experience(
            "exp-001", "TS2345 type error", confidence=0.35, days_old=60
        )
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()

        assert len(index.clusters) == 0

    async def test_decay_formula_correctness(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", "TS2345 type error", days_old=10)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()

        from meta_learning.shared.io import list_all_experiences

        updated = list_all_experiences(tmp_config)
        expected = 0.6 * (0.95**10)
        assert updated[0].confidence == pytest.approx(expected, abs=0.01)

    async def test_decay_base_config_respected(self, tmp_config, stub_llm):
        tmp_config.confidence.decay_base = 0.99
        exp = _make_experience("exp-001", "TS2345 type error", days_old=10)
        write_experience(exp, tmp_config)

        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()

        from meta_learning.shared.io import list_all_experiences

        updated = list_all_experiences(tmp_config)
        expected = 0.6 * (0.99**10)
        assert updated[0].confidence == pytest.approx(expected, abs=0.01)

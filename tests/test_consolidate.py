from datetime import datetime

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
) -> Experience:
    return Experience(
        id=exp_id,
        task_type=task_type,
        created_at=datetime.now(),
        source_signal=f"sig-for-{exp_id}",
        confidence=confidence,
        scene=f"Scene for {exp_id}",
        failure_signature=failure_sig,
        root_cause="some root cause",
        resolution="some resolution",
        meta_insight="some insight",
    )


@pytest.mark.asyncio
class TestConsolidate:
    async def test_empty_pool(self, tmp_config, stub_llm):
        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()
        assert index.clusters == []

    async def test_single_experience_no_cluster(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", "TS2345 type error")
        write_experience(exp, tmp_config)
        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()
        assert len(index.clusters) == 0

    async def test_similar_experiences_cluster(self, tmp_config, stub_llm):
        exp1 = _make_experience("exp-001", "TS2345 type error generic")
        exp2 = _make_experience("exp-002", "TS2345 type error inference")
        write_experience(exp1, tmp_config)
        write_experience(exp2, tmp_config)
        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()
        assert len(index.clusters) >= 1

    async def test_different_task_types_separate(self, tmp_config, stub_llm):
        exp1 = _make_experience("exp-001", "error in deploy", TaskType.DEVOPS)
        exp2 = _make_experience("exp-002", "error in code", TaskType.CODING)
        write_experience(exp1, tmp_config)
        write_experience(exp2, tmp_config)
        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()
        assert len(index.clusters) == 0

    async def test_pruned_experiences_excluded(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", "some error", confidence=0.2)
        write_experience(exp, tmp_config)
        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()
        assert len(index.clusters) == 0

    async def test_ready_for_taxonomy(self, tmp_config, stub_llm):
        for i in range(4):
            exp = _make_experience(f"exp-{i + 1:03d}", "TS2345 type error generic")
            write_experience(exp, tmp_config)
        cons = Consolidator(tmp_config, stub_llm)
        await cons.consolidate()
        ready = cons.get_clusters_ready_for_taxonomy()
        assert len(ready) >= 1
        assert len(ready[0].experience_ids) >= 3

    async def test_no_placeholder_clusters_in_index(self, tmp_config, stub_llm):
        for i in range(4):
            exp = _make_experience(f"exp-{i + 1:03d}", "TS2345 type error generic")
            write_experience(exp, tmp_config)
        cons = Consolidator(tmp_config, stub_llm)
        index = await cons.consolidate()
        for cluster in index.clusters:
            assert cluster.cluster_id != "__placeholder__"
            assert cluster.failure_signature_pattern != ""
            assert len(cluster.experience_ids) > 0

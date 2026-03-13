from datetime import datetime

import pytest

from meta_learning.layer3.cross_task_miner import CrossTaskMiner, _root_causes_overlap
from meta_learning.layer3.memory_architect import MemoryArchitect
from meta_learning.layer3.new_capability import NewCapabilityDetector
from meta_learning.layer3.orchestrator import Layer3Orchestrator
from meta_learning.shared.io import (
    load_latest_layer3_result,
    write_experience,
)
from meta_learning.shared.models import (
    Experience,
    TaskType,
)


def _make_experience(
    exp_id: str,
    task_type: TaskType,
    root_cause: str,
    failure_sig: str | None = None,
    confidence: float = 0.6,
    promoted_to: str | None = None,
) -> Experience:
    return Experience(
        id=exp_id,
        task_type=task_type,
        created_at=datetime.now(),
        source_signal=f"sig-for-{exp_id}",
        confidence=confidence,
        scene=f"Scene for {exp_id}",
        failure_signature=failure_sig or f"error in {exp_id}",
        root_cause=root_cause,
        resolution="some resolution",
        meta_insight="some insight",
        promoted_to=promoted_to,
    )


class TestRootCausesOverlap:
    def test_similar_causes_match(self):
        assert _root_causes_overlap(
            "Missing type annotation causes inference failure",
            "Missing explicit annotation leads to inference error",
        )

    def test_different_causes_no_match(self):
        assert not _root_causes_overlap(
            "Database connection timeout",
            "CSS selector specificity conflict",
        )

    def test_empty_causes_no_match(self):
        assert not _root_causes_overlap("", "")

    def test_short_words_ignored(self):
        assert not _root_causes_overlap("a b c", "a b c")


@pytest.mark.asyncio
class TestCrossTaskMiner:
    async def test_empty_pool(self, tmp_config, stub_llm):
        miner = CrossTaskMiner(tmp_config, stub_llm)
        patterns = await miner.mine_patterns()
        assert patterns == []

    async def test_insufficient_experiences(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", TaskType.CODING, "some root cause")
        write_experience(exp, tmp_config)
        miner = CrossTaskMiner(tmp_config, stub_llm)
        patterns = await miner.mine_patterns()
        assert patterns == []

    async def test_single_type_no_cross_pattern(self, tmp_config, stub_llm):
        for i in range(5):
            exp = _make_experience(
                f"exp-{i + 1:03d}", TaskType.CODING, "missing type annotation"
            )
            write_experience(exp, tmp_config)
        miner = CrossTaskMiner(tmp_config, stub_llm)
        patterns = await miner.mine_patterns()
        assert patterns == []

    async def test_cross_type_shared_root_cause(self, tmp_config, stub_llm):
        root = "missing configuration validation before deployment"
        for i in range(3):
            write_experience(
                _make_experience(f"exp-cod-{i + 1:03d}", TaskType.CODING, root),
                tmp_config,
            )
        for i in range(3):
            write_experience(
                _make_experience(f"exp-dev-{i + 1:03d}", TaskType.DEVOPS, root),
                tmp_config,
            )
        miner = CrossTaskMiner(tmp_config, stub_llm)
        patterns = await miner.mine_patterns()
        assert len(patterns) >= 1
        assert len(patterns[0].affected_task_types) >= 2

    async def test_pattern_has_valid_fields(self, tmp_config, stub_llm):
        root = "missing configuration validation before deployment"
        for i in range(3):
            write_experience(
                _make_experience(f"exp-cod-{i + 1:03d}", TaskType.CODING, root),
                tmp_config,
            )
        for i in range(3):
            write_experience(
                _make_experience(f"exp-dev-{i + 1:03d}", TaskType.DEVOPS, root),
                tmp_config,
            )
        miner = CrossTaskMiner(tmp_config, stub_llm)
        patterns = await miner.mine_patterns()
        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_id.startswith("ctp-")
        assert p.description
        assert p.shared_root_cause
        assert p.meta_strategy
        assert p.confidence > 0
        assert len(p.source_experience_ids) >= 2


@pytest.mark.asyncio
class TestNewCapabilityDetector:
    async def test_no_gaps_when_empty(self, tmp_config, stub_llm):
        detector = NewCapabilityDetector(tmp_config, stub_llm)
        gaps = await detector.detect_gaps()
        assert gaps == []

    async def test_no_gaps_when_all_promoted(self, tmp_config, stub_llm):
        for i in range(5):
            exp = _make_experience(
                f"exp-{i + 1:03d}",
                TaskType.CODING,
                "root cause",
                promoted_to="tax-001",
            )
            write_experience(exp, tmp_config)
        detector = NewCapabilityDetector(tmp_config, stub_llm)
        gaps = await detector.detect_gaps()
        assert gaps == []

    async def test_detects_gap_for_unpromoted_type(self, tmp_config, stub_llm):
        for i in range(5):
            exp = _make_experience(
                f"exp-{i + 1:03d}",
                TaskType.DEBUGGING,
                "null reference handling",
            )
            write_experience(exp, tmp_config)
        detector = NewCapabilityDetector(tmp_config, stub_llm)
        gaps = await detector.detect_gaps()
        assert len(gaps) >= 1
        assert gaps[0].gap_id.startswith("gap-")

    async def test_gap_has_valid_fields(self, tmp_config, stub_llm):
        for i in range(5):
            write_experience(
                _make_experience(f"exp-{i + 1:03d}", TaskType.DEBUGGING, "null ref"),
                tmp_config,
            )
        detector = NewCapabilityDetector(tmp_config, stub_llm)
        gaps = await detector.detect_gaps()
        assert len(gaps) >= 1
        g = gaps[0]
        assert g.gap_type in {"failure", "frequency", "efficiency"}
        assert g.description
        assert g.suggested_action
        assert g.priority > 0


@pytest.mark.asyncio
class TestMemoryArchitect:
    async def test_empty_pool(self, tmp_config, stub_llm):
        architect = MemoryArchitect(tmp_config, stub_llm)
        recs = await architect.optimize()
        assert recs == []

    async def test_high_confidence_extract(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", TaskType.CODING, "root", confidence=0.9)
        write_experience(exp, tmp_config)
        architect = MemoryArchitect(tmp_config, stub_llm)
        recs = await architect.optimize()
        extract_recs = [r for r in recs if r.action.value == "extract"]
        assert len(extract_recs) >= 1

    async def test_low_confidence_prune(self, tmp_config, stub_llm):
        exp = _make_experience("exp-001", TaskType.CODING, "root", confidence=0.3)
        write_experience(exp, tmp_config)
        architect = MemoryArchitect(tmp_config, stub_llm)
        recs = await architect.optimize()
        prune_recs = [r for r in recs if r.action.value == "prune"]
        assert len(prune_recs) >= 1

    async def test_mixed_confidences(self, tmp_config, stub_llm):
        write_experience(
            _make_experience("exp-hi", TaskType.CODING, "root", confidence=0.9),
            tmp_config,
        )
        write_experience(
            _make_experience("exp-lo", TaskType.CODING, "root", confidence=0.3),
            tmp_config,
        )
        architect = MemoryArchitect(tmp_config, stub_llm)
        recs = await architect.optimize()
        actions = {r.action.value for r in recs}
        assert "extract" in actions
        assert "prune" in actions


@pytest.mark.asyncio
class TestLayer3Orchestrator:
    async def test_empty_pipeline(self, tmp_config, stub_llm):
        orch = Layer3Orchestrator(tmp_config, stub_llm)
        result = await orch.run_pipeline()
        assert result.cross_task_patterns == []
        assert result.capability_gaps == []
        assert result.memory_recommendations == []
        assert result.timestamp is not None

    async def test_pipeline_saves_result(self, tmp_config, stub_llm):
        for i in range(3):
            write_experience(
                _make_experience(
                    f"exp-{i + 1:03d}", TaskType.CODING, "shared root cause"
                ),
                tmp_config,
            )
        orch = Layer3Orchestrator(tmp_config, stub_llm)
        await orch.run_pipeline()

        loaded = load_latest_layer3_result(tmp_config)
        assert loaded is not None
        assert loaded.timestamp is not None

    async def test_pipeline_saves_state(self, tmp_config, stub_llm):
        orch = Layer3Orchestrator(tmp_config, stub_llm)
        await orch.run_pipeline()
        last_run = orch._load_last_run_time()
        assert last_run is not None

    async def test_full_pipeline_with_data(self, tmp_config, stub_llm):
        root = "missing validation before operation"
        for i in range(3):
            write_experience(
                _make_experience(f"exp-cod-{i + 1:03d}", TaskType.CODING, root),
                tmp_config,
            )
        for i in range(3):
            write_experience(
                _make_experience(f"exp-dev-{i + 1:03d}", TaskType.DEVOPS, root),
                tmp_config,
            )
        for i in range(3):
            write_experience(
                _make_experience(
                    f"exp-dbg-{i + 1:03d}",
                    TaskType.DEBUGGING,
                    "null reference",
                    confidence=0.9,
                ),
                tmp_config,
            )

        orch = Layer3Orchestrator(tmp_config, stub_llm)
        result = await orch.run_pipeline()

        assert result.timestamp is not None
        assert len(result.memory_recommendations) > 0

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from meta_learning.layer3.cross_task_miner import CrossTaskMiner
from meta_learning.layer3.memory_architect import MemoryArchitect
from meta_learning.layer3.new_capability import NewCapabilityDetector
from meta_learning.shared.io import save_layer3_result
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import Layer3Result, MetaLearningConfig

logger = logging.getLogger(__name__)

STATE_FILE = "layer3_state.json"


class Layer3Orchestrator:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm
        self._cross_task_miner = CrossTaskMiner(config, llm)
        self._capability_detector = NewCapabilityDetector(config, llm)
        self._memory_architect = MemoryArchitect(config, llm)

    async def run_pipeline(self) -> Layer3Result:
        logger.info("Layer 3 pipeline starting")

        logger.info("Step 1: Mining cross-task patterns")
        patterns = await self._cross_task_miner.mine_patterns()
        logger.info("Found %d cross-task patterns", len(patterns))

        logger.info("Step 2: Detecting capability gaps")
        gaps = await self._capability_detector.detect_gaps()
        logger.info("Found %d capability gaps", len(gaps))

        logger.info("Step 3: Optimizing memory architecture")
        recommendations = await self._memory_architect.optimize()
        logger.info("Generated %d memory recommendations", len(recommendations))

        result = Layer3Result(
            cross_task_patterns=patterns,
            capability_gaps=gaps,
            memory_recommendations=recommendations,
            timestamp=datetime.now(),
        )

        save_layer3_result(result, self._config)
        self._save_last_run_time()

        logger.info(
            "Layer 3 pipeline complete: patterns=%d, gaps=%d, recommendations=%d",
            len(patterns),
            len(gaps),
            len(recommendations),
        )
        return result

    def _state_path(self) -> Path:
        return Path(self._config.signal_buffer_path) / STATE_FILE

    def _load_last_run_time(self) -> datetime | None:
        state_path = self._state_path()
        if not state_path.exists():
            return None
        try:
            with open(state_path) as f:
                data = json.load(f)
            return datetime.fromisoformat(data["last_l3_run"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _save_last_run_time(self) -> None:
        state_path = self._state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if state_path.exists():
            try:
                with open(state_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

        existing["last_l3_run"] = datetime.now().isoformat()
        with open(state_path, "w") as f:
            json.dump(existing, f)

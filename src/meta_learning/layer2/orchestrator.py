from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from meta_learning.layer2.consolidate import Consolidator
from meta_learning.layer2.materialize import Materializer
from meta_learning.layer2.skill_evolve import SkillEvolver
from meta_learning.layer2.taxonomy import TaxonomyBuilder
from meta_learning.shared.io import list_pending_signals
from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    DetectionChannel,
    ExperienceCluster,
    MetaLearningConfig,
)

logger = logging.getLogger(__name__)

STATE_FILE = "layer2_state.json"


class Layer2Orchestrator:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm
        self._materializer = Materializer(config, llm)
        self._consolidator = Consolidator(config, llm)
        self._taxonomy_builder = TaxonomyBuilder(config, llm)
        self._skill_evolver = SkillEvolver(config, llm)

    def _trace_path(self) -> Path:
        return Path(self._config.workspace_root).expanduser() / "audit" / "layer2_trace.jsonl"

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        trace_path = self._trace_path()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(),
            "event": event,
            "experiment_id": self._config.experiment.experiment_id if self._config.experiment.enabled else None,
            "experiment_group": self._config.experiment.group.value if self._config.experiment.enabled else None,
            **payload,
        }
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def should_trigger(self) -> bool:
        pending = list_pending_signals(self._config)
        if not pending:
            return False

        if any(DetectionChannel.USER_CORRECTION in s.detection_channels for s in pending):
            return True

        if len(pending) >= self._config.layer2.trigger.min_pending_signals:
            return True

        last_run = self._load_last_run_time()
        if last_run is None:
            return True

        hours_since = (datetime.now() - last_run).total_seconds() / 3600
        if hours_since >= self._config.layer2.trigger.max_hours_since_last:
            return True

        return False

    async def run_pipeline(self) -> Layer2Result:
        exp_cfg = self._config.experiment
        if exp_cfg.enabled:
            logger.info(
                "Layer 2 pipeline starting (experiment=%s, group=%s)",
                exp_cfg.experiment_id,
                exp_cfg.group.value,
            )
        else:
            logger.info("Layer 2 pipeline starting")

        pending_signals = list_pending_signals(self._config)
        signal_trigger_map = {s.signal_id: s.detection_channels for s in pending_signals}

        logger.info("Step 0: Materializing signals")
        self._trace("step_start", {"step": "materialize"})
        new_experiences = await self._materializer.materialize_all_pending()
        logger.info("Materialized %d experiences", len(new_experiences))
        self._trace("step_done", {"step": "materialize", "materialized_count": len(new_experiences)})

        fast_track_exps = [
            e for e in new_experiences
            if DetectionChannel.USER_CORRECTION in (signal_trigger_map.get(e.source_signal) or [])
        ]
        fast_track_taxonomy = []
        if fast_track_exps:
            logger.info(
                "Step 1a: Fast-track %d USER_CORRECTION experiences (skip clustering)",
                len(fast_track_exps),
            )
            self._trace("step_start", {"step": "fast_track", "count": len(fast_track_exps)})
            fast_track_clusters = [
                ExperienceCluster(
                    cluster_id=f"ft-{e.id}",
                    task_type=e.task_type,
                    failure_signature_pattern=e.failure_signature or "user_correction",
                    experience_ids=[e.id],
                )
                for e in fast_track_exps
            ]
            fast_track_taxonomy = await self._taxonomy_builder.build_from_clusters(
                fast_track_clusters
            )
            logger.info("Fast-track produced %d taxonomy entries", len(fast_track_taxonomy))
            self._trace("step_done", {"step": "fast_track", "taxonomy": len(fast_track_taxonomy)})

        logger.info("Step 1: Consolidating experiences")
        self._trace("step_start", {"step": "consolidate"})
        index = await self._consolidator.consolidate()
        logger.info("Consolidation complete, %d total clusters", len(index.clusters))
        self._trace("step_done", {"step": "consolidate", "cluster_count": len(index.clusters)})

        ready_clusters = self._consolidator.get_clusters_ready_for_taxonomy()
        logger.info(
            "Step 2: Building taxonomy from %d ready clusters", len(ready_clusters)
        )
        self._trace("step_start", {"step": "taxonomy", "ready_clusters": len(ready_clusters)})
        cluster_taxonomy = await self._taxonomy_builder.build_from_clusters(
            ready_clusters
        )
        new_taxonomy_entries = fast_track_taxonomy + cluster_taxonomy
        logger.info("Created %d new taxonomy entries (%d fast-track + %d clustered)",
                     len(new_taxonomy_entries), len(fast_track_taxonomy), len(cluster_taxonomy))
        self._trace(
            "step_done",
            {"step": "taxonomy", "new_taxonomy_entries": len(new_taxonomy_entries)},
        )

        logger.info("Step 3: Evolving skills")
        self._trace("step_start", {"step": "skill_evolve"})
        skill_results = await self._skill_evolver.evolve_from_taxonomy(
            new_taxonomy_entries
        )
        logger.info("Skill evolution complete, %d results", len(skill_results))
        self._trace(
            "step_done",
            {
                "step": "skill_evolve",
                "skill_results": len(skill_results),
                "skill_updates": len([r for r in skill_results if r.action.value != "none"]),
            },
        )

        result = Layer2Result(
            materialized_count=len(new_experiences),
            total_clusters=len(index.clusters),
            new_taxonomy_entries=len(new_taxonomy_entries),
            skill_updates=len([r for r in skill_results if r.action.value != "none"]),
            timestamp=datetime.now(),
            experiment_id=exp_cfg.experiment_id if exp_cfg.enabled else None,
            experiment_group=exp_cfg.group.value if exp_cfg.enabled else None,
        )
        logger.info("Layer 2 pipeline complete: %s", result)
        self._trace(
            "pipeline_complete",
            {
                "materialized_count": result.materialized_count,
                "total_clusters": result.total_clusters,
                "new_taxonomy_entries": result.new_taxonomy_entries,
                "skill_updates": result.skill_updates,
            },
        )

        self.mark_completed(result)

        if result.new_taxonomy_entries > 0:
            self._auto_sync_nobot()

        return result

    def _auto_sync_nobot(self) -> None:
        try:
            from meta_learning.sync_nobot import sync_taxonomy_to_nobot_workspace

            import os
            nobot_workspace = os.path.expanduser("~/.deskclaw/nanobot/workspace")
            skills_path = str(Path(nobot_workspace) / "skills")

            if not Path(nobot_workspace).exists():
                logger.debug("Nanobot workspace not found at %s, skipping auto-sync", nobot_workspace)
                return

            sync_result = sync_taxonomy_to_nobot_workspace(self._config, skills_path)
            logger.info(
                "Auto-synced taxonomy to nanobot: %d entries, top-%d in SKILL.md",
                sync_result.total_entries,
                sync_result.top_n_in_skill,
            )
            self._trace("auto_sync_nobot", {"total_entries": sync_result.total_entries})
        except Exception:
            logger.warning("Auto-sync to nanobot failed (non-fatal)", exc_info=True)

    def _state_path(self) -> Path:
        return Path(self._config.signal_buffer_path) / STATE_FILE

    @staticmethod
    def load_state(config: MetaLearningConfig) -> dict:
        state_path = Path(config.signal_buffer_path) / STATE_FILE
        if not state_path.exists():
            return {"status": "idle"}
        try:
            with open(state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {"status": "idle"}

    def _load_last_run_time(self) -> datetime | None:
        data = self.load_state(self._config)
        lr = data.get("last_run")
        if lr is None:
            return None
        try:
            return datetime.fromisoformat(lr)
        except (ValueError, TypeError):
            return None

    def _write_state(self, data: dict) -> None:
        state_path = self._state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def mark_running(self) -> None:
        self._write_state({
            "status": "running",
            "started_at": datetime.now().isoformat(),
        })

    def mark_completed(self, result: Layer2Result) -> None:
        self._write_state({
            "status": "completed",
            "last_run": result.timestamp.isoformat(),
            "completed_at": result.timestamp.isoformat(),
            "result": {
                "materialized_count": result.materialized_count,
                "new_taxonomy_entries": result.new_taxonomy_entries,
                "skill_updates": result.skill_updates,
            },
        })

    def mark_failed(self, error: str) -> None:
        self._write_state({
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error,
        })


class Layer2Result:
    def __init__(
        self,
        materialized_count: int,
        total_clusters: int,
        new_taxonomy_entries: int,
        skill_updates: int,
        timestamp: datetime,
        experiment_id: str | None = None,
        experiment_group: str | None = None,
    ) -> None:
        self.materialized_count = materialized_count
        self.total_clusters = total_clusters
        self.new_taxonomy_entries = new_taxonomy_entries
        self.skill_updates = skill_updates
        self.timestamp = timestamp
        self.experiment_id = experiment_id
        self.experiment_group = experiment_group

    def __repr__(self) -> str:
        base = (
            f"Layer2Result(materialized={self.materialized_count}, "
            f"clusters={self.total_clusters}, "
            f"taxonomy={self.new_taxonomy_entries}, "
            f"skills={self.skill_updates})"
        )
        if self.experiment_id:
            return (
                f"{base[:-1]}, exp={self.experiment_id}, group={self.experiment_group})"
            )
        return base

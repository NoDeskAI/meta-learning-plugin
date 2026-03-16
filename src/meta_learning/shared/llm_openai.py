"""OpenAI-compatible LLM implementation for meta-learning pipeline.

Supports any OpenAI-compatible API (OpenAI, OneRouter, local proxies, etc.)
by configuring base_url and api_key via MetaLearningConfig.llm settings.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    CapabilityAnalysis,
    ConsolidateJudgment,
    CrossTaskAnalysis,
    Experience,
    MaterializeResult,
    MemoryAction,
    MemoryAnalysis,
    MemoryRecommendation,
    MetaLearningConfig,
    Signal,
    SkillEvolveResult,
    SkillUpdateAction,
    TaskType,
    TaxonomyEntry,
    TaxonomyExtraction,
)

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    def __init__(self, config: MetaLearningConfig) -> None:
        self._config = config
        self._base_url = os.environ.get(
            "META_LEARNING_LLM_BASE_URL",
            "https://api.openai.com/v1",
        )
        self._api_key = os.environ.get("META_LEARNING_LLM_API_KEY", "")
        self._model = config.llm.model
        self._temperature = config.llm.temperature
        self._max_tokens = config.llm.max_tokens

    async def _chat(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        raw = await self._chat(
            system + "\n\nRespond ONLY with valid JSON, no markdown fences.",
            user,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        return json.loads(cleaned)

    async def materialize_signal(
        self,
        signal: Signal,
        session_context: str,
    ) -> MaterializeResult:
        system = """You are a meta-learning analyst. Given a learning signal and session context,
extract structured experience data. Respond with JSON containing:
- scene: string (what the user was trying to do)
- failure_signature: string|null (the specific error pattern, null if none)
- root_cause: string (why the issue occurred)
- resolution: string (how it was resolved)
- meta_insight: string (transferable lesson learned)
- task_type: one of "coding"|"devops"|"writing"|"debugging"|"configuration"|"_unclassified"
"""
        user = f"""Signal ID: {signal.signal_id}
Trigger: {signal.trigger_reason}
Task: {signal.task_summary}
Keywords: {', '.join(signal.keywords)}
Error: {signal.error_snapshot or 'N/A'}
Resolution: {signal.resolution_snapshot or 'N/A'}
User feedback: {signal.user_feedback or 'N/A'}

Session context (truncated):
{session_context[:3000]}"""

        try:
            data = await self._chat_json(system, user)
            return MaterializeResult(
                scene=data.get("scene", signal.task_summary),
                failure_signature=data.get("failure_signature"),
                root_cause=data.get("root_cause", "Unknown"),
                resolution=data.get("resolution", signal.resolution_snapshot or "Unknown"),
                meta_insight=data.get("meta_insight", ""),
                task_type=_parse_task_type(data.get("task_type", "_unclassified")),
            )
        except Exception as e:
            logger.warning("materialize_signal LLM call failed: %s", e)
            return _fallback_materialize(signal)

    async def judge_same_class(
        self,
        exp_a: Experience,
        exp_b: Experience,
    ) -> ConsolidateJudgment:
        system = """You judge whether two experiences describe the same class of problem.
Respond with JSON: {"same_class": true/false, "reason": "brief explanation"}"""

        user = f"""Experience A:
- Type: {exp_a.task_type.value}
- Scene: {exp_a.scene}
- Failure: {exp_a.failure_signature or 'N/A'}
- Root cause: {exp_a.root_cause}

Experience B:
- Type: {exp_b.task_type.value}
- Scene: {exp_b.scene}
- Failure: {exp_b.failure_signature or 'N/A'}
- Root cause: {exp_b.root_cause}"""

        try:
            data = await self._chat_json(system, user)
            return ConsolidateJudgment(
                same_class=bool(data.get("same_class", False)),
                reason=data.get("reason", "LLM judgment"),
            )
        except Exception as e:
            logger.warning("judge_same_class LLM call failed: %s", e)
            return ConsolidateJudgment(same_class=False, reason=f"LLM error: {e}")

    async def extract_taxonomy(
        self,
        experiences: list[Experience],
    ) -> TaxonomyExtraction:
        system = """You extract a taxonomy entry from a cluster of similar experiences.
Respond with JSON:
- name: string (concise category name)
- trigger: string (when does this error pattern occur)
- fix_sop: string (standard fix procedure, step by step)
- prevention: string (how to prevent this in the future)
- keywords: string[] (5-10 indexing keywords for quick matching)"""

        exp_summaries = "\n\n".join(
            f"[{e.id}] {e.scene}\n  Failure: {e.failure_signature}\n  Root cause: {e.root_cause}\n  Resolution: {e.resolution}"
            for e in experiences[:10]
        )
        user = f"Experiences ({len(experiences)} total):\n{exp_summaries}"

        try:
            data = await self._chat_json(system, user)
            return TaxonomyExtraction(
                name=data.get("name", "Unknown pattern"),
                trigger=data.get("trigger", "Unknown"),
                fix_sop=data.get("fix_sop", ""),
                prevention=data.get("prevention", ""),
                keywords=data.get("keywords", []),
            )
        except Exception as e:
            logger.warning("extract_taxonomy LLM call failed: %s", e)
            sigs = [e.failure_signature for e in experiences if e.failure_signature]
            return TaxonomyExtraction(
                name=sigs[0] if sigs else "Unknown",
                trigger=sigs[0] if sigs else "Unknown",
                fix_sop="",
                prevention="",
                keywords=[],
            )

    async def evaluate_skill_update(
        self,
        taxonomy_entry: TaxonomyEntry,
        existing_skill_content: str | None,
    ) -> SkillEvolveResult:
        system = """You evaluate whether a taxonomy entry warrants creating or updating a skill.
Respond with JSON:
- action: one of "create"|"append"|"replace"|"merge"|"split"|"none"
- target_skill: string|null (skill name if updating)
- changes_description: string
- new_content: string|null (full skill content if create/replace, or appended section if append)
- version_bump: string|null (e.g. "1.0.0" for new, null for no change)"""

        user = f"""Taxonomy entry:
- ID: {taxonomy_entry.id}
- Name: {taxonomy_entry.name}
- Trigger: {taxonomy_entry.trigger}
- Fix SOP: {taxonomy_entry.fix_sop}
- Prevention: {taxonomy_entry.prevention}
- Confidence: {taxonomy_entry.confidence}
- Source experiences: {len(taxonomy_entry.source_exps)}

Existing skill content:
{existing_skill_content or 'None (no existing skill)'}"""

        try:
            data = await self._chat_json(system, user)
            action = data.get("action", "none")
            return SkillEvolveResult(
                action=_parse_skill_action(action),
                target_skill=data.get("target_skill"),
                changes_description=data.get("changes_description", ""),
                new_content=data.get("new_content"),
                version_bump=data.get("version_bump"),
            )
        except Exception as e:
            logger.warning("evaluate_skill_update LLM call failed: %s", e)
            return SkillEvolveResult(
                action=SkillUpdateAction.NONE,
                changes_description=f"LLM error: {e}",
            )

    async def analyze_cross_task_patterns(
        self,
        experience_groups: list[list[Experience]],
    ) -> list[CrossTaskAnalysis]:
        if not experience_groups:
            return []

        system = """You analyze groups of experiences from different task types to find shared root causes.
For each group, respond with JSON array of objects:
- description: string
- shared_root_cause: string
- meta_strategy: string (prevention strategy)
- confidence: float (0.0-1.0)"""

        group_texts = []
        for i, group in enumerate(experience_groups):
            types = {e.task_type.value for e in group}
            causes = [e.root_cause for e in group]
            group_texts.append(
                f"Group {i+1} (types: {', '.join(types)}):\n"
                + "\n".join(f"  - {c}" for c in causes[:5])
            )
        user = "\n\n".join(group_texts)

        try:
            data = await self._chat_json(system, user)
            results = data if isinstance(data, list) else [data]
            return [
                CrossTaskAnalysis(
                    description=r.get("description", ""),
                    shared_root_cause=r.get("shared_root_cause", ""),
                    meta_strategy=r.get("meta_strategy", ""),
                    confidence=float(r.get("confidence", 0.5)),
                )
                for r in results
            ]
        except Exception as e:
            logger.warning("analyze_cross_task_patterns LLM call failed: %s", e)
            return []

    async def analyze_capability_gaps(
        self,
        ungrouped_experiences: list[Experience],
        existing_taxonomy_keywords: list[str],
    ) -> list[CapabilityAnalysis]:
        if not ungrouped_experiences:
            return []

        system = """You identify capability gaps from experiences not covered by existing taxonomy.
Respond with JSON array of objects:
- description: string (what the gap is)
- suggested_action: string (what skill to create)
- priority: float (0.0-1.0)"""

        exp_summaries = "\n".join(
            f"- [{e.id}] {e.task_type.value}: {e.scene[:100]}"
            for e in ungrouped_experiences[:20]
        )
        user = f"""Ungrouped experiences ({len(ungrouped_experiences)} total):
{exp_summaries}

Existing taxonomy keywords: {', '.join(existing_taxonomy_keywords[:30])}"""

        try:
            data = await self._chat_json(system, user)
            results = data if isinstance(data, list) else [data]
            return [
                CapabilityAnalysis(
                    description=r.get("description", ""),
                    suggested_action=r.get("suggested_action", ""),
                    priority=float(r.get("priority", 0.5)),
                )
                for r in results
            ]
        except Exception as e:
            logger.warning("analyze_capability_gaps LLM call failed: %s", e)
            return []

    async def analyze_memory(
        self,
        high_confidence_experiences: list[Experience],
        low_confidence_experiences: list[Experience],
    ) -> MemoryAnalysis:
        recommendations: list[MemoryRecommendation] = []

        for exp in high_confidence_experiences:
            recommendations.append(
                MemoryRecommendation(
                    action=MemoryAction.EXTRACT,
                    target=exp.id,
                    reason=f"High confidence ({exp.confidence:.2f}) — promote to long-term memory",
                    content=exp.meta_insight,
                )
            )

        for exp in low_confidence_experiences:
            recommendations.append(
                MemoryRecommendation(
                    action=MemoryAction.PRUNE,
                    target=exp.id,
                    reason=f"Low confidence ({exp.confidence:.2f}) — candidate for removal",
                )
            )

        return MemoryAnalysis(recommendations=recommendations)


def _parse_task_type(value: str) -> TaskType:
    try:
        return TaskType(value)
    except ValueError:
        return TaskType.UNCLASSIFIED


def _parse_skill_action(value: str) -> SkillUpdateAction:
    try:
        return SkillUpdateAction(value.lower())
    except ValueError:
        return SkillUpdateAction.NONE


def _fallback_materialize(signal: Signal) -> MaterializeResult:
    task_type = TaskType.UNCLASSIFIED
    for tt in TaskType:
        if tt.value != "_unclassified" and tt.value in signal.task_summary.lower():
            task_type = tt
            break

    return MaterializeResult(
        scene=signal.task_summary,
        failure_signature=signal.error_snapshot,
        root_cause=f"Derived from: {signal.resolution_snapshot or 'unknown'}",
        resolution=signal.resolution_snapshot or "No resolution captured",
        meta_insight=f"Signal {signal.signal_id}: {', '.join(signal.keywords)}",
        task_type=task_type,
    )

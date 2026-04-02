"""OpenAI-compatible LLM implementation for meta-learning pipeline.

Supports any OpenAI-compatible API (OpenAI, OneRouter, local proxies, etc.)
by configuring base_url and api_key via MetaLearningConfig.llm settings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    CapabilityAnalysis,
    ConsolidateJudgment,
    CrossTaskAnalysis,
    TriggerReason,
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
            "https://llm-gateway-api.nodesk.tech/default/v1",
        )
        self._api_key = os.environ.get(
            "META_LEARNING_LLM_API_KEY",
            "nd-9f27abd1325015b7932ea4c8b54c4fdc889f0496c1f5f2b3bf24e80fd7f19895",
        )
        self._model = config.llm.model
        self._temperature = config.llm.temperature
        self._max_tokens = config.llm.max_tokens

    def _append_io_audit(self, event: str, payload: dict[str, Any]) -> None:
        audit_dir = Path(self._config.workspace_root).expanduser() / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "provider": "openai",
            "model": self._model,
            **payload,
        }
        with open(audit_dir / "llm_io_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _chat(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
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
        try:
            parsed = json.loads(cleaned)
            self._append_io_audit(
                "json_parse_ok",
                {"mode": "direct", "response_chars": len(cleaned)},
            )
            return parsed
        except json.JSONDecodeError as e:
            extracted = _extract_first_json_block(cleaned)
            if extracted is None:
                logger.warning(
                    "JSON parse failed, no JSON block found (model=%s, chars=%d). Raw:\n%s",
                    self._model, len(cleaned), cleaned[:1000],
                )
                self._append_io_audit(
                    "json_parse_failed",
                    {
                        "mode": "no_json_block",
                        "error": str(e),
                        "response_chars": len(cleaned),
                        "response_head": cleaned[:500],
                    },
                )
                raise
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError as e2:
                logger.warning(
                    "JSON parse failed on extracted block (model=%s, raw_chars=%d, extracted_chars=%d). "
                    "Raw:\n%s\n---Extracted:\n%s",
                    self._model, len(cleaned), len(extracted),
                    cleaned[:500], extracted[:500],
                )
                self._append_io_audit(
                    "json_parse_failed",
                    {
                        "mode": "extracted_block_failed",
                        "error": str(e2),
                        "response_chars": len(cleaned),
                        "response_head": cleaned[:500],
                        "extracted_head": extracted[:500],
                    },
                )
                raise
            self._append_io_audit(
                "json_parse_ok",
                {"mode": "extracted_block", "response_chars": len(cleaned)},
            )
            return parsed

    async def materialize_signal(
        self,
        signal: Signal,
        session_context: str,
    ) -> MaterializeResult:
        system = """You are a meta-learning analyst for AI agents. Given a learning signal
and session context, extract structured experience data. Respond with JSON containing:
- scene: string (what the agent was tasked with — read the FULL context carefully)
- failure_signature: string|null (the specific STRATEGY error the agent made. Focus on procedural mistakes, not parameter-level details.)
- root_cause: string (WHY the agent made this strategy error — what verification/step/constraint did it miss?)
- resolution: string (what the CORRECT strategy/procedure should be — describe the right approach)
- meta_insight: string (transferable lesson: a GENERAL rule applicable across similar tasks)
- task_type: one of "coding"|"devops"|"writing"|"debugging"|"configuration"|"customer_service"|"professional_document"|"_unclassified"

ANALYSIS GUIDELINES:
1. Lines prefixed with [agent_tool] show tool invocations with args and results.
2. Lines prefixed with [user_tool] show system-side tool results.
3. The "User feedback" or "[evaluation_feedback]" field may contain QA supervisor / LLM evaluator feedback — use this as a diagnostic clue.
4. Focus on STRATEGY-LEVEL patterns (verify before act, follow domain conventions, complete all deliverables, check requirements) not parameter-level details (specific IDs, amounts, filenames).
5. Domain-specific strategy patterns:
   - Customer service: verification discipline, policy compliance, multi-step completeness, topic switching.
   - Professional documents: requirement coverage, domain standard adherence, formatting conventions, data accuracy, citation quality.
   - Coding/DevOps: testing discipline, dependency management, error handling, edge case coverage.
"""
        if (
            signal.trigger_reason == TriggerReason.USER_CORRECTION
            and signal.user_feedback
        ):
            system += """

CRITICAL — USER CORRECTION MODE:
This signal was triggered because the user corrected the agent. Switch from
failure-analysis mode to INSTRUCTION EXTRACTION mode:

1. The user's feedback is the GROUND TRUTH — do not reinterpret it.
2. resolution MUST use the user's exact terminology and action verbs.
   Example: if user said "先备份" → resolution = "backup the file before modifying",
   NOT "verify the file state" or "read existing content".
3. If user_feedback contains a specific directive (path, command, sequence),
   preserve it verbatim in resolution.
4. If user_feedback is vague (e.g. just "不对"), use session context to infer
   what was wrong, but anchor resolution on the user's implied intent.
5. meta_insight = a directly executable rule from the user's words.
   Do NOT generalize a specific user preference into abstract advice.
"""
        if signal.trigger_reason == TriggerReason.USER_CORRECTION:
            user = f"""**User correction (GROUND TRUTH):** {signal.user_feedback or 'N/A'}

Signal ID: {signal.signal_id}
Task: {signal.task_summary}
Keywords: {', '.join(signal.keywords)}
Error: {signal.error_snapshot or 'N/A'}
Resolution: {signal.resolution_snapshot or 'N/A'}

Session context:
{session_context[:6000]}"""
        else:
            user = f"""Signal ID: {signal.signal_id}
Trigger: {signal.trigger_reason}
Task: {signal.task_summary}
Keywords: {', '.join(signal.keywords)}
Error: {signal.error_snapshot or 'N/A'}
Resolution: {signal.resolution_snapshot or 'N/A'}
User feedback: {signal.user_feedback or 'N/A'}

Session context:
{session_context[:6000]}"""

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
            self._append_io_audit(
                "materialize_fallback",
                {
                    "signal_id": signal.signal_id,
                    "reason": str(e),
                },
            )
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

CRITICAL DISTINCTION — two kinds of knowledge to extract:
1. **User persistent preferences**: specific values, paths, conventions, or constraints the user stated
   (e.g. "~/projects/", "always use feature branches", "never use bare except").
   These MUST be preserved verbatim in fix_sop and prevention — they are NOT "task-specific parameters",
   they are cross-session user rules.
2. **Strategy patterns**: general procedural steps (e.g. "verify deliverables before stopping").
   These should be concrete and actionable, not abstract advice.

When in doubt, PRESERVE the specific value from the experiences. It is far worse to abstract away
a concrete user preference into vague advice than to include a specific value that might not apply.

Respond with JSON:
- name: string (concise pattern name, e.g. "Always Create Projects Under ~/projects/", "Use Feature Branches Before Committing")
- trigger: string (when this rule applies — describe the situation type)
- fix_sop: string (a CONCRETE step-by-step checklist with SPECIFIC values from the experiences. e.g.:
  "1. Create the project under ~/projects/ using `mkdir -p ~/projects/<project_name>`\\n2. Verify with `ls ~/projects/`"
  NOT "Scan conversation history for path constraints" — the agent has no access to past conversations)
- prevention: string (a specific, directly executable rule with concrete values, e.g.
  "Always create new projects under ~/projects/, never in the current working directory"
  NOT "Check if the user has a preferred directory")
- keywords: string[] (5-10 keywords for matching)

IMPORTANT:
- fix_sop and prevention MUST contain CONCRETE ACTIONS an agent can execute RIGHT NOW without
  needing to look up past conversations or ask the user. The taxonomy IS the memory.
- Include specific paths, branch naming conventions, exception types, tool commands, etc.
  from the experiences — these are the learned knowledge, not noise to be generalized away.
- The entry must still be TRANSFERABLE: applicable to similar future tasks, not just the exact
  task in the experiences."""

        exp_summaries = "\n\n".join(
            f"[{e.id}] {e.scene}\n  Failure: {e.failure_signature}\n  Root cause: {e.root_cause}\n  Resolution: {e.resolution}"
            for e in experiences[:10]
        )
        user = f"Experiences ({len(experiences)} total):\n{exp_summaries}"

        if len(experiences) == 1:
            system += """

SINGLE EXPERIENCE — PRESERVE SPECIFICS:
There is only ONE experience. Preserve its resolution and meta_insight as closely
as possible in fix_sop and prevention. Do NOT over-generalize a single data point.
If the resolution contains a specific action verb (e.g. "backup", "create under X"),
it MUST appear in prevention.
"""

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
    summary_lower = signal.task_summary.lower()
    if any(kw in summary_lower for kw in ("customer", "airline", "reservation", "flight", "cancel")):
        task_type = TaskType.CUSTOMER_SERVICE
    else:
        for tt in TaskType:
            if tt.value != "_unclassified" and tt.value in summary_lower:
                task_type = tt
                break

    resolution = signal.resolution_snapshot or signal.user_feedback or "No resolution captured"
    root_cause = signal.user_feedback or f"Derived from: {signal.resolution_snapshot or 'unknown'}"

    return MaterializeResult(
        scene=signal.task_summary,
        failure_signature=signal.error_snapshot,
        root_cause=root_cause,
        resolution=resolution,
        meta_insight=signal.user_feedback or f"Signal {signal.signal_id}: {', '.join(signal.keywords)}",
        task_type=task_type,
    )


def _extract_first_json_block(text: str) -> str | None:
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start in starts:
        stack: list[str] = []
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                opener = stack.pop()
                if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                    break
                if not stack:
                    return text[start : idx + 1]
    return None

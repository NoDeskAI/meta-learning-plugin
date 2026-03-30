"""Adapter: convert GDPVal evaluation feedback into meta-learning signals.

Maps ClawWork LLMEvaluator output (score, feedback text) to the
``capture_signal`` interface, preserving only strategy-level patterns
that transfer across tasks within the same occupation.
"""

from __future__ import annotations

import re
from typing import Any


_RUBRIC_DIMENSIONS = {
    "completeness": "missing required deliverables, incomplete sections, or unaddressed requirements",
    "correctness": "inaccurate data, wrong citations, logical errors, or incorrect calculations",
    "quality": "poor formatting, unclear organization, or unprofessional presentation",
    "domain_standards": "failure to follow industry-specific best practices or conventions",
}

_PARAM_PATTERN = re.compile(
    r"(\$[\d,]+\.?\d*|[A-Z]{2,}-\d{3,}|paragraph \d+\.\d+|section \d+\.\d+|page \d+)",
    re.IGNORECASE,
)


def extract_dimension_failures(feedback: str) -> list[str]:
    """Extract rubric dimension failures from LLM evaluator feedback.

    Looks for named dimensions (completeness, correctness, quality,
    domain_standards) and their associated failure descriptions.
    Filters out parameter-level details (specific dollar amounts,
    section numbers, page references).
    """
    corrections: list[str] = []
    feedback_lower = feedback.lower()

    for dim, desc in _RUBRIC_DIMENSIONS.items():
        if dim in feedback_lower:
            for line in feedback.split("\n"):
                line_lower = line.lower().strip()
                if dim in line_lower and any(
                    kw in line_lower
                    for kw in [
                        "miss", "lack", "no ", "not ", "absent", "incomplete",
                        "incorrect", "wrong", "error", "fail", "poor", "weak",
                    ]
                ):
                    cleaned = _PARAM_PATTERN.sub("[REDACTED]", line.strip())
                    if len(cleaned) > 20:
                        corrections.append(f"[{dim}] {cleaned}")

    if not corrections:
        for line in feedback.split("\n"):
            line_s = line.strip()
            if any(
                kw in line_s.lower()
                for kw in [
                    "missing", "not provided", "absent", "lacks",
                    "incorrect", "should have", "failed to",
                ]
            ) and len(line_s) > 20:
                cleaned = _PARAM_PATTERN.sub("[REDACTED]", line_s)
                corrections.append(cleaned)
                if len(corrections) >= 5:
                    break

    return corrections


def build_signal_from_gdpval(
    task: dict[str, Any],
    evaluation: dict[str, Any],
    execution: dict[str, Any],
    tool_calls: list[str] | None = None,
) -> dict[str, Any]:
    """Build a dict of kwargs suitable for ``capture_signal()``.

    Parameters
    ----------
    task:
        GDPVal task dict (task_id, occupation, sector, prompt, ...).
    evaluation:
        Evaluation result dict from ``_evaluate_task`` (has_evaluation,
        evaluation_score, feedback, ...).
    execution:
        Execution info dict (iterations, tool_calls count, time_sec).
    tool_calls:
        Optional list of tool names used during execution.

    Returns
    -------
    Dict ready to splat into ``capture_signal(**result)``.
    """
    occupation = task.get("occupation", "unknown")
    sector = task.get("sector", "unknown")
    task_id = task.get("task_id", "?")
    score = evaluation.get("evaluation_score", 0.0)
    score_10 = evaluation.get("score_10", score * 10)
    feedback = evaluation.get("feedback", "")
    has_eval = evaluation.get("has_evaluation", False)

    prompt_preview = task.get("prompt", "")[:200]
    description = (
        f"[gdpval][{occupation}][{sector}] {prompt_preview} "
        f"| score={score_10:.1f}/10"
    )

    errors: list[str] = []
    if has_eval and score < 0.6:
        errors.append(
            f"Task scored {score_10:.1f}/10 (below 0.6 threshold). "
            f"Feedback: {feedback[:500]}"
        )
    elif has_eval and score < 0.8:
        errors.append(
            f"Task scored {score_10:.1f}/10 (acceptable but not good). "
            f"Feedback: {feedback[:500]}"
        )

    is_success = has_eval and score >= 0.8
    corrections = extract_dimension_failures(feedback) if (feedback and not is_success) else []

    resolution = ""
    if is_success:
        resolution = (
            f"Task completed well (score={score_10:.1f}/10). "
            f"{execution.get('iterations', 0)} iterations, "
            f"{execution.get('tool_calls', 0)} tool calls."
        )

    tools = tool_calls or []
    iters = execution.get("iterations", 0)

    return {
        "task_description": description,
        "session_id": f"gdpval_{task_id}",
        "errors_encountered": errors,
        "errors_fixed": is_success,
        "user_corrections": corrections,
        "tools_used": tools or ["shell"],
        "new_tools": [],
        "resolution_snapshot": resolution or f"score={score_10:.1f}/10, iters={iters}",
        "step_count": max(iters, 1),
    }


def build_session_from_execution(
    workspace: str,
    session_id: str,
    task: dict[str, Any],
    feedback: str,
    execution: dict[str, Any],
) -> None:
    """Write a synthetic session JSONL file for meta-learning to consume.

    Since GDPVal tasks are agent-only (no conversation partner), we
    construct a session from: task prompt → agent actions → evaluation.
    """
    import json
    from pathlib import Path

    sessions_dir = Path(workspace) / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with open(sessions_dir / f"{session_id}.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "role": "user",
            "content": f"[task_prompt] {task.get('prompt', '')[:2000]}",
        }, ensure_ascii=False) + "\n")

        f.write(json.dumps({
            "role": "assistant",
            "content": (
                f"Executed task: {execution.get('iterations', 0)} iterations, "
                f"{execution.get('tool_calls', 0)} tool calls, "
                f"{execution.get('time_sec', 0):.0f}s wall time."
            ),
        }, ensure_ascii=False) + "\n")

        if feedback:
            f.write(json.dumps({
                "role": "user",
                "content": f"[evaluation_feedback] {feedback[:3000]}",
            }, ensure_ascii=False) + "\n")

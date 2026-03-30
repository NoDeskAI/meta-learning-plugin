#!/usr/bin/env python3
"""Analyze A/B trend experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    xs = [float(i + 1) for i in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


def _extract_claw_runs(path: Path) -> dict[str, Any]:
    rows = _load_json(path)
    acc = [float(r.get("score", 0.0)) for r in rows]
    dur = [float(r.get("duration_s", 0.0)) for r in rows]
    return {
        "count": len(rows),
        "accuracy_series": acc,
        "duration_series": dur,
        "accuracy_slope": _linear_slope(acc),
        "duration_slope": _linear_slope(dur),
        "mean_accuracy": (sum(acc) / len(acc)) if acc else 0.0,
        "mean_duration": (sum(dur) / len(dur)) if dur else 0.0,
    }


def _extract_tau_runs(path: Path) -> dict[str, Any]:
    rows = _load_json(path)
    acc = [1.0 if float(r.get("reward", 0.0)) >= 1.0 else 0.0 for r in rows]
    dur = [float(r.get("duration_s", 0.0)) for r in rows]
    reward = [float(r.get("reward", 0.0)) for r in rows]
    return {
        "count": len(rows),
        "accuracy_series": acc,
        "reward_series": reward,
        "duration_series": dur,
        "accuracy_slope": _linear_slope(acc),
        "reward_slope": _linear_slope(reward),
        "duration_slope": _linear_slope(dur),
        "mean_accuracy": (sum(acc) / len(acc)) if acc else 0.0,
        "mean_reward": (sum(reward) / len(reward)) if reward else 0.0,
        "mean_duration": (sum(dur) / len(dur)) if dur else 0.0,
    }


def _trend_judgement(stats: dict[str, Any]) -> str:
    up_acc = stats.get("accuracy_slope", 0.0) > 0.0
    down_dur = stats.get("duration_slope", 0.0) < 0.0
    if up_acc and down_dur:
        return "both_improving"
    if up_acc:
        return "accuracy_only"
    if down_dur:
        return "speed_only"
    return "no_clear_gain"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trend AB experiment result directory")
    parser.add_argument("--result-root", required=True, help="Path like abtest/results/trend_ab_<runid>")
    args = parser.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    claw_a = _extract_claw_runs(root / "claw_A_runs.json")
    claw_b = _extract_claw_runs(root / "claw_B_runs.json")
    tau_a = _extract_tau_runs(root / "tau3_A_runs.json")
    tau_b = _extract_tau_runs(root / "tau3_B_runs.json")

    report = {
        "result_root": str(root),
        "claw_wfl004": {
            "A": claw_a,
            "B": claw_b,
            "A_judgement": _trend_judgement(claw_a),
            "B_judgement": _trend_judgement(claw_b),
        },
        "tau3_airline_task35": {
            "A": tau_a,
            "B": tau_b,
            "A_judgement": _trend_judgement(tau_a),
            "B_judgement": _trend_judgement(tau_b),
        },
    }
    (root / "trend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# A/B Trend Report",
        "",
        f"- result_root: `{root}`",
        "",
        "## claw-bench wfl-004",
        f"- A: acc_slope={claw_a['accuracy_slope']:.6f}, dur_slope={claw_a['duration_slope']:.6f}, judgement={report['claw_wfl004']['A_judgement']}",
        f"- B: acc_slope={claw_b['accuracy_slope']:.6f}, dur_slope={claw_b['duration_slope']:.6f}, judgement={report['claw_wfl004']['B_judgement']}",
        "",
        "## tau3 airline task35",
        f"- A: acc_slope={tau_a['accuracy_slope']:.6f}, dur_slope={tau_a['duration_slope']:.6f}, judgement={report['tau3_airline_task35']['A_judgement']}",
        f"- B: acc_slope={tau_b['accuracy_slope']:.6f}, dur_slope={tau_b['duration_slope']:.6f}, judgement={report['tau3_airline_task35']['B_judgement']}",
        "",
    ]
    (root / "trend_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {root / 'trend_report.json'}")
    print(f"Wrote: {root / 'trend_report.md'}")


if __name__ == "__main__":
    main()

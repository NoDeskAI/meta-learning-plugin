#!/usr/bin/env python3
"""A/B test analyzer for meta-learning experiment signals.

Loads all signals from signal_buffer, groups by experiment_group,
and computes per-group metrics:
  - repeat_error_rate: fraction of error_recovery signals whose keywords
    overlap with at least one earlier signal in the same group
  - task_success_rate: fraction of signals where errors were encountered
    AND subsequently fixed (trigger_reason != efficiency_anomaly and
    error_snapshot is present implies error path; no error_snapshot implies success)

Usage:
    python -m reports.ab_test_analyzer --workspace ~/.openclaw/workspace
    python -m reports.ab_test_analyzer --signal-dir /path/to/signal_buffer
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_signals(signal_dir: Path) -> list[dict]:
    if not signal_dir.exists():
        return []
    signals = []
    for p in sorted(signal_dir.glob("sig-*.yaml")):
        with open(p) as f:
            raw = yaml.safe_load(f)
        if raw:
            signals.append(raw)
    return signals


def group_signals(signals: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sig in signals:
        group = sig.get("experiment_group") or "unassigned"
        groups[group].append(sig)
    return dict(groups)


def compute_repeat_error_rate(signals: list[dict]) -> float:
    error_signals = [s for s in signals if s.get("trigger_reason") == "error_recovery"]
    if len(error_signals) < 2:
        return 0.0

    repeat_count = 0
    seen_keyword_sets: list[set[str]] = []

    for sig in error_signals:
        current_kws = set(kw.lower() for kw in sig.get("keywords", []))
        if not current_kws:
            seen_keyword_sets.append(current_kws)
            continue

        is_repeat = False
        for prev_kws in seen_keyword_sets:
            if not prev_kws:
                continue
            overlap = current_kws & prev_kws
            if len(overlap) / min(len(current_kws), len(prev_kws)) >= 0.3:
                is_repeat = True
                break

        if is_repeat:
            repeat_count += 1
        seen_keyword_sets.append(current_kws)

    return repeat_count / len(error_signals)


def compute_task_success_rate(signals: list[dict]) -> float:
    if not signals:
        return 0.0

    success_count = 0
    for sig in signals:
        trigger = sig.get("trigger_reason", "")
        has_error = sig.get("error_snapshot") is not None
        if trigger == "error_recovery" and has_error:
            success_count += 1
        elif trigger in ("user_correction", "new_tool"):
            success_count += 1
        elif trigger == "efficiency_anomaly":
            pass

    return success_count / len(signals)


def format_report(grouped: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("A/B Test Analysis Report")
    lines.append("=" * 60)

    for group, signals in sorted(grouped.items()):
        repeat_rate = compute_repeat_error_rate(signals)
        success_rate = compute_task_success_rate(signals)
        error_count = sum(
            1 for s in signals if s.get("trigger_reason") == "error_recovery"
        )

        lines.append("")
        lines.append(f"Group: {group}")
        lines.append(f"  Total signals:      {len(signals)}")
        lines.append(f"  Error recoveries:   {error_count}")
        lines.append(f"  Repeat error rate:  {repeat_rate:.2%}")
        lines.append(f"  Task success rate:  {success_rate:.2%}")

    total = sum(len(sigs) for sigs in grouped.values())
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Total signals across all groups: {total}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A/B test signal analyzer")
    parser.add_argument(
        "--signal-dir",
        type=Path,
        help="Direct path to signal_buffer directory",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace root (signal_buffer is a subdirectory)",
    )
    args = parser.parse_args(argv)

    if args.signal_dir:
        signal_dir = args.signal_dir
    elif args.workspace:
        signal_dir = args.workspace / "signal_buffer"
    else:
        signal_dir = Path.home() / ".openclaw" / "workspace" / "signal_buffer"

    signals = load_signals(signal_dir)
    if not signals:
        print(f"No signals found in {signal_dir}", file=sys.stderr)
        sys.exit(1)

    grouped = group_signals(signals)
    print(format_report(grouped))


if __name__ == "__main__":
    main()

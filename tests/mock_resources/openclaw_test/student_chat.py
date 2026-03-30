#!/usr/bin/env python3
"""Mock student_chat.py for local AB pipeline verification."""

from __future__ import annotations

import argparse
import os


def _done_count(total: int, group_a: bool, stage_formal: bool, round_idx: int) -> int:
    if group_a:
        base = total - (1 if stage_formal else 0)
        # Make formal rounds slightly improving for A.
        return min(total, max(0, base + max(0, round_idx - 1)))
    base = total - (2 if stage_formal else 1)
    return max(0, base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    gateway = os.environ.get("OPENCLAW_GATEWAY_URL", "")
    group_a = "group-a" in gateway
    stage_formal = args.num_problems >= 20
    round_idx = int(os.environ.get("MOCK_ROUND_INDEX", "1"))
    done = _done_count(args.num_problems, group_a, stage_formal, round_idx)

    print("#" * 60)
    print(f"# Summary: {done}/{args.num_problems} problems completed within turn limit")
    print("#" * 60)


if __name__ == "__main__":
    main()

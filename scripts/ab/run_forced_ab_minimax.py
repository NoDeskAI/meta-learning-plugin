#!/usr/bin/env python3
"""Run forced A/B claw-bench with per-group lock and unique output.

This avoids accidental duplicate B runs and output directory collisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Iterable

from claw_bench.cli.run import run_cmd


DEFAULT_TASKS = "wfl-004,wfl-008,wfl-009,wfl-010,wfl-013,wfl-014"


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GroupLock:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.acquired = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            old_pid = int(payload.get("pid", -1))
            if _is_pid_running(old_pid):
                raise RuntimeError(
                    f"Lock already held: {self.lock_path} (pid={old_pid})"
                )
            self.lock_path.unlink()
        payload = {
            "pid": os.getpid(),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.lock_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.acquired = True

    def release(self) -> None:
        if self.acquired and self.lock_path.exists():
            self.lock_path.unlink()
        self.acquired = False


def _parse_groups(raw: str) -> list[str]:
    groups: list[str] = []
    for part in raw.split(","):
        item = part.strip().upper()
        if not item:
            continue
        if item not in {"A", "B"}:
            raise ValueError(f"Unsupported group: {item}")
        if item not in groups:
            groups.append(item)
    if not groups:
        raise ValueError("No valid groups were provided.")
    return groups


def _build_output_path(output_root: Path, group: str, run_id: str) -> Path:
    return output_root / f"clawbench_L4_forced_{group}_minimax_{run_id}"


def _run_group(
    *,
    group: str,
    run_id: str,
    output_root: Path,
    tasks: str,
    skills: str,
    runs: int,
    parallel: int,
    timeout: int,
    agent_url: str,
    agent_name: str,
    mcp_servers: str,
) -> Path:
    output_path = _build_output_path(output_root, group, run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RuntimeError(
            f"Refuse to overwrite existing output directory: {output_path}. "
            "Use a different run-id."
        )
    run_cmd(
        framework="nanobot",
        model="minimax-m2.5",
        tasks=tasks,
        skills=skills,
        model_tier=None,
        runs=runs,
        parallel=parallel,
        timeout=timeout,
        output=output_path,
        dry_run=False,
        sandbox=False,
        tier=None,
        mcp_servers=mcp_servers or None,
        memory_modules=None,
        claw_id=None,
        agent_url=agent_url,
        agent_name=agent_name,
        resume=False,
    )
    return output_path


def _iter_group_settings(
    groups: Iterable[str],
    a_url: str,
    b_url: str,
    mcp_servers_a: str,
    mcp_servers_b: str,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for group in groups:
        if group == "A":
            result.append(
                {
                    "group": "A",
                    "agent_url": a_url,
                    "agent_name": "NobotBridgeAForcedMiniMax",
                    "mcp_servers": mcp_servers_a,
                }
            )
        else:
            result.append(
                {
                    "group": "B",
                    "agent_url": b_url,
                    "agent_name": "NobotBridgeBForcedMiniMax",
                    "mcp_servers": mcp_servers_b,
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forced A/B minimax benchmark safely")
    parser.add_argument("--groups", default="A,B", help="Comma-separated groups to run (A,B)")
    parser.add_argument("--run-id", default="", help="Custom run id; default uses UTC timestamp")
    parser.add_argument("--output-root", default="abtest/results", help="Result root directory")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Task list for claw-bench")
    parser.add_argument("--skills", default="vanilla")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--agent-url-a", default="http://127.0.0.1:5063")
    parser.add_argument("--agent-url-b", default="http://127.0.0.1:5064")
    parser.add_argument("--lock-root", default="/tmp/nobot_ab_minimax/.locks")
    parser.add_argument(
        "--mcp-servers-a",
        default="meta_learning",
        help="Comma-separated MCP server names declared for group A",
    )
    parser.add_argument(
        "--mcp-servers-b",
        default="",
        help="Comma-separated MCP server names declared for group B",
    )
    args = parser.parse_args()

    groups = _parse_groups(args.groups)
    run_id = args.run_id.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    lock_root = Path(args.lock_root).expanduser().resolve()

    print(f"[run] run_id={run_id}")
    print(f"[run] groups={','.join(groups)}")
    print(f"[run] output_root={output_root}")
    print(f"[run] lock_root={lock_root}")
    print(f"[run] tasks={args.tasks}")

    for item in _iter_group_settings(
        groups,
        args.agent_url_a,
        args.agent_url_b,
        args.mcp_servers_a.strip(),
        args.mcp_servers_b.strip(),
    ):
        group = item["group"]
        lock = GroupLock(lock_root / f"{group}.lock")
        output_path = _build_output_path(output_root, group, run_id)
        print(
            f"[group:{group}] output={output_path} "
            f"mcp_servers={item.get('mcp_servers', '') or '(none)'}"
        )
        lock.acquire()
        try:
            _run_group(
                group=group,
                run_id=run_id,
                output_root=output_root,
                tasks=args.tasks,
                skills=args.skills,
                runs=args.runs,
                parallel=args.parallel,
                timeout=args.timeout,
                agent_url=item["agent_url"],
                agent_name=item["agent_name"],
                mcp_servers=item.get("mcp_servers", ""),
            )
        finally:
            lock.release()
        print(f"[group:{group}] done")


if __name__ == "__main__":
    main()

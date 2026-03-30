#!/usr/bin/env python3
"""Prepare Nobot A/B config files from local base config.

This script avoids committing secrets by reading the local base config and
emitting A/B variants under /tmp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_agents_md() -> str:
    return """# Benchmark MCP Policy

When solving benchmark tasks:
1. Call `quick_think` before risky operations.
2. After task completion, call `capture_signal` with:
   - `errors_encountered` / `errors_fixed`
   - `resolution_snapshot`
   - `image_snapshots` when applicable
   - `step_count`
3. If enough signals accumulated, call `run_layer2(force=true)` periodically.
"""


def _parse_disabled_groups(raw: str) -> set[str]:
    normalized = raw.strip().upper()
    if not normalized or normalized == "NONE":
        return set()
    parts = {part.strip() for part in normalized.split(",") if part.strip()}
    allowed = {"A", "B"}
    unknown = parts - allowed
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported group in --disable-meta-learning-for: {joined}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Nobot A/B configs")
    parser.add_argument(
        "--base-config",
        default=str(Path.home() / ".deskclaw/nanobot/config.json"),
        help="Base Nobot config path",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/nobot_ab",
        help="Output directory for generated A/B files",
    )
    parser.add_argument("--mcp-port-a", type=int, default=18811)
    parser.add_argument("--mcp-port-b", type=int, default=18812)
    parser.add_argument("--bridge-port-a", type=int, default=5061)
    parser.add_argument("--bridge-port-b", type=int, default=5062)
    parser.add_argument(
        "--disable-meta-learning-for",
        default="none",
        help="Comma-separated groups to disable meta-learning MCP for (A,B,none)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    disabled_groups = _parse_disabled_groups(args.disable_meta_learning_for)

    with open(Path(args.base_config).expanduser().resolve(), encoding="utf-8") as f:
        base = json.load(f)

    summary_rows: list[dict[str, str]] = []
    for group, mcp_port, bridge_port in [
        ("A", args.mcp_port_a, args.bridge_port_a),
        ("B", args.mcp_port_b, args.bridge_port_b),
    ]:
        cfg = json.loads(json.dumps(base))
        tools = cfg.setdefault("tools", {})
        mcp_servers = tools.setdefault("mcp_servers", {})
        if group in disabled_groups:
            mcp_servers.pop("meta_learning", None)
            meta_enabled = False
            mcp_url = "-"
        else:
            mcp_servers["meta_learning"] = {
                "type": "streamableHttp",
                "url": f"http://127.0.0.1:{mcp_port}/mcp",
                "tool_timeout": 45,
                "enabled_tools": ["*"],
            }
            meta_enabled = True
            mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        cfg_path = out_dir / f"nobot.config.{group}.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        workspace = out_dir / group / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "AGENTS.md").write_text(build_agents_md(), encoding="utf-8")

        bridge_meta = {
            "group": group,
            "bridge_port": bridge_port,
            "bridge_api_key": "nobot-bridge-key",
            "config_path": str(cfg_path),
            "workspace": str(workspace),
            "mcp_url": mcp_url,
            "meta_learning_enabled": meta_enabled,
        }
        with open(out_dir / f"bridge.{group}.json", "w", encoding="utf-8") as f:
            json.dump(bridge_meta, f, ensure_ascii=False, indent=2)
        summary_rows.append(
            {
                "group": group,
                "meta_learning_enabled": str(meta_enabled),
                "bridge_port": str(bridge_port),
                "config_path": str(cfg_path),
                "mcp_url": mcp_url,
            }
        )

    print(f"Prepared A/B configs in: {out_dir}")
    print("AB config summary:")
    for row in summary_rows:
        print(
            f"  Group {row['group']}: "
            f"meta_learning_enabled={row['meta_learning_enabled']}, "
            f"bridge_port={row['bridge_port']}, "
            f"mcp_url={row['mcp_url']}, "
            f"config={row['config_path']}"
        )


if __name__ == "__main__":
    main()

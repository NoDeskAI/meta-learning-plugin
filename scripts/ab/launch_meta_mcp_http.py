#!/usr/bin/env python3
"""Launch meta-learning MCP server with streamable HTTP transport.

Usage:
  venv/bin/python3 scripts/ab/launch_meta_mcp_http.py --config abtest/config.meta-learning.A.yaml --workspace /tmp/nobot_ab/A/workspace --port 18811
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch meta-learning MCP (streamable-http)")
    parser.add_argument("--config", required=True, help="Path to meta-learning YAML config")
    parser.add_argument("--workspace", required=True, help="Workspace root path")
    parser.add_argument("--port", type=int, required=True, help="HTTP port for MCP server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    args = parser.parse_args()

    config_path = str(Path(args.config).expanduser().resolve())
    workspace_path = str(Path(args.workspace).expanduser().resolve())

    os.environ["META_LEARNING_CONFIG"] = config_path
    os.environ["META_LEARNING_WORKSPACE"] = workspace_path
    os.environ["FASTMCP_HOST"] = args.host
    os.environ["FASTMCP_PORT"] = str(args.port)

    # Import AFTER env setup so FastMCP picks up host/port settings.
    from meta_learning.mcp_server import mcp

    # Explicitly set runtime settings to avoid environment loading edge cases.
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

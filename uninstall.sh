#!/usr/bin/env bash
set -euo pipefail

DESKCLAW_ROOT="${HOME}/.deskclaw"
NANOBOT_CONFIG="${DESKCLAW_ROOT}/nanobot/config.json"
WORKSPACE="${DESKCLAW_ROOT}/nanobot/workspace"
DATA_DIR="${WORKSPACE}/meta-learning-data"
SKILLS_DIR="${WORKSPACE}/skills/meta-learning"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "This will uninstall the meta-learning MCP plugin."
echo ""
echo "The following will be removed:"
echo "  - MCP registration in nanobot/config.json"
echo "  - Data directory: ${DATA_DIR}"
echo "  - Skills directory: ${SKILLS_DIR}"
echo "  - Virtual environment: ${SCRIPT_DIR}/.venv"
echo ""
read -rp "Continue? [y/N] " confirm
if [[ "${confirm}" != [yY] ]]; then
    echo "Aborted."
    exit 0
fi

# ---------- remove MCP registration from config.json ----------

if [ -f "${NANOBOT_CONFIG}" ]; then
    python3 -c "
import json, sys

path = '${NANOBOT_CONFIG}'
with open(path) as f:
    cfg = json.load(f)

servers = cfg.get('tools', {}).get('mcp_servers', {})
if 'meta-learning' in servers:
    del servers['meta-learning']
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print('Removed meta-learning from', path)
else:
    print('meta-learning not found in', path, '— skipping.')
" 2>/dev/null || echo "WARNING: Failed to patch ${NANOBOT_CONFIG}. Remove \"meta-learning\" from tools.mcp_servers manually."
fi

# ---------- remove data & skills ----------

if [ -d "${DATA_DIR}" ]; then
    rm -rf "${DATA_DIR}"
    echo "Removed ${DATA_DIR}"
fi

if [ -d "${SKILLS_DIR}" ]; then
    rm -rf "${SKILLS_DIR}"
    echo "Removed ${SKILLS_DIR}"
fi

# ---------- remove venv ----------

if [ -d "${SCRIPT_DIR}/.venv" ]; then
    rm -rf "${SCRIPT_DIR}/.venv"
    echo "Removed ${SCRIPT_DIR}/.venv"
fi

# ---------- done ----------

echo ""
echo "============================================================"
echo "  Uninstall complete!"
echo "============================================================"
echo ""
echo "Restart DeskClaw / Nobot to finish."
echo ""
echo "To also remove the plugin source code, run:"
echo "  rm -rf ${SCRIPT_DIR}"

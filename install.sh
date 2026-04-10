#!/usr/bin/env bash
set -euo pipefail

DESKCLAW_ROOT="${HOME}/.deskclaw"
WORKSPACE="${DESKCLAW_ROOT}/nanobot/workspace"
DATA_DIR="${WORKSPACE}/meta-learning-data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- preflight checks ----------

if [ ! -d "${DESKCLAW_ROOT}" ]; then
    echo "ERROR: DeskClaw not found at ${DESKCLAW_ROOT}"
    echo "Please install DeskClaw / Nobot first."
    exit 1
fi

python_bin=""
for cmd in python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(sys.version_info[:2] >= (3,12))' 2>/dev/null || echo "False")
        if [ "$ver" = "True" ]; then
            python_bin="$cmd"
            break
        fi
    fi
done

if [ -z "$python_bin" ]; then
    echo "ERROR: Python >= 3.12 not found. Please install Python 3.12+."
    exit 1
fi

echo "Using Python: $($python_bin --version)"

# ---------- venv & dependencies ----------

cd "${SCRIPT_DIR}"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &>/dev/null; then
        uv venv .venv --python "$python_bin"
    else
        "$python_bin" -m venv .venv
    fi
fi

echo "Installing dependencies..."
if command -v uv &>/dev/null; then
    uv pip install --python .venv/bin/python -e ".[mcp]"
else
    .venv/bin/pip install -e ".[mcp]"
fi

# ---------- data directory & config ----------

mkdir -p "${DATA_DIR}"

if [ ! -f "${DATA_DIR}/config.yaml" ]; then
    cp "${SCRIPT_DIR}/config.deskclaw.yaml" "${DATA_DIR}/config.yaml"
    echo "Created config at ${DATA_DIR}/config.yaml"
else
    echo "Config already exists at ${DATA_DIR}/config.yaml — skipping."
fi

# ---------- bootstrap SKILL.md ----------

SKILL_DIR="${WORKSPACE}/skills/meta-learning"
SKILL_PATH="${SKILL_DIR}/SKILL.md"

if [ ! -f "${SKILL_PATH}" ]; then
    mkdir -p "${SKILL_DIR}"
    "${SCRIPT_DIR}/.venv/bin/python" -c \
        "from meta_learning.sync_nobot import render_bootstrap_skill_md; print(render_bootstrap_skill_md(), end='')" \
        > "${SKILL_PATH}"
    echo "Created bootstrap SKILL.md at ${SKILL_PATH}"
else
    echo "SKILL.md already exists at ${SKILL_PATH} — skipping."
fi

# ---------- inject AGENTS.md instructions ----------

AGENTS_MD="${WORKSPACE}/AGENTS.md"

"${SCRIPT_DIR}/.venv/bin/python" -c "
from meta_learning.sync_nobot import inject_agents_md
changed = inject_agents_md('${AGENTS_MD}')
if changed:
    print('Injected meta-learning instructions into ${AGENTS_MD}')
else:
    print('AGENTS.md already up to date — skipping.')
"

# ---------- register MCP in config.json ----------

NANOBOT_CONFIG="${DESKCLAW_ROOT}/nanobot/config.json"
MCP_CMD="${SCRIPT_DIR}/.venv/bin/meta-learning-mcp"

"${SCRIPT_DIR}/.venv/bin/python" -c "
import json, sys
from pathlib import Path

config_path = Path('${NANOBOT_CONFIG}')
if not config_path.exists():
    print('WARNING: ${NANOBOT_CONFIG} not found — skipping MCP registration.')
    print('You will need to manually add the meta-learning MCP entry.')
    sys.exit(0)

with open(config_path, encoding='utf-8') as f:
    cfg = json.load(f)

servers = cfg.setdefault('tools', {}).setdefault('mcp_servers', {})
new_entry = {
    'type': 'stdio',
    'command': '${MCP_CMD}',
    'args': [],
    'tool_timeout': 120,
    'env': {
        'META_LEARNING_WORKSPACE': '${DATA_DIR}',
        'META_LEARNING_CONFIG': '${DATA_DIR}/config.yaml',
        'META_LEARNING_SESSIONS_ROOT': '${WORKSPACE}/sessions',
    },
}

if 'meta-learning' in servers and servers['meta-learning'] == new_entry:
    print('MCP already registered in config.json — skipping.')
    sys.exit(0)

servers['meta-learning'] = new_entry
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write('\n')
print('Registered meta-learning MCP in ${NANOBOT_CONFIG}')
"

# ---------- done ----------

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
echo "Restart DeskClaw / Nobot to activate the plugin."

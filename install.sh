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

# ---------- output MCP registration ----------

VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
MCP_CMD="${SCRIPT_DIR}/.venv/bin/meta-learning-mcp"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
echo "Add the following to ~/.deskclaw/nanobot/config.json"
echo "under \"tools\" -> \"mcp_servers\":"
echo ""
cat <<EOF
"meta-learning": {
  "type": "stdio",
  "command": "${MCP_CMD}",
  "args": [],
  "tool_timeout": 120,
  "env": {
    "META_LEARNING_WORKSPACE": "${DATA_DIR}",
    "META_LEARNING_CONFIG": "${DATA_DIR}/config.yaml",
    "META_LEARNING_SESSIONS_ROOT": "${WORKSPACE}/sessions"
  }
}
EOF
echo ""
echo "Then restart DeskClaw / Nobot to activate the plugin."

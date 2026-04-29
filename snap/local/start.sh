#!/bin/bash
set -e

# ── Directories ───────────────────────────────────────────────────────────────
SOCKET_DIR="$SNAP_DATA/package-run/mind"
mkdir -p "$SOCKET_DIR"
mkdir -p "$SNAP_COMMON/logs"
mkdir -p "$SNAP_COMMON/prometheus"

# ── Default config ─────────────────────────────────────────────────────────────
# Copy bundled model_config.yaml to writable SNAP_COMMON on first install
if [ ! -f "$SNAP_COMMON/model_config.yaml" ]; then
    cp "$SNAP/model_config.yaml" "$SNAP_COMMON/model_config.yaml"
fi

# ── Environment ───────────────────────────────────────────────────────────────
export SOCKET_PATH="$SOCKET_DIR/web.sock"
export DATABASE_URL="sqlite+aiosqlite:///$SNAP_COMMON/mind.db"
export LOG_DIR="$SNAP_COMMON/logs"
export PROMETHEUS_MULTIPROC_DIR="$SNAP_COMMON/prometheus"
export JWT_SECRET="${JWT_SECRET:-change-me-in-production}"

# ── Run ────────────────────────────────────────────────────────────────────────
# Run from $SNAP so relative paths (app/static, app/templates) resolve correctly
cd "$SNAP"

# Redirect stderr to a log file so startup crashes are always captured
exec python3 -u -m uvicorn app.main:app \
    --uds "$SOCKET_PATH" \
    --root-path /mind \
    2>> "$SNAP_COMMON/logs/startup.log"

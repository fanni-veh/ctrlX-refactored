#!/bin/bash
set -e

# ── Directories ───────────────────────────────────────────────────────────────
SOCKET_DIR="$SNAP_DATA/package-run/mind"
mkdir -p "$SOCKET_DIR"
mkdir -p "$SNAP_COMMON/logs"
mkdir -p "$SNAP_COMMON/prometheus"

# ── Environment ───────────────────────────────────────────────────────────────
export SOCKET_PATH="$SOCKET_DIR/web.sock"
export DATABASE_URL="sqlite+aiosqlite:///$SNAP_COMMON/mind.db"
export LOG_DIR="$SNAP_COMMON/logs"
export PROMETHEUS_MULTIPROC_DIR="$SNAP_COMMON/prometheus"

# Run from $SNAP so relative paths (app/static, app/templates) resolve correctly
cd "$SNAP"

exec python3 -m uvicorn app.main:app --uds "$SOCKET_PATH" --root-path /mind

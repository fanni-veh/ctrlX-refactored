#!/bin/bash
# MIND snap startup script
# Runs inside the snap sandbox — $SNAP (read-only) and $SNAP_COMMON (writable) are set by snapd.
set -e

# ── Writable directories ───────────────────────────────────────────────────
mkdir -p "$SNAP_COMMON/logs"
mkdir -p "$SNAP_COMMON/prometheus_metrics"

# ── Seed model_config.yaml to SNAP_COMMON on first install ────────────────
if [ ! -f "$SNAP_COMMON/model_config.yaml" ]; then
    cp "$SNAP/model_config.yaml" "$SNAP_COMMON/model_config.yaml"
fi

# ── Environment variables read by pydantic-settings ───────────────────────
export LOG_DIR="$SNAP_COMMON/logs"
export PROMETHEUS_MULTIPROC_DIR="$SNAP_COMMON/prometheus_metrics"

# Default to SQLite stored in SNAP_COMMON if no DATABASE_URL is provided.
# For PostgreSQL set DATABASE_URL via:
#   snap set mind database-url="postgresql+asyncpg://user:pass@host:5432/dbname"
if [ -z "$DATABASE_URL" ]; then
    export DATABASE_URL="sqlite+aiosqlite:///$SNAP_COMMON/mind.db"
fi

# JWT_SECRET must be provided via snap configuration:
#   snap set mind jwt-secret="<your-secret>"
if [ -z "$JWT_SECRET" ]; then
    echo "ERROR: JWT_SECRET is not set. Run: snap set mind jwt-secret=<secret>" >&2
    exit 1
fi

# ── Change to the snap root so relative paths (templates, static) resolve ─
cd "$SNAP"

# ── Run database migrations ───────────────────────────────────────────────
"$SNAP/bin/python3" -m alembic upgrade head

# ── Start Gunicorn with Uvicorn workers ───────────────────────────────────
exec "$SNAP/bin/gunicorn" app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --log-level info

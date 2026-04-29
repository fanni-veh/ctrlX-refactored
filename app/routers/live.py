"""
Live motor capture API — prefix: /api/live

Wraps the LiveCapture collector so a running ctrlX drive can be used as a
data source instead of (or alongside) imported ZIP files.

Endpoints:
  GET    /api/live/status    Current session status (idle if no session)
  POST   /api/live/connect   Test ctrlX credentials without starting capture
  POST   /api/live/start     Start a new capture session for an application
  POST   /api/live/stop      Stop capture and persist cycles to the database
  DELETE /api/live           Discard capture without saving
"""

import asyncio
import datetime as dt
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app import database, models
from app.scripts.auth import get_effective_user
from app.scripts.database_helper import add_cycles
from app.live_capture.collector import CompletedCycle, LiveCapture

logger = logging.getLogger("live_capture")
router = APIRouter(prefix="/api/live", tags=["Live Capture"])

# Key used to store the active LiveCapture instance on app.state
_CAPTURE_KEY = "_live_capture"


def _get_capture(request: Request) -> Optional[LiveCapture]:
    return getattr(request.app.state, _CAPTURE_KEY, None)


def _set_capture(request: Request, capture: Optional[LiveCapture]):
    setattr(request.app.state, _CAPTURE_KEY, capture)


# ── Request models ────────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    ctrlx_ip: str = "192.168.1.1"
    username: str = "boschrexroth"
    password: str
    axis_name: str = "Axis_1"


class StartRequest(BaseModel):
    application_id: int
    ctrlx_ip: str = "192.168.1.1"
    username: str = "boschrexroth"
    password: str
    axis_name: str = "Axis_1"
    # Classification for ALL cycles in this session
    classification: str = "unknown"   # "good" / "bad" / "unknown"
    # Motion parameters — mirror TestParameters in MACSIoTSingleMotor
    target_pos: float    = 90.0
    home_pos: float      = 0.0
    velocity: float      = 3600.0
    acceleration: float  = 3600.0
    time_on_s: float     = 2.0
    time_pause_s: float  = 0.5
    load_torque: float   = 0.0
    spike_torque: float  = 0.0
    spike_start_s: float = 0.0
    spike_length_s: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status(
    request: Request,
    user: models.User = Depends(get_effective_user),
):
    """Return the current capture session status, or an idle stub if none is active."""
    capture = _get_capture(request)
    if capture is None:
        return {"connected": False, "running": False, "cycles_buffered": 0}
    return {"connected": True, **capture.get_status()}


@router.post("/connect")
async def connect(
    body: ConnectRequest,
    user: models.User = Depends(get_effective_user),
):
    """
    Test authentication with the ctrlX CORE without starting a capture session.
    Returns 401 if credentials are wrong or the device is unreachable.
    """
    capture = LiveCapture(
        ctrlx_ip=body.ctrlx_ip,
        username=body.username,
        password=body.password,
        axis_name=body.axis_name,
        classification="unknown",
    )
    ok = await asyncio.get_event_loop().run_in_executor(None, capture.authenticate)
    capture.dl.close()
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not authenticate with ctrlX CORE — check IP and credentials",
        )
    return {"status": "ok", "message": "Authentication successful"}


@router.post("/start")
async def start_capture(
    body: StartRequest,
    request: Request,
    db: AsyncSession = Depends(database.get_db),
    user: models.User = Depends(get_effective_user),
):
    """
    Authenticate, power on the motor and begin capturing cycles.
    Only one session at a time is supported.
    The motor will automatically run repeated home→target→hold→home cycles
    until /api/live/stop is called.
    """
    existing = _get_capture(request)
    if existing and existing._running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A capture session is already running. Call /api/live/stop first.",
        )

    # Verify the application exists and the user has access to it
    stmt = select(models.Application).where(
        models.Application.id == body.application_id,
        ~models.Application.disabled,
    )
    if not user.isAdmin():
        stmt = stmt.join(models.Motor).where(
            models.Motor.users.any(models.User.id == user.id)
        )
    application = (await db.scalars(stmt)).one_or_none()
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application {body.application_id} not found or access denied",
        )

    capture = LiveCapture(
        ctrlx_ip=body.ctrlx_ip,
        username=body.username,
        password=body.password,
        axis_name=body.axis_name,
        classification=body.classification,
        target_pos=body.target_pos,
        home_pos=body.home_pos,
        velocity=body.velocity,
        acceleration=body.acceleration,
        time_on_s=body.time_on_s,
        time_pause_s=body.time_pause_s,
        load_torque=body.load_torque,
        spike_torque=body.spike_torque,
        spike_start_s=body.spike_start_s,
        spike_length_s=body.spike_length_s,
    )

    # Authenticate first
    ok = await asyncio.get_event_loop().run_in_executor(None, capture.authenticate)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not authenticate with ctrlX CORE",
        )

    # Start threads (returns quickly — actual motion runs in the background)
    await asyncio.get_event_loop().run_in_executor(None, capture.start)
    _set_capture(request, capture)

    logger.info(
        "Live capture started for application %s by user %s",
        body.application_id, user.email,
    )
    return {
        "status":         "started",
        "application_id": body.application_id,
        "classification": body.classification,
    }


@router.post("/stop")
async def stop_capture(
    request: Request,
    application_id: int = Query(..., description="Application ID to save cycles under"),
    db: AsyncSession = Depends(database.get_db)
):
    """
    Stop the running capture session and save all buffered cycles to the database.
    After this call the cycles are available as regular CycleData records and can
    be used for training via POST /api/train?application_id=<id>.
    """
    capture = _get_capture(request)
    if capture is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active capture session",
        )

    # Stop threads (blocks until they exit)
    await asyncio.get_event_loop().run_in_executor(None, capture.stop)
    _set_capture(request, None)

    completed = capture.completed_cycles
    if not completed:
        return {"cycles_saved": 0, "points_saved": 0}

    cycles_saved, points_saved = await _flush_cycles_to_db(completed, application_id, db)
    logger.info(
        "Saved %d live cycles (%d points) for application %s",
        cycles_saved, points_saved, application_id
    )
    return {"cycles_saved": cycles_saved, "points_saved": points_saved}


@router.delete("")
async def discard_capture(
    request: Request,
    user: models.User = Depends(get_effective_user),
):
    """Stop the active capture session without saving any data."""
    capture = _get_capture(request)
    if capture:
        await asyncio.get_event_loop().run_in_executor(None, capture.stop)
        _set_capture(request, None)
    return {"status": "discarded"}


# ── Database persistence ──────────────────────────────────────────────────────

async def _flush_cycles_to_db(
    completed: list[CompletedCycle],
    application_id: int,
    db: AsyncSession,
) -> tuple[int, int]:
    """
    Convert buffered CompletedCycle objects into CycleData + MeasuringPoint rows.

    Each CompletedCycle becomes one CycleData record.
    Each CyclicSample becomes rows in a long-format DataFrame (timestamp/field/value)
    which is then passed to the existing add_cycles() helper — the same path used
    by the ZIP import, so all downstream pipeline code works unchanged.

    Returns (cycles_saved, points_saved).
    """
    orm_cycles: list[tuple[models.CycleData, list]] = []

    for cc in completed:
        classification = models.CycleData.Classification.from_str(cc.classification)
        cycle = models.CycleData(
            application_id=application_id,
            classification=classification,
            testCycleCounter=cc.cycle_counter,
            cycleConfig=cc.cycle_config,
            driveConfig=cc.drive_config,
        )
        db.add(cycle)
        orm_cycles.append((cycle, cc.samples))

    # Flush to get auto-generated cycle IDs without committing yet
    await db.flush()

    # Build long-format DataFrames and attach them for add_cycles()
    cycles_for_add: list[models.CycleData] = []
    for cycle, samples in orm_cycles:
        rows = []
        for s in samples:
            ts = s.timestamp.replace(tzinfo=dt.timezone.utc)
            for field, value in [
                ("act_current",         s.act_current),
                ("act_torque",          s.act_torque),
                ("act_velocity",        s.act_velocity),
                ("act_following_error", s.act_following_error),
                ("act_position",        s.act_position),
                ("cmd_position",        s.cmd_position),
                ("cmd_velocity",        s.cmd_velocity),
                ("temp_motor",          s.temp_motor),
                ("temp_power_stage",    s.temp_power_stage),
            ]:
                if value is not None:
                    rows.append({"timestamp": ts, "field": field, "value": float(value)})

        cycle.dataframe = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["timestamp", "field", "value"]
        )
        cycles_for_add.append(cycle)

    await db.commit()
    points_saved = await add_cycles(db, cycles_for_add, logger=logger)
    return len(cycles_for_add), points_saved

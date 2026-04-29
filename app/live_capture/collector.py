"""
Live motor data collector.

Adapted from MACSIoTSingleMotor — runs an automatic test sequence on a ctrlX
CORE drive (home → target → hold → home) and buffers each completed cycle in
memory.  When stop() is called the caller can read completed_cycles and flush
them to the database via the /api/live/stop endpoint.

Threading model (mirrors the original MACS .mc state machines):
  _cyclic_thread  — reads sensor data every 5 ms  (PDO_PERIOD_S)
  _toggle_thread  — reads slow temperature data every 1 s
  _test_thread    — drives the motion sequence; one pass = one cycle
"""

import dataclasses
import datetime
import logging
import threading
import time
from typing import Optional

from app.live_capture.datalayer import CtrlXDataLayer

logger = logging.getLogger("live_capture.collector")

# Timing constants (seconds) — mirror MACS timer periods
PDO_PERIOD_S    = 0.005   # 5 ms cyclic data collection
TOGGLE_PERIOD_S = 1.0     # 1 s slow-data (temperature) read

# EtherCAT path for the averaged current value — {AXIS} is replaced at init
_CURRENT_NODE_TMPL = (
    "fieldbuses/ethercat/master/instances/ethercatmaster"
    "/realtime_data/input/data/IDX/{AXIS}"
    "/CSP_Inputs.Current_actual_value_averaged"
)

# Drive state bits (mirrors USER_PARAM(PARAM_0_STATE) in the original)
STA_POWER_ON    = 0x10
STA_POWER_OFF   = 0x20
STA_MOVING      = 0x01
STA_IN_POSITION = 0x02
STA_ERROR       = 0x40


@dataclasses.dataclass
class CyclicSample:
    """One timestamped sensor snapshot, maps 1-to-1 with a MeasuringPoint row."""
    timestamp: datetime.datetime
    act_current: Optional[float]
    act_torque: Optional[float]
    act_velocity: Optional[float]
    act_following_error: Optional[float]
    act_position: Optional[float]
    cmd_position: Optional[float]
    cmd_velocity: Optional[float]
    temp_motor: Optional[float]
    temp_power_stage: Optional[float]


@dataclasses.dataclass
class CompletedCycle:
    """One finished home→target→home cycle ready to persist to the database."""
    classification: str      # "good" / "bad" / "unknown"
    cycle_counter: int
    cycle_config: dict       # motion parameters (maps to CycleData.cycleConfig)
    drive_config: dict       # drive identity / gains (maps to CycleData.driveConfig)
    samples: list            # list[CyclicSample]


class LiveCapture:
    """
    Captures real-time motor cycles from a ctrlX CORE drive.

    Typical usage:
        capture = LiveCapture(ctrlx_ip="192.168.1.1", username="boschrexroth",
                              password="...", axis_name="Axis_1",
                              classification="good", target_pos=90.0, ...)
        ok = capture.authenticate()   # test credentials
        capture.start()               # powers on motor, begins test sequence
        time.sleep(30)
        capture.stop()                # powers off motor, joins threads
        cycles = capture.completed_cycles   # list[CompletedCycle]
    """

    def __init__(
        self,
        ctrlx_ip: str,
        username: str,
        password: str,
        axis_name: str,
        classification: str,
        target_pos: float = 90.0,
        home_pos: float = 0.0,
        velocity: float = 3600.0,
        acceleration: float = 3600.0,
        time_on_s: float = 2.0,
        time_pause_s: float = 0.5,
        load_torque: float = 0.0,
        spike_torque: float = 0.0,
        spike_start_s: float = 0.0,
        spike_length_s: float = 0.0,
    ):
        self.axis_name = axis_name
        self.classification = classification
        self.dl = CtrlXDataLayer(ctrlx_ip, username, password)

        self._axis_base = f"motion/axs/{axis_name}"
        self._current_node = _CURRENT_NODE_TMPL.replace("{AXIS}", axis_name)

        # Test parameters (mirror TestParameters in the original)
        self._target_pos   = target_pos
        self._home_pos     = home_pos
        self._velocity     = velocity
        self._acceleration = acceleration
        self._time_on_s    = time_on_s
        self._time_pause_s = time_pause_s
        self._load_torque  = load_torque
        self._spike_torque = spike_torque
        self._spike_start_s  = spike_start_s
        self._spike_length_s = spike_length_s

        # Drive identity read at startup
        self._rated_torque = 1

        # Thread synchronisation
        self._lock    = threading.Lock()
        self._running = False
        self._state   = STA_POWER_OFF
        self._error_info: str = ""

        # Real-time data updated by _cyclic_thread
        self._act_position: float = 0.0
        self._act_velocity: float = 0.0
        self._act_current_mA: float = 0.0
        self._act_follow_err: float = 0.0
        self._temp_motor: float = 0.0
        self._temp_stage: float = 0.0

        # Samples being recorded for the current cycle
        self._recording = False
        self._current_samples: list[CyclicSample] = []

        # Finished cycles waiting to be flushed to the database
        self._completed_cycles: list[CompletedCycle] = []
        self._cycle_counter = 0

        self._threads: list[threading.Thread] = []
        self._test_thread: Optional[threading.Thread] = None

    # ── Public interface ─────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Test credentials and obtain an access token. Returns True on success."""
        return bool(self.dl.get_token())

    def start(self):
        """
        Read drive identity, start background threads, and begin the test sequence.
        Assumes authenticate() has already been called successfully.
        """
        logger.info("LiveCapture: starting (axis=%s, classification=%s)",
                    self.axis_name, self.classification)
        self._running = True
        self._state   = STA_POWER_OFF

        # Read rated torque for current→torque conversion (best-effort)
        try:
            state = self._read_safe(f"{self._axis_base}/state/values/actual", {})
            rated = state.get("ratedTorque", 1) if isinstance(state, dict) else 1
            self._rated_torque = rated if rated and rated > 0 else 1
        except Exception:
            self._rated_torque = 1

        t_cyclic = threading.Thread(target=self._cyclic_thread, daemon=True, name="lc-cyclic")
        t_toggle = threading.Thread(target=self._toggle_thread, daemon=True, name="lc-toggle")
        self._threads = [t_cyclic, t_toggle]
        for t in self._threads:
            t.start()

        self._test_thread = threading.Thread(target=self._test_sequence, daemon=True, name="lc-test")
        self._test_thread.start()

    def stop(self):
        """Stop the test sequence and all background threads, then power off the motor."""
        logger.info("LiveCapture: stopping")
        self._running  = False
        self._recording = False
        with self._lock:
            self._current_samples = []  # discard any partial cycle in progress

        if self._test_thread:
            self._test_thread.join(timeout=10.0)
        for t in self._threads:
            t.join(timeout=2.0)

        try:
            # Abort any in-progress motion first
            self.dl.write_node(f"motion/axs/{self.axis_name}/cmd/abort", True, "bool8")
            # Wait for axis to leave DISCRETE_MOTION (poll state)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                state_val = self._read_safe(f"{self._axis_base}/state/values/actual", {})
                if isinstance(state_val, dict):
                    axs_state = state_val.get("axsState", "")
                    if axs_state not in ("DISCRETE_MOTION", "CONTINUOUS_MOTION", "HOMING"):
                        break
                time.sleep(0.1)
            self.dl.write_node(f"motion/axs/{self.axis_name}/cmd/power", False, "bool8")
        except Exception as e:
            logger.warning("LiveCapture: power-off sequence failed: %s", e)

        self.dl.close()
        logger.info("LiveCapture: stopped — %d cycles collected", len(self._completed_cycles))

    @property
    def completed_cycles(self) -> list[CompletedCycle]:
        return self._completed_cycles

    def get_status(self) -> dict:
        return {
            "running":         self._running,
            "cycles_buffered": len(self._completed_cycles),
            "cycle_counter":   self._cycle_counter,
            "state":           hex(self._state),
            "error":           self._error_info,
            "act_position":    round(self._act_position, 3),
            "act_velocity":    round(self._act_velocity, 1),
            "temp_motor_c":    round(self._temp_motor / 10, 1),
        }

    # ── Background threads ───────────────────────────────────────────────────

    def _cyclic_thread(self):
        """
        Reads sensor data every 5 ms.
        Mirrors UpdateCyclicDataArray() called from SIG_PERIOD_5MS in the original.
        """
        while self._running:
            t0 = time.monotonic()
            try:
                state = self._read_safe(f"{self._axis_base}/state/values/actual", {})
                if isinstance(state, dict):
                    self._act_position = float(state.get("actualPos", 0))
                    self._act_velocity = float(state.get("actualVel", 0))

                raw = self._read_safe(self._current_node, 0)
                self._act_current_mA = float(raw) if raw else 0.0

                if self._recording:
                    sample = CyclicSample(
                        timestamp=datetime.datetime.utcnow(),
                        act_current=self._act_current_mA,
                        act_torque=self._act_current_mA * self._rated_torque / 1000,
                        act_velocity=self._act_velocity,
                        act_following_error=self._act_follow_err,
                        act_position=self._act_position,
                        cmd_position=self._target_pos,
                        cmd_velocity=self._velocity,
                        temp_motor=self._temp_motor,
                        temp_power_stage=self._temp_stage,
                    )
                    with self._lock:
                        self._current_samples.append(sample)
            except Exception as e:
                logger.debug("Cyclic read error: %s", e)

            elapsed = time.monotonic() - t0
            remainder = PDO_PERIOD_S - elapsed
            if remainder > 0:
                time.sleep(remainder)

    def _toggle_thread(self):
        """
        Reads slow-changing temperature data every 1 s.
        Mirrors SIG_X_TOGGLE in the original.
        """
        while self._running:
            try:
                raw_motor = self._read_safe(
                    f"{self._axis_base}/state/values/actual/tempMotor", 0)
                raw_stage = self._read_safe(
                    f"{self._axis_base}/state/values/actual/tempPowerStage", 0)
                self._temp_motor = float(raw_motor) if raw_motor else 0.0
                self._temp_stage = float(raw_stage) if raw_stage else 0.0
            except Exception:
                pass
            time.sleep(TOGGLE_PERIOD_S)

    # ── Test sequence ────────────────────────────────────────────────────────

    def _test_sequence(self):
        """
        Runs repeated home→target→hold→home cycles until stop() is called.
        Each completed cycle is appended to _completed_cycles.

        Mirrors TestSequenceMachine from the original, simplified for one motor.
        """
        # Power on — fatal if this fails
        try:
            self.dl.write_node(f"motion/axs/{self.axis_name}/cmd/power", True, "bool8")
            self._state = (self._state & ~STA_POWER_OFF) | STA_POWER_ON
            time.sleep(1.0)
        except Exception as e:
            self._error_info = str(e)
            self._running = False
            logger.error("LiveCapture: power-on failed: %s", e)
            return

        # Homing (set current position as zero) — non-fatal, motor runs from wherever it is
        try:
            self.dl.write_node(
                f"motion/axs/{self.axis_name}/cmd/set-position",
                {"axsPos": "0", "buffered": False},
            )
            time.sleep(0.5)
        except Exception as e:
            logger.warning("LiveCapture: homing skipped (%s) — continuing from current position", e)

        while self._running:
            self._cycle_counter += 1
            logger.info("LiveCapture: cycle %d (%s)", self._cycle_counter, self.classification)

            cycle_config = {
                "targetPos":      self._target_pos,
                "homePos":        self._home_pos,
                "velocity":       self._velocity,
                "rampUp":         self._acceleration,
                "rampDown":       self._acceleration,
                "timeOn":         self._time_on_s,
                "timePause":      self._time_pause_s,
                "loadTorque":     self._load_torque,
                "spikeTorque":    self._spike_torque,
                "spikeStartTime": self._spike_start_s,
                "spikeLength":    self._spike_length_s,
            }
            drive_config = {
                "ratedTorque": self._rated_torque,
                "axisName":    self.axis_name,
            }

            # Start recording this cycle
            with self._lock:
                self._current_samples = []
            self._recording = True

            # Move to target position
            try:
                self.dl.write_node(
                    f"motion/axs/{self.axis_name}/cmd/pos-abs",
                    {
                        "axsPos":   str(self._target_pos),
                        "buffered": False,
                        "lim": {
                            "vel":    str(self._velocity),
                            "acc":    str(self._acceleration),
                            "dec":    str(self._acceleration),
                            "jrkAcc": "0",
                            "jrkDec": "0",
                        },
                    },
                )
                self._state = (self._state & ~STA_IN_POSITION) | STA_MOVING
            except Exception as e:
                self._recording = False
                self._error_info = str(e)
                logger.error("LiveCapture: move-to-target failed: %s", e)
                break

            self._wait_position(self._target_pos, timeout=15.0)

            # Hold at target
            time.sleep(self._time_on_s)

            # Return home
            try:
                self.dl.write_node(
                    f"motion/axs/{self.axis_name}/cmd/pos-abs",
                    {
                        "axsPos":   str(self._home_pos),
                        "buffered": False,
                        "lim": {
                            "vel":    str(self._velocity),
                            "acc":    str(self._acceleration),
                            "dec":    str(self._acceleration),
                            "jrkAcc": "0",
                            "jrkDec": "0",
                        },
                    },
                )
            except Exception as e:
                self._recording = False
                self._error_info = str(e)
                logger.error("LiveCapture: return-home failed: %s", e)
                break

            self._wait_position(self._home_pos, timeout=15.0)
            self._recording = False

            # Store completed cycle
            with self._lock:
                samples_snapshot = list(self._current_samples)

            min_samples = max(5, int(self._time_on_s / 0.1 / 2))  # at least half the hold time at ~100ms/read
            if len(samples_snapshot) >= min_samples:
                self._completed_cycles.append(CompletedCycle(
                    classification=self.classification,
                    cycle_counter=self._cycle_counter,
                    cycle_config=cycle_config,
                    drive_config=drive_config,
                    samples=samples_snapshot,
                ))
                logger.info("LiveCapture: cycle %d done — %d samples buffered",
                            self._cycle_counter, len(samples_snapshot))
            else:
                logger.warning("LiveCapture: cycle %d discarded — only %d samples (min %d)",
                               self._cycle_counter, len(samples_snapshot), min_samples)

            time.sleep(self._time_pause_s)

    def _wait_position(self, target_deg: float, timeout: float = 15.0, tolerance: float = 1.0):
        """
        Block until the motor reaches target_deg ± tolerance, or timeout expires.
        Mirrors _wait_for_position() / SIG_TARGET_REACHED in the original.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self._running:
            if abs(self._act_position - target_deg) <= tolerance:
                return
            time.sleep(0.05)
        if self._running:
            logger.warning("LiveCapture: timeout waiting for %.1f°", target_deg)

    # ── Data layer helpers ───────────────────────────────────────────────────

    def _read_safe(self, path: str, default=0):
        try:
            return self.dl.read_node(path).get("value", default)
        except Exception:
            return default

import logging
from pathlib import Path
import threading
from typing import Dict, Callable, Optional
import yaml

from app import Classification, RunTask
from app.scripts.tsa_logging import create_logger


class CycleBatchCollector:
    """
    Collects incoming cycles per Application ID into batches.
    After a configurable idle time without new cycles, triggers an action callback with the batch statistics.
    Tracks train cycles (good/bad) and predict cycles (unknown) separately.
    """

    def __init__(self, on_idle_callback: Callable[[int, RunTask, int], None], logger: Optional[logging.Logger] = None):
        """
        Args:
            on_idle_callback: Callback function called when no more cycles arrive.
                              Receives (application_id, train_cycles, predict_cycles)
            logger: Logger instance for logging.
        """
        self._lock = threading.Lock()
        self._timers: Dict[int, Dict[str, threading.Timer]] = {}  # application_id -> task -> timer
        self._train_counts: Dict[int, int] = {}  # application_id -> count of train cycles (good/bad)
        self._predict_counts: Dict[int, int] = {}  # application_id -> count of predict cycles (unknown)
        self._on_idle_callback = on_idle_callback
        self._logger = logger or create_logger(__name__, output_file="mqtt_client")

    def on_cycle_received(self, application_id: int, classification: Classification) -> None:
        """
        Called when a new cycle is received for an application.
        Resets the timer for this application.

        Args:
            application_id: The application ID that received a cycle.
            classification: The classification of the cycle (good/bad/unknown).
        """
        with self._lock:
            # First, check if rule exists for this application to get wait time
            rules_file = Path(__file__).parent / "rule_engine" / "rules" / f"{application_id}.yml"

            if rules_file.exists():
                with open(rules_file, "r") as f:
                    rules_cfg = yaml.safe_load(f)
                    action_delay_time_seconds = rules_cfg.get("action_delay_time_seconds", 5)

                # Cancel existing timer if any
                task = RunTask.PREDICTION if classification == Classification.UNKNOWN else RunTask.TRAIN
                if application_id in self._timers and task.value in self._timers[application_id]:
                    self._timers[application_id][task.value].cancel()

                # Update cycle counts based on classification
                if classification == Classification.UNKNOWN:
                    self._predict_counts[application_id] = self._predict_counts.get(application_id, 0) + 1
                else:  # good or bad -> train cycle
                    self._train_counts[application_id] = self._train_counts.get(application_id, 0) + 1

                # Start new timer
                timer = threading.Timer(action_delay_time_seconds, self._on_timer_expired, args=[application_id, task])
                timer.daemon = True
                timer.start()
                self._timers.setdefault(application_id, {})[task.value] = timer

                train = self._train_counts.get(application_id, 0)
                predict = self._predict_counts.get(application_id, 0)
                self._logger.debug("Cycle received for app_id: %d (train=%d, predict=%d), starting %0.1fs timer",
                                   application_id, train, predict, action_delay_time_seconds)

    def _on_timer_expired(self, application_id: int, task: RunTask) -> None:
        """
        Called when the timer expires for an application (no more cycles received).
        """
        with self._lock:
            # Clean up timer reference
            if application_id in self._timers and task.value in self._timers[application_id]:
                del self._timers[application_id][task.value]
            cycles = self._train_counts.pop(application_id, 0) if task == RunTask.TRAIN else self._predict_counts.pop(application_id, 0)

        self._logger.info("No more %s cycles for app_id: %d after %d cycles, triggering action",
                          task.name.lower(), application_id, cycles)
        try:
            self._on_idle_callback(application_id, task, cycles)
        except Exception:
            self._logger.exception("Error in on_idle_callback for app_id: %d", application_id)

    def shutdown(self) -> None:
        """
        Cancel all pending timers and clean up.
        """
        with self._lock:
            for timers in self._timers.values():
                for timer in timers.values():
                    timer.cancel()
            self._timers.clear()
            self._train_counts.clear()
            self._predict_counts.clear()

from datetime import datetime, timedelta


class Trigger:
    def match(self, state: dict, now: datetime) -> bool:
        raise NotImplementedError


class MinNewCycles(Trigger):
    """Trigger when new_cycles >= value (minimum threshold)."""

    def __init__(self, value: int):
        self.value = value

    def match(self, state, now):
        return state.get("new_cycles", 0) >= self.value


class Interval(Trigger):
    """Trigger when interval has passed since last run AND at least 1 new cycle exists."""

    def __init__(self, value: timedelta):
        self.interval = value

    def match(self, state, now):
        # Must have at least 1 new cycle
        new_cycles = state.get("new_cycles", 0)
        if new_cycles < 1:
            return False

        last_run = state.get("last_run")
        if last_run is None:
            # No previous run - trigger immediately
            return True

        # Parse datetime if it's a string
        if isinstance(last_run, str):
            last_run = datetime.fromisoformat(last_run)

        return now - last_run >= self.interval


TRIGGERS = {
    "min_new_cycles": MinNewCycles,
    "interval": Interval,
}

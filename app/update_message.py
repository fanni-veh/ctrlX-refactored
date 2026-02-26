from dataclasses import dataclass
from app import RunState, RunTask


@dataclass
class UpdateMessage():
    status: RunState
    task: RunTask
    metadata: dict = None

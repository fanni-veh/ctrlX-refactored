import json
from pathlib import Path
from uuid import UUID
from fastapi import UploadFile
from uuid_extensions import uuid7str
import os
from app import models
from app.scripts.tsa_logging import create_run_logger
from app.config import setting
import logging
from enum import Enum


class TskParam:

    class TaskName(Enum):
        PREVIEW = "preview"
        TRAIN = "train"
        PREDICT = "prediction"

    run_id: str = None
    user_id: int = None
    file: bytes = None
    application_id: int = None
    train_run_id: str = None  # Specific run_id to use for predication
    application: models.Application = None
    output_path: str = None
    log_level: int = None
    dataset_name: str = None
    using_cycle_ids: set[int] = set()
    filters: list[str] = []
    logger: logging.Logger
    pipeline_config: dict = None  # Could be used to pass custom config for the pipeline

    def __init__(self, user_id: int, task_for_filename: TaskName, task_id: str = None) -> None:
        if task_id:
            try:
                UUID(task_id)
            except ValueError:
                raise ValueError("Run id invalid")
        self.run_id = task_id or uuid7str()
        self.user_id = user_id
        self.output_path = os.path.join(setting.log_dir, self.run_id)
        self.log_level = logging.DEBUG
        self.logger = create_run_logger(self.run_id, task_for_filename.value)
        self.logger.info("User %d started %s task with run_id: %s", user_id, task_for_filename.value, self.run_id)

    def __del__(self):
        self.finish()

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "hasFile": True if self.file else False,
            "hasApplication": True if self.application else False,
            "application_id": self.application_id,
            "train_run_id": self.train_run_id,
            "log_level": self.log_level,
            "dataset_name": self.dataset_name,
            "using_cycle_ids": list(self.using_cycle_ids),
            "filters": self.filters,
        }

    def read_file(self, filename: str) -> str:
        """
        Read a file from the output path securely.
        """
        # Sanitize and validate filename
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            raise ValueError("Invalid filename.")

        base_path = os.path.abspath(self.output_path)
        file_path = os.path.join(base_path, safe_filename)

        # Final safety check
        if not file_path.startswith(base_path + os.sep):
            raise ValueError("Access to this path is not allowed.")

        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                raw_data = file.read()
                if raw_data:
                    self.file = raw_data
        return self.file

    def write_file(self, file: UploadFile) -> float:
        """
        Write an uploaded file to the output path securely.
        Returns the size of the file in MB.
        """
        # Validate and sanitize filename
        name = os.path.basename(file.filename or "")
        if not name or name in (".", "..") or name != file.filename:
            raise ValueError("Invalid filename.")

        base = Path(self.output_path).resolve()
        base.mkdir(parents=True, exist_ok=True)

        path = (base / name).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            raise ValueError("Access to this path is not allowed.")

        chunks = []
        total = 0
        with open(path, "wb", buffering=1024 * 1024) as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                chunks.append(chunk)
                total += len(chunk)

        self.file = b"".join(chunks)
        size_mb = total / (1024 * 1024)
        self.logger.debug("Received file: %s with %.2f MB", name, size_mb)
        return self.file

    def finish(self):
        # Close logger
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def dump(self):
        base = Path(self.output_path).resolve()
        targetPath = (base / "tskparam.json").resolve()

        with open(targetPath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

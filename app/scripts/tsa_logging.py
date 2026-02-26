from logging.handlers import RotatingFileHandler
import logging
import os
from pathlib import Path
import time
from typing import Optional
from app.config import setting


def create_logger(name, level=logging.DEBUG, output_file: Optional[str] = None, maxFileSizeMb=50, backup_count=5) -> logging.Logger:
    """
    Create a logger that writes to a rotating file.

    :param name: the name of the logger
    :param level: the logging level
    :param output_file: the output file name (without path and extension)
    :type output_file: Optional[str]
    :param maxFileSizeMb: the maximum size of the log file in megabytes before rotation
    :param backup_count: the number of backup files to keep
    :return: a configured Logger instance
    :rtype: Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not os.path.exists(setting.log_dir):
        os.makedirs(setting.log_dir)

    target_name = output_file if output_file else name
    file_handler = RotatingFileHandler(
        f"{setting.log_dir}/{target_name}.log",
        maxBytes=maxFileSizeMb * 1024 * 1024,
        backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log_level = getattr(logging, setting.log_level.upper(), logging.INFO)
    file_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    return logger


def create_run_logger(task_id: str, task_for_filename: str, level=logging.DEBUG) -> logging.Logger:
    """
    Securely set up a file logger for a run.
    Prevents path traversal or injection via task_id.
    """
    base = Path(setting.log_dir).resolve()
    # Sanitize and validate task_id
    safe_task_id = os.path.basename(task_id)
    if not safe_task_id or safe_task_id in (".", "..") or safe_task_id != task_id:
        raise ValueError("Invalid task_id.")

    path = (base / safe_task_id).resolve()
    # Ensure path is inside log_dir
    try:
        path.relative_to(base)
    except ValueError:
        raise ValueError("Access to this path is not allowed.")

    path.mkdir(parents=True, exist_ok=True)
    # Create a unique filename with timestamp
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    filename = path / f"{current_time}_{task_for_filename}.log"

    logger = logging.getLogger(str(filename))
    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger

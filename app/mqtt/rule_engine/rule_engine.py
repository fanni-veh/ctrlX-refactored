from collections import defaultdict
import threading
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import httpx
from cachetools import TTLCache
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session
from app import RunTask
from app.mqtt.rule_engine.rule_evaluator import evaluate
from app.config import setting
import yaml
from app.models import Application, CycleData, Run, cycle_run
from app.mqtt.sync_database import sync_session_manager
from app.scripts.tsa_logging import create_logger

# ==================== Configuration ====================
logger = create_logger('rule_engine', level=logging.DEBUG, output_file='rule_engine', maxFileSizeMb=5, backup_count=5)

API_TOKEN: Dict[str, Any] = {
    'token': None,
    'created': 0
}

# Cache for application metrics from DB (application_id -> metrics dict)
# This cache persists between auto_action calls to avoid repeated DB queries
# TTL: 1 hour (3600 seconds)
_metrics_cache = TTLCache(maxsize=1000, ttl=3600)
_metrics_cache_lock = threading.Lock()

# Locks per application_id to ensure only one thread can trigger auto action at a time
_trigger_locks: Dict[int, threading.Lock] = defaultdict(threading.Lock)


def trigger_auto_action(application_id: int, task: RunTask, cycles: int) -> None:
    """
    Triggered when no more cycles are received for an application after the idle period.
    Runs in separate thread, performs DB queries and HTTP calls.
    """
    if application_id < 1 or cycles < 1 or not task:
        logger.warning("Invalid parameters: app_id=%d, task=%s, cycles=%d", application_id, task, cycles)
        return
    # Get or create lock for this application_id
    with _trigger_locks[application_id]:
        logger.info("Auto action triggered for app_id: %d, task=%s, cycles=%d", application_id, task.name.lower(), cycles)
        try:
            # Build initial metrics from batch counts + cached data
            metrics = _get_cached_metrics(application_id)
            if not metrics:
                with sync_session_manager.session() as db:
                    metrics = _fetch_application_metrics(db, application_id)
            if not metrics:
                logger.warning("No metrics found in DB for app_id: %d", application_id)
                return

            # Update cached metrics with new batch counts
            metrics = metrics.copy()
            key = "unused_predict_cycles" if task == RunTask.PREDICTION else "unused_train_cycles"
            metrics[key] = metrics.get(key, 0) + cycles
            _update_cached_metrics(application_id, metrics)
            logger.debug("Metrics updated for app_id %d: %s", application_id, metrics)

            trigger_result = _should_trigger_action(application_id, task, metrics)
            if trigger_result:
                matched_rule, user_id = trigger_result
                filter_used = matched_rule.get("filter_used_cycles", True)
                _trigger_rest_call(application_id, task, user_id=user_id, filter_used=filter_used)
                metrics[key] = 0
                _update_cached_metrics(application_id, metrics)
                logger.debug("Reset %s to 0 in cache for app_id %d after triggering REST call", key, application_id)
        except Exception as e:
            logger.exception("Error in auto_action for app_id %d: %s", application_id, e)


def _get_cached_metrics(application_id: int) -> Optional[Dict[str, Any]]:
    """Get cached metrics for an application (thread-safe) or None"""
    with _metrics_cache_lock:
        return _metrics_cache.get(application_id)


def _update_cached_metrics(application_id: int, metrics: Dict[str, Any]) -> None:
    """Update cached metrics for an application (thread-safe)."""
    with _metrics_cache_lock:
        _metrics_cache[application_id] = metrics


def _fetch_application_metrics(db: Session, application_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch metrics for an application from the database.
    Includes cycle counts, run information, and cycles used in runs.

    Args:
        db: Database session.
        application_id: The application ID.

    Returns:
        Dictionary with application metrics or None if not found.
    """
    try:
        start = time.time()
        # Get application info
        app: Optional[Application] = db.get(Application, application_id)
        if not app:
            logger.warning("Application %d not found", application_id)
            return None

        # Get last successful train run
        last_runs = db.execute(
            select(Run.task, func.max(Run.time_created))
            .where(
                Run.application_id == application_id,
                Run.task.in_([Run.Task.TRAIN, Run.Task.PREDICTION]),
                Run.state == Run.State.SUCCESS,
                ~Run.disabled
            )
            .group_by(Run.task)
        ).all()

        last_train_time = last_predict_time = None
        for task, time_created in last_runs:
            if task == Run.Task.TRAIN:
                last_train_time = time_created
            else:
                last_predict_time = time_created

        # Count unused predict cycles (unknown classification, not in any non-disabled run)
        used_cycle_ids = (
            select(cycle_run.c.cycle_id)
            .join(Run, Run.id == cycle_run.c.run_id)
            .where(~Run.disabled)
        )
        cycle_counts = db.execute(
            select(
                func.sum(case((CycleData.classification == CycleData.Classification.UNKNOWN, 1), else_=0)),
                func.sum(case((CycleData.classification.in_([CycleData.Classification.GOOD, CycleData.Classification.BAD]), 1), else_=0))
            )
            .where(
                CycleData.application_id == application_id,
                ~CycleData.disabled,
                ~CycleData.id.in_(used_cycle_ids)
            )
        ).one()

        end = time.time()
        logger.debug("Fetching metrics from DB for app_id %d took %.4f seconds", application_id, end - start)
        return {
            "unused_predict_cycles": cycle_counts[0] or 0,
            "unused_train_cycles": cycle_counts[1] or 0,
            "last_train_run": last_train_time.isoformat() if last_train_time else None,
            "last_predict_run": last_predict_time.isoformat() if last_predict_time else None
        }
    except Exception as e:
        logger.exception("Error fetching metrics for app_id %d: %s", application_id, e)
        return None


def _should_trigger_action(application_id: int, action_type: RunTask, metrics: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Check if the action should be triggered based on rules configuration.
    Uses first-match-wins logic when multiple rules are configured.

    Returns:
        Tuple of (matched_rule_dict, user_id) if action should be triggered, else None.
    """

    rules_cfg = get_rules(application_id)
    if not rules_cfg or "actions" not in rules_cfg:
        logger.warning("Invalid or empty rules configuration for app_id %d", application_id)
        return None

    user_id = rules_cfg.get("user_id")
    if not user_id:
        logger.warning("No user_id found in rules for app_id %d", application_id)
        return None

    action_cfg = rules_cfg.get("actions", {}).get(action_type.value)
    if not action_cfg:
        logger.debug("No '%s' action configured for app_id %d", action_type.value, application_id)
        return None

    # Normalize to list of rules (support both old single-rule and new list format)
    rules_list = action_cfg if isinstance(action_cfg, list) else [action_cfg]
    now = datetime.now(timezone.utc)
    is_predict = action_type == RunTask.PREDICTION
    last_run_key = "last_predict_run" if is_predict else "last_train_run"
    unused_cycles_key = "unused_predict_cycles" if is_predict else "unused_train_cycles"
    cycle_count = metrics.get(unused_cycles_key, 0)

    state = {
        "application_id": application_id,
        "new_cycles": cycle_count,
        "last_run": metrics.get(last_run_key),
    }
    # First-match-wins: evaluate rules in order
    for rule in rules_list:
        rule_name = rule.get("name", "unnamed")

        wrapper = {
            "logic": rule.get("logic", "AND"),
            "triggers": rule.get("triggers", [])
        }

        result = evaluate(wrapper, state, now, logger)
        # logger.debug("Rule '%s' evaluation for app_id %d, action '%s': %s (cycles=%d)", rule_name, application_id, action_type.value, result, cycle_count)

        if result:
            logger.info("Rule '%s' matched for app_id %d, action '%s'", rule_name, application_id, action_type.value)
            return rule, user_id

    return None


def _get_auth_cookie() -> Dict[str, str]:
    """Get or refresh the access token for REST calls."""
    token_age = time.time() - float(API_TOKEN.get('created', 0))
    token_missing = not API_TOKEN.get('token')
    if token_missing or token_age >= (setting.access_token_expire_minutes - 1) * 60:
        logger.debug("Refreshing access token for auto_action")
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"http://{setting.tsa_hostname}/login",
                json={"email": setting.mqtt_service_username, "password": setting.mqtt_service_password}
            )
            response.raise_for_status()
            token_data = response.json()
            API_TOKEN['token'] = token_data.get('access_token')
            API_TOKEN['created'] = time.time()
            logger.debug("Access token refreshed successfully")
    return {'access_token': API_TOKEN['token']}


def _trigger_rest_call(application_id: int, task: RunTask, user_id: int, filter_used: bool = True) -> None:
    """
    Trigger a REST call to start the action(predict/train).

    Args:
        application_id: The application ID.
        task: The type of action(predict/train).
        user_id: The user ID from the rules configuration.
        filter_used: Whether to filter out cycles already used in previous runs.
    """

    # Prepare REST call
    action_endpoint = "train" if task == RunTask.TRAIN else "predict"

    cookies = _get_auth_cookie()
    headers = {
        'X-On-Behalf-Of': str(user_id)
    }
    url = f"http://{setting.tsa_hostname}/api/{action_endpoint}"
    params = {"application_id": application_id,
              "filter_used": filter_used}

    logger.info("Triggering %s action for app_id %d via REST call to %s (On-Behalf-Of=%s, filter_used=%s)",
                action_endpoint, application_id, url, user_id, filter_used)
    with httpx.Client(timeout=10.0) as client:
        response = client.post(url, params=params, cookies=cookies, headers=headers)
        response.raise_for_status()


def _get_rules_file_path(application_id: int) -> Path:
    return Path(__file__).parent / "rules" / f"{application_id}.yml"


def get_rules(application_id: int) -> dict:
    rules_file = _get_rules_file_path(application_id)
    if not rules_file.exists():
        # Rule file does not exist, return default empty rules
        return {
            "actions": {
                "train": [],
                "prediction": []
            }
        }

    return yaml.safe_load(rules_file.read_text())


def write_rules(application_id: int, rules: Optional[dict], user_id: int) -> None:
    rules_file = _get_rules_file_path(application_id)
    empty = rules is None or all(
        len(rules.get("actions", {}).get(action_type, [])) == 0
        for action_type in ["train", "prediction"]
    )
    if empty:
        if rules_file.exists():
            try:
                rules_file.unlink()
                logger.info("Rules file deleted for app_id %d by user %d", application_id, user_id)
            except Exception:
                logger.exception("Failed to delete rules file for app_id %d", application_id)
                raise
        return

    _validate_rules_structure(rules)
    try:
        rules['user_id'] = user_id
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        with rules_file.open('w') as f:
            yaml.safe_dump(rules, f, default_flow_style=False)
        logger.info("Rules file written for app_id %d at %s by user %d", application_id, rules_file, user_id)
    except Exception as e:
        logger.exception("Failed to write rules file for app_id %d: %s", application_id, e)
        raise


def _validate_rules_structure(rules: dict) -> None:
    """Validate rules structure."""
    if not isinstance(rules, dict):
        raise ValueError("Rules must be a dictionary")

    if "actions" not in rules or not isinstance(rules["actions"], dict):
        raise ValueError("actions is required and must be a dictionary")

    # Validate each action type
    for action_type in ["train", "prediction"]:
        if action_type in rules["actions"]:
            action_rules = rules["actions"][action_type]
            if not isinstance(action_rules, list):
                raise ValueError(f"actions.{action_type} must be a list")

            for rule in action_rules:
                if not isinstance(rule, dict):
                    raise ValueError(f"Each rule in actions.{action_type} must be a dictionary")

                if "triggers" not in rule or not isinstance(rule["triggers"], list):
                    raise ValueError("Each rule must have a 'triggers' list")

                if "logic" in rule and rule["logic"] not in ["AND", "OR"]:
                    raise ValueError("logic must be 'AND' or 'OR'")

                # Validate triggers
                for trigger in rule["triggers"]:
                    if not isinstance(trigger, dict) or len(trigger) != 1:
                        raise ValueError("Each trigger must be a dictionary with exactly one key-value pair")

                    trigger_type = list(trigger.keys())[0]
                    if trigger_type not in ["min_new_cycles", "interval"]:
                        raise ValueError(f"Invalid trigger type: {trigger_type}")

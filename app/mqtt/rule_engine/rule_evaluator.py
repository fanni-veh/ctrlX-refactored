import logging
from app.mqtt.rule_engine.triggers import TRIGGERS
from datetime import datetime
from datetime import timedelta


def evaluate(node: dict, state: dict, now: datetime, logger: logging.Logger) -> bool:
    """
    Recursively evaluate a trigger node.
    node can be:
    - simple trigger as {trigger_type: value}, e.g., {"min_new_cycles": 10}
    - logic node with 'logic' (AND/OR) and list of triggers
    """
    # Check if this is a simple trigger (single key-value pair, not "logic" or "triggers")
    if "logic" not in node and "triggers" not in node:
        # Simple trigger: {trigger_type: value}
        if len(node) != 1:
            logger.warning("Invalid trigger format (expected single key-value): %s", node)
            return False

        trigger_type = list(node.keys())[0]
        value = node[trigger_type]

        cls = TRIGGERS.get(trigger_type)
        if not cls:
            logger.warning("Unknown trigger type: %s", trigger_type)
            return False

        try:
            if trigger_type == "interval":
                value = parse_duration(value)
            return cls(value).match(state, now)
        except Exception:
            logger.exception("Error evaluating trigger %s=%s", trigger_type, value)
            return False

    # logic node
    logic = node.get("logic", "AND").upper()
    triggers = node.get("triggers", [])
    if not triggers:
        logger.warning("Logic node without triggers: %s", node)
        return False

    results = [evaluate(t, state, now, logger) for t in triggers]

    if logic == "OR":
        return any(results)
    return all(results)


def parse_duration(value: str) -> timedelta:
    """
    Parse duration strings like '3d' or '5h' into timedelta
    """
    if not value:
        raise ValueError("Duration string is empty")

    unit = value[-1].lower()  # last character
    try:
        amount = int(value[:-1])
    except ValueError:
        raise ValueError(f"Invalid numeric value in duration: {value}")

    if unit == "d":
        return timedelta(days=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "w":
        return timedelta(weeks=amount)

    raise ValueError(f"Unsupported duration unit: {value} - The allowed units are h=hours, d=days, w=weeks")

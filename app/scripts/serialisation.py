import math


def make_json_serializable(obj):
    """
    Replace NaN-values with None and convert not JSON-compatible values to strings.
    """
    if hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif not isinstance(obj, (int, float, str, bool, type(None))):
        return str(obj)
    return obj

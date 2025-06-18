"""Utility functions for Junior AI Assistant."""

from typing import Any, Dict, Optional


def format_error_response(error: Exception, context: str = "") -> str:
    """Format an error response for user display."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"❌ {context}: {error_msg}"
    else:
        return f"❌ Error ({error_type}): {error_msg}"


def validate_temperature(temperature: float) -> float:
    """Validate and constrain temperature parameter."""
    if not isinstance(temperature, (int, float)):
        raise ValueError("Temperature must be a number")
    
    # Constrain to valid range
    return max(0.0, min(1.0, float(temperature)))


def safe_json_loads(data: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON data with error handling."""
    import json
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}", file=sys.stderr)
        return None


def format_ai_name(ai_name: str) -> str:
    """Format AI name for display."""
    return ai_name.upper()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
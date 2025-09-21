import os
import json
import redis
import datetime
from typing import Any

def get_current_datetime_str() -> str:
    """Get the current date and time as a formatted string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

def get_cwd() -> str:
    """Get the current working directory."""
    # get this file directory
    return os.path.dirname(os.path.abspath(__file__))

def chunk_response(text: str, size: int = 20):
    """Yield text in small chunks for streaming effect."""
    for i in range(0, len(text), size):
        yield text[i : i + size]

def _convert_pydantic_recursive(obj: Any) -> Any:
    """
    Recursively convert Pydantic models to dictionaries.
    """
    if hasattr(obj, "model_dump"):
        return _convert_pydantic_recursive(obj.model_dump())
    if isinstance(obj, dict):
        return {
            k: _convert_pydantic_recursive(v) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_convert_pydantic_recursive(item) for item in obj]
    return obj


def safe_jsondumps(obj, indent=None):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dumps(obj, default=default, indent=indent)

def _ensure_serializable(data: Any) -> Any:
    """
    Ensure that data is JSON serializable by converting complex objects.
    """
    if hasattr(data, "model_dump"):
        return _convert_pydantic_recursive(data.model_dump())
    if hasattr(data, "content") and hasattr(data, "type"):
        return {
            "content": data.content,
            "type": data.type,
            "additional_kwargs": getattr(data, "additional_kwargs", {})
        }
    if isinstance(data, dict):
        return {k: _ensure_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_ensure_serializable(item) for item in data]
    # Handle potential bytes that might come from tool outputs
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return f"<bytes data len={len(data)}>"  # Placeholder for non-utf8 bytes
    return data


def get_redis_instance():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    return redis.Redis(host=redis_host, port=redis_port, password=redis_password)
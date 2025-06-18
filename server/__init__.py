"""Server lifecycle and JSON-RPC handling."""

from .lifecycle import (
    ServerLifecycle,
    cleanup_resources
)

from .json_rpc import (
    send_response,
    send_error,
    parse_json_rpc
)

__all__ = [
    'ServerLifecycle',
    'cleanup_resources',
    'send_response',
    'send_error',
    'parse_json_rpc'
]
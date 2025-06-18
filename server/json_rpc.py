"""JSON-RPC protocol handling utilities.

This module implements JSON-RPC 2.0 protocol utilities for the MCP server.
It provides functions for sending responses, handling errors, and parsing
incoming messages according to the JSON-RPC specification.

Key functions:
- send_response: Send JSON-RPC response to stdout
- send_error: Send properly formatted error responses
- parse_json_rpc: Parse incoming JSON-RPC messages
- create_result_response: Create standard result responses
- create_tool_result: Format tool results for MCP protocol

Error codes follow JSON-RPC 2.0 specification:
- PARSE_ERROR (-32700): Invalid JSON
- INVALID_REQUEST (-32600): Invalid request structure
- METHOD_NOT_FOUND (-32601): Unknown method
- INVALID_PARAMS (-32602): Invalid method parameters
- INTERNAL_ERROR (-32603): Internal server error
"""

import json
import sys
from typing import Dict, Any, Optional


def send_response(response: Dict[str, Any]) -> None:
    """Send a JSON-RPC response to stdout."""
    print(json.dumps(response))
    sys.stdout.flush()


def send_error(request_id: Any, code: int, message: str, data: Optional[Any] = None) -> None:
    """Send a JSON-RPC error response."""
    error_response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    
    if data is not None:
        error_response["error"]["data"] = data
    
    send_response(error_response)


def parse_json_rpc(message: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON-RPC message."""
    try:
        return json.loads(message)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return None


def create_result_response(request_id: Any, result: Any) -> Dict[str, Any]:
    """Create a JSON-RPC result response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }


def create_tool_result(text: str) -> Dict[str, Any]:
    """Create a tool result with text content."""
    return {
        "content": [{"type": "text", "text": text}]
    }


# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
"""Request handlers for Junior AI Assistant."""

from .mcp_protocol import (
    MCPProtocolHandler,
    ServerInfo,
    AIClientInfo,
    PatternDetectionConfig
)

from .tool_dispatcher import (
    ToolDispatcher,
    register_tool_handler,
    get_tool_handler
)

from .base import (
    BaseHandler,
    HandlerContext,
    HandlerRegistry
)

__all__ = [
    'MCPProtocolHandler',
    'ServerInfo',
    'AIClientInfo',
    'PatternDetectionConfig',
    'ToolDispatcher',
    'register_tool_handler',
    'get_tool_handler',
    'BaseHandler',
    'HandlerContext',
    'HandlerRegistry'
]
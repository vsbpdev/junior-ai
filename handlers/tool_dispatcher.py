"""Tool dispatcher for routing tool calls to appropriate handlers.

This module implements the central dispatching mechanism for tool calls
in the Junior AI Assistant. It routes incoming tool requests to the
appropriate handler based on tool name registration.

Key components:
- ToolDispatcher: Main dispatcher that routes tool calls to handlers
- Special handler support: Allows registration of custom handlers for
  specific tools outside the normal handler hierarchy
- Helper functions: Convenience functions for handler registration and lookup

The dispatcher provides:
- Automatic routing based on tool name
- Support for special/custom handlers
- Consistent error handling and reporting
- Handler registration and discovery
"""

from typing import Dict, Any, Optional, Callable
from .base import HandlerRegistry


class ToolDispatcher:
    """Dispatches tool calls to appropriate handlers."""
    
    def __init__(self, registry: HandlerRegistry):
        """Initialize the dispatcher with a handler registry."""
        self.registry = registry
        self._special_handlers: Dict[str, Callable] = {}
    
    def register_special_handler(self, tool_name: str, handler: Callable[[Dict[str, Any]], Any]):
        """Register a special handler for a specific tool."""
        self._special_handlers[tool_name] = handler
    
    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Route a tool call to the appropriate handler."""
        # Check for special handlers first
        if tool_name in self._special_handlers:
            try:
                return self._special_handlers[tool_name](arguments)
            except Exception as e:
                return f"❌ Error in special handler for '{tool_name}': {str(e)}"
        
        # Use registry to find handler
        try:
            return self.registry.handle_tool_call(tool_name, arguments)
        except ValueError as e:
            # No handler found
            return f"❌ {str(e)}"
        except Exception as e:
            # Handler error
            return f"❌ Error handling tool '{tool_name}': {str(e)}"


def register_tool_handler(registry: HandlerRegistry, handler: Any) -> None:
    """Register a tool handler with the registry."""
    registry.register(handler)


def get_tool_handler(registry: HandlerRegistry, tool_name: str) -> Optional[Any]:
    """Get a handler for a specific tool."""
    return registry.get_handler(tool_name)
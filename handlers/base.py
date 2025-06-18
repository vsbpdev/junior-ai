"""Base handler classes and interfaces for the Junior AI Assistant.

This module provides the foundational classes and interfaces for building
tool handlers in the modular server architecture. It implements a registry
pattern for dynamic handler discovery and routing.

Key classes:
- HandlerContext: Shared context passed to all handlers containing AI clients,
  pattern engines, and other shared resources
- BaseHandler: Abstract base class that all tool handlers must inherit from
- HandlerRegistry: Central registry for registering and routing tool calls
  to appropriate handlers

The handler system supports:
- Dynamic tool registration
- Shared resource management through context
- Consistent error handling
- Tool routing and dispatch
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class HandlerContext:
    """Context passed to handlers containing shared resources."""
    ai_clients: Dict[str, Any]
    pattern_engine: Optional[Any] = None
    response_manager: Optional[Any] = None
    ai_consultation_manager: Optional[Any] = None
    async_pipeline: Optional[Any] = None
    text_pipeline: Optional[Any] = None
    credentials: Optional[Dict[str, Any]] = None


class BaseHandler(ABC):
    """Base class for all tool handlers."""
    
    def __init__(self, context: HandlerContext):
        """Initialize handler with context."""
        self.context = context
        self.ai_clients = context.ai_clients
        self.pattern_engine = context.pattern_engine
        self.response_manager = context.response_manager
        self.ai_consultation_manager = context.ai_consultation_manager
        self.async_pipeline = context.async_pipeline
        self.text_pipeline = context.text_pipeline
        self.credentials = context.credentials
    
    @abstractmethod
    def get_tool_names(self) -> List[str]:
        """Return list of tool names this handler supports."""
        pass
    
    @abstractmethod
    def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a tool call."""
        pass


class HandlerRegistry:
    """Registry for tool handlers."""
    
    def __init__(self):
        """Initialize the registry."""
        self._handlers: Dict[str, BaseHandler] = {}
        self._tool_to_handler: Dict[str, BaseHandler] = {}
    
    def register(self, handler: BaseHandler) -> None:
        """Register a handler."""
        handler_name = handler.__class__.__name__
        self._handlers[handler_name] = handler
        
        # Map tool names to handler
        for tool_name in handler.get_tool_names():
            if tool_name in self._tool_to_handler:
                raise ValueError(f"Tool {tool_name} already registered")
            self._tool_to_handler[tool_name] = handler
    
    def get_handler(self, tool_name: str) -> Optional[BaseHandler]:
        """Get handler for a tool."""
        return self._tool_to_handler.get(tool_name)
    
    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a tool call by routing to appropriate handler."""
        handler = self.get_handler(tool_name)
        if not handler:
            raise ValueError(f"No handler registered for tool: {tool_name}")
        
        return handler.handle(tool_name, arguments)
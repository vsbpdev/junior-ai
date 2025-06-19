#!/usr/bin/env python3
"""Junior AI Assistant MCP Server - Simplified Modular Version.

This is the main entry point for the Junior AI Assistant MCP server.
The server has been refactored into a clean modular architecture with
clear separation of concerns.

Architecture overview:
- core/: Core utilities, configuration, and AI client management
- ai/: AI calling and response formatting functionality
- handlers/: Tool handlers implementing business logic
- pattern/: Pattern detection engine management
- server/: Server lifecycle and JSON-RPC protocol handling

The server supports:
- Multiple AI providers (Gemini, Grok, OpenAI, DeepSeek, OpenRouter)
- Pattern detection with AI consultation
- Collaborative AI tools (debates, consensus, etc.)
- Async pattern caching for performance
- Manual override controls
- Comprehensive error handling and graceful degradation

The modular design allows for:
- Easy addition of new AI providers
- Clear separation between protocol handling and business logic
- Testable components with minimal coupling
- Optional features that gracefully degrade when unavailable
"""

import os
import sys
import json
import asyncio
import signal
from typing import Dict, Any

# Ensure unbuffered output for MCP
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Core imports
from core import (
    __version__,
    load_credentials,
    initialize_all_clients,
    AI_CLIENTS
)
from core.ai_clients import initialize_all_clients_async

# Handler imports
from handlers.mcp_protocol import (
    MCPProtocolHandler,
    ServerInfo,
    AIClientInfo,
    PatternDetectionConfig
)
from handlers.base import HandlerContext, HandlerRegistry
from handlers.ai_tools import AIToolsHandler
from handlers.collaborative_tools import CollaborativeToolsHandler

# Pattern detection imports (optional)
try:
    from pattern import PatternEngineManager
    PATTERN_AVAILABLE = True
except ImportError:
    PATTERN_AVAILABLE = False
    print("Pattern detection modules not available", file=sys.stderr)


class JuniorAIServer:
    """Main server class for Junior AI Assistant."""
    
    def __init__(self):
        """Initialize the server."""
        self.running = False
        self.protocol_handler = None
        self.handler_registry = HandlerRegistry()
        self.pattern_manager = None
        self._initialized = False
        
    async def _initialize_components(self):
        """Initialize all server components."""
        # Load credentials
        credentials = load_credentials()
        
        # Initialize AI clients (both sync and async)
        ai_clients = await initialize_all_clients_async()
        
        # Create handler context
        context = HandlerContext(
            ai_clients=ai_clients,
            credentials=credentials
        )
        
        # Initialize pattern detection if available
        if PATTERN_AVAILABLE:
            try:
                self.pattern_manager = PatternEngineManager(
                    credentials=credentials,
                    ai_callers={
                        name: lambda p, t=0.7, ai=name: self._call_ai_wrapper(ai, p, t)
                        for name in ai_clients.keys()
                    }
                )
                
                # Add pattern components to context
                context.pattern_engine = self.pattern_manager.get_pattern_engine()
                context.response_manager = self.pattern_manager.get_response_manager()
                context.ai_consultation_manager = self.pattern_manager.get_ai_consultation_manager()
                context.async_pipeline = self.pattern_manager.get_async_pipeline()
                context.text_pipeline = self.pattern_manager.get_text_pipeline()
                
                # Import and register pattern handlers
                from handlers.pattern_tools import PatternToolsHandler
                from handlers.cache_tools import CacheToolsHandler
                
                self.handler_registry.register(PatternToolsHandler(context))
                self.handler_registry.register(CacheToolsHandler(context))
                
            except Exception as e:
                print(f"Failed to initialize pattern detection: {e}", file=sys.stderr)
        
        # Register core handlers
        self.handler_registry.register(AIToolsHandler(context))
        self.handler_registry.register(CollaborativeToolsHandler(context))
        
        # Initialize MCP protocol handler
        ai_client_info = {
            name: AIClientInfo(name=name, client=client, type="gemini" if name == "gemini" else "openai")
            for name, client in ai_clients.items()
        }
        
        pattern_config = PatternDetectionConfig()
        if PATTERN_AVAILABLE and self.pattern_manager:
            config = self.pattern_manager.get_pattern_config()
            pattern_config = PatternDetectionConfig(
                enabled=config.get('enabled', True),
                default_junior=config.get('default_junior', 'openrouter'),
                accuracy_mode=config.get('accuracy_mode', True)
            )
        
        self.protocol_handler = MCPProtocolHandler(
            server_info=ServerInfo(name="junior-ai-assistant-mcp", version=__version__),
            ai_clients=ai_client_info,
            pattern_detection_available=PATTERN_AVAILABLE and bool(self.pattern_manager),
            ai_consultation_available=PATTERN_AVAILABLE and self.pattern_manager and self.pattern_manager.get_ai_consultation_manager() is not None,
            async_cache_available=PATTERN_AVAILABLE and self.pattern_manager and self.pattern_manager.get_async_pipeline() is not None,
            pattern_config=pattern_config
        )
    
    def _call_ai_wrapper(self, ai_name: str, prompt: str, temperature: float = 0.7) -> str:
        """Wrapper for AI calls to match expected signature."""
        from ai.caller import call_ai
        return call_ai(ai_name, prompt, temperature)
    
    def send_response(self, response: Dict[str, Any]):
        """Send a JSON-RPC response."""
        print(json.dumps(response))
        sys.stdout.flush()
    
    def send_error(self, request_id: Any, code: int, message: str):
        """Send a JSON-RPC error response."""
        self.send_response({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        })
    
    async def handle_request(self, request: Dict[str, Any]):
        """Handle a JSON-RPC request."""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        try:
            if method == 'initialize':
                response = self.protocol_handler.handle_initialize(request_id)
            elif method == 'tools/list':
                response = self.protocol_handler.handle_tools_list(request_id)
            elif method == 'tools/call':
                tool_name = params.get('name')
                arguments = params.get('arguments', {})
                
                # Handle server_status specially
                if tool_name == 'server_status':
                    result = self._handle_server_status()
                else:
                    # Route to appropriate handler
                    result = self.handler_registry.handle_tool_call(tool_name, arguments)
                
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": result}]
                    }
                }
            else:
                self.send_error(request_id, -32601, f"Method not found: {method}")
                return
                
            self.send_response(response)
            
        except Exception as e:
            print(f"Error handling request: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.send_error(request_id, -32603, str(e))
    
    def _handle_server_status(self) -> str:
        """Handle server status request."""
        status = {
            "server": "Junior AI Assistant for Claude Code",
            "version": __version__,
            "ai_models": {}
        }
        
        # Add AI model status
        for ai_name, client in AI_CLIENTS.items():
            status["ai_models"][ai_name] = {
                "available": True,
                "type": "gemini" if ai_name == "gemini" else "openai-compatible"
            }
        
        # Add pattern detection status
        if PATTERN_AVAILABLE and self.pattern_manager:
            status["pattern_detection"] = self.pattern_manager.get_status()
        else:
            status["pattern_detection"] = {"available": False}
        
        # Format as text response
        lines = [
            f"# {status['server']} v{status['version']}",
            "",
            "## Available AI Models:"
        ]
        
        for ai_name, ai_status in status["ai_models"].items():
            lines.append(f"- **{ai_name.upper()}**: ✅ Available ({ai_status['type']})")
        
        if not status["ai_models"]:
            lines.append("❌ No AI models available")
        
        # Add pattern detection status
        lines.extend(["", "## Pattern Detection:"])
        pd_status = status.get("pattern_detection", {})
        if pd_status.get("available"):
            lines.append("✅ Pattern detection enabled")
            if pd_status.get("components"):
                for comp, comp_status in pd_status["components"].items():
                    if comp_status.get("available"):
                        lines.append(f"  - {comp}: ✅")
        else:
            lines.append("❌ Pattern detection not available")
        
        return "\n".join(lines)
    
    async def run(self):
        """Run the main server loop."""
        # Initialize components if not already done
        if not self._initialized:
            await self._initialize_components()
            self._initialized = True
        
        self.running = True
        print(f"Junior AI Assistant MCP Server v{__version__} started", file=sys.stderr)
        
        try:
            while self.running:
                try:
                    loop = asyncio.get_running_loop()
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        request = json.loads(line)
                        await self.handle_request(request)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON: {e}", file=sys.stderr)
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in main loop: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        print("Shutting down server...", file=sys.stderr)
        
        # Clean up async AI clients
        try:
            from core.ai_clients import cleanup_async_ai_clients
            await cleanup_async_ai_clients()
        except Exception as e:
            print(f"Error cleaning up async AI clients: {e}", file=sys.stderr)
        
        if self.pattern_manager:
            try:
                await self.pattern_manager.shutdown()
            except Exception as e:
                print(f"Error during pattern manager shutdown: {e}", file=sys.stderr)
        
        self.running = False
        print("Server shutdown complete", file=sys.stderr)
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...", file=sys.stderr)
        self.running = False


async def main():
    """Main entry point."""
    server = JuniorAIServer()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, server.handle_signal)
    signal.signal(signal.SIGTERM, server.handle_signal)
    
    # Run the server
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
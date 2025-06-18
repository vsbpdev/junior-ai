#!/usr/bin/env python3
"""
Junior AI MCP Server - Simplified Main Entry Point
Version: 3.0.0
"""

import sys
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# Core modules
from core.json_rpc import JSONRPCMessage, create_response, create_error_response
from core.logging_config import logger
from core.utils import get_server_info

# AI modules
from ai.client_manager import AIClientManager
from ai.tools import AIToolHandler
from ai.collaborative import CollaborativeAIHandler

# Pattern detection modules (optional)
try:
    from pattern.engine import PatternEngine
    from pattern.handlers import PatternHandler
    PATTERN_DETECTION_AVAILABLE = True
except ImportError:
    PATTERN_DETECTION_AVAILABLE = False
    logger.warning("Pattern detection modules not available")


class JuniorAIServer:
    """Simplified MCP server implementation."""
    
    def __init__(self):
        """Initialize the server with all components."""
        self.server_info = get_server_info()
        self.running = False
        
        # Initialize AI components
        self.ai_manager = AIClientManager()
        self.ai_tools = AIToolHandler(self.ai_manager)
        self.collaborative = CollaborativeAIHandler(self.ai_manager)
        
        # Initialize pattern detection if available
        if PATTERN_DETECTION_AVAILABLE:
            self.pattern_engine = PatternEngine()
            self.pattern_handler = PatternHandler(
                self.pattern_engine,
                self.ai_manager,
                self.collaborative
            )
        else:
            self.pattern_engine = None
            self.pattern_handler = None
        
        # Build capabilities
        self._capabilities = self._build_capabilities()
        
    def _build_capabilities(self) -> Dict[str, Any]:
        """Build server capabilities."""
        capabilities = {
            "name": self.server_info["name"],
            "version": self.server_info["version"],
            "description": self.server_info["description"],
            "tools": []
        }
        
        # Always available tools
        base_tools = [
            {
                "name": "server_status",
                "description": "Get server status and available AI models",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
        
        # Add AI tools for each available client
        for ai_name in self.ai_manager.get_available_ais():
            base_tools.extend(self.ai_tools.get_tools_for_ai(ai_name))
        
        # Add collaborative tools if multiple AIs available
        if len(self.ai_manager.get_available_ais()) >= 2:
            base_tools.extend(self.collaborative.get_tools())
        
        # Add pattern detection tools if available
        if self.pattern_handler:
            base_tools.extend(self.pattern_handler.get_tools())
        
        capabilities["tools"] = base_tools
        return capabilities
    
    async def handle_initialize(self, message: JSONRPCMessage) -> None:
        """Handle initialize request."""
        response = {
            "protocolVersion": "0.1.0",
            "capabilities": self._capabilities,
            "serverInfo": self.server_info
        }
        
        await self.send_response(create_response(message.id, response))
        
        # Send initialized notification
        await self.send_response({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        })
    
    async def handle_tools_list(self, message: JSONRPCMessage) -> None:
        """Handle tools/list request."""
        await self.send_response(
            create_response(message.id, {"tools": self._capabilities["tools"]})
        )
    
    async def handle_tool_call(self, message: JSONRPCMessage) -> None:
        """Handle tools/call request."""
        tool_name = message.params.get("name")
        arguments = message.params.get("arguments", {})
        
        try:
            # Server status tool
            if tool_name == "server_status":
                result = await self._get_server_status()
            
            # AI tools
            elif self.ai_tools.is_ai_tool(tool_name):
                result = await self.ai_tools.handle_tool(tool_name, arguments)
            
            # Collaborative tools
            elif self.collaborative.is_collaborative_tool(tool_name):
                result = await self.collaborative.handle_tool(tool_name, arguments)
            
            # Pattern detection tools
            elif self.pattern_handler and self.pattern_handler.is_pattern_tool(tool_name):
                result = await self.pattern_handler.handle_tool(tool_name, arguments)
            
            else:
                await self.send_response(
                    create_error_response(
                        message.id,
                        -32601,
                        f"Unknown tool: {tool_name}"
                    )
                )
                return
            
            await self.send_response(create_response(message.id, result))
            
        except Exception as e:
            logger.error(f"Error in tool {tool_name}: {str(e)}", exc_info=True)
            await self.send_response(
                create_error_response(
                    message.id,
                    -32603,
                    f"Tool execution error: {str(e)}"
                )
            )
    
    async def _get_server_status(self) -> Dict[str, Any]:
        """Get current server status."""
        available_ais = self.ai_manager.get_available_ais()
        
        status = {
            "status": "running",
            "version": self.server_info["version"],
            "available_ais": available_ais,
            "ai_count": len(available_ais),
            "pattern_detection": PATTERN_DETECTION_AVAILABLE,
            "uptime": str(datetime.now() - self.start_time) if hasattr(self, 'start_time') else "0:00:00",
            "capabilities": {
                "multi_ai_support": len(available_ais) >= 2,
                "pattern_detection": PATTERN_DETECTION_AVAILABLE,
                "ai_clients": {ai: "active" for ai in available_ais}
            }
        }
        
        if self.pattern_engine:
            status["pattern_stats"] = self.pattern_engine.get_stats()
        
        return status
    
    async def handle_message(self, message_str: str) -> None:
        """Handle incoming JSON-RPC message."""
        try:
            # Parse message
            try:
                data = json.loads(message_str)
                message = JSONRPCMessage(data)
            except json.JSONDecodeError as e:
                await self.send_response(
                    create_error_response(None, -32700, f"Parse error: {str(e)}")
                )
                return
            except ValueError as e:
                await self.send_response(
                    create_error_response(None, -32600, f"Invalid request: {str(e)}")
                )
                return
            
            # Route to appropriate handler
            if message.method == "initialize":
                await self.handle_initialize(message)
            elif message.method == "tools/list":
                await self.handle_tools_list(message)
            elif message.method == "tools/call":
                await self.handle_tool_call(message)
            else:
                await self.send_response(
                    create_error_response(
                        message.id,
                        -32601,
                        f"Method not found: {message.method}"
                    )
                )
        
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            await self.send_response(
                create_error_response(
                    message.id if 'message' in locals() else None,
                    -32603,
                    f"Internal error: {str(e)}"
                )
            )
    
    async def send_response(self, response: Dict[str, Any]) -> None:
        """Send JSON-RPC response."""
        response_str = json.dumps(response)
        sys.stdout.write(response_str + "\n")
        sys.stdout.flush()
    
    async def run(self) -> None:
        """Main server loop."""
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"Junior AI MCP Server v{self.server_info['version']} starting...")
        logger.info(f"Available AIs: {', '.join(self.ai_manager.get_available_ais())}")
        logger.info(f"Pattern detection: {'enabled' if PATTERN_DETECTION_AVAILABLE else 'disabled'}")
        
        try:
            while self.running:
                try:
                    # Read line from stdin
                    line = await asyncio.get_running_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    
                    if not line:  # EOF
                        break
                    
                    line = line.strip()
                    if line:
                        await self.handle_message(line)
                
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down server...")
        self.running = False
        
        # Cleanup AI clients
        await self.ai_manager.cleanup()
        
        # Cleanup pattern engine if available
        if self.pattern_engine:
            await self.pattern_engine.cleanup()
        
        logger.info("Server shutdown complete")


async def main():
    """Main entry point."""
    server = JuniorAIServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
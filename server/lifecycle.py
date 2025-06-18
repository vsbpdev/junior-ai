"""Server lifecycle management for Junior AI Assistant.

This module manages the server's lifecycle including startup, shutdown,
and graceful termination. It provides signal handling for clean shutdown
and resource cleanup mechanisms.

Key components:
- ServerLifecycle: Main lifecycle manager handling server state and
  shutdown coordination
- cleanup_resources: Utility function for cleaning up multiple resources
  with different cleanup interfaces

The lifecycle manager supports:
- Signal handling (SIGINT, SIGTERM) for graceful shutdown
- Registration of cleanup handlers
- Async and sync cleanup handler support
- Resource cleanup with multiple interface patterns (cleanup, close, shutdown)
- Event loop integration for async operations
"""

import sys
import signal
import asyncio
from typing import Callable, Any


class ServerLifecycle:
    """Manages server lifecycle including startup, shutdown, and signal handling."""
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self.running = False
        self.cleanup_handlers = []
        self._signal_handlers_set = False
    
    def add_cleanup_handler(self, handler: Callable[[], Any]) -> None:
        """Add a cleanup handler to be called on shutdown."""
        self.cleanup_handlers.append(handler)
    
    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        if self._signal_handlers_set:
            return
            
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, initiating shutdown...", file=sys.stderr)
            self.running = False
            # Set a flag to exit the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    lambda: setattr(self, 'shutdown_requested', True)
                )
            except RuntimeError:
                # No running loop, just set the flag
                self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._signal_handlers_set = True
    
    async def cleanup(self) -> None:
        """Run all cleanup handlers."""
        print("Running cleanup handlers...", file=sys.stderr)
        
        for handler in self.cleanup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                print(f"Error in cleanup handler: {e}", file=sys.stderr)
        
        print("Cleanup complete", file=sys.stderr)
    
    def start(self) -> None:
        """Mark server as running."""
        self.running = True
        self.shutdown_requested = False
    
    def stop(self) -> None:
        """Mark server as stopped."""
        self.running = False
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running and not getattr(self, 'shutdown_requested', False)


async def cleanup_resources(*resources) -> None:
    """Clean up multiple resources."""
    for resource in resources:
        if resource is None:
            continue
            
        try:
            # Check for various cleanup methods
            if hasattr(resource, 'cleanup'):
                if asyncio.iscoroutinefunction(resource.cleanup):
                    await resource.cleanup()
                else:
                    resource.cleanup()
            elif hasattr(resource, 'close'):
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
            elif hasattr(resource, 'shutdown'):
                if asyncio.iscoroutinefunction(resource.shutdown):
                    await resource.shutdown()
                else:
                    resource.shutdown()
        except Exception as e:
            print(f"Error cleaning up resource {type(resource).__name__}: {e}", file=sys.stderr)
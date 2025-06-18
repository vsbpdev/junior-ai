#!/usr/bin/env python3
"""Test script to verify modular server components."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    modules = [
        ("core", "Core utilities"),
        ("core.config", "Configuration"),
        ("core.ai_clients", "AI client management"),
        ("core.utils", "Utilities"),
        ("ai", "AI communication"),
        ("ai.caller", "AI calling"),
        ("ai.response_formatter", "Response formatting"),
        ("handlers", "Request handlers"),
        ("handlers.base", "Base handler classes"),
        ("handlers.mcp_protocol", "MCP protocol"),
        ("handlers.ai_tools", "AI tool handlers"),
        ("handlers.collaborative_tools", "Collaborative tools"),
        ("pattern", "Pattern detection (optional)"),
        ("server", "Server modules"),
        ("server.json_rpc", "JSON-RPC handling"),
        ("server.lifecycle", "Server lifecycle")
    ]
    
    results = []
    for module_name, description in modules:
        try:
            __import__(module_name)
            results.append((module_name, description, "✅"))
        except ImportError as e:
            results.append((module_name, description, f"❌ {str(e)}"))
    
    # Print results
    print("\nModule Import Results:")
    print("-" * 60)
    for module, desc, status in results:
        print(f"{module:<30} {desc:<25} {status}")
    
    return all(status == "✅" for _, _, status in results[:7])  # Core modules must work


def test_server_initialization():
    """Test server initialization without running."""
    print("\n\nTesting server initialization...")
    
    try:
        # Set minimal environment for testing
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        from server import JuniorAIServer
        
        # Try to create server instance
        server = JuniorAIServer()
        print("✅ Server instance created successfully")
        
        # Check components
        print("\nServer components:")
        print(f"- Protocol handler: {'✅' if server.protocol_handler else '❌'}")
        print(f"- Handler registry: {'✅' if server.handler_registry else '❌'}")
        print(f"- Pattern manager: {'✅' if server.pattern_manager else '❌ (optional)'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_lines():
    """Count lines in new vs old server."""
    print("\n\nLine count comparison:")
    
    files = {
        "server.py": "New modular server",
        "server_backup.py": "Original monolithic server"
    }
    
    for filename, description in files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = len(f.readlines())
            print(f"{description:<30} {lines:>5} lines")
    
    # Count module lines
    module_lines = 0
    module_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                filepath = os.path.join(root, file)
                # Only count our new modules
                if any(filepath.startswith(f"./{d}/") for d in ['core', 'ai', 'handlers', 'pattern', 'server']):
                    with open(filepath, 'r') as f:
                        lines = len(f.readlines())
                    module_lines += lines
                    module_files.append((filepath, lines))
    
    print(f"\nTotal module lines: {module_lines}")
    print("\nModule breakdown:")
    for filepath, lines in sorted(module_files):
        print(f"  {filepath:<40} {lines:>5} lines")


def main():
    """Run all tests."""
    print("=== Junior AI Server Modularization Test ===\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test server initialization
    server_ok = test_server_initialization()
    
    # Count lines
    count_lines()
    
    # Summary
    print("\n\n=== Summary ===")
    if imports_ok and server_ok:
        print("✅ Modularization successful!")
        print("✅ Server can be initialized")
        print("✅ All core modules are working")
    else:
        print("❌ Some issues found, but this is expected during refactoring")
    
    return 0 if imports_ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Test cleanup error handling in async AI client."""

import asyncio
import logging
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ai.async_client import AsyncOpenAICompatibleClient, AIProvider

# Set up logging to capture warnings
logging.basicConfig(level=logging.DEBUG)


async def test_cleanup_with_exception():
    """Test that cleanup handles exceptions gracefully."""
    print("Testing cleanup error handling...")
    
    client = AsyncOpenAICompatibleClient(
        api_key="test",
        model="test",
        provider=AIProvider.OPENAI
    )
    
    # Mock a client with a close method that raises an exception
    mock_client = Mock()
    mock_close = AsyncMock(side_effect=RuntimeError("Connection already closed"))
    mock_client.close = mock_close
    
    client._client = mock_client
    client._initialized = True
    
    # This should not raise an exception
    try:
        await client.cleanup()
        print("✅ Cleanup succeeded despite close() exception")
        
        # Verify the client is marked as not initialized
        assert not client._initialized, "Client should be marked as not initialized"
        print("✅ Client properly marked as not initialized")
        
        # Verify close was attempted
        mock_close.assert_called_once()
        print("✅ close() was called as expected")
        
        return True
    except Exception as e:
        print(f"❌ Cleanup raised an exception: {e}")
        return False


async def test_cleanup_in_context_manager():
    """Test cleanup error handling in context manager."""
    print("\nTesting cleanup in context manager...")
    
    # Create a client that will fail during cleanup
    client = AsyncOpenAICompatibleClient(
        api_key="test",
        model="test",
        provider=AIProvider.GROK
    )
    
    # Mock initialization
    client._initialized = True
    
    # Mock a client with failing close
    mock_client = Mock()
    mock_client.close = AsyncMock(side_effect=ConnectionError("Network error"))
    client._client = mock_client
    
    # Use context manager - should not raise exception on exit
    try:
        async with client:
            print("✅ Context manager entered successfully")
            # Simulate some work
            pass
        
        print("✅ Context manager exited successfully despite cleanup error")
        return True
    except Exception as e:
        print(f"❌ Context manager raised an exception: {e}")
        return False


async def test_multiple_client_cleanup():
    """Test that one client's cleanup failure doesn't affect others."""
    print("\nTesting multiple client cleanup...")
    
    clients = []
    
    # Create multiple clients with different behaviors
    for i in range(3):
        client = AsyncOpenAICompatibleClient(
            api_key=f"test-{i}",
            model="test",
            provider=AIProvider.DEEPSEEK
        )
        
        mock_client = Mock()
        if i == 1:
            # Middle client will fail
            mock_client.close = AsyncMock(side_effect=Exception("Cleanup failed"))
        else:
            # Others will succeed
            mock_client.close = AsyncMock()
        
        client._client = mock_client
        client._initialized = True
        clients.append(client)
    
    # Clean up all clients
    cleanup_results = []
    for i, client in enumerate(clients):
        try:
            await client.cleanup()
            cleanup_results.append(True)
            print(f"✅ Client {i} cleanup completed")
        except Exception as e:
            cleanup_results.append(False)
            print(f"❌ Client {i} cleanup failed: {e}")
    
    # All cleanups should complete (no exceptions)
    if all(cleanup_results):
        print("✅ All client cleanups completed without raising exceptions")
        return True
    else:
        print("❌ Some client cleanups raised exceptions")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Async AI Client Cleanup Error Handling")
    print("=" * 50)
    
    tests = [
        test_cleanup_with_exception,
        test_cleanup_in_context_manager,
        test_multiple_client_cleanup
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\nCleanup error handling is working correctly:")
        print("- Exceptions during close() are caught and logged")
        print("- Client is marked as uninitialized regardless of errors")
        print("- Context manager exits cleanly")
        print("- Multiple client cleanup is fault-tolerant")
    else:
        print(f"❌ {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
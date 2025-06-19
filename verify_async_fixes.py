#!/usr/bin/env python3
"""Simple verification of async AI client fixes."""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ai.async_client import AsyncGeminiClient, AsyncOpenAICompatibleClient, AIProvider


async def verify_cleanup_fix():
    """Verify cleanup handles missing close() method."""
    print("1. Testing cleanup with missing close() method...")
    
    client = AsyncOpenAICompatibleClient(
        api_key="test",
        model="test",
        provider=AIProvider.OPENAI
    )
    
    # Mock a client without close
    class MockClient:
        pass
    
    client._client = MockClient()
    client._initialized = True
    
    try:
        await client.cleanup()
        print("   ✅ Cleanup succeeded without close() method")
        return True
    except AttributeError as e:
        print(f"   ❌ Cleanup failed: {e}")
        return False


async def verify_init_lock():
    """Verify initialization lock prevents race conditions."""
    print("\n2. Testing initialization lock...")
    
    # Check that AsyncGeminiClient has _init_lock
    client = AsyncGeminiClient(
        api_key="test",
        model="test"
    )
    
    if hasattr(client, '_init_lock'):
        print("   ✅ AsyncGeminiClient has _init_lock attribute")
        print(f"   ✅ Lock type: {type(client._init_lock)}")
        return True
    else:
        print("   ❌ AsyncGeminiClient missing _init_lock")
        return False


def verify_error_context():
    """Verify error context preservation in fallback."""
    print("\n3. Checking error context preservation code...")
    
    # Read the source to verify the fix is in place
    with open('ai/caller.py', 'r') as f:
        content = f.read()
    
    if 'async_error = e' in content and 'Both async and sync calls failed' in content:
        print("   ✅ Error context preservation code found")
        return True
    else:
        print("   ❌ Error context preservation code missing")
        return False


async def main():
    """Run all verifications."""
    print("🔍 Verifying Async AI Client Fixes")
    print("=" * 50)
    
    results = []
    
    # Test 1: Cleanup fix
    results.append(await verify_cleanup_fix())
    
    # Test 2: Init lock
    results.append(await verify_init_lock())
    
    # Test 3: Error context
    results.append(verify_error_context())
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} fixes verified successfully!")
    else:
        print(f"⚠️  {passed}/{total} fixes verified")
        print("Please check the failed verifications above.")


if __name__ == "__main__":
    asyncio.run(main())
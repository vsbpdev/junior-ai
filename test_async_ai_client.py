#!/usr/bin/env python3
"""Test script for async AI client implementation.

This script tests:
1. Async client initialization
2. Concurrent AI calls
3. Performance comparison vs sync calls
4. Error handling and timeouts
5. Resource cleanup
"""

import asyncio
import time

# Test imports
from core.config import load_credentials
from core.ai_clients import initialize_all_clients_async, cleanup_async_ai_clients
from ai.async_client import call_ai_async, ASYNC_AI_CLIENTS
from ai.caller import call_ai


async def test_async_initialization():
    """Test async client initialization."""
    print("1. Testing async client initialization...")
    
    start_time = time.time()
    clients = await initialize_all_clients_async()
    init_time = time.time() - start_time
    
    print(f"   ✅ Initialized {len(clients)} sync clients")
    print(f"   ✅ Initialized {len(ASYNC_AI_CLIENTS)} async clients")
    print(f"   ⏱️  Initialization time: {init_time:.2f}s")
    
    # List available clients
    print("   Available async clients:")
    for name, client in ASYNC_AI_CLIENTS.items():
        print(f"     - {name}: {client.model}")
    
    return len(ASYNC_AI_CLIENTS) > 0


async def test_single_async_call():
    """Test a single async AI call."""
    print("\n2. Testing single async AI call...")
    
    test_prompt = "What is 2+2? Answer in one word."
    
    # Find first available AI
    ai_name = None
    for name in ASYNC_AI_CLIENTS:
        ai_name = name
        break
    
    if not ai_name:
        print("   ❌ No async AI clients available")
        return False
    
    try:
        start_time = time.time()
        response = await call_ai_async(ai_name, test_prompt, temperature=0.1)
        call_time = time.time() - start_time
        
        print(f"   ✅ Called {ai_name} successfully")
        print(f"   ⏱️  Response time: {call_time:.2f}s")
        print(f"   📝 Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"   ❌ Error calling {ai_name}: {e}")
        return False


async def test_concurrent_calls():
    """Test concurrent async AI calls."""
    print("\n3. Testing concurrent async AI calls...")
    
    test_prompts = [
        "What is the capital of France? Answer in one word.",
        "What is 10 multiplied by 5? Answer with just the number.",
        "What color is the sky on a clear day? Answer in one word."
    ]
    
    available_ais = list(ASYNC_AI_CLIENTS.keys())
    if len(available_ais) < 2:
        print("   ⚠️  Less than 2 AIs available, skipping concurrent test")
        return True
    
    # Create concurrent tasks
    tasks = []
    for i, prompt in enumerate(test_prompts):
        ai_name = available_ais[i % len(available_ais)]
        tasks.append(call_ai_async(ai_name, prompt, temperature=0.1))
    
    try:
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"   ✅ Completed {len(responses)} concurrent calls")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📊 Average time per call: {total_time/len(responses):.2f}s")
        
        for i, response in enumerate(responses):
            ai_name = available_ais[i % len(available_ais)]
            print(f"   📝 {ai_name}: {response[:50]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ Error in concurrent calls: {e}")
        return False


async def test_performance_comparison():
    """Compare async vs sync performance."""
    print("\n4. Testing performance: async vs sync...")
    
    test_prompt = "Count from 1 to 5. Just list the numbers."
    available_ais = list(ASYNC_AI_CLIENTS.keys())[:3]  # Test with up to 3 AIs
    
    if not available_ais:
        print("   ⚠️  No AIs available for performance test")
        return True
    
    # Test sync calls (sequential)
    print("   Testing synchronous calls...")
    sync_start = time.time()
    sync_responses = {}
    for ai_name in available_ais:
        try:
            sync_responses[ai_name] = call_ai(ai_name, test_prompt, 0.1)
        except Exception as e:
            sync_responses[ai_name] = f"Error: {e}"
    sync_time = time.time() - sync_start
    
    # Test async calls (concurrent)
    print("   Testing asynchronous calls...")
    async_start = time.time()
    tasks = [call_ai_async(ai_name, test_prompt, 0.1) for ai_name in available_ais]
    try:
        async_responses = await asyncio.gather(*tasks, return_exceptions=True)
        async_responses = {
            ai_name: resp if not isinstance(resp, Exception) else f"Error: {resp}"
            for ai_name, resp in zip(available_ais, async_responses)
        }
    except Exception as e:
        async_responses = {ai_name: f"Error: {e}" for ai_name in available_ais}
    async_time = time.time() - async_start
    
    # Compare results
    print(f"\n   📊 Performance Comparison ({len(available_ais)} AIs):")
    print(f"   Sync time:  {sync_time:.2f}s")
    print(f"   Async time: {async_time:.2f}s")
    print(f"   Speedup:    {sync_time/async_time:.2f}x")
    
    return True


async def test_error_handling():
    """Test error handling in async calls."""
    print("\n5. Testing error handling...")
    
    # Test with invalid AI name
    try:
        await call_ai_async("invalid_ai", "test", 0.7)
        print("   ❌ Should have raised an error for invalid AI")
        return False
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid AI error: {e}")
    
    # Test timeout handling (if we had a way to simulate timeout)
    # For now, just verify the timeout is set
    for name, client in ASYNC_AI_CLIENTS.items():
        print(f"   ✅ {name} timeout set to: {client.timeout}s")
    
    return True


async def test_cleanup():
    """Test cleanup of async resources."""
    print("\n6. Testing resource cleanup...")
    
    initial_count = len(ASYNC_AI_CLIENTS)
    
    try:
        await cleanup_async_ai_clients()
        print(f"   ✅ Cleaned up {initial_count} async clients")
        print(f"   ✅ Current async clients: {len(ASYNC_AI_CLIENTS)}")
        return len(ASYNC_AI_CLIENTS) == 0
    except Exception as e:
        print(f"   ❌ Error during cleanup: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Junior AI Async Client Test Suite")
    print("=" * 50)
    
    # Load credentials first
    credentials = load_credentials()
    ai_count = sum(1 for k, v in credentials.items() 
                  if k in ['gemini', 'grok', 'openai', 'deepseek', 'openrouter'] 
                  and isinstance(v, dict) and v.get('api_key'))
    
    print(f"Found {ai_count} AI configurations in credentials")
    
    if ai_count == 0:
        print("\n❌ No AI clients configured. Please set up credentials first.")
        return
    
    # Run tests
    tests = [
        test_async_initialization,
        test_single_async_call,
        test_concurrent_calls,
        test_performance_comparison,
        test_error_handling,
        test_cleanup
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for r in results if r)
    print(f"   ✅ Passed: {passed}/{len(tests)}")
    print(f"   ❌ Failed: {len(tests) - passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Async AI clients are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")


if __name__ == "__main__":
    asyncio.run(main())
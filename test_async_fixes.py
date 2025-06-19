#!/usr/bin/env python3
"""Test script to verify the async AI client fixes.

Tests:
1. Resource cleanup with missing close() method
2. Concurrent initialization race condition
3. Error context preservation in fallback
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ai.async_client import AsyncGeminiClient, AsyncOpenAICompatibleClient, AIProvider
from ai.caller import call_ai_async


class TestAsyncClientFixes(unittest.TestCase):
    """Test the three critical fixes in async AI client implementation."""
    
    def test_cleanup_without_close_method(self):
        """Test that cleanup handles clients without close() method gracefully."""
        async def test():
            # Create client
            client = AsyncOpenAICompatibleClient(
                api_key="test-key",
                model="test-model",
                provider=AIProvider.OPENAI
            )
            
            # Mock a client without close() method
            mock_client = Mock()
            # Explicitly remove close attribute
            if hasattr(mock_client, 'close'):
                delattr(mock_client, 'close')
            
            client._client = mock_client
            client._initialized = True
            
            # This should not raise AttributeError
            try:
                await client.cleanup()
                # Success - no exception
                return True
            except AttributeError:
                # This should not happen with our fix
                return False
        
        result = asyncio.run(test())
        self.assertTrue(result, "Cleanup should handle missing close() method")
    
    def test_concurrent_initialization_safety(self):
        """Test that concurrent initialization doesn't cause race conditions."""
        async def test():
            # Track configuration calls
            configure_calls = []
            
            # Mock genai.configure to track calls
            with patch('google.generativeai.configure') as mock_configure:
                mock_configure.side_effect = lambda **kwargs: configure_calls.append(kwargs)
                
                with patch('google.generativeai.GenerativeModel'):
                    client = AsyncGeminiClient(
                        api_key="test-key",
                        model="test-model"
                    )
                    
                    # Launch multiple concurrent initializations
                    tasks = [client.initialize() for _ in range(10)]
                    await asyncio.gather(*tasks)
                    
                    # Should only configure once despite concurrent calls
                    return len(configure_calls)
        
        # Need to mock the module
        with patch.dict('sys.modules', {'google.generativeai': Mock()}):
            result = asyncio.run(test())
            self.assertEqual(result, 1, "genai.configure should only be called once")
    
    def test_error_context_preservation(self):
        """Test that both async and sync errors are preserved in fallback."""
        async def test():
            # Mock async client to fail
            with patch('core.ai_clients.get_async_ai_client') as mock_get_async:
                mock_get_async.return_value = Mock()  # Client exists
                
                with patch('ai.caller.async_call_ai') as mock_async_call:
                    mock_async_call.side_effect = ValueError("Async error: rate limit")
                    
                    # Mock sync call to also fail
                    with patch('ai.caller.call_ai') as mock_sync_call:
                        mock_sync_call.side_effect = KeyError("Sync error: invalid key")
                        
                        # This should raise RuntimeError with both error contexts
                        try:
                            await call_ai_async("test_ai", "test prompt")
                            return None, None  # Should not reach here
                        except RuntimeError as e:
                            return str(e), e.__cause__
        
        # Mock has_async_support to return True
        with patch('ai.caller.has_async_support', return_value=True):
            with patch('ai.caller.HAS_ASYNC_CLIENTS', True):
                error_msg, cause = asyncio.run(test())
                
                # Verify both errors are in the message
                self.assertIsNotNone(error_msg)
                self.assertIn("Async error: rate limit", error_msg)
                self.assertIn("Sync error: invalid key", error_msg)
                self.assertIsInstance(cause, KeyError)


class TestAsyncClientIntegration(unittest.TestCase):
    """Integration tests for async client functionality."""
    
    def test_async_timeout_handling(self):
        """Test that timeouts are properly enforced."""
        async def test():
            client = AsyncOpenAICompatibleClient(
                api_key="test-key",
                model="test-model",
                provider=AIProvider.OPENAI,
                timeout=0.1  # Very short timeout
            )
            
            # Mock slow API call
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            
            async def slow_create(**kwargs):
                await asyncio.sleep(1.0)  # Longer than timeout
                return mock_response
            
            mock_client.chat.completions.create = slow_create
            client._client = mock_client
            client._initialized = True
            
            # Should raise timeout error
            with self.assertRaises(asyncio.TimeoutError):
                await client.generate("test prompt")
        
        asyncio.run(test())


if __name__ == "__main__":
    print("🧪 Testing Async AI Client Fixes")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)
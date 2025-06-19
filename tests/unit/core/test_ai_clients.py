"""Unit tests for core.ai_clients module"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest

from core import ai_clients


@pytest.mark.unit
class TestInitializeGeminiClient:
    """Test suite for initialize_gemini_client function"""
    
    def test_initialize_gemini_success(self, mock_gemini_client, reset_global_state):
        """Test successful Gemini client initialization"""
        api_key = "test-key"
        model_name = "gemini-pro"
        
        with patch.dict('sys.modules', {'google.generativeai': mock_gemini_client}):
            result = ai_clients.initialize_gemini_client(api_key, model_name)
        
        assert result is not None
        mock_gemini_client.configure.assert_called_once_with(api_key=api_key)
        mock_gemini_client.GenerativeModel.assert_called_once_with(model_name)
    
    def test_initialize_gemini_exception(self, mock_gemini_client, reset_global_state):
        """Test Gemini initialization with exception"""
        api_key = "test-key"
        mock_gemini_client.configure.side_effect = Exception("API Error")
        
        with patch.dict('sys.modules', {'google.generativeai': mock_gemini_client}):
            result = ai_clients.initialize_gemini_client(api_key)
        
        assert result is None


@pytest.mark.unit
class TestInitializeOpenAICompatibleClient:
    """Test suite for initialize_openai_compatible_client function"""
    
    def test_initialize_openai_success(self, mock_openai_client, reset_global_state):
        """Test successful OpenAI client initialization"""
        service_name = "openai"
        api_key = "test-key"
        
        with patch.dict('sys.modules', {'openai': Mock(OpenAI=mock_openai_client)}):
            result = ai_clients.initialize_openai_compatible_client(
                service_name, api_key
            )
        
        assert result is not None
        mock_openai_client.assert_called_once_with(api_key=api_key)
    
    def test_initialize_grok_with_base_url(self, mock_openai_client, reset_global_state):
        """Test Grok client initialization with custom base URL"""
        service_name = "grok"
        api_key = "test-key"
        base_url = "https://api.x.ai/v1"
        model_name = "grok-beta"
        
        with patch.dict('sys.modules', {'openai': Mock(OpenAI=mock_openai_client)}):
            result = ai_clients.initialize_openai_compatible_client(
                service_name, api_key, base_url, model_name
            )
        
        assert result is not None
        mock_openai_client.assert_called_once_with(
            api_key=api_key,
            base_url=base_url
        )
    
    def test_initialize_deepseek_custom_model(self, mock_openai_client, reset_global_state):
        """Test DeepSeek client with custom model"""
        service_name = "deepseek"
        api_key = "test-key"
        base_url = "https://api.deepseek.com/v1"
        model_name = "deepseek-chat"
        
        with patch.dict('sys.modules', {'openai': Mock(OpenAI=mock_openai_client)}):
            result = ai_clients.initialize_openai_compatible_client(
                service_name, api_key, base_url, model_name
            )
        
        assert result is not None
    
    def test_initialize_client_import_error(self, reset_global_state):
        """Test client initialization when openai not installed"""
        service_name = "openai"
        api_key = "test-key"
        
        with patch.dict('sys.modules', {'openai': None}):
            result = ai_clients.initialize_openai_compatible_client(
                service_name, api_key
            )
        
        assert result is None


@pytest.mark.unit
class TestInitializeAllClients:
    """Test suite for initialize_all_clients function"""
    
    def test_initialize_all_clients_success(self, mock_credentials, reset_global_state):
        """Test successful initialization of all clients"""
        with patch('core.config.get_credentials', return_value=mock_credentials):
            with patch('core.ai_clients.initialize_gemini_client', return_value=Mock()) as mock_gemini:
                with patch('core.ai_clients.initialize_openai_compatible_client', return_value=Mock()) as mock_openai:
                    result = ai_clients.initialize_all_clients()
        
        # Verify all clients were attempted
        mock_gemini.assert_called_once_with("test-gemini-key", "gemini-pro")
        
        # OpenAI-compatible clients should be called 4 times
        assert mock_openai.call_count == 4
        
        # Verify the calls for each AI service
        expected_calls = [
            call("grok", "test-grok-key", "https://api.x.ai/v1", model_name="grok-beta"),
            call("openai", "test-openai-key", model_name="gpt-4o-mini"),
            call("deepseek", "test-deepseek-key", "https://api.deepseek.com/v1", model_name="deepseek-chat"),
            call("openrouter", "test-openrouter-key", "https://openrouter.ai/api/v1", model_name="openai/gpt-4")
        ]
        mock_openai.assert_has_calls(expected_calls, any_order=True)
        
        # Should return the global AI_CLIENTS dict
        assert result == ai_clients.AI_CLIENTS
    
    def test_initialize_all_clients_partial_success(self, mock_credentials, reset_global_state):
        """Test initialization when some clients fail"""
        with patch('core.config.get_credentials', return_value=mock_credentials):
            with patch('core.ai_clients.initialize_gemini_client', return_value=None):
                with patch('core.ai_clients.initialize_openai_compatible_client', side_effect=[Mock(), None, Mock(), None]):
                    ai_clients.initialize_all_clients()
        
        # Should have 2 successful clients
        assert len(ai_clients.AI_CLIENTS) == 2
    
    def test_initialize_all_clients_all_fail(self, mock_credentials, reset_global_state):
        """Test initialization when all clients fail"""
        with patch('core.config.get_credentials', return_value=mock_credentials):
            with patch('core.ai_clients.initialize_gemini_client', return_value=None):
                with patch('core.ai_clients.initialize_openai_compatible_client', return_value=None):
                    ai_clients.initialize_all_clients()
        
        assert len(ai_clients.AI_CLIENTS) == 0


@pytest.mark.unit
class TestGetAIClient:
    """Test suite for get_ai_client function"""
    
    def test_get_ai_client_exists(self, reset_global_state):
        """Test retrieving existing AI client"""
        mock_client = Mock()
        ai_clients.AI_CLIENTS["openai"] = {"client": mock_client}
        
        result = ai_clients.get_ai_client("openai")
        
        assert result == mock_client
    
    def test_get_ai_client_not_exists(self, reset_global_state):
        """Test retrieving non-existent AI client"""
        result = ai_clients.get_ai_client("nonexistent")
        
        assert result is None
    
    def test_get_ai_client_case_insensitive(self, reset_global_state):
        """Test case-insensitive client retrieval"""
        mock_client = Mock()
        ai_clients.AI_CLIENTS["openai"] = {"client": mock_client}
        
        result = ai_clients.get_ai_client("OpenAI")
        
        assert result == mock_client


@pytest.mark.unit
class TestAsyncSupport:
    """Test suite for async AI client support"""
    
    def test_import_async_support_success(self):
        """Test successful import of async support"""
        mock_module = MagicMock()
        
        with patch('sys.path', ['/test/path']):
            with patch('importlib.import_module', return_value=mock_module):
                result = ai_clients._import_async_support()
        
        assert result == mock_module
    
    def test_import_async_support_failure(self):
        """Test failed import of async support"""
        with patch('importlib.import_module', side_effect=ImportError):
            result = ai_clients._import_async_support()
        
        assert result is None
    
    def test_has_async_support_true(self):
        """Test async support detection when available"""
        with patch('core.ai_clients._import_async_support', return_value=Mock()):
            result = ai_clients.has_async_support()
        
        assert result is True
    
    def test_has_async_support_false(self):
        """Test async support detection when not available"""
        with patch('core.ai_clients._import_async_support', return_value=None):
            result = ai_clients.has_async_support()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_all_clients_async(self, mock_credentials, reset_global_state):
        """Test async client initialization"""
        mock_async_module = MagicMock()
        mock_async_module.create_async_clients = MagicMock(return_value={
            "openai": Mock(),
            "gemini": Mock()
        })
        
        with patch('core.ai_clients._import_async_support', return_value=mock_async_module):
            await ai_clients.initialize_all_clients_async(mock_credentials)
        
        assert len(ai_clients.ASYNC_AI_CLIENTS) == 2
        assert "openai" in ai_clients.ASYNC_AI_CLIENTS
        assert "gemini" in ai_clients.ASYNC_AI_CLIENTS
    
    @pytest.mark.asyncio
    async def test_cleanup_async_ai_clients(self, reset_global_state):
        """Test async client cleanup"""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        ai_clients.ASYNC_AI_CLIENTS = {
            "openai": mock_client1,
            "gemini": mock_client2
        }
        
        await ai_clients.cleanup_async_ai_clients()
        
        mock_client1.cleanup.assert_called_once()
        mock_client2.cleanup.assert_called_once()
        assert len(ai_clients.ASYNC_AI_CLIENTS) == 0
    
    def test_get_async_ai_client(self, reset_global_state):
        """Test retrieving async AI client"""
        mock_client = Mock()
        ai_clients.ASYNC_AI_CLIENTS["openai"] = mock_client
        
        result = ai_clients.get_async_ai_client("openai")
        
        assert result == mock_client


@pytest.mark.unit
class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_get_available_ai_clients(self, reset_global_state):
        """Test getting list of available AI clients"""
        ai_clients.AI_CLIENTS = {
            "openai": {"client": Mock()},
            "gemini": {"client": Mock()},
            "grok": {"client": Mock()}
        }
        
        result = ai_clients.get_available_ai_clients()
        
        assert sorted(result) == ["gemini", "grok", "openai"]
    
    def test_get_available_ai_clients_empty(self, reset_global_state):
        """Test getting available clients when none initialized"""
        result = ai_clients.get_available_ai_clients()
        
        assert result == []


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_initialize_client_with_none_credentials(self, reset_global_state):
        """Test initialization with None credentials"""
        result = ai_clients.initialize_gemini_client(None)
        
        assert result is None
    
    def test_initialize_client_with_empty_credentials(self, reset_global_state):
        """Test initialization with empty credentials"""
        result = ai_clients.initialize_gemini_client({})
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_async_client_cleanup_with_exception(self, reset_global_state):
        """Test async cleanup when client raises exception"""
        mock_client = AsyncMock()
        mock_client.cleanup.side_effect = Exception("Cleanup failed")
        ai_clients.ASYNC_AI_CLIENTS = {"openai": mock_client}
        
        # Should not raise exception
        await ai_clients.cleanup_async_ai_clients()
        
        assert len(ai_clients.ASYNC_AI_CLIENTS) == 0


# Helper class for async testing
class AsyncMock(MagicMock):
    async def cleanup(self):
        pass
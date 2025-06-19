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
        mock_gemini_client = Mock()
        mock_openai_client = Mock()
        
        with patch('core.ai_clients.get_credentials', return_value=mock_credentials):
            with patch('core.ai_clients.initialize_gemini_client', return_value=mock_gemini_client) as mock_gemini:
                with patch('core.ai_clients.initialize_openai_compatible_client', return_value=mock_openai_client) as mock_openai:
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
        
        # Verify clients were stored directly in AI_CLIENTS
        assert len(ai_clients.AI_CLIENTS) == 5  # gemini + 4 openai-compatible
        assert ai_clients.AI_CLIENTS['gemini'] == mock_gemini_client
        assert ai_clients.AI_CLIENTS['grok'] == mock_openai_client
    
    def test_initialize_all_clients_partial_success(self, mock_credentials, reset_global_state):
        """Test initialization when some clients fail"""
        mock_client = Mock()
        
        with patch('core.ai_clients.get_credentials', return_value=mock_credentials):
            with patch('core.ai_clients.initialize_gemini_client', return_value=None):
                with patch('core.ai_clients.initialize_openai_compatible_client', side_effect=[mock_client, None, mock_client, None]):
                    ai_clients.initialize_all_clients()
        
        # Should have 2 successful clients (grok and deepseek)
        assert len(ai_clients.AI_CLIENTS) == 2
        assert 'grok' in ai_clients.AI_CLIENTS
        assert 'deepseek' in ai_clients.AI_CLIENTS
    
    def test_initialize_all_clients_all_fail(self, mock_credentials, reset_global_state):
        """Test initialization when all clients fail"""
        with patch('core.ai_clients.get_credentials', return_value=mock_credentials):
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
        ai_clients.AI_CLIENTS["openai"] = mock_client
        
        result = ai_clients.get_ai_client("openai")
        
        assert result == mock_client
    
    def test_get_ai_client_not_exists(self, reset_global_state):
        """Test retrieving non-existent AI client"""
        result = ai_clients.get_ai_client("nonexistent")
        
        assert result is None
    
    def test_get_ai_client_case_insensitive(self, reset_global_state):
        """Test case-insensitive client retrieval"""
        mock_client = Mock()
        ai_clients.AI_CLIENTS["openai"] = mock_client
        
        result = ai_clients.get_ai_client("OpenAI")
        
        # Note: Current implementation is case-sensitive, so this should return None
        assert result is None


@pytest.mark.unit
class TestAsyncSupport:
    """Test suite for async AI client support"""
    
    def test_import_async_support_success(self):
        """Test successful import of async support"""
        mock_module = MagicMock()
        mock_module.ASYNC_AI_CLIENTS = {}
        
        with patch('sys.path', ['/test/path']):
            with patch.dict('sys.modules', {'ai.async_client': mock_module}):
                result = ai_clients._import_async_support()
        
        assert result is True  # Function returns True on success
    
    def test_import_async_support_failure(self):
        """Test failed import of async support"""
        with patch.dict('sys.modules', {'ai.async_client': None}):
            result = ai_clients._import_async_support()
        
        assert result is False  # Function returns False on failure
    
    def test_has_async_support_true(self):
        """Test async support detection when available"""
        # Reset the global flag first
        ai_clients.HAS_ASYNC_SUPPORT = False
        
        def mock_import_async_support():
            ai_clients.HAS_ASYNC_SUPPORT = True
            return True
        
        with patch('core.ai_clients._import_async_support', side_effect=mock_import_async_support):
            result = ai_clients.has_async_support()
        
        assert result is True
    
    def test_has_async_support_false(self):
        """Test async support detection when not available"""
        # Reset the global flag first
        ai_clients.HAS_ASYNC_SUPPORT = False
        
        def mock_import_async_support_fail():
            ai_clients.HAS_ASYNC_SUPPORT = False
            return False
        
        with patch('core.ai_clients._import_async_support', side_effect=mock_import_async_support_fail):
            result = ai_clients.has_async_support()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_all_clients_async(self, mock_credentials, reset_global_state):
        """Test async client initialization"""
        # Mock the sync client initialization first
        mock_sync_clients = {"openai": Mock(), "gemini": Mock()}
        
        # Mock async initialization function
        mock_async_init = MagicMock()
        
        with patch('core.ai_clients.initialize_all_clients', return_value=mock_sync_clients):
            with patch('core.ai_clients._import_async_support', return_value=True):
                with patch('core.ai_clients.get_credentials', return_value=mock_credentials):
                    with patch.dict('sys.modules', {'ai.async_client': MagicMock(initialize_async_ai_clients=mock_async_init)}):
                        result = await ai_clients.initialize_all_clients_async()
        
        # Should return sync clients
        assert result == mock_sync_clients
        
        # Should have called async initialization
        mock_async_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_async_ai_clients(self, reset_global_state):
        """Test async client cleanup"""
        # Mock cleanup function
        mock_cleanup_func = MagicMock()
        
        with patch('core.ai_clients._import_async_support', return_value=True):
            with patch.dict('sys.modules', {'ai.async_client': MagicMock(cleanup_async_ai_clients=mock_cleanup_func)}):
                await ai_clients.cleanup_async_ai_clients()
        
        # Should have called the cleanup function
        mock_cleanup_func.assert_called_once()
    
    def test_get_async_ai_client(self, reset_global_state):
        """Test retrieving async AI client"""
        mock_client = Mock()
        # Directly modify the global dict since it's imported dynamically
        with patch.dict('core.ai_clients.ASYNC_AI_CLIENTS', {"openai": mock_client}):
            with patch('core.ai_clients.HAS_ASYNC_SUPPORT', True):
                result = ai_clients.get_async_ai_client("openai")
        
        assert result == mock_client


@pytest.mark.unit
class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_get_available_ai_clients(self, reset_global_state):
        """Test getting list of available AI clients"""
        ai_clients.AI_CLIENTS = {
            "openai": Mock(),
            "gemini": Mock(),
            "grok": Mock()
        }
        
        result = ai_clients.get_available_ai_clients()
        
        assert result == ai_clients.AI_CLIENTS  # Returns the actual dictionary
    
    def test_get_available_ai_clients_empty(self, reset_global_state):
        """Test getting available clients when none initialized"""
        result = ai_clients.get_available_ai_clients()
        
        assert result == {}  # Returns empty dictionary, not list


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_initialize_client_with_none_credentials(self, reset_global_state):
        """Test initialization with None credentials"""
        result = ai_clients.initialize_gemini_client(None, 'gemini-2.0-flash')
        
        assert result is None
    
    def test_initialize_client_with_empty_credentials(self, reset_global_state):
        """Test initialization with empty credentials"""
        result = ai_clients.initialize_gemini_client('', 'gemini-2.0-flash')
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_async_client_cleanup_with_exception(self, reset_global_state):
        """Test async cleanup when cleanup function raises exception"""
        mock_cleanup_func = MagicMock(side_effect=Exception("Cleanup failed"))
        
        with patch('core.ai_clients._import_async_support', return_value=True):
            with patch.dict('sys.modules', {'ai.async_client': MagicMock(cleanup_async_ai_clients=mock_cleanup_func)}):
                # Should not raise exception
                await ai_clients.cleanup_async_ai_clients()
        
        # Should have attempted cleanup
        mock_cleanup_func.assert_called_once()


# Helper class for async testing
class AsyncMock(MagicMock):
    async def cleanup(self):
        pass
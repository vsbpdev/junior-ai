"""Shared fixtures and mocks for core module tests"""

import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_credentials():
    """Mock credentials for testing"""
    return {
        "model": "gpt-4",
        "gemini": {
            "api_key": "test-gemini-key",
            "model": "gemini-pro"
        },
        "openai": {
            "api_key": "test-openai-key",
            "model": "gpt-4o-mini"
        },
        "grok": {
            "api_key": "test-grok-key",
            "base_url": "https://api.x.ai/v1",
            "model": "grok-beta"
        },
        "deepseek": {
            "api_key": "test-deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        },
        "openrouter": {
            "api_key": "test-openrouter-key",
            "model": "openai/gpt-4"
        },
        "pattern_detection": {
            "enabled": True,
            "sensitivity": "medium",
            "min_matches": 2,
            "context_lines": 3,
            "categories": {
                "security": {"enabled": True, "sensitivity": "high"},
                "uncertainty": {"enabled": True},
                "algorithm": {"enabled": True},
                "gotcha": {"enabled": True},
                "architecture": {"enabled": True}
            }
        },
        "ai_consultation_preferences": {
            "strategy": "smart",
            "require_consensus": False,
            "min_ai_responses": 1,
            "timeout": 30
        },
        "ai_consultation": {
            "strategy": "smart",
            "require_consensus": False,
            "min_ai_responses": 1,
            "timeout": 30
        }
    }


@pytest.fixture
def temp_credentials_file(tmp_path, mock_credentials):
    """Create a temporary credentials file"""
    cred_file = tmp_path / "credentials.json"
    cred_file.write_text(json.dumps(mock_credentials, indent=2))
    return str(cred_file)


@pytest.fixture
def mock_gemini_client():
    """Mock Google Gemini client"""
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_client.GenerativeModel.return_value = mock_model
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock_client = MagicMock()
    mock_client.return_value = mock_client
    return mock_client


@pytest.fixture
def mock_secure_credential_manager():
    """Mock SecureCredentialManager"""
    mock_manager = MagicMock()
    mock_manager.load_credentials.return_value = None
    mock_manager.migrate_from_json.return_value = True
    return mock_manager


@pytest.fixture
def reset_global_state():
    """Reset global state before and after tests"""
    # Import modules to reset their state
    import core.config
    import core.ai_clients
    
    # Reset before test
    core.config.CREDENTIALS = None
    core.ai_clients.AI_CLIENTS = {}
    core.ai_clients.ASYNC_AI_CLIENTS = {}
    
    yield
    
    # Reset after test
    core.config.CREDENTIALS = None
    core.ai_clients.AI_CLIENTS = {}
    core.ai_clients.ASYNC_AI_CLIENTS = {}


@pytest.fixture
def mock_ai_sdks():
    """Mock all AI SDK imports"""
    with patch.dict('sys.modules', {
        'google.generativeai': MagicMock(),
        'openai': MagicMock(),
        'google': MagicMock(),
        'google.generativeai': MagicMock()
    }):
        yield
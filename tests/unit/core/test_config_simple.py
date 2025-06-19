"""Simplified unit tests for core.config module"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core import config


@pytest.mark.unit
class TestGetCredentials:
    """Test suite for get_credentials function"""
    
    def test_get_credentials_lazy_loading(self, mock_credentials, reset_global_state):
        """Test lazy loading of credentials"""
        # CREDENTIALS should be None initially
        assert config.CREDENTIALS is None
        
        with patch('core.config.load_credentials', return_value=mock_credentials):
            result = config.get_credentials()
            
        assert result == mock_credentials
    
    def test_get_credentials_uses_cache(self, mock_credentials, reset_global_state):
        """Test that get_credentials uses cached value"""
        config.CREDENTIALS = mock_credentials
        
        with patch('core.config.load_credentials') as mock_load:
            result = config.get_credentials()
            
        assert result == mock_credentials
        mock_load.assert_not_called()


@pytest.mark.unit 
class TestPatternDetectionConfig:
    """Test suite for get_pattern_detection_config function"""
    
    def test_get_pattern_detection_config_full(self, mock_credentials, reset_global_state):
        """Test getting full pattern detection config"""
        with patch('core.config.get_credentials', return_value=mock_credentials):
            result = config.get_pattern_detection_config()
            
        assert result == mock_credentials["pattern_detection"]
        assert result["enabled"] is True
        assert result["sensitivity"] == "medium"
    
    def test_get_pattern_detection_config_missing(self, reset_global_state):
        """Test default pattern detection config when missing"""
        creds_without_pattern = {"model": "gpt-4"}
        
        with patch('core.config.get_credentials', return_value=creds_without_pattern):
            result = config.get_pattern_detection_config()
            
        assert result == {}
    
    def test_get_pattern_detection_config_partial(self, reset_global_state):
        """Test pattern detection config with partial values"""
        partial_creds = {
            "pattern_detection": {
                "enabled": False
            }
        }
        
        with patch('core.config.get_credentials', return_value=partial_creds):
            result = config.get_pattern_detection_config()
            
        assert result["enabled"] is False


@pytest.mark.unit
class TestAIConsultationConfig:
    """Test suite for get_ai_consultation_config function"""
    
    def test_get_ai_consultation_config_full(self, mock_credentials, reset_global_state):
        """Test getting full AI consultation config"""
        with patch('core.config.get_credentials', return_value=mock_credentials):
            result = config.get_ai_consultation_config()
            
        assert result == mock_credentials["ai_consultation_preferences"]
        assert result["strategy"] == "smart"
        assert result["timeout"] == 30
    
    def test_get_ai_consultation_config_missing(self, reset_global_state):
        """Test default AI consultation config when missing"""
        creds_without_ai_consultation = {"model": "gpt-4"}
        
        with patch('core.config.get_credentials', return_value=creds_without_ai_consultation):
            result = config.get_ai_consultation_config()
            
        assert result == {}
    
    def test_get_ai_consultation_config_partial(self, reset_global_state):
        """Test AI consultation config with partial values"""
        partial_creds = {
            "ai_consultation_preferences": {
                "strategy": "all",
                "timeout": 60
            }
        }
        
        with patch('core.config.get_credentials', return_value=partial_creds):
            result = config.get_ai_consultation_config()
            
        assert result["strategy"] == "all"
        assert result["timeout"] == 60


@pytest.mark.unit
class TestGlobalState:
    """Test global state management"""
    
    def test_credentials_caching(self, mock_credentials, reset_global_state):
        """Test that credentials are cached properly"""
        # Set up initial credentials to test caching
        config.CREDENTIALS = mock_credentials
        
        with patch('core.config.load_credentials') as mock_load:
            # First call should use cache
            result1 = config.get_credentials()
            # Second call should also use cache
            result2 = config.get_credentials()
            
        assert result1 == result2
        assert result1 is result2  # Same object reference
        mock_load.assert_not_called()  # Should not call load_credentials due to caching
    
    def test_config_version(self):
        """Test that version is available"""
        assert hasattr(config, '__version__')
        assert isinstance(config.__version__, str)
        assert config.__version__ == "2.1.0"
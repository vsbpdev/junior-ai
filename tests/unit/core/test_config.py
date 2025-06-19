"""Unit tests for core.config module"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

import pytest

from core import config


@pytest.mark.unit
class TestLoadCredentials:
    """Test suite for load_credentials function"""
    
    
    def test_load_credentials_invalid_json(self, tmp_path, reset_global_state):
        """Test handling of invalid JSON in credentials file"""
        cred_file = tmp_path / "bad_credentials.json"
        cred_file.write_text("invalid json content")
        
        with patch('core.config.CREDENTIALS_FILE', str(cred_file)):
            with patch('sys.exit') as mock_exit:
                config.load_credentials()
                
        mock_exit.assert_called_once_with(1)
    
    


@pytest.mark.unit
class TestGetCredentials:
    """Test suite for get_credentials function"""
    
    
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
        config.CREDENTIALS = mock_credentials
        
        result = config.get_pattern_detection_config()
        
        assert result == mock_credentials["pattern_detection"]
        assert result["enabled"] is True
        assert result["sensitivity"] == "medium"
    
    def test_get_pattern_detection_config_missing(self, reset_global_state):
        """Test default pattern detection config when missing"""
        config.CREDENTIALS = {"model": "gpt-4"}
        
        result = config.get_pattern_detection_config()
        
        assert result == {}
    
    def test_get_pattern_detection_config_partial(self, reset_global_state):
        """Test pattern detection config with partial values"""
        config.CREDENTIALS = {
            "pattern_detection": {
                "enabled": False
            }
        }
        
        result = config.get_pattern_detection_config()
        
        assert result["enabled"] is False


@pytest.mark.unit
class TestAIConsultationConfig:
    """Test suite for get_ai_consultation_config function"""
    
    def test_get_ai_consultation_config_full(self, mock_credentials, reset_global_state):
        """Test getting full AI consultation config"""
        config.CREDENTIALS = mock_credentials
        
        result = config.get_ai_consultation_config()
        
        assert result == mock_credentials["ai_consultation"]
        assert result["strategy"] == "smart"
        assert result["timeout"] == 30
    
    def test_get_ai_consultation_config_missing(self, reset_global_state):
        """Test default AI consultation config when missing"""
        config.CREDENTIALS = {"model": "gpt-4"}
        
        result = config.get_ai_consultation_config()
        
        assert result == {}
    
    def test_get_ai_consultation_config_partial(self, reset_global_state):
        """Test AI consultation config with partial values"""
        config.CREDENTIALS = {
            "ai_consultation": {
                "strategy": "all",
                "timeout": 60
            }
        }
        
        result = config.get_ai_consultation_config()
        
        assert result["strategy"] == "all"
        assert result["timeout"] == 60



@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_credentials_file_permission_error(self, reset_global_state):
        """Test handling of permission errors when reading credentials"""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', side_effect=PermissionError):
                with patch('sys.exit') as mock_exit:
                    config.load_credentials()
                    
        mock_exit.assert_called_once_with(1)
    
    def test_credentials_file_unicode_error(self, tmp_path, reset_global_state):
        """Test handling of unicode errors in credentials file"""
        cred_file = tmp_path / "unicode_credentials.json"
        cred_file.write_bytes(b'\xff\xfe')  # Invalid UTF-8
        
        with patch('core.config.CREDENTIALS_FILE', str(cred_file)):
            with patch('sys.exit') as mock_exit:
                config.load_credentials()
                
        mock_exit.assert_called_once_with(1)
    

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
    
    def test_load_credentials_success(self, temp_credentials_file, mock_credentials, reset_global_state):
        """Test successful credential loading from JSON file"""
        mock_manager = MagicMock()
        mock_manager.load_credentials.return_value = mock_credentials
        
        with patch('core.config.SecureCredentialManager', return_value=mock_manager):
            result = config.load_credentials()
            
        assert result == mock_credentials
        assert config.CREDENTIALS == mock_credentials
    
    def test_load_credentials_with_secure_storage(self, mock_credentials, reset_global_state):
        """Test credential loading with secure storage available"""
        mock_manager = MagicMock()
        mock_manager.load_credentials.return_value = mock_credentials
        
        with patch('core.config.SecureCredentialManager', return_value=mock_manager):
            result = config.load_credentials()
            
        assert result == mock_credentials
        assert config.CREDENTIALS == mock_credentials
        mock_manager.load_credentials.assert_called_once()
    
    def test_load_credentials_migration_needed(self, temp_credentials_file, mock_credentials, mock_secure_credential_manager, reset_global_state):
        """Test credential migration from JSON to secure storage"""
        mock_secure_credential_manager.load_credentials.return_value = None
        
        with patch('core.config.CREDENTIALS_FILE', temp_credentials_file):
            with patch('core.config._import_secure_credentials', return_value=mock_secure_credential_manager):
                result = config.load_credentials()
        
        assert result == mock_credentials
        mock_secure_credential_manager.migrate_from_json.assert_called_once_with(temp_credentials_file)
    
    def test_load_credentials_no_file_exits(self, reset_global_state):
        """Test that system exits when no credentials file exists"""
        with patch('pathlib.Path.exists', return_value=False):
            with patch('sys.exit') as mock_exit:
                config.load_credentials()
                
        mock_exit.assert_called_once_with(1)
    
    def test_load_credentials_invalid_json(self, tmp_path, reset_global_state):
        """Test handling of invalid JSON in credentials file"""
        cred_file = tmp_path / "bad_credentials.json"
        cred_file.write_text("invalid json content")
        
        with patch('core.config.CREDENTIALS_FILE', str(cred_file)):
            with patch('sys.exit') as mock_exit:
                config.load_credentials()
                
        mock_exit.assert_called_once_with(1)
    
    def test_load_credentials_secure_import_fails(self, temp_credentials_file, mock_credentials, reset_global_state):
        """Test fallback when secure credentials import fails"""
        with patch('core.config.CREDENTIALS_FILE', temp_credentials_file):
            with patch('core.config._import_secure_credentials', side_effect=ImportError):
                result = config.load_credentials()
                
        assert result == mock_credentials
    
    def test_load_credentials_caching(self, temp_credentials_file, mock_credentials, reset_global_state):
        """Test that credentials are cached after first load"""
        with patch('core.config.CREDENTIALS_FILE', temp_credentials_file):
            # First load
            result1 = config.load_credentials()
            # Second load should use cache
            result2 = config.load_credentials()
            
        assert result1 == result2
        assert result1 is result2  # Same object reference


@pytest.mark.unit
class TestGetCredentials:
    """Test suite for get_credentials function"""
    
    def test_get_credentials_lazy_loading(self, temp_credentials_file, mock_credentials, reset_global_state):
        """Test lazy loading of credentials"""
        with patch('core.config.CREDENTIALS_FILE', temp_credentials_file):
            # CREDENTIALS should be None initially
            assert config.CREDENTIALS is None
            
            # get_credentials should trigger loading
            result = config.get_credentials()
            
            assert result == mock_credentials
            assert config.CREDENTIALS == mock_credentials
    
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
        
        assert result == {
            "enabled": True,
            "sensitivity": "medium"
        }
    
    def test_get_pattern_detection_config_partial(self, reset_global_state):
        """Test pattern detection config with partial values"""
        config.CREDENTIALS = {
            "pattern_detection": {
                "enabled": False
            }
        }
        
        result = config.get_pattern_detection_config()
        
        assert result["enabled"] is False
        assert result["sensitivity"] == "medium"  # Default value


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
        
        assert result == {
            "strategy": "smart",
            "require_consensus": False,
            "min_ai_responses": 1,
            "timeout": 30
        }
    
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
        assert result["require_consensus"] is False  # Default value
        assert result["min_ai_responses"] == 1  # Default value


@pytest.mark.unit
class TestImportSecureCredentials:
    """Test suite for _import_secure_credentials function"""
    
    def test_import_secure_credentials_success(self):
        """Test successful import of secure credentials module"""
        mock_module = MagicMock()
        mock_manager = MagicMock()
        mock_module.SecureCredentialManager = mock_manager
        
        with patch('importlib.import_module', return_value=mock_module):
            result = config._import_secure_credentials()
            
        assert result == mock_manager
    
    def test_import_secure_credentials_failure(self):
        """Test failed import of secure credentials module"""
        with patch('importlib.import_module', side_effect=ImportError):
            result = config._import_secure_credentials()
            
        assert result is None
    
    def test_import_secure_credentials_attribute_error(self):
        """Test import with missing SecureCredentialManager class"""
        mock_module = MagicMock()
        del mock_module.SecureCredentialManager
        
        with patch('importlib.import_module', return_value=mock_module):
            result = config._import_secure_credentials()
            
        assert result is None


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
    
    def test_concurrent_credential_loading(self, temp_credentials_file, mock_credentials, reset_global_state):
        """Test thread safety of credential loading"""
        import threading
        
        results = []
        
        def load_creds():
            with patch('core.config.CREDENTIALS_FILE', temp_credentials_file):
                results.append(config.get_credentials())
        
        threads = [threading.Thread(target=load_creds) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should get the same credentials object
        assert all(r == mock_credentials for r in results)
        assert all(r is results[0] for r in results)  # Same object reference
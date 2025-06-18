#!/usr/bin/env python3
"""
Comprehensive test suite for secure credential management system.

Tests all credential providers, security features, and migration capabilities.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from secure_credentials import (
    SecureCredentialManager,
    CredentialBackend,
    EnvironmentProvider,
    KeyringProvider,
    EncryptedFileProvider,
    PlainJsonProvider,
    check_credential_security,
    load_credentials
)


class TestEnvironmentProvider(unittest.TestCase):
    """Test environment variable credential provider."""
    
    def setUp(self):
        self.provider = EnvironmentProvider()
        # Store original environment
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_load_single_ai_credentials(self):
        """Test loading credentials for a single AI."""
        os.environ['JUNIOR_AI_GEMINI_API_KEY'] = 'test-gemini-key'
        os.environ['JUNIOR_AI_GEMINI_MODEL'] = 'gemini-2.0-flash'
        os.environ['JUNIOR_AI_GEMINI_ENABLED'] = 'true'
        
        creds = self.provider.load('gemini')
        self.assertIsNotNone(creds)
        self.assertEqual(creds['api_key'], 'test-gemini-key')
        self.assertEqual(creds['model'], 'gemini-2.0-flash')
        self.assertTrue(creds['enabled'])
    
    def test_load_all_credentials(self):
        """Test loading all AI credentials."""
        # Set up multiple AI credentials
        os.environ['JUNIOR_AI_GEMINI_API_KEY'] = 'test-gemini-key'
        os.environ['JUNIOR_AI_GROK_API_KEY'] = 'test-grok-key'
        os.environ['JUNIOR_AI_GROK_BASE_URL'] = 'https://api.x.ai/v1'
        os.environ['JUNIOR_AI_PATTERN_ENABLED'] = 'true'
        os.environ['JUNIOR_AI_PATTERN_SENSITIVITY'] = 'high'
        
        creds = self.provider.load('all')
        self.assertIn('gemini', creds)
        self.assertIn('grok', creds)
        self.assertEqual(creds['grok']['base_url'], 'https://api.x.ai/v1')
        self.assertIn('pattern_detection', creds)
        self.assertEqual(creds['pattern_detection']['sensitivity']['global_level'], 'high')
    
    def test_load_nonexistent_credentials(self):
        """Test loading when no credentials exist."""
        creds = self.provider.load('nonexistent')
        self.assertIsNone(creds)
    
    def test_is_available(self):
        """Test that environment provider is always available."""
        self.assertTrue(self.provider.is_available())
    
    def test_save_not_supported(self):
        """Test that save returns False for environment provider."""
        self.assertFalse(self.provider.save('test', {'key': 'value'}))
    
    def test_delete_not_supported(self):
        """Test that delete returns False for environment provider."""
        self.assertFalse(self.provider.delete('test'))


class TestPlainJsonProvider(unittest.TestCase):
    """Test plain JSON credential provider."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.creds_file = Path(self.temp_dir) / "credentials.json"
        self.provider = PlainJsonProvider(self.creds_file)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_credentials(self):
        """Test saving and loading credentials."""
        test_creds = {
            'api_key': 'test-key',
            'model': 'test-model',
            'enabled': True
        }
        
        # Save credentials
        self.assertTrue(self.provider.save('test_ai', test_creds))
        self.assertTrue(self.creds_file.exists())
        
        # Load credentials
        loaded = self.provider.load('test_ai')
        self.assertEqual(loaded, test_creds)
    
    def test_save_and_load_all_credentials(self):
        """Test saving and loading all credentials."""
        all_creds = {
            'gemini': {'api_key': 'gemini-key'},
            'grok': {'api_key': 'grok-key'}
        }
        
        self.assertTrue(self.provider.save('all', all_creds))
        loaded = self.provider.load('all')
        self.assertEqual(loaded, all_creds)
    
    def test_delete_specific_credential(self):
        """Test deleting specific credentials."""
        # Save multiple credentials
        self.provider.save('ai1', {'key': 'value1'})
        self.provider.save('ai2', {'key': 'value2'})
        
        # Delete one
        self.assertTrue(self.provider.delete('ai1'))
        
        # Verify deletion
        self.assertIsNone(self.provider.load('ai1'))
        self.assertIsNotNone(self.provider.load('ai2'))
    
    def test_delete_all_credentials(self):
        """Test deleting all credentials."""
        self.provider.save('test', {'key': 'value'})
        self.assertTrue(self.creds_file.exists())
        
        self.assertTrue(self.provider.delete('all'))
        self.assertFalse(self.creds_file.exists())
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        self.assertIsNone(self.provider.load('test'))
    
    def test_is_available(self):
        """Test that plain JSON provider is always available."""
        self.assertTrue(self.provider.is_available())


class TestEncryptedFileProvider(unittest.TestCase):
    """Test encrypted file credential provider."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.enc_file = Path(self.temp_dir) / "credentials.enc"
        self.provider = EncryptedFileProvider(self.enc_file)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(not EncryptedFileProvider(None).is_available(), 
                     "cryptography not installed")
    def test_save_and_load_encrypted(self):
        """Test saving and loading encrypted credentials."""
        test_creds = {
            'api_key': 'super-secret-key',
            'model': 'test-model'
        }
        
        # Save encrypted
        self.assertTrue(self.provider.save('test_ai', test_creds))
        self.assertTrue(self.enc_file.exists())
        self.assertTrue(self.provider.key_file.exists())
        
        # Verify file is encrypted (not readable as JSON)
        with open(self.enc_file, 'rb') as f:
            content = f.read()
            self.assertNotIn(b'super-secret-key', content)
        
        # Load and verify
        loaded = self.provider.load('test_ai')
        self.assertEqual(loaded, test_creds)
    
    @unittest.skipIf(not EncryptedFileProvider(None).is_available(), 
                     "cryptography not installed")
    def test_encryption_key_persistence(self):
        """Test that encryption key persists across provider instances."""
        self.provider.save('test', {'key': 'value'})
        
        # Create new provider instance
        new_provider = EncryptedFileProvider(self.enc_file)
        loaded = new_provider.load('test')
        self.assertEqual(loaded, {'key': 'value'})
    
    @unittest.skipIf(not EncryptedFileProvider(None).is_available(), 
                     "cryptography not installed")
    def test_delete_encrypted_credentials(self):
        """Test deleting encrypted credentials."""
        self.provider.save('test', {'key': 'value'})
        
        # Delete all
        self.assertTrue(self.provider.delete('all'))
        self.assertFalse(self.enc_file.exists())
        self.assertFalse(self.provider.key_file.exists())
    
    def test_is_available_without_cryptography(self):
        """Test availability check."""
        # This will return True/False based on whether cryptography is installed
        # We're just testing that the method works
        result = self.provider.is_available()
        self.assertIsInstance(result, bool)


class TestKeyringProvider(unittest.TestCase):
    """Test OS keyring credential provider."""
    
    def setUp(self):
        self.provider = KeyringProvider()
    
    @patch('keyring.get_password')
    @patch('keyring.set_password')
    @patch('keyring.delete_password')
    def test_keyring_operations(self, mock_delete, mock_set, mock_get):
        """Test keyring operations with mocks."""
        # Configure mocks
        mock_get.return_value = '{"api_key": "test-key"}'
        mock_set.return_value = None
        mock_delete.return_value = None
        
        # Test load
        creds = self.provider.load('test')
        self.assertEqual(creds, {'api_key': 'test-key'})
        mock_get.assert_called_with('junior-ai-assistant', 'test')
        
        # Test save
        self.assertTrue(self.provider.save('test', {'api_key': 'new-key'}))
        mock_set.assert_called_with(
            'junior-ai-assistant', 
            'test', 
            '{"api_key": "new-key"}'
        )
        
        # Test delete
        self.assertTrue(self.provider.delete('test'))
        mock_delete.assert_called_with('junior-ai-assistant', 'test')
    
    @patch('keyring.get_password', side_effect=Exception("Keyring error"))
    def test_keyring_error_handling(self, mock_get):
        """Test error handling for keyring operations."""
        result = self.provider.load('test')
        self.assertIsNone(result)


class TestSecureCredentialManager(unittest.TestCase):
    """Test the main credential manager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.legacy_file = Path(self.temp_dir) / "credentials.json"
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        # Clean up environment variables
        for key in list(os.environ.keys()):
            if key.startswith('JUNIOR_AI_'):
                del os.environ[key]
    
    def test_load_credentials_priority_order(self):
        """Test that credentials are loaded in priority order."""
        # Set up credentials in multiple backends
        os.environ['JUNIOR_AI_GEMINI_API_KEY'] = 'env-key'
        
        plain_creds = {'gemini': {'api_key': 'json-key'}}
        with open(self.legacy_file, 'w') as f:
            json.dump(plain_creds, f)
        
        # Manager should prefer environment variables
        manager = SecureCredentialManager(legacy_file_path=self.legacy_file)
        creds = manager.load_credentials()
        
        self.assertIn('gemini', creds)
        self.assertEqual(creds['gemini']['api_key'], 'env-key')
        self.assertEqual(manager.active_backend, CredentialBackend.ENVIRONMENT)
    
    def test_fallback_to_plain_json(self):
        """Test fallback to plain JSON when no secure credentials exist."""
        plain_creds = {'gemini': {'api_key': 'json-key'}}
        with open(self.legacy_file, 'w') as f:
            json.dump(plain_creds, f)
        
        manager = SecureCredentialManager(
            providers=[CredentialBackend.PLAIN_JSON],
            legacy_file_path=self.legacy_file
        )
        creds = manager.load_credentials()
        
        self.assertEqual(creds['gemini']['api_key'], 'json-key')
        self.assertEqual(manager.active_backend, CredentialBackend.PLAIN_JSON)
    
    def test_migrate_from_plain_json(self):
        """Test migration from plain JSON to secure storage."""
        # Create legacy credentials
        legacy_creds = {
            'gemini': {'api_key': 'legacy-key'},
            'pattern_detection': {'enabled': True}
        }
        with open(self.legacy_file, 'w') as f:
            json.dump(legacy_creds, f)
        
        # Test migration to encrypted storage
        manager = SecureCredentialManager(legacy_file_path=self.legacy_file)
        
        if CredentialBackend.ENCRYPTED_FILE in manager.get_available_backends():
            success = manager.migrate_from_plain_json(
                target_backend=CredentialBackend.ENCRYPTED_FILE,
                delete_original=False
            )
            self.assertTrue(success)
            
            # Verify original still exists
            self.assertTrue(self.legacy_file.exists())
            
            # Create new manager and verify it loads from encrypted
            new_manager = SecureCredentialManager(legacy_file_path=self.legacy_file)
            creds = new_manager.load_credentials()
            self.assertEqual(creds['gemini']['api_key'], 'legacy-key')
    
    def test_save_credentials(self):
        """Test saving credentials to specific backend."""
        manager = SecureCredentialManager(
            providers=[CredentialBackend.PLAIN_JSON],
            legacy_file_path=self.legacy_file
        )
        
        test_creds = {'test_ai': {'api_key': 'test-key'}}
        self.assertTrue(manager.save_credentials(
            test_creds,
            backend=CredentialBackend.PLAIN_JSON
        ))
        
        # Verify saved
        loaded = manager.load_credentials()
        self.assertEqual(loaded, test_creds)
    
    def test_get_available_backends(self):
        """Test listing available backends."""
        manager = SecureCredentialManager()
        backends = manager.get_available_backends()
        
        # Environment and PlainJSON should always be available
        self.assertIn(CredentialBackend.ENVIRONMENT, backends)
        self.assertIn(CredentialBackend.PLAIN_JSON, backends)


class TestSecurityChecks(unittest.TestCase):
    """Test security assessment functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.legacy_file = Path(self.temp_dir) / "credentials.json"
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        # Clean up environment variables
        for key in list(os.environ.keys()):
            if key.startswith('JUNIOR_AI_'):
                del os.environ[key]
    
    @patch('secure_credentials.SecureCredentialManager')
    def test_check_credential_security(self, mock_manager_class):
        """Test security assessment function."""
        # Mock the manager
        mock_manager = MagicMock()
        mock_manager.get_available_backends.return_value = [
            CredentialBackend.ENVIRONMENT,
            CredentialBackend.PLAIN_JSON
        ]
        mock_manager.load_credentials.return_value = {'test': 'creds'}
        mock_manager.active_backend = CredentialBackend.ENVIRONMENT
        mock_manager.legacy_file_path = self.legacy_file
        mock_manager_class.return_value = mock_manager
        
        # Create legacy file
        with open(self.legacy_file, 'w') as f:
            json.dump({}, f)
        
        assessment = check_credential_security()
        
        self.assertEqual(assessment['active_backend'], 'environment')
        self.assertEqual(assessment['security_level'], 'high')
        self.assertTrue(assessment['has_credentials'])
        self.assertTrue(assessment['plain_json_exists'])
    
    def test_security_levels(self):
        """Test security level assessment for different backends."""
        test_cases = [
            (CredentialBackend.ENVIRONMENT, 'high'),
            (CredentialBackend.KEYRING, 'high'),
            (CredentialBackend.ENCRYPTED_FILE, 'medium'),
            (CredentialBackend.PLAIN_JSON, 'low'),
        ]
        
        for backend, expected_level in test_cases:
            with patch('secure_credentials.SecureCredentialManager') as mock_class:
                mock_manager = MagicMock()
                mock_manager.active_backend = backend
                mock_manager.load_credentials.return_value = {'test': 'data'}
                mock_manager.get_available_backends.return_value = [backend]
                mock_manager.legacy_file_path.exists.return_value = False
                mock_class.return_value = mock_manager
                
                assessment = check_credential_security()
                self.assertEqual(
                    assessment['security_level'], 
                    expected_level,
                    f"Wrong security level for {backend}"
                )


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.creds_file = Path(self.temp_dir) / "credentials.json"
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_credentials_function(self):
        """Test the convenience load_credentials function."""
        # Create test credentials
        test_creds = {
            'gemini': {'api_key': 'test-key'},
            'pattern_detection': {'enabled': True}
        }
        with open(self.creds_file, 'w') as f:
            json.dump(test_creds, f)
        
        # Test loading
        loaded = load_credentials(self.creds_file)
        self.assertEqual(loaded, test_creds)
    
    def test_empty_credentials_handling(self):
        """Test handling of empty or missing credentials."""
        # Test with non-existent file
        loaded = load_credentials(Path(self.temp_dir) / "nonexistent.json")
        self.assertEqual(loaded, {})
        
        # Test with empty file
        self.creds_file.write_text("{}")
        loaded = load_credentials(self.creds_file)
        self.assertEqual(loaded, {})


class TestEnvironmentVariableIntegration(unittest.TestCase):
    """Test .env file integration with python-dotenv."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env_file = Path(self.temp_dir) / ".env"
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_dotenv_loading(self):
        """Test loading from .env file."""
        # Create .env file
        env_content = """
JUNIOR_AI_GEMINI_API_KEY=dotenv-gemini-key
JUNIOR_AI_GEMINI_ENABLED=true
JUNIOR_AI_PATTERN_ENABLED=true
JUNIOR_AI_PATTERN_SENSITIVITY=high
"""
        self.env_file.write_text(env_content)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            # Create provider (should load .env)
            provider = EnvironmentProvider()
            creds = provider.load('all')
            
            self.assertIn('gemini', creds)
            self.assertEqual(creds['gemini']['api_key'], 'dotenv-gemini-key')
            self.assertIn('pattern_detection', creds)
            
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main(verbosity=2)
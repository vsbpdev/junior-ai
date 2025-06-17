"""
Secure credential management system for Junior AI Assistant.

This module provides multiple secure storage backends for API keys and sensitive
configuration data, replacing the plain text credentials.json approach.

Supported backends:
1. Environment variables (highest priority)
2. OS keyring/keychain 
3. Encrypted file storage
4. Plain text JSON (backward compatibility, lowest priority)

The system uses a fallback chain to try each backend in order until credentials
are found.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from abc import ABC, abstractmethod
import base64
import hashlib
import platform
import fcntl
import tempfile

# Third-party imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False

from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)


class CredentialBackend(Enum):
    """Available credential storage backends."""
    ENVIRONMENT = "environment"
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"
    PLAIN_JSON = "plain_json"


class CredentialProvider(ABC):
    """Abstract base class for credential providers."""
    
    @abstractmethod
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load credentials for a specific key."""
        pass
    
    @abstractmethod
    def save(self, key: str, credentials: Dict[str, Any]) -> bool:
        """Save credentials for a specific key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete credentials for a specific key."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available on the current system."""
        pass


class EnvironmentProvider(CredentialProvider):
    """Load credentials from environment variables."""
    
    def __init__(self):
        # Load .env file if it exists
        load_dotenv()
        self.prefix = "JUNIOR_AI_"
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load credentials from environment variables.
        
        Environment variable naming convention:
        - AI API keys: JUNIOR_AI_{AI_NAME}_API_KEY
        - AI models: JUNIOR_AI_{AI_NAME}_MODEL
        - Pattern detection: JUNIOR_AI_PATTERN_{SETTING}
        """
        credentials = {}
        
        # Special handling for different credential types
        if key == "all":
            # Load all AI credentials
            for ai_name in ["GEMINI", "GROK", "OPENAI", "DEEPSEEK", "OPENROUTER"]:
                api_key_var = f"{self.prefix}{ai_name}_API_KEY"
                if api_key_var in os.environ:
                    credentials[ai_name.lower()] = {
                        "api_key": os.environ[api_key_var],
                        "enabled": os.environ.get(f"{self.prefix}{ai_name}_ENABLED", "true").lower() == "true",
                        "model": os.environ.get(f"{self.prefix}{ai_name}_MODEL", self._get_default_model(ai_name.lower()))
                    }
                    
                    # Add base_url for providers that need it
                    if ai_name in ["GROK", "DEEPSEEK", "OPENROUTER"]:
                        base_url_var = f"{self.prefix}{ai_name}_BASE_URL"
                        if base_url_var in os.environ:
                            credentials[ai_name.lower()]["base_url"] = os.environ[base_url_var]
            
            # Load pattern detection settings
            pattern_enabled = os.environ.get(f"{self.prefix}PATTERN_ENABLED", "true").lower() == "true"
            if pattern_enabled:
                credentials["pattern_detection"] = {
                    "enabled": True,
                    "sensitivity": {
                        "global_level": os.environ.get(f"{self.prefix}PATTERN_SENSITIVITY", "medium")
                    }
                }
        
        elif key.upper() in ["GEMINI", "GROK", "OPENAI", "DEEPSEEK", "OPENROUTER"]:
            # Load specific AI credentials
            api_key_var = f"{self.prefix}{key.upper()}_API_KEY"
            if api_key_var in os.environ:
                credentials = {
                    "api_key": os.environ[api_key_var],
                    "enabled": os.environ.get(f"{self.prefix}{key.upper()}_ENABLED", "true").lower() == "true",
                    "model": os.environ.get(f"{self.prefix}{key.upper()}_MODEL", self._get_default_model(key.lower()))
                }
                
                # Add base_url for providers that need it
                if key.upper() in ["GROK", "DEEPSEEK", "OPENROUTER"]:
                    base_url_var = f"{self.prefix}{key.upper()}_BASE_URL"
                    if base_url_var in os.environ:
                        credentials["base_url"] = os.environ[base_url_var]
        
        return credentials if credentials else None
    
    def save(self, key: str, credentials: Dict[str, Any]) -> bool:
        """Environment variables cannot be saved at runtime."""
        logger.warning("Cannot save credentials to environment variables at runtime")
        return False
    
    def delete(self, key: str) -> bool:
        """Environment variables cannot be deleted at runtime."""
        logger.warning("Cannot delete environment variables at runtime")
        return False
    
    def is_available(self) -> bool:
        """Environment variables are always available."""
        return True
    
    def _get_default_model(self, ai_name: str) -> str:
        """Get default model for an AI provider."""
        defaults = {
            "gemini": "gemini-2.0-flash",
            "grok": "grok-3",
            "openai": "gpt-4o",
            "deepseek": "deepseek-chat",
            "openrouter": "openai/gpt-4o"
        }
        return defaults.get(ai_name, "")


class KeyringProvider(CredentialProvider):
    """Store credentials in OS keyring/keychain."""
    
    def __init__(self):
        self.service_name = "junior-ai-assistant"
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load credentials from OS keyring."""
        if not self.is_available():
            return None
        
        try:
            # Keyring stores strings, so we JSON encode/decode
            stored_data = keyring.get_password(self.service_name, key)
            if stored_data:
                return json.loads(stored_data)
        except Exception as e:
            logger.error(f"Failed to load from keyring: {e}")
        
        return None
    
    def save(self, key: str, credentials: Dict[str, Any]) -> bool:
        """Save credentials to OS keyring."""
        if not self.is_available():
            return False
        
        try:
            # Convert to JSON string for storage
            keyring.set_password(self.service_name, key, json.dumps(credentials))
            return True
        except Exception as e:
            logger.error(f"Failed to save to keyring: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete credentials from OS keyring."""
        if not self.is_available():
            return False
        
        try:
            keyring.delete_password(self.service_name, key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from keyring: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if keyring is available and functional."""
        if not HAS_KEYRING:
            return False
        
        try:
            # Test keyring functionality
            keyring.get_keyring()
            return True
        except Exception:
            return False


class EncryptedFileProvider(CredentialProvider):
    """Store credentials in an encrypted file."""
    
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or Path.home() / ".junior-ai" / "credentials.enc"
        self.key_file = self.file_path.with_suffix(".key")
        
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create a new one."""
        if self.key_file.exists():
            return self.key_file.read_bytes()
        
        # Generate new key
        key = Fernet.generate_key()
        
        # Ensure directory exists
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save key with restricted permissions
        self.key_file.write_bytes(key)
        if platform.system() != "Windows":
            os.chmod(self.key_file, 0o600)
        
        return key
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt credentials from file."""
        if not self.is_available() or not self.file_path.exists():
            return None
        
        try:
            # Read encrypted data
            encrypted_data = self.file_path.read_bytes()
            
            # Get encryption key
            encryption_key = self._get_or_create_key()
            fernet = Fernet(encryption_key)
            
            # Decrypt data
            decrypted_data = fernet.decrypt(encrypted_data)
            all_credentials = json.loads(decrypted_data.decode())
            
            # Return specific key or all credentials
            if key == "all":
                return all_credentials
            return all_credentials.get(key)
            
        except Exception as e:
            logger.error(f"Failed to load encrypted credentials: {e}")
            return None
    
    def save(self, key: str, credentials: Dict[str, Any]) -> bool:
        """Encrypt and save credentials to file with atomic write."""
        if not self.is_available():
            return False
        
        temp_file = None
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temp file in same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.file_path.parent,
                prefix='.credentials_',
                suffix='.tmp'
            )
            temp_file = Path(temp_path)
            
            # Lock the file during write (Unix only)
            if platform.system() != "Windows":
                fcntl.flock(temp_fd, fcntl.LOCK_EX)
            
            try:
                # Load existing credentials or start fresh
                if self.file_path.exists():
                    all_credentials = self.load("all") or {}
                else:
                    all_credentials = {}
                
                # Update with new credentials
                if key == "all":
                    all_credentials = credentials
                else:
                    all_credentials[key] = credentials
                
                # Get encryption key and encrypt
                encryption_key = self._get_or_create_key()
                fernet = Fernet(encryption_key)
                encrypted_data = fernet.encrypt(json.dumps(all_credentials).encode())
                
                # Write to temp file
                os.write(temp_fd, encrypted_data)
                os.fsync(temp_fd)  # Ensure data is written to disk
                
            finally:
                # Always close the file descriptor
                os.close(temp_fd)
            
            # Set permissions before rename
            if platform.system() != "Windows":
                os.chmod(temp_file, 0o600)
            
            # Atomic rename
            temp_file.replace(self.file_path)
            temp_file = None  # Mark as successfully moved
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save encrypted credentials: {e}")
            # Clean up temp file if it exists
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            return False
    
    def delete(self, key: str) -> bool:
        """Delete specific credentials or all credentials."""
        if not self.is_available() or not self.file_path.exists():
            return False
        
        try:
            if key == "all":
                # Delete everything
                self.file_path.unlink()
                if self.key_file.exists():
                    self.key_file.unlink()
                return True
            else:
                # Delete specific key
                all_credentials = self.load("all") or {}
                if key in all_credentials:
                    del all_credentials[key]
                    return self.save("all", all_credentials)
                
        except Exception as e:
            logger.error(f"Failed to delete encrypted credentials: {e}")
        
        return False
    
    def is_available(self) -> bool:
        """Check if cryptography library is available."""
        return HAS_CRYPTOGRAPHY


class PlainJsonProvider(CredentialProvider):
    """Legacy plain JSON storage for backward compatibility."""
    
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or Path(__file__).parent / "credentials.json"
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load credentials from plain JSON file."""
        if not self.file_path.exists():
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                all_credentials = json.load(f)
            
            if key == "all":
                return all_credentials
            return all_credentials.get(key)
            
        except Exception as e:
            logger.error(f"Failed to load JSON credentials: {e}")
            return None
    
    def save(self, key: str, credentials: Dict[str, Any]) -> bool:
        """Save credentials to plain JSON file."""
        try:
            # Load existing or start fresh
            if self.file_path.exists():
                all_credentials = self.load("all") or {}
            else:
                all_credentials = {}
            
            # Update with new credentials
            if key == "all":
                all_credentials = credentials
            else:
                all_credentials[key] = credentials
            
            # Save to file
            with open(self.file_path, 'w') as f:
                json.dump(all_credentials, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON credentials: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete credentials from JSON file."""
        if not self.file_path.exists():
            return False
        
        try:
            if key == "all":
                self.file_path.unlink()
                return True
            else:
                all_credentials = self.load("all") or {}
                if key in all_credentials:
                    del all_credentials[key]
                    return self.save("all", all_credentials)
                    
        except Exception as e:
            logger.error(f"Failed to delete JSON credentials: {e}")
        
        return False
    
    def is_available(self) -> bool:
        """Plain JSON is always available."""
        return True


class SecureCredentialManager:
    """Main credential manager that orchestrates multiple providers."""
    
    def __init__(self, 
                 providers: Optional[List[CredentialBackend]] = None,
                 legacy_file_path: Optional[Path] = None):
        """Initialize credential manager with provider priority chain.
        
        Args:
            providers: List of providers in priority order (default: all available)
            legacy_file_path: Path to legacy credentials.json for migration
        """
        self.legacy_file_path = legacy_file_path or Path(__file__).parent / "credentials.json"
        
        # Initialize providers in priority order
        if providers is None:
            providers = [
                CredentialBackend.ENVIRONMENT,
                CredentialBackend.KEYRING,
                CredentialBackend.ENCRYPTED_FILE,
                CredentialBackend.PLAIN_JSON
            ]
        
        self.providers: Dict[CredentialBackend, CredentialProvider] = {}
        
        # Initialize each provider
        for backend in providers:
            if backend == CredentialBackend.ENVIRONMENT:
                self.providers[backend] = EnvironmentProvider()
            elif backend == CredentialBackend.KEYRING:
                self.providers[backend] = KeyringProvider()
            elif backend == CredentialBackend.ENCRYPTED_FILE:
                self.providers[backend] = EncryptedFileProvider()
            elif backend == CredentialBackend.PLAIN_JSON:
                self.providers[backend] = PlainJsonProvider(self.legacy_file_path)
        
        # Track which backend provided credentials
        self.active_backend: Optional[CredentialBackend] = None
        
    def load_credentials(self) -> Dict[str, Any]:
        """Load credentials from the first available provider.
        
        Returns:
            Dictionary containing all credentials
        """
        for backend, provider in self.providers.items():
            if provider.is_available():
                credentials = provider.load("all")
                if credentials:
                    self.active_backend = backend
                    logger.info(f"Loaded credentials from {backend.value}")
                    return credentials
        
        # No credentials found
        logger.warning("No credentials found in any provider")
        return {}
    
    def save_credentials(self, credentials: Dict[str, Any], 
                        backend: Optional[CredentialBackend] = None) -> bool:
        """Save credentials to specified backend or the most secure available.
        
        Args:
            credentials: Credentials to save
            backend: Specific backend to use (optional)
            
        Returns:
            True if save was successful
        """
        # Use specified backend or find the most secure available
        if backend:
            if backend in self.providers and self.providers[backend].is_available():
                return self.providers[backend].save("all", credentials)
            else:
                logger.error(f"Backend {backend.value} is not available")
                return False
        
        # Try backends in order of security (skip environment)
        for backend_type in [CredentialBackend.KEYRING, 
                           CredentialBackend.ENCRYPTED_FILE,
                           CredentialBackend.PLAIN_JSON]:
            if backend_type in self.providers:
                provider = self.providers[backend_type]
                if provider.is_available():
                    if provider.save("all", credentials):
                        logger.info(f"Saved credentials to {backend_type.value}")
                        return True
        
        logger.error("Failed to save credentials to any backend")
        return False
    
    def migrate_from_plain_json(self, 
                               target_backend: Optional[CredentialBackend] = None,
                               delete_original: bool = False) -> bool:
        """Migrate credentials from plain JSON to secure storage.
        
        Args:
            target_backend: Specific backend to migrate to
            delete_original: Whether to delete the original file after migration
            
        Returns:
            True if migration was successful
        """
        # Check if plain JSON exists
        if not self.legacy_file_path.exists():
            logger.info("No legacy credentials.json to migrate")
            return True
        
        # Load from plain JSON
        plain_provider = PlainJsonProvider(self.legacy_file_path)
        credentials = plain_provider.load("all")
        
        if not credentials:
            logger.warning("Failed to load legacy credentials")
            return False
        
        # Save to secure backend
        if self.save_credentials(credentials, target_backend):
            logger.info("Successfully migrated credentials")
            
            if delete_original:
                try:
                    self.legacy_file_path.unlink()
                    logger.info("Deleted original credentials.json")
                except Exception as e:
                    logger.error(f"Failed to delete original file: {e}")
            
            return True
        
        return False
    
    def get_available_backends(self) -> List[CredentialBackend]:
        """Get list of available credential backends."""
        available = []
        for backend, provider in self.providers.items():
            if provider.is_available():
                available.append(backend)
        return available
    
    def get_active_backend(self) -> Optional[CredentialBackend]:
        """Get the backend that provided the current credentials."""
        return self.active_backend


# Convenience function for backward compatibility
def load_credentials(file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load credentials using the secure credential manager.
    
    This function maintains backward compatibility with the old
    credential loading approach while using secure storage.
    
    Args:
        file_path: Optional path to legacy credentials.json
        
    Returns:
        Dictionary containing all credentials
    """
    manager = SecureCredentialManager(legacy_file_path=file_path)
    return manager.load_credentials()


# Helper function to check credential security
def check_credential_security() -> Dict[str, Any]:
    """Check the security status of stored credentials.
    
    Returns:
        Dictionary with security assessment
    """
    manager = SecureCredentialManager()
    available_backends = manager.get_available_backends()
    credentials = manager.load_credentials()
    
    assessment = {
        "available_backends": [b.value for b in available_backends],
        "active_backend": manager.active_backend.value if manager.active_backend else None,
        "has_credentials": bool(credentials),
        "plain_json_exists": manager.legacy_file_path.exists(),
        "security_level": "low"  # Default
    }
    
    # Determine security level
    if manager.active_backend == CredentialBackend.ENVIRONMENT:
        assessment["security_level"] = "high"
    elif manager.active_backend == CredentialBackend.KEYRING:
        assessment["security_level"] = "high"
    elif manager.active_backend == CredentialBackend.ENCRYPTED_FILE:
        assessment["security_level"] = "medium"
    elif manager.active_backend == CredentialBackend.PLAIN_JSON:
        assessment["security_level"] = "low"
        assessment["recommendation"] = "Consider migrating to secure storage"
    
    return assessment
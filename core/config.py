"""Configuration management for Junior AI Assistant."""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Version information
__version__ = "2.1.0"

# Global credentials storage
CREDENTIALS: Optional[Dict[str, Any]] = None

# Credentials file path
CREDENTIALS_FILE = Path.home() / '.config' / 'claude-mcp-servers' / 'junior-ai' / 'credentials.json'


def load_credentials() -> Dict[str, Any]:
    """Load credentials using SecureCredentialManager."""
    global CREDENTIALS
    
    if CREDENTIALS is not None:
        return CREDENTIALS
    
    try:
        # Import secure credential manager
        from secure_credentials import SecureCredentialManager
        
        # Initialize credential manager
        manager = SecureCredentialManager()
        
        # Check if migration is needed
        if CREDENTIALS_FILE.exists():
            # Check if already using secure storage
            try:
                # Try to load from secure storage first
                test_creds = manager.load_credentials()
                if test_creds:
                    # Already migrated
                    CREDENTIALS = test_creds
                    return CREDENTIALS
            except Exception:
                pass
            
            # Need to migrate
            print("Migrating existing credentials to secure storage...", file=sys.stderr)
            with open(CREDENTIALS_FILE, 'r') as f:
                old_creds = json.load(f)
            
            # Try to save to secure storage
            if hasattr(manager, 'migrate_credentials'):
                if manager.migrate_credentials(old_creds, CREDENTIALS_FILE):
                    print("✅ Successfully migrated credentials to secure storage", file=sys.stderr)
                else:
                    print("⚠️ Migration failed, will continue with existing credentials", file=sys.stderr)
        
        # Load credentials (from secure storage or fallback)
        CREDENTIALS = manager.load_credentials()
        
        if not CREDENTIALS:
            print("❌ No credentials found. Please run setup.sh first.", file=sys.stderr)
            sys.exit(1)
            
        return CREDENTIALS
        
    except Exception as e:
        print(f"Error loading credentials: {e}", file=sys.stderr)
        
        # Fallback to direct JSON loading
        try:
            if CREDENTIALS_FILE.exists():
                with open(CREDENTIALS_FILE, 'r') as f:
                    CREDENTIALS = json.load(f)
                    return CREDENTIALS
        except Exception as fallback_error:
            print(f"Fallback loading also failed: {fallback_error}", file=sys.stderr)
        
        print("❌ Could not load credentials. Please run setup.sh", file=sys.stderr)
        sys.exit(1)


def get_credentials() -> Dict[str, Any]:
    """Get loaded credentials, loading them if necessary."""
    if CREDENTIALS is None:
        return load_credentials()
    return CREDENTIALS


def get_pattern_detection_config() -> Dict[str, Any]:
    """Get pattern detection configuration from credentials."""
    creds = get_credentials()
    return creds.get('pattern_detection', {})


def get_ai_consultation_config() -> Dict[str, Any]:
    """Get AI consultation configuration from credentials."""
    creds = get_credentials()
    return creds.get('ai_consultation_preferences', {})
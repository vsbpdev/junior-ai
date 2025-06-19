"""Shared test configuration and fixtures for junior-ai tests."""

import json
import tempfile
import os
from pathlib import Path

def get_test_config():
    """Get a complete test configuration for pattern detection."""
    return {
        "pattern_detection": {
            "enabled": True,
            "sensitivity": {
                "global_level": "medium",
                "levels": {
                    "low": {
                        "confidence_threshold": 0.9,
                        "context_multiplier": 0.8,
                        "min_matches_for_consultation": 3,
                        "severity_threshold": "high"
                    },
                    "medium": {
                        "confidence_threshold": 0.7,
                        "context_multiplier": 1.0,
                        "min_matches_for_consultation": 2,
                        "severity_threshold": "medium"
                    },
                    "high": {
                        "confidence_threshold": 0.5,
                        "context_multiplier": 1.2,
                        "min_matches_for_consultation": 1,
                        "severity_threshold": "low"
                    },
                    "maximum": {
                        "confidence_threshold": 0.3,
                        "context_multiplier": 1.5,
                        "min_matches_for_consultation": 1,
                        "severity_threshold": "low"
                    }
                },
                "category_overrides": {}
            }
        }
    }

def create_test_config_file():
    """Create a temporary config file for testing."""
    config = get_test_config()
    
    # Create temp file
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f)
        return path
    except Exception:
        os.unlink(path)
        raise

def mock_config_path(monkeypatch, tmp_path):
    """Mock the configuration path for tests."""
    config_file = tmp_path / "test_credentials.json"
    config_file.write_text(json.dumps(get_test_config()))
    
    # Mock the Path.home() to return tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    
    return config_file
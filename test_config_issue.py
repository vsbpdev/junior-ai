#!/usr/bin/env python3
"""Test to demonstrate the configuration parsing issue."""

import json
import tempfile
import os
import logging
from pattern_detection import EnhancedPatternDetectionEngine

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_empty_levels():
    """Test configuration with empty levels dictionary."""
    config = {
        "pattern_detection": {
            "enabled": True,
            "sensitivity": {
                "global_level": "medium",
                "levels": {}  # Empty levels dictionary - this causes the error
            }
        }
    }
    
    # Write config to temp file in current directory
    config_path = 'test_config_temp.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Debug: print the config
    print(f"Written config to {config_path}")
    print("Config content:")
    print(json.dumps(config, indent=2))
    
    try:
        # This should fail with ConfigurationError at line 551
        engine = EnhancedPatternDetectionEngine(config_path=config_path)
        print("ERROR: Engine created successfully (should have failed)")
    except Exception as e:
        print(f"Expected error: {type(e).__name__}: {e}")
        print(f"This happens because 'levels' is empty, so 'medium' is not found")
    finally:
        os.unlink(config_path)

def test_proper_config():
    """Test configuration with proper levels."""
    print("Creating config with all levels...")
    config = {
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
            },
            "pattern_categories": {
                "security": {
                    "enabled": True,
                    "custom_keywords": [],
                    "custom_patterns": []
                }
            }
        }
    }
    
    # Write config to temp file in current directory
    config_path = 'test_config_temp.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Debug: print the config
    print(f"Written config to {config_path}")
    print("Config content:")
    print(json.dumps(config, indent=2))
    
    try:
        # Verify file exists and can be read
        print(f"\nVerifying file exists: {os.path.exists(config_path)}")
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            print("File loaded successfully")
            levels = loaded_config.get('pattern_detection', {}).get('sensitivity', {}).get('levels', {})
            print(f"Levels in loaded config: {list(levels.keys())}")
            print(f"'medium' in levels: {'medium' in levels}")
        
        engine = EnhancedPatternDetectionEngine(config_path=config_path)
        print("SUCCESS: Engine created with proper config")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
    finally:
        os.unlink(config_path)

if __name__ == "__main__":
    print("Testing empty levels configuration...")
    test_empty_levels()
    print("\nTesting proper configuration...")
    test_proper_config()
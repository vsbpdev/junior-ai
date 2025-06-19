#!/usr/bin/env python3
"""
Suggested fix for the pattern detection configuration issue.
Add this to _parse_sensitivity_config method after line 546.
"""

from typing import Dict

# These would be imported from the actual module structure
class ConfigurationError(Exception):
    """Configuration error exception."""
    pass

class SensitivitySettings:
    """Sensitivity settings class."""
    pass

def _parse_sensitivity_config_fixed(self, config: Dict) -> 'SensitivitySettings':
    """Parse and validate sensitivity configuration"""
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a JSON object")
    
    pattern_config = config.get('pattern_detection', {})
    if not isinstance(pattern_config, dict):
        raise ConfigurationError("pattern_detection must be an object")
    
    sensitivity_config = pattern_config.get('sensitivity', {})
    if not isinstance(sensitivity_config, dict):
        raise ConfigurationError("sensitivity must be an object")
    
    # Get and validate global level
    global_level = sensitivity_config.get('global_level', 'medium')
    self._validate_sensitivity_level(global_level)
    
    # Get and validate levels configuration
    levels = sensitivity_config.get('levels', {})
    if not isinstance(levels, dict):
        raise ConfigurationError("levels must be an object")
    
    # NEW FIX: If levels is empty, provide defaults
    if not levels:
        levels = {
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
        }
    
    if global_level not in levels:
        raise ConfigurationError(f"Global level '{global_level}' not found in levels configuration")
    
    # Rest of the method continues as before...
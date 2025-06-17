#!/usr/bin/env python3
"""
Test script for manual override functionality in Junior AI Assistant
Tests pattern detection enable/disable, category controls, and keyword management
"""

import json
import os
import tempfile
import shutil
from pattern_detection import EnhancedPatternDetectionEngine, PatternCategory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_config():
    """Create a temporary test configuration file in current directory"""
    config = {
        "pattern_detection": {
            "enabled": True,
            "manual_override": {
                "allow_disable_detection": True,
                "allow_force_consultation": True,
                "allow_custom_patterns": True,
                "force_consultation_keywords": [],
                "global_exclusions": [],
                "bypass_patterns": []
            },
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
                    "severity_override": None,
                    "custom_keywords": [],
                    "custom_patterns": [],
                    "disabled_keywords": [],
                    "sensitivity_multiplier": 1.0
                },
                "uncertainty": {
                    "enabled": True,
                    "severity_override": None,
                    "custom_keywords": [],
                    "custom_patterns": [],
                    "disabled_keywords": [],
                    "sensitivity_multiplier": 1.0
                },
                "algorithm": {
                    "enabled": True,
                    "severity_override": None,
                    "custom_keywords": [],
                    "custom_patterns": [],
                    "disabled_keywords": [],
                    "sensitivity_multiplier": 1.0
                },
                "gotcha": {
                    "enabled": True,
                    "severity_override": None,
                    "custom_keywords": [],
                    "custom_patterns": [],
                    "disabled_keywords": [],
                    "sensitivity_multiplier": 1.0
                },
                "architecture": {
                    "enabled": True,
                    "severity_override": None,
                    "custom_keywords": [],
                    "custom_patterns": [],
                    "disabled_keywords": [],
                    "sensitivity_multiplier": 1.0
                }
            }
        }
    }
    
    # Create temporary config file in current directory
    test_config_path = "test_manual_override_config.json"
    with open(test_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return test_config_path

def test_global_enable_disable(engine):
    """Test global pattern detection enable/disable"""
    print("\n=== Testing Global Enable/Disable ===")
    
    test_text = "password = 'admin123'  # TODO: fix this security issue"
    
    # Test with pattern detection enabled
    print("1. Pattern detection enabled:")
    matches = engine.detect_patterns(test_text)
    print(f"   Found {len(matches)} patterns")
    assert len(matches) > 0, "Should detect patterns when enabled"
    
    # Disable pattern detection
    print("2. Disabling pattern detection...")
    success = engine.set_pattern_detection_enabled(False)
    assert success, "Failed to disable pattern detection"
    
    # Test with pattern detection disabled
    matches = engine.detect_patterns(test_text)
    print(f"   Found {len(matches)} patterns")
    assert len(matches) == 0, "Should not detect patterns when disabled"
    
    # Re-enable pattern detection
    print("3. Re-enabling pattern detection...")
    success = engine.set_pattern_detection_enabled(True)
    assert success, "Failed to re-enable pattern detection"
    
    matches = engine.detect_patterns(test_text)
    print(f"   Found {len(matches)} patterns")
    assert len(matches) > 0, "Should detect patterns when re-enabled"
    
    print("✅ Global enable/disable test passed!")

def test_category_enable_disable(engine):
    """Test per-category enable/disable"""
    print("\n=== Testing Category Enable/Disable ===")
    
    # Test texts for different categories
    test_cases = [
        ("security", "password = 'admin123'"),
        ("uncertainty", "TODO: implement this feature"),
        ("algorithm", "This O(n^2) algorithm might be slow"),
        ("gotcha", "Be careful of race condition here"),
        ("architecture", "Should we use repository pattern?")
    ]
    
    for category, test_text in test_cases:
        print(f"\n1. Testing {category} category:")
        
        # Verify patterns are detected initially
        matches = engine.detect_patterns(test_text)
        category_matches = [m for m in matches if m.category.value == category]
        print(f"   Initial: Found {len(category_matches)} {category} patterns")
        assert len(category_matches) > 0, f"Should detect {category} patterns initially"
        
        # Disable the category
        print(f"2. Disabling {category} category...")
        success = engine.set_category_enabled(category, False)
        assert success, f"Failed to disable {category} category"
        
        # Verify no patterns are detected
        matches = engine.detect_patterns(test_text)
        category_matches = [m for m in matches if m.category.value == category]
        print(f"   After disable: Found {len(category_matches)} {category} patterns")
        assert len(category_matches) == 0, f"Should not detect {category} patterns when disabled"
        
        # Re-enable the category
        print(f"3. Re-enabling {category} category...")
        success = engine.set_category_enabled(category, True)
        assert success, f"Failed to re-enable {category} category"
        
        matches = engine.detect_patterns(test_text)
        category_matches = [m for m in matches if m.category.value == category]
        print(f"   After re-enable: Found {len(category_matches)} {category} patterns")
        assert len(category_matches) > 0, f"Should detect {category} patterns when re-enabled"
    
    print("\n✅ Category enable/disable test passed!")

def test_custom_keywords(engine):
    """Test adding and removing custom keywords"""
    print("\n=== Testing Custom Keywords ===")
    
    # Test custom keywords for security category
    custom_keywords = ["mySecretKey", "customAuth", "specialToken"]
    
    print("1. Adding custom keywords to security category:")
    for keyword in custom_keywords:
        print(f"   - {keyword}")
    
    success = engine.add_custom_keywords("security", custom_keywords)
    assert success, "Failed to add custom keywords"
    
    # Test detection with custom keywords
    test_text = "const mySecretKey = 'abc123'; // This is our customAuth token"
    matches = engine.detect_patterns(test_text)
    security_matches = [m for m in matches if m.category.value == "security"]
    
    print(f"2. Testing detection with custom keywords:")
    print(f"   Found {len(security_matches)} security patterns")
    for match in security_matches:
        print(f"   - Keyword: {match.keyword}")
    
    # Verify custom keywords were detected
    detected_keywords = [m.keyword.lower() for m in security_matches]
    assert "mysecretkey" in detected_keywords, "Custom keyword 'mySecretKey' not detected"
    assert "customauth" in detected_keywords, "Custom keyword 'customAuth' not detected"
    
    # Get all keywords to verify they were added
    all_keywords = engine.get_all_keywords("security")
    print(f"3. Total keywords in security category: {len(all_keywords)}")
    
    # Remove custom keywords
    print("4. Removing custom keywords...")
    success = engine.remove_keywords("security", custom_keywords)
    assert success, "Failed to remove custom keywords"
    
    # Test detection without custom keywords
    matches = engine.detect_patterns(test_text)
    security_matches = [m for m in matches if m.category.value == "security"]
    detected_keywords = [m.keyword.lower() for m in security_matches]
    
    print(f"5. After removal: Found {len(security_matches)} security patterns")
    assert "mysecretkey" not in detected_keywords, "Custom keyword should not be detected after removal"
    assert "customauth" not in detected_keywords, "Custom keyword should not be detected after removal"
    
    print("✅ Custom keywords test passed!")

def test_custom_patterns(engine):
    """Test adding custom regex patterns"""
    print("\n=== Testing Custom Patterns ===")
    
    # Add a custom pattern for detecting version numbers
    custom_pattern = r'v\d+\.\d+\.\d+'
    
    print("1. Adding custom pattern to uncertainty category:")
    print(f"   Pattern: {custom_pattern}")
    
    success = engine.add_custom_pattern("uncertainty", custom_pattern)
    assert success, "Failed to add custom pattern"
    
    # Test detection with custom pattern
    test_text = "Current version: v2.3.1 // TODO: update to v3.0.0"
    matches = engine.detect_patterns(test_text)
    
    print(f"2. Testing detection with custom pattern:")
    print(f"   Found {len(matches)} total patterns")
    for match in matches:
        print(f"   - {match.category.value}: {match.keyword}")
    
    # Verify version numbers were detected
    version_matches = [m for m in matches if m.keyword.startswith('v') and '.' in m.keyword]
    assert len(version_matches) > 0, "Custom pattern for version numbers not detected"
    
    print("✅ Custom patterns test passed!")

def test_persistence(config_path):
    """Test that settings persist across engine instances"""
    print("\n=== Testing Configuration Persistence ===")
    
    # Create first engine instance
    engine1 = EnhancedPatternDetectionEngine(config_path=config_path)
    
    # Make changes
    print("1. Making configuration changes in first engine:")
    engine1.set_pattern_detection_enabled(False)
    engine1.set_category_enabled("security", False)
    engine1.add_custom_keywords("algorithm", ["customAlgo", "specialSort"])
    engine1.update_sensitivity(global_level="high")
    
    # Create second engine instance with same config
    engine2 = EnhancedPatternDetectionEngine(config_path=config_path)
    
    print("2. Verifying changes in second engine:")
    
    # Verify pattern detection is disabled
    assert not engine2.is_pattern_detection_enabled(), "Pattern detection should be disabled"
    print("   ✓ Pattern detection disabled state persisted")
    
    # Verify category is disabled
    test_text = "password = 'admin123'"
    matches = engine2.detect_patterns(test_text)
    security_matches = [m for m in matches if m.category.value == "security"]
    assert len(security_matches) == 0, "Security category should be disabled"
    print("   ✓ Category disabled state persisted")
    
    # Verify custom keywords
    keywords = engine2.get_all_keywords("algorithm")
    assert "customalgo" in [k.lower() for k in keywords], "Custom keywords should persist"
    print("   ✓ Custom keywords persisted")
    
    # Verify sensitivity level
    info = engine2.get_sensitivity_info()
    assert info["global_level"] == "high", "Sensitivity level should persist"
    print("   ✓ Sensitivity level persisted")
    
    print("✅ Configuration persistence test passed!")

def main():
    """Run all manual override tests"""
    print("🧪 Junior AI Assistant - Manual Override Testing")
    print("=" * 50)
    
    # Create temporary config file
    config_path = create_test_config()
    
    try:
        # Create pattern detection engine
        engine = EnhancedPatternDetectionEngine(config_path=config_path)
        
        # Run tests
        test_global_enable_disable(engine)
        test_category_enable_disable(engine)
        test_custom_keywords(engine)
        test_custom_patterns(engine)
        test_persistence(config_path)
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
    finally:
        # Clean up temporary config file
        if os.path.exists(config_path):
            os.unlink(config_path)
            print(f"\n🧹 Cleaned up temporary config: {config_path}")

if __name__ == "__main__":
    main()
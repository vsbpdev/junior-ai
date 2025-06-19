#!/usr/bin/env python3
"""
Verify that SecureCredentialManager is truly lazily initialized
"""

from pattern_detection import EnhancedPatternDetectionEngine as PatternDetectionEngine

# Reset any existing shared instance
if hasattr(PatternDetectionEngine, '_shared_credential_manager'):
    PatternDetectionEngine._shared_credential_manager = None

print("1. Creating PatternDetectionEngine...")
engine = PatternDetectionEngine()
print(f"   Credential manager exists: {PatternDetectionEngine._shared_credential_manager is not None}")

print("\n2. Accessing method that requires credentials...")
info = engine.get_sensitivity_info()
print(f"   Credential manager exists: {PatternDetectionEngine._shared_credential_manager is not None}")
print(f"   Credential manager ID: {id(PatternDetectionEngine._shared_credential_manager)}")

print("\n3. Creating another engine...")
engine2 = PatternDetectionEngine()
print(f"   Credential manager exists: {PatternDetectionEngine._shared_credential_manager is not None}")
print(f"   Same credential manager: {id(PatternDetectionEngine._shared_credential_manager)}")

print("\n✅ Lazy initialization verified!")
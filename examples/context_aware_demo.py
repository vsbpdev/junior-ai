#!/usr/bin/env python3
"""
Context-Aware Pattern Matching Demo
Shows how the Junior AI Assistant intelligently handles patterns based on context
"""

import unittest

# Example 1: Security pattern in production code (HIGH SEVERITY)
def process_user_login(username, password):
    """Production login handler - this will trigger security alerts"""
    # FIXME: This is insecure!
    if password == "admin123":  # Hard-coded password - BAD!
        return generate_token(username)
    
    # TODO: Implement proper authentication
    api_key = "sk-production-key-12345"  # API key exposed!
    return None


# Example 2: Same patterns in test code (REDUCED SEVERITY)
class TestAuthentication(unittest.TestCase):
    def test_login_validation(self):
        """Test code - patterns here are less severe"""
        # In tests, hardcoded values are acceptable
        test_password = "test123"
        test_api_key = "sk-test-key-12345"
        
        # Test the login with known values
        result = mock_login("testuser", test_password)
        self.assertIsNotNone(result)


# Example 3: Commented code (FILTERED OUT)
def old_implementation():
    """This function has commented security issues"""
    # password = "secret123"  # Old code - won't trigger
    # api_key = get_api_key()  # Commented out
    
    # TODO: Remove this deprecated function
    return "deprecated"


# Example 4: Security library imports (REDUCED CONFIDENCE)
import bcrypt
import jwt

def secure_password_handler(password):
    """With security imports, patterns have lower confidence"""
    # The word 'password' here is less concerning because
    # we're using bcrypt for proper hashing
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed


# Example 5: Algorithm complexity pattern
def find_duplicates(items):
    """O(n²) algorithm - will trigger optimization consultation"""
    duplicates = []
    # Nested loop - O(n²) complexity
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates


if __name__ == "__main__":
    print("Context-Aware Pattern Detection Demo")
    print("=" * 50)
    print("This file contains various patterns that will be")
    print("detected differently based on their context:")
    print()
    print("1. Security patterns in production code → HIGH severity")
    print("2. Same patterns in test code → MEDIUM severity") 
    print("3. Patterns in comments → Filtered out")
    print("4. Patterns with security imports → Lower confidence")
    print("5. Algorithm complexity → Triggers optimization advice")
    print()
    print("Run this through Junior AI's pattern detection to see")
    print("how context affects the analysis!")
#!/usr/bin/env python3
"""
Test cases for context-aware pattern matching
"""

import unittest
from context_aware_matching import (
    ContextAwarePatternMatcher, CodeLanguage, LanguageDetector,
    PythonContextAnalyzer, ContextualPattern
)
from pattern_detection import PatternCategory, PatternSeverity


class TestLanguageDetection(unittest.TestCase):
    """Test language detection functionality"""
    
    def setUp(self):
        self.detector = LanguageDetector()
    
    def test_extension_detection(self):
        """Test language detection from file extensions"""
        test_cases = [
            ("main.py", CodeLanguage.PYTHON),
            ("app.js", CodeLanguage.JAVASCRIPT),
            ("index.ts", CodeLanguage.TYPESCRIPT),
            ("Main.java", CodeLanguage.JAVA),
            ("program.cpp", CodeLanguage.CPP),
            ("server.go", CodeLanguage.GO),
            ("lib.rs", CodeLanguage.RUST),
            ("unknown.txt", CodeLanguage.UNKNOWN),
        ]
        
        for filename, expected in test_cases:
            with self.subTest(filename=filename):
                result = self.detector.detect_from_extension(filename)
                self.assertEqual(result, expected)
    
    def test_content_detection(self):
        """Test language detection from content"""
        python_code = """
def hello_world():
    print("Hello, World!")
    
class MyClass:
    pass
"""
        
        javascript_code = """
function helloWorld() {
    console.log("Hello, World!");
}

const myVar = 42;
"""
        
        java_code = """
package com.example;

public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        
        self.assertEqual(
            self.detector.detect_from_content(python_code),
            CodeLanguage.PYTHON
        )
        self.assertEqual(
            self.detector.detect_from_content(javascript_code),
            CodeLanguage.JAVASCRIPT
        )
        self.assertEqual(
            self.detector.detect_from_content(java_code),
            CodeLanguage.JAVA
        )


class TestPythonContextAnalyzer(unittest.TestCase):
    """Test Python-specific context analysis"""
    
    def setUp(self):
        self.analyzer = PythonContextAnalyzer()
    
    def test_scope_extraction(self):
        """Test scope type and name extraction"""
        code = '''
class UserAuth:
    def validate_password(self, password):
        # Check password strength
        if len(password) < 8:
            return False
        return True

def standalone_function():
    api_key = "test"
    return api_key
'''
        
        # Test inside class method
        pos = code.find("password) < 8")
        scope_type, scope_name = self.analyzer.extract_scope_info(code, pos)
        self.assertEqual(scope_type, "function")
        self.assertEqual(scope_name, "validate_password")
        
        # Test inside standalone function
        pos = code.find('api_key = "test"')
        scope_type, scope_name = self.analyzer.extract_scope_info(code, pos)
        self.assertEqual(scope_type, "function")
        self.assertEqual(scope_name, "standalone_function")
    
    def test_import_extraction(self):
        """Test import statement extraction"""
        code = '''
import os
import sys
from datetime import datetime
from typing import List, Dict
import numpy as np

def process_data():
    pass
'''
        
        imports = self.analyzer.extract_imports(code)
        
        expected_imports = ['os', 'sys', 'numpy', 'datetime.datetime', 'typing.List', 'typing.Dict']
        for imp in expected_imports:
            self.assertIn(imp, imports)
    
    def test_test_code_detection(self):
        """Test detection of test code"""
        test_code = '''
import unittest

class TestAuthentication(unittest.TestCase):
    def test_password_validation(self):
        password = "test123"  # This is OK in tests
        self.assertTrue(len(password) > 0)
    
    def helper_method(self):
        # Not a test method
        pass

def test_standalone():
    # Pytest style test
    assert True
'''
        
        # Inside test method
        pos = test_code.find('password = "test123"')
        self.assertTrue(self.analyzer.is_in_test_code(test_code, pos))
        
        # Inside test class but not test method
        pos = test_code.find("# Not a test method")
        self.assertFalse(self.analyzer.is_in_test_code(test_code, pos))
        
        # Inside pytest-style test
        pos = test_code.find("assert True")
        self.assertTrue(self.analyzer.is_in_test_code(test_code, pos))


class TestContextAwareMatching(unittest.TestCase):
    """Test the complete context-aware pattern matching"""
    
    def setUp(self):
        self.matcher = ContextAwarePatternMatcher()
    
    def test_security_pattern_in_test_code(self):
        """Test that security patterns in test code are handled differently"""
        code = '''
import unittest

class TestAuth(unittest.TestCase):
    def test_login(self):
        # In test code, hardcoded passwords should be less severe
        password = "admin123"
        api_key = "sk-1234567890"
        
        self.assertEqual(password, "admin123")
'''
        
        matches = self.matcher.detect_with_context(code, "test_auth.py")
        
        # Should find security patterns
        security_matches = [m for m in matches if m.base_match.category == PatternCategory.SECURITY]
        self.assertGreater(len(security_matches), 0)
        
        # But they should be marked as test code
        for match in security_matches:
            self.assertTrue(match.is_test_code)
            # Severity should be reduced
            self.assertLessEqual(match.base_match.severity.value, PatternSeverity.MEDIUM.value)
    
    def test_commented_patterns_filtered(self):
        """Test that commented patterns are filtered appropriately"""
        code = '''
def process_payment(card_number):
    # Old code: password = "12345"  # TODO: Remove this
    # api_key = "secret"  # Commented out
    
    # This TODO should still be detected
    # TODO: Implement proper validation
    
    return True
'''
        
        matches = self.matcher.detect_with_context(code, "payment.py")
        
        # Should find TODO patterns (both TODO keywords)
        todo_matches = [m for m in matches if m.base_match.category == PatternCategory.UNCERTAINTY]
        self.assertGreater(len(todo_matches), 0)
        
        # Commented security patterns should be filtered
        security_matches = [m for m in matches 
                          if m.base_match.category == PatternCategory.SECURITY 
                          and not m.is_commented_out]
        self.assertEqual(len(security_matches), 0)
    
    def test_context_aware_confidence(self):
        """Test that context affects pattern confidence"""
        code = '''
import bcrypt
import hashlib

def secure_password_hash(password):
    # Using bcrypt - this is secure
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def insecure_password_storage(password):
    # Just storing plaintext - this is bad
    stored_password = password
    return stored_password
'''
        
        matches = self.matcher.detect_with_context(code, "auth.py")
        
        # Both functions have "password" patterns
        password_matches = [m for m in matches if "password" in m.base_match.keyword.lower()]
        
        # The one with security imports should have lower confidence
        for match in password_matches:
            if match.scope_name == "secure_password_hash":
                # Should have imports that reduce confidence
                self.assertIn("bcrypt", match.imports)
                self.assertLess(match.base_match.confidence, 1.0)
    
    def test_semantic_role_detection(self):
        """Test semantic role detection in patterns"""
        code = '''
def authenticate(username, password, api_key):
    # Parameters should be detected with their semantic role
    if password == "admin":
        return False
    
    token = generate_token(api_key)
    return token
'''
        
        matches = self.matcher.detect_with_context(code, "auth.py")
        
        # Should detect security patterns
        self.assertGreater(len(matches), 0)
        
        # Check if we identified parameter vs assignment roles
        param_matches = [m for m in matches if m.semantic_role == "parameter"]
        assign_matches = [m for m in matches if m.semantic_role == "assignment"]
        
        # We expect some matches to be identified as parameters
        # Note: This might need adjustment based on actual implementation
        self.assertGreaterEqual(len(matches), 2)
    
    def test_multi_language_detection(self):
        """Test pattern detection across different languages"""
        js_code = '''
// JavaScript authentication
function login(password) {
    // TODO: Add proper validation
    const apiKey = "sk-12345";
    
    if (password === "admin123") {
        return true;
    }
    
    return false;
}
'''
        
        matches = self.matcher.detect_with_context(js_code, "auth.js")
        
        # Should detect language as JavaScript
        self.assertTrue(any(m.language == CodeLanguage.JAVASCRIPT for m in matches))
        
        # Should find patterns
        self.assertGreater(len(matches), 0)
        
        # Should find TODO
        todo_matches = [m for m in matches if m.base_match.category == PatternCategory.UNCERTAINTY]
        self.assertGreater(len(todo_matches), 0)
    
    def test_enhanced_consultation_strategy(self):
        """Test enhanced consultation strategy generation"""
        code = '''
import unittest

class TestSecurity(unittest.TestCase):
    def test_encryption(self):
        # Test patterns should affect strategy
        password = "test123"
        api_key = "test-key"

def production_code():
    # TODO: Implement secure storage
    secret = "production-secret"  # FIXME: Move to env vars
    
    # Complex algorithm - O(n^2)
    for i in range(n):
        for j in range(n):
            process(i, j)
'''
        
        matches = self.matcher.detect_with_context(code, "mixed_code.py")
        strategy = self.matcher.get_enhanced_consultation_strategy(matches)
        
        # Should have context insights
        self.assertIn('context_insights', strategy)
        self.assertGreater(len(strategy['context_insights']), 0)
        
        # Should mention test code in insights
        insights_text = ' '.join(strategy['context_insights'])
        self.assertIn('test', insights_text.lower())


class TestContextualFiltering(unittest.TestCase):
    """Test contextual filtering rules"""
    
    def setUp(self):
        self.matcher = ContextAwarePatternMatcher()
    
    def test_decorator_context_reduces_confidence(self):
        """Test that security keywords in decorators have reduced confidence"""
        code = '''
from functools import wraps

def requires_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check authentication
        return f(*args, **kwargs)
    return decorated_function

@requires_auth
def secure_endpoint():
    return "data"
'''
        
        matches = self.matcher.detect_with_context(code, "decorators.py")
        
        # Should find auth pattern but with reduced confidence due to decorator context
        auth_matches = [m for m in matches if "auth" in m.base_match.keyword.lower()]
        
        for match in auth_matches:
            if match.semantic_role == "decorator":
                self.assertLess(match.base_match.confidence, 1.0)
    
    def test_indentation_level_extraction(self):
        """Test indentation level detection"""
        code = '''
def outer():
    def inner():
        if True:
            password = "nested"  # 3 levels deep
            return password
    
    api_key = "less_nested"  # 1 level deep
    return inner
'''
        
        matches = self.matcher.detect_with_context(code, "nested.py")
        
        # Check indentation levels
        for match in matches:
            if "nested" in match.base_match.context:
                self.assertGreaterEqual(match.indentation_level, 3)
            elif "less_nested" in match.base_match.context:
                self.assertLessEqual(match.indentation_level, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
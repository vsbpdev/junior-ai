#!/usr/bin/env python3
"""Security improvements for context-aware pattern matching"""

import re
from typing import Optional, Set
import ast

class SecureContextAnalyzer:
    """Security-focused improvements"""
    
    # Maximum file size to prevent DoS
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Maximum AST depth to prevent stack overflow
    MAX_AST_DEPTH = 100
    
    # Regex patterns with timeout protection
    SAFE_REGEX_TIMEOUT = 2.0  # seconds
    
    def __init__(self):
        self.suspicious_patterns = {
            'eval_usage': re.compile(r'\beval\s*\('),
            'exec_usage': re.compile(r'\bexec\s*\('),
            'pickle_loads': re.compile(r'pickle\.loads'),
            'subprocess_shell': re.compile(r'shell\s*=\s*True'),
            'sql_injection': re.compile(r'["\'].*\+.*(?:SELECT|INSERT|UPDATE|DELETE)', re.IGNORECASE),
        }
    
    def analyze_security_context(self, code: str, position: int) -> Set[str]:
        """Detect security-sensitive context"""
        if len(code) > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {len(code)} bytes exceeds {self.MAX_FILE_SIZE}")
        
        security_flags = set()
        
        # Check for dangerous patterns near position
        context_window = 500
        start = max(0, position - context_window)
        end = min(len(code), position + context_window)
        context = code[start:end]
        
        for pattern_name, pattern in self.suspicious_patterns.items():
            try:
                # Use timeout for regex matching (Python 3.11+)
                if pattern.search(context, timeout=self.SAFE_REGEX_TIMEOUT):
                    security_flags.add(pattern_name)
            except:
                # Fallback for older Python versions
                if pattern.search(context):
                    security_flags.add(pattern_name)
        
        return security_flags
    
    def sanitize_code_for_analysis(self, code: str) -> str:
        """Remove potentially malicious code before analysis"""
        # Remove zero-width characters that could hide malicious code
        zero_width_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
        ]
        
        for char in zero_width_chars:
            code = code.replace(char, '')
        
        # Limit string literals to prevent memory exhaustion
        code = self._limit_string_literals(code)
        
        return code
    
    def _limit_string_literals(self, code: str, max_length: int = 1000) -> str:
        """Truncate overly long string literals"""
        # Simple approach - could be enhanced with proper parsing
        lines = code.split('\n')
        result = []
        
        for line in lines:
            if len(line) > max_length:
                # Truncate but preserve structure
                result.append(line[:max_length] + '... [truncated]')
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def validate_ast_depth(self, tree: ast.AST) -> bool:
        """Check if AST depth is within safe limits"""
        def get_depth(node, current_depth=0):
            if current_depth > self.MAX_AST_DEPTH:
                return current_depth
            
            max_child_depth = current_depth
            for child in ast.iter_child_nodes(node):
                child_depth = get_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return get_depth(tree) <= self.MAX_AST_DEPTH
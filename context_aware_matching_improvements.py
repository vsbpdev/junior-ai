#!/usr/bin/env python3
"""Performance improvements for context-aware pattern matching"""

import ast
from functools import lru_cache
from typing import Dict, Optional, Tuple, List
import hashlib

class OptimizedPythonContextAnalyzer:
    """Optimized Python context analyzer with caching"""
    
    def __init__(self):
        self.tree_cache: Dict[str, ast.AST] = {}
        self.scope_cache: Dict[str, List[Tuple[int, int, str, Optional[str]]]] = {}
        
    @lru_cache(maxsize=1024)
    def _get_code_hash(self, code: str) -> str:
        """Generate hash for code caching"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    
    def analyze(self, code: str, position: int) -> Optional['SyntaxContext']:
        """Analyze with caching"""
        code_hash = self._get_code_hash(code)
        
        # Try to use cached AST
        if code_hash not in self.tree_cache:
            try:
                self.tree_cache[code_hash] = ast.parse(code)
            except SyntaxError:
                return None
                
        tree = self.tree_cache[code_hash]
        # ... rest of analysis
        
    def extract_scope_info_batch(self, code: str, positions: List[int]) -> Dict[int, Tuple[str, Optional[str]]]:
        """Extract scope info for multiple positions at once"""
        code_hash = self._get_code_hash(code)
        
        # Build scope map if not cached
        if code_hash not in self.scope_cache:
            self._build_scope_map(code, code_hash)
            
        # Look up positions in scope map
        results = {}
        for pos in positions:
            line_num = code[:pos].count('\n') + 1
            results[pos] = self._find_scope_for_line(code_hash, line_num)
            
        return results
    
    def _build_scope_map(self, code: str, code_hash: str):
        """Pre-build scope map for entire file"""
        try:
            tree = ast.parse(code)
            scopes = []
            
            def visit_node(node, depth=0, parent_scope=None):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    scope_info = None
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scope_info = (node.lineno, node.end_lineno or node.lineno, "function", node.name)
                    elif isinstance(node, ast.ClassDef):
                        scope_info = (node.lineno, node.end_lineno or node.lineno, "class", node.name)
                    
                    if scope_info:
                        scopes.append(scope_info)
                        parent_scope = scope_info
                
                for child in ast.iter_child_nodes(node):
                    visit_node(child, depth + 1, parent_scope)
            
            visit_node(tree)
            self.scope_cache[code_hash] = sorted(scopes)
        except:
            self.scope_cache[code_hash] = []
#!/usr/bin/env python3
"""
Context-Aware Pattern Matching Module for Junior AI Assistant
Enhances pattern detection with syntax and semantic understanding
"""

import re
import ast
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import tokenize
import io
from pathlib import Path

from pattern_detection import (
    PatternCategory, PatternSeverity, PatternMatch,
    PatternDefinition, EnhancedPatternDetectionEngine
)


class CodeLanguage(Enum):
    """Supported programming languages for syntax-aware detection"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class ContextualPattern:
    """Extended pattern with contextual information"""
    base_match: PatternMatch
    language: CodeLanguage
    scope_type: str  # function, class, module, etc.
    scope_name: Optional[str]
    surrounding_vars: List[str]
    control_flow: List[str]  # if, for, try, etc.
    imports: List[str]
    comments_nearby: List[str]
    indentation_level: int
    is_test_code: bool
    is_commented_out: bool
    semantic_role: Optional[str]  # parameter, return value, condition, etc.


@dataclass
class SyntaxContext:
    """Syntax information for a code position"""
    node_type: str
    parent_nodes: List[str]
    siblings: List[str]
    depth: int
    attributes: Dict[str, Any]


class LanguageDetector:
    """Detect programming language from file extension or content"""
    
    EXTENSION_MAP = {
        '.py': CodeLanguage.PYTHON,
        '.js': CodeLanguage.JAVASCRIPT,
        '.jsx': CodeLanguage.JAVASCRIPT,
        '.ts': CodeLanguage.TYPESCRIPT,
        '.tsx': CodeLanguage.TYPESCRIPT,
        '.java': CodeLanguage.JAVA,
        '.cpp': CodeLanguage.CPP,
        '.cc': CodeLanguage.CPP,
        '.cxx': CodeLanguage.CPP,
        '.go': CodeLanguage.GO,
        '.rs': CodeLanguage.RUST,
    }
    
    CONTENT_PATTERNS = {
        CodeLanguage.PYTHON: [
            r'^\s*def\s+\w+\s*\(',
            r'^\s*class\s+\w+[\s\(:]',
            r'^\s*import\s+\w+',
            r'^\s*from\s+\w+\s+import',
        ],
        CodeLanguage.JAVASCRIPT: [
            r'^\s*function\s+\w+\s*\(',
            r'^\s*const\s+\w+\s*=',
            r'^\s*let\s+\w+\s*=',
            r'^\s*var\s+\w+\s*=',
            r'=>',
        ],
        CodeLanguage.JAVA: [
            r'^\s*public\s+class\s+\w+',
            r'^\s*private\s+\w+\s+\w+',
            r'^\s*package\s+[\w\.]+;',
        ],
    }
    
    @classmethod
    def detect_from_extension(cls, filename: str) -> CodeLanguage:
        """Detect language from file extension"""
        ext = Path(filename).suffix.lower()
        return cls.EXTENSION_MAP.get(ext, CodeLanguage.UNKNOWN)
    
    @classmethod
    def detect_from_content(cls, content: str) -> CodeLanguage:
        """Detect language from content patterns"""
        lines = content.split('\n')[:50]  # Check first 50 lines
        
        scores = {lang: 0 for lang in cls.CONTENT_PATTERNS}
        
        for lang, patterns in cls.CONTENT_PATTERNS.items():
            for pattern in patterns:
                for line in lines:
                    if re.match(pattern, line):
                        scores[lang] += 1
        
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            if best_lang[1] > 0:
                return best_lang[0]
        
        return CodeLanguage.UNKNOWN


class PythonContextAnalyzer:
    """Analyze Python code context using AST"""
    
    def __init__(self):
        self.tree = None
        self.lines = []
    
    def analyze(self, code: str, position: int) -> Optional[SyntaxContext]:
        """Analyze syntax context at a specific position"""
        try:
            self.tree = ast.parse(code)
            self.lines = code.split('\n')
            
            # Find line and column for position
            line_num = code[:position].count('\n') + 1
            col_offset = position - code.rfind('\n', 0, position) - 1
            
            # Find the node at this position
            node_info = self._find_node_at_position(self.tree, line_num, col_offset)
            
            if node_info:
                node, parents = node_info
                return SyntaxContext(
                    node_type=type(node).__name__,
                    parent_nodes=[type(p).__name__ for p in parents],
                    siblings=self._get_siblings(node, parents),
                    depth=len(parents),
                    attributes=self._extract_attributes(node)
                )
        except SyntaxError:
            # Code has syntax errors, fall back to basic analysis
            pass
        
        return None
    
    def _find_node_at_position(self, node, target_line, target_col, parents=None):
        """Find AST node at specific line and column"""
        if parents is None:
            parents = []
        
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            # Check if position is within this node
            if (node.lineno <= target_line and 
                (not hasattr(node, 'end_lineno') or target_line <= node.end_lineno)):
                
                # Found a candidate, but check children first
                for child in ast.iter_child_nodes(node):
                    result = self._find_node_at_position(child, target_line, target_col, parents + [node])
                    if result:
                        return result
                
                # No child contains the position, so this node is the best match
                return (node, parents)
        
        # Check children
        for child in ast.iter_child_nodes(node):
            result = self._find_node_at_position(child, target_line, target_col, parents + [node])
            if result:
                return result
        
        return None
    
    def _get_siblings(self, node, parents):
        """Get sibling nodes"""
        if not parents:
            return []
        
        parent = parents[-1]
        siblings = []
        
        for child in ast.iter_child_nodes(parent):
            if child != node:
                siblings.append(type(child).__name__)
        
        return siblings
    
    def _extract_attributes(self, node):
        """Extract relevant attributes from AST node"""
        attrs = {}
        
        if isinstance(node, ast.FunctionDef):
            attrs['name'] = node.name
            attrs['args'] = [arg.arg for arg in node.args.args]
            attrs['decorators'] = [self._get_decorator_name(d) for d in node.decorator_list]
        elif isinstance(node, ast.ClassDef):
            attrs['name'] = node.name
            attrs['bases'] = [self._get_name(base) for base in node.bases]
        elif isinstance(node, ast.Assign):
            attrs['targets'] = [self._get_name(t) for t in node.targets]
        elif isinstance(node, ast.Call):
            attrs['func'] = self._get_name(node.func)
            attrs['arg_count'] = len(node.args)
        
        return attrs
    
    def _get_name(self, node):
        """Extract name from various node types"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif hasattr(node, 'id'):
            return node.id
        return type(node).__name__
    
    def _get_decorator_name(self, node):
        """Extract decorator name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return "unknown"
    
    def extract_scope_info(self, code: str, position: int) -> Tuple[str, Optional[str]]:
        """Extract scope type and name at position"""
        try:
            tree = ast.parse(code)
            
            # Find line and column for position
            line_num = code[:position].count('\n') + 1
            
            # Find all scopes that contain this line
            scopes = []
            
            def visit_node(node, depth=0):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    if node.lineno <= line_num <= (node.end_lineno or node.lineno):
                        if isinstance(node, ast.FunctionDef):
                            scopes.append((depth, "function", node.name))
                        elif isinstance(node, ast.AsyncFunctionDef):
                            scopes.append((depth, "async_function", node.name))
                        elif isinstance(node, ast.ClassDef):
                            scopes.append((depth, "class", node.name))
                
                # Visit children
                for child in ast.iter_child_nodes(node):
                    visit_node(child, depth + 1)
            
            visit_node(tree)
            
            # Return the deepest scope (most specific)
            if scopes:
                scopes.sort(key=lambda x: x[0], reverse=True)
                return (scopes[0][1], scopes[0][2])
                
        except:
            pass
        
        return ("module", None)
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract all imports from code"""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
        except SyntaxError:
            # Fallback to regex
            import_pattern = re.compile(r'^\s*(?:from\s+([\w\.]+)\s+)?import\s+([\w\s,]+)', re.MULTILINE)
            for match in import_pattern.finditer(code):
                if match.group(1):
                    imports.append(match.group(1))
                imports.extend(name.strip() for name in match.group(2).split(','))
        
        return imports
    
    def is_in_test_code(self, code: str, position: int) -> bool:
        """Check if position is within test code"""
        # First check the scope
        scope_type, scope_name = self.extract_scope_info(code, position)
        
        # For methods, check if they start with test_
        if scope_type == "function" and scope_name:
            if scope_name.startswith('test_'):
                return True
        
        # For classes, check if they contain Test
        if scope_type == "class" and scope_name:
            if 'Test' in scope_name:
                # Now check if we're in a test method within this test class
                # Get the actual method we're in
                try:
                    tree = ast.parse(code)
                    line_num = code[:position].count('\n') + 1
                    
                    # Find the specific method within the test class
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and 'Test' in node.name:
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    if hasattr(item, 'lineno') and hasattr(item, 'end_lineno'):
                                        if item.lineno <= line_num <= (item.end_lineno or item.lineno):
                                            # We're in this method
                                            return item.name.startswith('test_')
                except:
                    pass
                    
                return False
        
        # Standalone test functions (pytest style)
        if scope_type == "function" and scope_name and scope_name.startswith('test_'):
            return True
        
        return False


class ContextAwarePatternMatcher:
    """Enhanced pattern matcher with context awareness"""
    
    def __init__(self, base_engine: Optional[EnhancedPatternDetectionEngine] = None):
        self.base_engine = base_engine or EnhancedPatternDetectionEngine()
        self.language_detector = LanguageDetector()
        self.analyzers = {
            CodeLanguage.PYTHON: PythonContextAnalyzer(),
        }
    
    def detect_with_context(self, text: str, filename: Optional[str] = None) -> List[ContextualPattern]:
        """Detect patterns with full context analysis"""
        # Detect language
        language = CodeLanguage.UNKNOWN
        if filename:
            language = self.language_detector.detect_from_extension(filename)
        if language == CodeLanguage.UNKNOWN:
            language = self.language_detector.detect_from_content(text)
        
        # Get base pattern matches
        base_matches = self.base_engine.detect_patterns(text)
        
        # Enhance with context
        contextual_matches = []
        for match in base_matches:
            context_match = self._enhance_with_context(match, text, language)
            
            # Apply context-based filtering
            if self._should_include_match(context_match):
                contextual_matches.append(context_match)
        
        return contextual_matches
    
    def _enhance_with_context(self, match: PatternMatch, text: str, language: CodeLanguage) -> ContextualPattern:
        """Add contextual information to a pattern match"""
        analyzer = self.analyzers.get(language)
        
        # Default context
        context = ContextualPattern(
            base_match=match,
            language=language,
            scope_type="unknown",
            scope_name=None,
            surrounding_vars=[],
            control_flow=[],
            imports=[],
            comments_nearby=[],
            indentation_level=0,
            is_test_code=False,
            is_commented_out=False,
            semantic_role=None
        )
        
        # Language-specific analysis
        if analyzer and language == CodeLanguage.PYTHON:
            # Extract scope information
            scope_type, scope_name = analyzer.extract_scope_info(text, match.start_pos)
            context.scope_type = scope_type
            context.scope_name = scope_name
            
            # Extract imports
            context.imports = analyzer.extract_imports(text)
            
            # Check if in test code
            context.is_test_code = analyzer.is_in_test_code(text, match.start_pos)
            
            # Analyze syntax context
            syntax_context = analyzer.analyze(text, match.start_pos)
            if syntax_context:
                context.semantic_role = self._determine_semantic_role(syntax_context)
        
        # Language-agnostic analysis
        context.is_commented_out = self._is_commented_out(text, match.start_pos, language)
        context.indentation_level = self._get_indentation_level(text, match.start_pos)
        context.comments_nearby = self._extract_nearby_comments(text, match.start_pos)
        context.control_flow = self._extract_control_flow(text, match.start_pos)
        context.surrounding_vars = self._extract_surrounding_variables(text, match.start_pos, language)
        
        return context
    
    def _should_include_match(self, context_match: ContextualPattern) -> bool:
        """Apply context-based filtering rules"""
        match = context_match.base_match
        
        # Keep TODO/FIXME patterns even in comments
        if match.category == PatternCategory.UNCERTAINTY:
            return True
        
        # Skip other patterns in comments 
        if context_match.is_commented_out:
            return False
        
        # Reduce severity for patterns in test code
        if context_match.is_test_code:
            if match.category == PatternCategory.SECURITY:
                # Security patterns in tests might be intentional
                match.severity = PatternSeverity.MEDIUM
                match.requires_multi_ai = False
        
        # Skip certain patterns based on imports
        if match.category == PatternCategory.SECURITY:
            # If proper security libraries are imported, some patterns might be ok
            security_libs = {'bcrypt', 'hashlib', 'cryptography', 'jwt', 'passlib'}
            if any(lib in imp for imp in context_match.imports for lib in security_libs):
                if match.keyword in ['password', 'hash', 'encrypt']:
                    match.confidence *= 0.7  # Reduce confidence
        
        # Context-specific rules
        if context_match.semantic_role == "decorator":
            # Decorators often contain security keywords legitimately
            if match.category == PatternCategory.SECURITY:
                match.confidence *= 0.8
        
        return True
    
    def _determine_semantic_role(self, syntax_context: SyntaxContext) -> Optional[str]:
        """Determine the semantic role of a code element"""
        node_type = syntax_context.node_type
        parent_types = syntax_context.parent_nodes
        
        if node_type == "arg" and "FunctionDef" in parent_types:
            return "parameter"
        elif node_type == "Return":
            return "return_value"
        elif node_type == "Compare" or "BoolOp" in parent_types:
            return "condition"
        elif node_type == "Assign":
            return "assignment"
        elif node_type == "Decorator":
            return "decorator"
        elif "ExceptHandler" in parent_types:
            return "exception_handler"
        
        return None
    
    def _is_commented_out(self, text: str, position: int, language: CodeLanguage) -> bool:
        """Check if position is within a comment"""
        line_start = text.rfind('\n', 0, position) + 1
        line_end = text.find('\n', position)
        if line_end == -1:
            line_end = len(text)
        
        line = text[line_start:line_end]
        pos_in_line = position - line_start
        
        # Language-specific comment detection
        if language in [CodeLanguage.PYTHON, CodeLanguage.RUST]:
            comment_pos = line.find('#')
            if comment_pos != -1 and comment_pos < pos_in_line:
                return True
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT, CodeLanguage.JAVA, CodeLanguage.CPP, CodeLanguage.GO]:
            # Single line comments
            comment_pos = line.find('//')
            if comment_pos != -1 and comment_pos < pos_in_line:
                return True
            
            # Multi-line comments (simplified check)
            before_pos = text[:position]
            after_pos = text[position:]
            
            last_open = before_pos.rfind('/*')
            last_close = before_pos.rfind('*/')
            
            if last_open > last_close:
                # We're inside a multi-line comment
                next_close = after_pos.find('*/')
                if next_close != -1:
                    return True
        
        return False
    
    def _get_indentation_level(self, text: str, position: int) -> int:
        """Get indentation level at position"""
        line_start = text.rfind('\n', 0, position) + 1
        line_end = text.find('\n', position)
        if line_end == -1:
            line_end = len(text)
            
        # Get the full line
        full_line = text[line_start:line_end]
        
        # Count leading spaces/tabs
        indent = 0
        for char in full_line:
            if char == ' ':
                indent += 1
            elif char == '\t':
                indent += 4  # Assume 4 spaces per tab
            else:
                break
        
        return indent // 4  # Return level, not spaces
    
    def _extract_nearby_comments(self, text: str, position: int, window: int = 200) -> List[str]:
        """Extract comments near the position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        nearby_text = text[start:end]
        
        comments = []
        
        # Simple comment extraction (can be enhanced per language)
        for line in nearby_text.split('\n'):
            line = line.strip()
            if line.startswith('#') or line.startswith('//'):
                comments.append(line.lstrip('#/').strip())
        
        return comments
    
    def _extract_control_flow(self, text: str, position: int) -> List[str]:
        """Extract control flow context"""
        # Get lines before position
        before_text = text[:position]
        lines = before_text.split('\n')[-20:]  # Last 20 lines
        
        control_flow = []
        control_keywords = ['if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'switch', 'case']
        
        for line in lines:
            stripped = line.strip()
            for keyword in control_keywords:
                if stripped.startswith(keyword + ' ') or stripped.startswith(keyword + ':'):
                    control_flow.append(keyword)
        
        return control_flow
    
    def _extract_surrounding_variables(self, text: str, position: int, language: CodeLanguage) -> List[str]:
        """Extract variable names near the position"""
        # Get surrounding context
        start = max(0, position - 500)
        end = min(len(text), position + 500)
        context = text[start:end]
        
        variables = set()
        
        # Language-specific variable patterns
        if language == CodeLanguage.PYTHON:
            # Simple pattern for Python variables
            var_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*=')
            for match in var_pattern.finditer(context):
                variables.add(match.group(1))
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            # JS/TS variable declarations
            var_pattern = re.compile(r'\b(?:let|const|var)\s+([a-zA-Z_$]\w*)')
            for match in var_pattern.finditer(context):
                variables.add(match.group(1))
        
        return list(variables)
    
    def get_enhanced_consultation_strategy(self, matches: List[ContextualPattern]) -> Dict[str, Any]:
        """Generate consultation strategy based on contextual matches"""
        base_matches = [m.base_match for m in matches]
        base_strategy = self.base_engine.get_consultation_strategy(base_matches)
        
        # Enhance strategy with context
        context_insights = []
        
        # Analyze patterns by context
        test_patterns = sum(1 for m in matches if m.is_test_code)
        commented_patterns = sum(1 for m in matches if m.is_commented_out)
        
        if test_patterns > 0:
            context_insights.append(f"{test_patterns} patterns found in test code")
        
        if commented_patterns > 0:
            context_insights.append(f"{commented_patterns} patterns in comments")
        
        # Language-specific insights
        languages = set(m.language.value for m in matches)
        if len(languages) > 1:
            context_insights.append(f"Mixed language codebase: {', '.join(languages)}")
        elif len(languages) == 1:
            context_insights.append(f"Language: {list(languages)[0]}")
        
        # Scope analysis
        scope_types = set(m.scope_type for m in matches)
        if 'function' in scope_types:
            func_names = [m.scope_name for m in matches if m.scope_type == 'function' and m.scope_name]
            if func_names:
                unique_funcs = list(set(func_names))[:3]
                context_insights.append(f"Patterns in functions: {', '.join(unique_funcs)}")
        
        # If no specific insights, ensure we have at least one
        if not context_insights:
            context_insights.append(f"Found {len(matches)} patterns")
        
        base_strategy['context_insights'] = context_insights
        
        return base_strategy


# Example usage and testing
if __name__ == "__main__":
    # Test the context-aware matcher
    matcher = ContextAwarePatternMatcher()
    
    test_code = '''
import hashlib
import bcrypt
from cryptography.fernet import Fernet

class UserAuthentication:
    def __init__(self):
        self.secret_key = "my-secret-key"  # TODO: Move to environment variable
        
    def hash_password(self, password):
        # Using bcrypt for secure password hashing
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    def verify_password(self, password, hashed):
        # This is secure because we're using bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
    def test_authentication(self):
        # In test code, hardcoded passwords are acceptable
        test_password = "test123"
        hashed = self.hash_password(test_password)
        assert self.verify_password(test_password, hashed)
    
    def encrypt_data(self, data):
        # FIXME: Need to implement proper key rotation
        cipher = Fernet(self.secret_key)
        return cipher.encrypt(data.encode())

# Some commented out code
# password = "admin123"  # Old insecure code
'''
    
    matches = matcher.detect_with_context(test_code, "auth.py")
    
    print("Context-Aware Pattern Detection Results")
    print("=" * 60)
    
    for match in matches:
        print(f"\nPattern: {match.base_match.keyword}")
        print(f"  Category: {match.base_match.category.value}")
        print(f"  Severity: {match.base_match.severity.name}")
        print(f"  Language: {match.language.value}")
        print(f"  Scope: {match.scope_type} {match.scope_name or ''}")
        print(f"  Is Test: {match.is_test_code}")
        print(f"  Is Comment: {match.is_commented_out}")
        print(f"  Semantic Role: {match.semantic_role}")
        print(f"  Imports: {', '.join(match.imports[:3])}")
    
    strategy = matcher.get_enhanced_consultation_strategy(matches)
    print(f"\nConsultation Strategy:")
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Context Insights: {', '.join(strategy['context_insights'])}")
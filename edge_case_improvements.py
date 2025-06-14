#!/usr/bin/env python3
"""Edge case handling improvements"""

from typing import Optional, List, Tuple
import unicodedata

class RobustTextProcessor:
    """Handle edge cases in text processing"""
    
    def normalize_text(self, text: str) -> str:
        """Normalize text to handle various encodings and special characters"""
        if not text:
            return ""
        
        # Handle different line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize Unicode (handle different representations of same character)
        text = unicodedata.normalize('NFC', text)
        
        # Handle BOM (Byte Order Mark)
        if text.startswith('\ufeff'):
            text = text[1:]
        
        return text
    
    def safe_position_to_line(self, text: str, position: int) -> Tuple[int, int]:
        """Convert position to line/column with boundary checks"""
        if not text:
            return (1, 0)
        
        # Clamp position to valid range
        position = max(0, min(position, len(text)))
        
        # Count lines up to position
        line_num = text[:position].count('\n') + 1
        
        # Find column
        line_start = text.rfind('\n', 0, position) + 1
        column = position - line_start
        
        return (line_num, column)
    
    def handle_mixed_indentation(self, line: str) -> int:
        """Handle files with mixed tabs and spaces"""
        indent = 0
        for char in line:
            if char == ' ':
                indent += 1
            elif char == '\t':
                # Use next tab stop (typically 4 or 8)
                indent = ((indent // 4) + 1) * 4
            else:
                break
        return indent
    
    def extract_context_safe(self, text: str, position: int, window: int = 150) -> str:
        """Extract context with proper boundary handling"""
        if not text:
            return ""
        
        # Ensure position is valid
        position = max(0, min(position, len(text) - 1))
        
        # Calculate boundaries
        start = max(0, position - window)
        end = min(len(text), position + window)
        
        # Try to align to line boundaries for better context
        if start > 0:
            # Find previous newline
            newline_pos = text.rfind('\n', 0, start)
            if newline_pos != -1 and start - newline_pos < 50:
                start = newline_pos + 1
        
        if end < len(text):
            # Find next newline
            newline_pos = text.find('\n', end)
            if newline_pos != -1 and newline_pos - end < 50:
                end = newline_pos
        
        return text[start:end]
    
    def handle_incomplete_code(self, code: str) -> Tuple[str, List[str]]:
        """Handle incomplete/malformed code gracefully"""
        warnings = []
        
        # Check for unclosed strings
        quote_counts = {
            "'": code.count("'") - code.count("\\'"),
            '"': code.count('"') - code.count('\\"'),
            '"""': code.count('"""'),
            "'''": code.count("'''"),
        }
        
        for quote, count in quote_counts.items():
            if len(quote) == 1 and count % 2 != 0:
                warnings.append(f"Unclosed {quote} quote detected")
            elif len(quote) == 3 and count % 2 != 0:
                warnings.append(f"Unclosed triple quote detected")
        
        # Check for unclosed brackets
        bracket_pairs = [('(', ')'), ('[', ']'), ('{', '}')]
        for open_b, close_b in bracket_pairs:
            open_count = code.count(open_b)
            close_count = code.count(close_b)
            if open_count != close_count:
                warnings.append(f"Mismatched {open_b}{close_b}: {open_count} open, {close_count} close")
        
        # Add synthetic closures for AST parsing
        if warnings:
            # This is a simplified approach - could be enhanced
            synthetic_code = code
            for quote, count in quote_counts.items():
                if len(quote) == 1 and count % 2 != 0:
                    synthetic_code += quote
            
            for open_b, close_b in bracket_pairs:
                diff = code.count(open_b) - code.count(close_b)
                if diff > 0:
                    synthetic_code += close_b * diff
            
            return synthetic_code, warnings
        
        return code, warnings
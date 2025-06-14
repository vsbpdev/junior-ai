#!/usr/bin/env python3
"""Enhanced context-aware pattern matching with all improvements"""

import ast
import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import hashlib
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

from pattern_detection import (
    PatternCategory, PatternSeverity, PatternMatch,
    PatternDefinition, EnhancedPatternDetectionEngine
)

logger = logging.getLogger(__name__)

class EnhancedContextAwarePatternMatcher:
    """Enhanced version with all improvements integrated"""
    
    def __init__(self, base_engine: Optional[EnhancedPatternDetectionEngine] = None):
        self.base_engine = base_engine or EnhancedPatternDetectionEngine()
        self.language_detector = LanguageDetector()
        
        # Performance: Cache for AST and analysis results
        self._ast_cache = {}
        self._analysis_cache = {}
        self._cache_lock = threading.Lock()
        
        # Security: Limits and validation
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        self.MAX_AST_DEPTH = 100
        self.ANALYSIS_TIMEOUT = 5.0  # seconds
        
        # Thread pool for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def detect_with_context(self, text: str, filename: Optional[str] = None) -> List['ContextualPattern']:
        """Enhanced detection with all improvements"""
        try:
            # Security: Validate input size
            if len(text) > self.MAX_FILE_SIZE:
                logger.warning(f"File too large: {len(text)} bytes")
                return self._fallback_detection(text, "File too large for full analysis")
            
            # Edge case: Normalize text
            text = self._normalize_text(text)
            
            # Performance: Check cache
            cache_key = self._get_cache_key(text, filename)
            if cache_key in self._analysis_cache:
                logger.debug("Using cached analysis results")
                return self._analysis_cache[cache_key]
            
            # Detect language
            language = self._detect_language_safe(text, filename)
            
            # Get base pattern matches with timeout
            try:
                future = self.executor.submit(self.base_engine.detect_patterns, text)
                base_matches = future.result(timeout=self.ANALYSIS_TIMEOUT)
            except TimeoutError:
                logger.warning("Pattern detection timeout")
                return self._fallback_detection(text, "Pattern detection timeout")
            
            # Performance: Batch process positions for context analysis
            positions = [match.start_pos for match in base_matches]
            context_map = self._batch_analyze_contexts(text, positions, language)
            
            # Enhance matches with context
            contextual_matches = []
            for match in base_matches:
                context_info = context_map.get(match.start_pos, {})
                context_match = self._create_contextual_match(match, context_info, language)
                
                # Apply filtering and adjustments
                if self._should_include_match_enhanced(context_match):
                    contextual_matches.append(context_match)
            
            # Cache results
            with self._cache_lock:
                self._analysis_cache[cache_key] = contextual_matches
            
            return contextual_matches
            
        except Exception as e:
            logger.error(f"Unexpected error in pattern detection: {e}")
            return self._fallback_detection(text, str(e))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing"""
        if not text:
            return ""
        
        # Handle line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove BOM
        if text.startswith('\ufeff'):
            text = text[1:]
        
        # Security: Remove zero-width characters
        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            text = text.replace(char, '')
        
        return text
    
    @lru_cache(maxsize=256)
    def _get_cache_key(self, text: str, filename: Optional[str]) -> str:
        """Generate cache key for analysis results"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{filename or 'unknown'}:{text_hash}"
    
    def _detect_language_safe(self, text: str, filename: Optional[str]) -> 'CodeLanguage':
        """Safely detect language with fallback"""
        try:
            if filename:
                lang = self.language_detector.detect_from_extension(filename)
                if lang != CodeLanguage.UNKNOWN:
                    return lang
            
            return self.language_detector.detect_from_content(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return CodeLanguage.UNKNOWN
    
    def _batch_analyze_contexts(self, text: str, positions: List[int], 
                               language: 'CodeLanguage') -> Dict[int, Dict[str, Any]]:
        """Analyze multiple positions in batch for performance"""
        context_map = {}
        
        # Group positions by proximity for efficient processing
        position_groups = self._group_nearby_positions(positions)
        
        for group in position_groups:
            try:
                # Analyze group together
                group_contexts = self._analyze_position_group(text, group, language)
                context_map.update(group_contexts)
            except Exception as e:
                logger.warning(f"Failed to analyze position group: {e}")
                # Fallback to basic context for this group
                for pos in group:
                    context_map[pos] = self._get_basic_context(text, pos)
        
        return context_map
    
    def _group_nearby_positions(self, positions: List[int], proximity: int = 500) -> List[List[int]]:
        """Group positions that are close together"""
        if not positions:
            return []
        
        sorted_positions = sorted(positions)
        groups = [[sorted_positions[0]]]
        
        for pos in sorted_positions[1:]:
            if pos - groups[-1][-1] <= proximity:
                groups[-1].append(pos)
            else:
                groups.append([pos])
        
        return groups
    
    def _should_include_match_enhanced(self, context_match: 'ContextualPattern') -> bool:
        """Enhanced filtering with security considerations"""
        match = context_match.base_match
        
        # Security: Check for security-sensitive context
        if hasattr(context_match, 'security_flags') and context_match.security_flags:
            # Increase severity for patterns in security-sensitive context
            if 'eval_usage' in context_match.security_flags:
                match.severity = PatternSeverity.CRITICAL
                match.requires_multi_ai = True
        
        # Apply base filtering rules
        return self._should_include_match(context_match)
    
    def _fallback_detection(self, text: str, reason: str) -> List['ContextualPattern']:
        """Fallback detection when full analysis fails"""
        logger.info(f"Using fallback detection: {reason}")
        
        # Use simple keyword search
        simple_matches = []
        patterns = {
            'password': PatternCategory.SECURITY,
            'api_key': PatternCategory.SECURITY,
            'TODO': PatternCategory.UNCERTAINTY,
            'FIXME': PatternCategory.UNCERTAINTY,
        }
        
        for keyword, category in patterns.items():
            for match in re.finditer(rf'\b{keyword}\b', text, re.IGNORECASE):
                simple_match = PatternMatch(
                    keyword=keyword,
                    category=category,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.5,
                    context=text[max(0, match.start()-50):match.end()+50],
                    start_pos=match.start(),
                    end_pos=match.end(),
                    line_number=text[:match.start()].count('\n') + 1,
                    requires_multi_ai=False
                )
                
                context_match = ContextualPattern(
                    base_match=simple_match,
                    language=CodeLanguage.UNKNOWN,
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
                
                simple_matches.append(context_match)
        
        return simple_matches
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        self._ast_cache.clear()
        self._analysis_cache.clear()
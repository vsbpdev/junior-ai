#!/usr/bin/env python3
"""
AI Consultation Manager Improvements
Performance, security, and reliability enhancements
"""

import concurrent.futures
import threading
import hashlib
import time
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging

from pattern_detection import PatternMatch, PatternSeverity, PatternCategory
from response_handlers import ConsultationResponse


class ConsultationErrorType(Enum):
    """Types of consultation errors"""
    AI_UNAVAILABLE = "ai_unavailable"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT = "rate_limit"
    INTERNAL_ERROR = "internal_error"
    PARTIAL_FAILURE = "partial_failure"


@dataclass
class ConsultationError:
    """Detailed error information"""
    error_type: ConsultationErrorType
    message: str
    ai_name: Optional[str] = None
    recoverable: bool = True
    retry_after: Optional[int] = None  # seconds
    details: Dict[str, Any] = field(default_factory=dict)


class PromptValidator:
    """Validate and sanitize prompts for security"""
    
    # Patterns that might indicate prompt injection
    SUSPICIOUS_PATTERNS = [
        r'ignore\s+previous\s+instructions',
        r'system\s+prompt',
        r'admin\s+mode',
        r'bypass\s+safety',
        r'</?\w+>',  # HTML/XML tags
        r'<script',
        r'javascript:',
        r'eval\(',
        r'exec\(',
    ]
    
    @classmethod
    def validate_prompt(cls, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate prompt for security issues"""
        if not prompt or not isinstance(prompt, str):
            return False, "Prompt must be a non-empty string"
        
        # Check length
        if len(prompt) > 50000:  # 50KB limit
            return False, "Prompt too long (max 50KB)"
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False, f"Prompt contains suspicious pattern: {pattern}"
        
        return True, None
    
    @classmethod
    def sanitize_context(cls, context: str, max_length: int = 10000) -> str:
        """Sanitize context for safe AI consumption"""
        if not context:
            return ""
        
        # Truncate if too long
        if len(context) > max_length:
            context = context[:max_length] + "... [truncated]"
        
        # Remove potential control characters
        context = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', context)
        
        # Escape special characters
        context = context.replace('\\', '\\\\')
        context = context.replace('"', '\\"')
        
        return context


class ParallelConsultationExecutor:
    """Execute AI consultations in parallel with proper error handling"""
    
    def __init__(self, ai_caller, logger, timeout: float = 30.0):
        self.ai_caller = ai_caller
        self.logger = logger
        self.timeout = timeout
        self.rate_limiters: Dict[str, 'RateLimiter'] = {}
    
    def execute_multi_consultation(self, 
                                 ai_names: List[str], 
                                 prompt: str,
                                 audit: Any) -> Dict[str, str]:
        """Execute multi-AI consultation in parallel with proper error handling"""
        responses = {}
        responses_lock = threading.Lock()
        
        def call_ai_with_timeout(ai_name: str) -> Tuple[str, Optional[str], Optional[ConsultationError]]:
            """Thread-safe AI call with timeout and rate limiting"""
            try:
                # Check rate limit
                if not self._check_rate_limit(ai_name):
                    error = ConsultationError(
                        error_type=ConsultationErrorType.RATE_LIMIT,
                        message=f"Rate limit exceeded for {ai_name}",
                        ai_name=ai_name,
                        retry_after=60
                    )
                    return ai_name, None, error
                
                # Record prompt in audit
                audit.prompts_sent[ai_name] = prompt
                
                # Execute with timeout
                start_time = time.time()
                
                if self.ai_caller:
                    # Use concurrent.futures for timeout
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.ai_caller, ai_name, prompt, 0.7)
                        try:
                            response = future.result(timeout=self.timeout)
                            elapsed = time.time() - start_time
                            
                            if not response.startswith("❌"):
                                self.logger.info(f"AI {ai_name} responded in {elapsed:.2f}s")
                                return ai_name, response, None
                            else:
                                error = ConsultationError(
                                    error_type=ConsultationErrorType.AI_UNAVAILABLE,
                                    message=response,
                                    ai_name=ai_name
                                )
                                return ai_name, None, error
                                
                        except concurrent.futures.TimeoutError:
                            error = ConsultationError(
                                error_type=ConsultationErrorType.TIMEOUT,
                                message=f"AI {ai_name} timed out after {self.timeout}s",
                                ai_name=ai_name
                            )
                            return ai_name, None, error
                else:
                    # Test mode
                    return ai_name, f"Test response from {ai_name}", None
                    
            except Exception as e:
                self.logger.error(f"Failed to call {ai_name}: {e}")
                error = ConsultationError(
                    error_type=ConsultationErrorType.INTERNAL_ERROR,
                    message=str(e),
                    ai_name=ai_name,
                    details={'exception_type': type(e).__name__}
                )
                return ai_name, None, error
        
        # Use ThreadPoolExecutor for parallel execution
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ai_names)) as executor:
            future_to_ai = {
                executor.submit(call_ai_with_timeout, ai_name): ai_name 
                for ai_name in ai_names
            }
            
            for future in concurrent.futures.as_completed(future_to_ai):
                ai_name, response, error = future.result()
                
                if response:
                    with responses_lock:
                        responses[ai_name] = response
                        audit.responses_received[ai_name] = response
                        audit.ais_consulted.append(ai_name)
                elif error:
                    errors.append(error)
        
        # Handle partial failures
        if errors and not responses:
            # Total failure
            raise ConsultationException(
                "All AI consultations failed",
                errors=errors
            )
        elif errors:
            # Partial failure - log but continue
            self.logger.warning(f"Partial consultation failure: {len(errors)} AIs failed")
            audit.metadata['partial_failures'] = [
                {'ai': e.ai_name, 'error': e.message} for e in errors
            ]
        
        return responses
    
    def _check_rate_limit(self, ai_name: str) -> bool:
        """Check if AI is within rate limits"""
        if ai_name not in self.rate_limiters:
            self.rate_limiters[ai_name] = RateLimiter(
                calls_per_minute=20,  # Configurable
                burst_size=5
            )
        return self.rate_limiters[ai_name].allow()


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, calls_per_minute: int, burst_size: int):
        self.rate = calls_per_minute / 60.0  # calls per second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def allow(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class ConsultationException(Exception):
    """Exception for consultation failures"""
    
    def __init__(self, message: str, errors: List[ConsultationError] = None):
        super().__init__(message)
        self.errors = errors or []


class OptimizedCacheManager:
    """Optimized cache with TTL and bounded size"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, 
            daemon=True
        )
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return value
                else:
                    # Expired
                    del self.cache[key]
                    self.access_order.remove(key)
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with TTL"""
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = (value, time.time())
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def _periodic_cleanup(self):
        """Periodically clean up expired entries"""
        while True:
            time.sleep(300)  # 5 minutes
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        with self.lock:
            current_time = time.time()
            expired = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp > self.ttl
            ]
            for key in expired:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)


class PatternHasher:
    """Generate deterministic hashes for pattern combinations"""
    
    @staticmethod
    def hash_patterns(patterns: List[PatternMatch]) -> str:
        """Generate hash for pattern list for caching"""
        pattern_data = []
        for p in sorted(patterns, key=lambda x: (x.start_pos, x.keyword)):
            pattern_data.append(
                f"{p.category.value}:{p.keyword}:{p.severity.value}"
            )
        
        pattern_str = '|'.join(pattern_data)
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def hash_context(context: str, max_length: int = 500) -> str:
        """Generate hash for context caching"""
        # Use first N chars for context hash to avoid huge contexts
        context_sample = context[:max_length] if len(context) > max_length else context
        return hashlib.sha256(context_sample.encode()).hexdigest()[:16]


def generate_fallback_recommendation(patterns: List[PatternMatch]) -> str:
    """Generate helpful fallback recommendation when AI consultation fails"""
    if not patterns:
        return "No specific patterns detected. General code review recommended."
    
    recommendations = []
    
    # Group patterns by category
    by_category = defaultdict(list)
    for pattern in patterns:
        by_category[pattern.category].append(pattern)
    
    # Generate category-specific recommendations
    for category, _ in by_category.items():
        if category == PatternCategory.SECURITY:
            recommendations.append(
                "🔒 Security: Review credential handling, validate inputs, "
                "and ensure proper authentication/authorization."
            )
        elif category == PatternCategory.ALGORITHM:
            recommendations.append(
                "⚡ Performance: Consider algorithm complexity, caching strategies, "
                "and potential optimization opportunities."
            )
        elif category == PatternCategory.UNCERTAINTY:
            recommendations.append(
                "❓ Uncertainty: Address TODO/FIXME items, clarify unclear logic, "
                "and add comprehensive documentation."
            )
        elif category == PatternCategory.GOTCHA:
            recommendations.append(
                "⚠️ Gotchas: Review edge cases, handle errors gracefully, "
                "and validate assumptions."
            )
        elif category == PatternCategory.ARCHITECTURE:
            recommendations.append(
                "🏗️ Architecture: Evaluate design patterns, modularity, "
                "and maintainability of the solution."
            )
    
    return "\n".join(recommendations)
#!/usr/bin/env python3
"""
Enhanced Error Handling for AI Consultation Manager
Detailed error tracking, recovery strategies, and fallback mechanisms
"""

import traceback
import sys
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1      # Degraded performance but functional
    MEDIUM = 2   # Some features unavailable
    HIGH = 3     # Major functionality impaired
    CRITICAL = 4 # System non-functional


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    CACHE = "use_cache"
    SKIP = "skip"
    FAIL = "fail"


@dataclass
class ErrorContext:
    """Comprehensive error context"""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = "unknown"
    operation: str = "unknown"
    ai_name: Optional[str] = None
    pattern_info: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    recovery_attempted: bool = False
    user_impact: Optional[str] = None


@dataclass
class RecoveryResult:
    """Result of recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    result: Optional[Any] = None
    fallback_message: Optional[str] = None
    performance_impact: Optional[str] = None


class ErrorHandler:
    """Comprehensive error handling with recovery"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies = self._init_recovery_strategies()
        self.error_thresholds = {
            ErrorSeverity.LOW: 10,      # Max 10 low severity errors
            ErrorSeverity.MEDIUM: 5,    # Max 5 medium severity errors
            ErrorSeverity.HIGH: 2,      # Max 2 high severity errors
            ErrorSeverity.CRITICAL: 1   # Any critical error
        }
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
    
    def _init_recovery_strategies(self) -> Dict[str, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different error types"""
        return {
            'timeout': [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            'rate_limit': [RecoveryStrategy.CACHE, RecoveryStrategy.DEGRADE],
            'ai_unavailable': [RecoveryStrategy.FALLBACK, RecoveryStrategy.SKIP],
            'validation_error': [RecoveryStrategy.FAIL],
            'partial_failure': [RecoveryStrategy.DEGRADE],
            'connection_error': [RecoveryStrategy.RETRY, RecoveryStrategy.CACHE],
            'unknown': [RecoveryStrategy.FALLBACK, RecoveryStrategy.FAIL]
        }
    
    def handle_error(self,
                    error: Exception,
                    context: Dict[str, Any],
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> RecoveryResult:
        """Handle error with context and attempt recovery"""
        # Create error context
        error_context = ErrorContext(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            component=context.get('component', 'unknown'),
            operation=context.get('operation', 'unknown'),
            ai_name=context.get('ai_name'),
            pattern_info=context.get('pattern_info'),
            stack_trace=traceback.format_exc(),
            retry_count=context.get('retry_count', 0)
        )
        
        # Log error
        self._log_error(error_context)
        
        # Store in history
        self.error_history.append(error_context)
        self._cleanup_error_history()
        
        # Update error counts
        self.error_counts[severity] += 1
        
        # Check if we should circuit break
        if self._should_circuit_break():
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FAIL,
                fallback_message="System experiencing too many errors. Circuit breaker activated."
            )
        
        # Attempt recovery
        return self._attempt_recovery(error, error_context, context)
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        log_message = (
            f"Error in {error_context.component}.{error_context.operation}: "
            f"{error_context.error_type} - {error_context.message}"
        )
        
        if error_context.ai_name:
            log_message += f" (AI: {error_context.ai_name})"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log stack trace for high/critical errors
        if error_context.severity.value >= ErrorSeverity.HIGH.value:
            self.logger.debug(f"Stack trace:\n{error_context.stack_trace}")
    
    def _should_circuit_break(self) -> bool:
        """Check if circuit breaker should activate"""
        for severity, count in self.error_counts.items():
            if count >= self.error_thresholds[severity]:
                return True
        return False
    
    def _attempt_recovery(self,
                         error: Exception,
                         error_context: ErrorContext,
                         context: Dict[str, Any]) -> RecoveryResult:
        """Attempt to recover from error"""
        # Determine error category
        error_category = self._categorize_error(error)
        
        # Get recovery strategies
        strategies = self.recovery_strategies.get(
            error_category,
            self.recovery_strategies['unknown']
        )
        
        # Try each strategy
        for strategy in strategies:
            if strategy == RecoveryStrategy.RETRY:
                if error_context.retry_count < 3:
                    return RecoveryResult(
                        success=True,
                        strategy_used=strategy,
                        fallback_message="Retrying operation..."
                    )
            
            elif strategy == RecoveryStrategy.FALLBACK:
                fallback_result = self._generate_fallback(error_context, context)
                return RecoveryResult(
                    success=True,
                    strategy_used=strategy,
                    result=fallback_result,
                    fallback_message="Using fallback response"
                )
            
            elif strategy == RecoveryStrategy.CACHE:
                cache_result = context.get('cache_lookup', lambda: None)()
                if cache_result:
                    return RecoveryResult(
                        success=True,
                        strategy_used=strategy,
                        result=cache_result,
                        performance_impact="Using cached result"
                    )
            
            elif strategy == RecoveryStrategy.DEGRADE:
                return RecoveryResult(
                    success=True,
                    strategy_used=strategy,
                    performance_impact="Running in degraded mode"
                )
            
            elif strategy == RecoveryStrategy.SKIP:
                return RecoveryResult(
                    success=True,
                    strategy_used=strategy,
                    fallback_message=f"Skipping {error_context.ai_name or 'component'}"
                )
        
        # No recovery possible
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.FAIL,
            fallback_message=f"Unable to recover from {error_context.error_type}"
        )
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for recovery strategy selection"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if 'timeout' in error_message:
            return 'timeout'
        elif 'rate limit' in error_message:
            return 'rate_limit'
        elif 'connection' in error_message or 'network' in error_message:
            return 'connection_error'
        elif 'validation' in error_message:
            return 'validation_error'
        elif any(ai in error_message for ai in ['gemini', 'grok', 'openai', 'deepseek']):
            return 'ai_unavailable'
        else:
            return 'unknown'
    
    def _generate_fallback(self,
                          error_context: ErrorContext,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate helpful fallback response"""
        pattern_info = error_context.pattern_info or {}
        
        fallback = {
            'error': True,
            'error_type': error_context.error_type,
            'message': "An error occurred during AI consultation",
            'recommendations': []
        }
        
        # Add pattern-specific recommendations
        if pattern_info.get('category') == 'security':
            fallback['recommendations'].extend([
                "Review security best practices",
                "Validate all inputs and outputs",
                "Consider using established security libraries"
            ])
        elif pattern_info.get('category') == 'algorithm':
            fallback['recommendations'].extend([
                "Analyze algorithm complexity",
                "Consider caching frequently computed results",
                "Profile code to identify bottlenecks"
            ])
        elif pattern_info.get('category') == 'architecture':
            fallback['recommendations'].extend([
                "Review SOLID principles",
                "Consider design patterns for this use case",
                "Ensure proper separation of concerns"
            ])
        else:
            fallback['recommendations'].extend([
                "Review the code for potential issues",
                "Consider breaking down complex logic",
                "Add comprehensive error handling"
            ])
        
        return fallback
    
    def _cleanup_error_history(self):
        """Clean up old error history"""
        # Keep last 1000 errors or last 24 hours
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Remove errors older than 24 hours
        cutoff_time = datetime.now().timestamp() - 86400
        self.error_history = [
            e for e in self.error_history
            if e.timestamp.timestamp() > cutoff_time
        ]
    
    def reset_error_counts(self):
        """Reset error counts (for circuit breaker reset)"""
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        summary = {
            'total_errors': len(self.error_history),
            'error_counts_by_severity': dict(self.error_counts),
            'circuit_breaker_active': self._should_circuit_break(),
            'recent_errors': []
        }
        
        # Add last 10 errors
        for error in self.error_history[-10:]:
            summary['recent_errors'].append({
                'timestamp': error.timestamp.isoformat(),
                'type': error.error_type,
                'severity': error.severity.name,
                'component': error.component,
                'message': error.message[:100]  # Truncate long messages
            })
        
        return summary


class ResilientAIConsultationManager:
    """AI Consultation Manager with resilient error handling"""
    
    def __init__(self,
                 base_manager: Any,
                 error_handler: ErrorHandler):
        self.base_manager = base_manager
        self.error_handler = error_handler
    
    def consult_with_recovery(self,
                             prompt: str,
                             context: str,
                             options: Dict[str, Any] = None) -> Any:
        """Consult with automatic error recovery"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Attempt consultation
                result = self.base_manager.consult_junior(prompt, context, options)
                
                # Reset error counts on success
                if retry_count > 0:
                    self.error_handler.logger.info(
                        f"Consultation succeeded after {retry_count} retries"
                    )
                
                return result
                
            except Exception as e:
                # Handle error with context
                error_context = {
                    'component': 'AIConsultationManager',
                    'operation': 'consult_junior',
                    'retry_count': retry_count,
                    'pattern_info': options.get('pattern_info') if options else None
                }
                
                recovery_result = self.error_handler.handle_error(
                    e, error_context, ErrorSeverity.HIGH
                )
                
                if recovery_result.success:
                    if recovery_result.strategy_used == RecoveryStrategy.RETRY:
                        retry_count += 1
                        continue
                    elif recovery_result.result:
                        return recovery_result.result
                    else:
                        # Use fallback
                        return {
                            'success': False,
                            'fallback': True,
                            'message': recovery_result.fallback_message,
                            'recommendations': recovery_result.result.get(
                                'recommendations', []
                            ) if recovery_result.result else []
                        }
                else:
                    # Recovery failed
                    raise ConsultationError(
                        f"Consultation failed: {recovery_result.fallback_message}",
                        original_error=e
                    )
        
        # Max retries exceeded
        raise ConsultationError(
            f"Consultation failed after {max_retries} retries"
        )


class ConsultationError(Exception):
    """Custom exception for consultation failures"""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
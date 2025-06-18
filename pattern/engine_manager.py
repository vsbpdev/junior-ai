"""Pattern Engine Manager - Centralized initialization and management of pattern detection components.

This module provides centralized management for all pattern detection
subsystems in Junior AI Assistant. It handles the initialization,
configuration, and lifecycle of pattern detection engines, response
managers, and AI consultation components.

Key responsibilities:
- Initialize and configure pattern detection engines (sync and async)
- Manage text processing pipelines
- Set up response and AI consultation managers
- Handle component lifecycle and cleanup
- Provide unified access to all pattern detection subsystems

Components managed:
- PatternDetectionEngine: Core synchronous pattern detection
- AsyncCachedPatternEngine: High-performance async pattern detection with caching
- TextProcessingPipeline: Text analysis and preprocessing
- PatternResponseManager: AI response orchestration for patterns
- AIConsultationManager: Intelligent AI selection and consultation

The manager ensures proper initialization order, handles dependencies
between components, and provides graceful degradation when optional
components are unavailable.
"""

import sys
import os
import json
import tempfile
from typing import Dict, Any, Optional, Callable


class PatternEngineManager:
    """Manages initialization and lifecycle of pattern detection components."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None, 
                 ai_callers: Optional[Dict[str, Callable]] = None):
        """Initialize the pattern engine manager."""
        self.credentials = credentials or {}
        self.ai_callers = ai_callers or {}
        
        # Component instances
        self.pattern_engine = None
        self.sync_pattern_engine = None
        self.async_pipeline = None
        self.text_pipeline = None
        self.response_manager = None
        self.ai_consultation_manager = None
        self.pattern_config = None
        
        # Import availability
        self.imports_available = {
            'pattern_detection': False,
            'async_pattern_cache': False,
            'text_processing': False,
            'response_handlers': False,
            'ai_consultation': False
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pattern detection components."""
        # Extract pattern configuration
        self.pattern_config = self.credentials.get('pattern_detection', {})
        
        # Initialize sync pattern engine
        self._initialize_sync_pattern_engine()
        
        # Initialize async components
        self._initialize_async_components()
        
        # Initialize text processing
        self._initialize_text_pipeline()
        
        # Initialize response managers
        self._initialize_response_managers()
        
        # Set primary pattern engine
        self._setup_primary_pattern_engine()
    
    def _initialize_sync_pattern_engine(self):
        """Initialize synchronous pattern detection engine."""
        try:
            from pattern_detection import PatternDetectionEngine
            self.imports_available['pattern_detection'] = True
            
            # Create temporary config file for pattern engine
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
                json.dump({'pattern_detection': self.pattern_config}, temp_config)
                temp_config_path = temp_config.name
            
            try:
                self.sync_pattern_engine = PatternDetectionEngine(temp_config_path)
                print("✅ Pattern detection engine initialized", file=sys.stderr)
            except Exception as e:
                print(f"❌ Failed to initialize pattern engine: {e}", file=sys.stderr)
            finally:
                os.unlink(temp_config_path)
                
        except ImportError:
            print("Pattern detection module not available", file=sys.stderr)
    
    def _initialize_async_components(self):
        """Initialize async pattern detection components."""
        try:
            from async_cached_pattern_engine import AsyncCachedPatternEngine
            from async_pattern_cache import AsyncPatternCache
            self.imports_available['async_pattern_cache'] = True
            
            if self.sync_pattern_engine:
                # Create async cache
                cache_config = self.pattern_config.get('async_cache', {})
                cache = AsyncPatternCache(
                    max_size=cache_config.get('max_size', 1000),
                    ttl_seconds=cache_config.get('ttl', 3600),
                    enable_deduplication=cache_config.get('deduplication', True)
                )
                
                # Create async cached engine
                self.async_pipeline = AsyncCachedPatternEngine(
                    self.sync_pattern_engine,
                    cache
                )
                print("✅ Async pattern detection initialized", file=sys.stderr)
                
        except ImportError:
            print("Async pattern cache not available", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing async components: {e}", file=sys.stderr)
    
    def _initialize_text_pipeline(self):
        """Initialize text processing pipeline."""
        try:
            from text_processing_pipeline import TextProcessingPipeline
            self.imports_available['text_processing'] = True
            
            if self.sync_pattern_engine:
                self.text_pipeline = TextProcessingPipeline(self.sync_pattern_engine)
                print("✅ Text processing pipeline initialized", file=sys.stderr)
                
        except ImportError:
            print("Text processing pipeline not available", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing text pipeline: {e}", file=sys.stderr)
    
    def _initialize_response_managers(self):
        """Initialize response and AI consultation managers."""
        # Initialize response manager
        try:
            from response_handlers import PatternResponseManager
            self.imports_available['response_handlers'] = True
            
            if self.ai_callers:
                self.response_manager = PatternResponseManager(self.ai_callers)
                print("✅ Pattern response manager initialized", file=sys.stderr)
                
        except ImportError:
            print("Response handlers not available", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing response manager: {e}", file=sys.stderr)
        
        # Initialize AI consultation manager
        try:
            from ai_consultation_manager import AIConsultationManager
            self.imports_available['ai_consultation'] = True
            
            if self.ai_callers:
                consultation_config = self.credentials.get('ai_consultation_preferences', {})
                self.ai_consultation_manager = AIConsultationManager(
                    self.ai_callers,
                    consultation_config
                )
                print("✅ AI consultation manager initialized", file=sys.stderr)
                
        except ImportError:
            print("AI consultation manager not available", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing AI consultation manager: {e}", file=sys.stderr)
    
    def _setup_primary_pattern_engine(self):
        """Set up the primary pattern engine (async if available, otherwise sync)."""
        if self.async_pipeline:
            self.pattern_engine = self.async_pipeline
        else:
            self.pattern_engine = self.sync_pattern_engine
    
    # Getter methods
    
    def get_pattern_engine(self) -> Optional[Any]:
        """Get the primary pattern engine (async or sync)."""
        return self.pattern_engine
    
    def get_sync_pattern_engine(self) -> Optional[Any]:
        """Get the synchronous pattern engine."""
        return self.sync_pattern_engine
    
    def get_async_pipeline(self) -> Optional[Any]:
        """Get the async pattern detection pipeline."""
        return self.async_pipeline
    
    def get_text_pipeline(self) -> Optional[Any]:
        """Get the text processing pipeline."""
        return self.text_pipeline
    
    def get_response_manager(self) -> Optional[Any]:
        """Get the pattern response manager."""
        return self.response_manager
    
    def get_ai_consultation_manager(self) -> Optional[Any]:
        """Get the AI consultation manager."""
        return self.ai_consultation_manager
    
    def get_pattern_config(self) -> Dict[str, Any]:
        """Get the pattern detection configuration."""
        return self.pattern_config
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all pattern detection components."""
        return {
            "available": bool(self.pattern_engine),
            "components": {
                "sync_engine": {"available": bool(self.sync_pattern_engine)},
                "async_pipeline": {"available": bool(self.async_pipeline)},
                "text_pipeline": {"available": bool(self.text_pipeline)},
                "response_manager": {"available": bool(self.response_manager)},
                "ai_consultation_manager": {"available": bool(self.ai_consultation_manager)}
            },
            "imports": self.imports_available
        }
    
    def is_pattern_detection_enabled(self) -> bool:
        """Check if pattern detection is enabled and available."""
        return bool(self.pattern_engine) and self.pattern_config.get('enabled', True)
    
    def update_ai_callers(self, ai_callers: Dict[str, Callable]):
        """Update AI callers for response and consultation managers."""
        self.ai_callers = ai_callers
        
        if self.response_manager:
            self.response_manager.ai_callers = ai_callers
        
        if self.ai_consultation_manager:
            self.ai_consultation_manager.ai_clients = ai_callers
    
    def shutdown(self):
        """Shutdown all components gracefully."""
        components = [
            self.async_pipeline,
            self.text_pipeline,
            self.response_manager,
            self.ai_consultation_manager
        ]
        
        for component in components:
            if component and hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    print(f"Error during component cleanup: {e}", file=sys.stderr)


# Module-level functions for convenience
_default_manager = None


def get_pattern_engine() -> Optional[Any]:
    """Get the default pattern engine."""
    global _default_manager
    if _default_manager:
        return _default_manager.get_pattern_engine()
    return None


def get_response_manager() -> Optional[Any]:
    """Get the default response manager."""
    global _default_manager
    if _default_manager:
        return _default_manager.get_response_manager()
    return None


def get_ai_consultation_manager() -> Optional[Any]:
    """Get the default AI consultation manager."""
    global _default_manager
    if _default_manager:
        return _default_manager.get_ai_consultation_manager()
    return None


def get_async_pipeline() -> Optional[Any]:
    """Get the default async pipeline."""
    global _default_manager
    if _default_manager:
        return _default_manager.get_async_pipeline()
    return None
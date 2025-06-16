#!/usr/bin/env python3
"""
Async Cached Pattern Detection Engine for Junior AI Assistant
Integrates async caching with pattern detection for high performance
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging

# Import pattern detection and caching components
try:
    from pattern_detection import (
        PatternDetectionEngine, 
        EnhancedPatternDetectionEngine,
        PatternMatch, 
        PatternCategory, 
        PatternSeverity,
        SensitivitySettings,
        PatternDetectionProtocol
    )
    from async_pattern_cache import AsyncPatternCache, get_global_cache
    PATTERN_DETECTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Pattern detection components not available: {e}")
    PATTERN_DETECTION_AVAILABLE = False


@dataclass
class AsyncPatternResult:
    """Result of async pattern detection with cache metadata"""
    patterns: List[PatternMatch]
    summary: Dict[str, Any]
    strategy: Dict[str, Any]
    processing_time: float
    was_cached: bool
    cache_age: Optional[float] = None
    was_deduplicated: bool = False
    sensitivity_level: str = "medium"
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AsyncCachedPatternEngine:
    """Async cached wrapper for pattern detection engines"""
    
    def __init__(self,
                 pattern_engine: Optional[PatternDetectionProtocol] = None,
                 cache: Optional[AsyncPatternCache] = None,
                 enable_cache: bool = True,
                 enable_deduplication: bool = True,
                 cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize async cached pattern engine
        
        Args:
            pattern_engine: Underlying pattern detection engine
            cache: Cache instance (uses global cache if None)
            enable_cache: Whether to enable caching
            enable_deduplication: Whether to enable request deduplication
            cache_config: Configuration for cache if creating new instance
        """
        if not PATTERN_DETECTION_AVAILABLE:
            raise ImportError("Pattern detection components not available")
        
        # Initialize pattern engine
        if pattern_engine is None:
            try:
                # Try to use enhanced engine first
                self.pattern_engine = EnhancedPatternDetectionEngine()
            except Exception:
                # Fallback to basic engine
                self.pattern_engine = PatternDetectionEngine()
        else:
            self.pattern_engine = pattern_engine
        
        # Initialize cache
        self.enable_cache = enable_cache
        self.enable_deduplication = enable_deduplication
        
        if enable_cache:
            if cache is None:
                if cache_config:
                    self.cache = AsyncPatternCache(**cache_config)
                else:
                    self.cache = get_global_cache()
            else:
                self.cache = cache
        else:
            self.cache = None
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_saves": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Lock for stats updates
        self._stats_lock = asyncio.Lock()
    
    async def detect_patterns_async(self, 
                                  text: str,
                                  sensitivity_level: Optional[str] = None) -> AsyncPatternResult:
        """
        Async pattern detection with caching and deduplication
        
        Args:
            text: Text to analyze
            sensitivity_level: Sensitivity level for detection
            
        Returns:
            AsyncPatternResult with patterns and metadata
        """
        start_time = time.time()
        
        # Get current sensitivity level
        if sensitivity_level is None:
            try:
                sensitivity_info = self.pattern_engine.get_sensitivity_info()
                sensitivity_level = sensitivity_info.get("global_level", "medium")
            except Exception:
                sensitivity_level = "medium"
        
        # Update request stats
        async with self._stats_lock:
            self.stats["total_requests"] += 1
        
        # Try cache first if enabled
        if self.enable_cache and self.cache:
            if self.enable_deduplication:
                # Use deduplication
                result, was_deduplicated = await self.cache.get_with_deduplication(
                    text,
                    sensitivity_level,
                    lambda: self._execute_pattern_detection(text, sensitivity_level)
                )
                
                if was_deduplicated:
                    async with self._stats_lock:
                        self.stats["deduplication_saves"] += 1
                    
                    processing_time = time.time() - start_time
                    return AsyncPatternResult(
                        patterns=result["patterns"],
                        summary=result["summary"],
                        strategy=result["strategy"],
                        processing_time=processing_time,
                        was_cached=result.get("was_cached", False),
                        cache_age=result.get("cache_age"),
                        was_deduplicated=True,
                        sensitivity_level=sensitivity_level
                    )
            else:
                # Check cache without deduplication
                cached_result = await self.cache.get(text, sensitivity_level)
                if cached_result:
                    async with self._stats_lock:
                        self.stats["cache_hits"] += 1
                    
                    processing_time = time.time() - start_time
                    patterns = await self._deserialize_patterns(cached_result["patterns"])
                    
                    return AsyncPatternResult(
                        patterns=patterns,
                        summary=cached_result["summary"],
                        strategy=cached_result["strategy"],
                        processing_time=processing_time,
                        was_cached=True,
                        cache_age=cached_result["cache_age"],
                        was_deduplicated=False,
                        sensitivity_level=sensitivity_level
                    )
        
        # Cache miss - execute pattern detection
        result = await self._execute_pattern_detection(text, sensitivity_level)
        
        processing_time = time.time() - start_time
        
        # Update stats
        async with self._stats_lock:
            if self.enable_cache:
                self.stats["cache_misses"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_requests"]
            )
        
        return AsyncPatternResult(
            patterns=result["patterns"],
            summary=result["summary"],
            strategy=result["strategy"],
            processing_time=processing_time,
            was_cached=False,
            was_deduplicated=False,
            sensitivity_level=sensitivity_level
        )
    
    async def _execute_pattern_detection(self, 
                                       text: str, 
                                       sensitivity_level: str) -> Dict[str, Any]:
        """
        Execute actual pattern detection and cache results
        
        Args:
            text: Text to analyze
            sensitivity_level: Current sensitivity level
            
        Returns:
            Dictionary with patterns, summary, and strategy
        """
        # Run pattern detection in executor to avoid blocking
        loop = asyncio.get_event_loop()
        patterns = await loop.run_in_executor(
            None, 
            self.pattern_engine.detect_patterns, 
            text
        )
        
        # Get summary and strategy
        summary = await loop.run_in_executor(
            None,
            self.pattern_engine.get_pattern_summary,
            patterns
        )
        
        strategy = await loop.run_in_executor(
            None,
            self.pattern_engine.get_consultation_strategy,
            patterns
        )
        
        # Cache the results if caching is enabled
        if self.enable_cache and self.cache:
            await self.cache.set(
                text=text,
                patterns=patterns,
                summary=summary,
                strategy=strategy,
                sensitivity_level=sensitivity_level
            )
        
        return {
            "patterns": patterns,
            "summary": summary,
            "strategy": strategy,
            "was_cached": False
        }
    
    async def _deserialize_patterns(self, serialized_patterns: List[Dict[str, Any]]) -> List[PatternMatch]:
        """
        Deserialize cached pattern data back to PatternMatch objects
        
        Args:
            serialized_patterns: List of serialized pattern dictionaries
            
        Returns:
            List of PatternMatch objects
        """
        patterns = []
        
        for pattern_data in serialized_patterns:
            try:
                pattern = PatternMatch(
                    category=PatternCategory(pattern_data["category"]),
                    severity=PatternSeverity(pattern_data["severity"]),
                    keyword=pattern_data["keyword"],
                    context=pattern_data["context"],
                    start_pos=pattern_data["start_pos"],
                    end_pos=pattern_data["end_pos"],
                    confidence=pattern_data["confidence"],
                    requires_multi_ai=pattern_data["requires_multi_ai"],
                    line_number=pattern_data.get("line_number"),
                    full_line=pattern_data.get("full_line")
                )
                patterns.append(pattern)
            except Exception as e:
                logging.warning(f"Failed to deserialize pattern: {e}")
                continue
        
        return patterns
    
    async def should_trigger_consultation_async(self, 
                                              patterns: List[PatternMatch],
                                              threshold: Optional[Any] = None) -> bool:
        """
        Async version of consultation trigger check
        
        Args:
            patterns: Detected patterns
            threshold: Optional threshold override
            
        Returns:
            True if consultation should be triggered
        """
        loop = asyncio.get_event_loop()
        
        if threshold is not None:
            return await loop.run_in_executor(
                None,
                self.pattern_engine.should_trigger_consultation,
                patterns,
                threshold
            )
        else:
            return await loop.run_in_executor(
                None,
                self.pattern_engine.should_trigger_consultation,
                patterns
            )
    
    async def get_pattern_summary_async(self, patterns: List[PatternMatch]) -> Dict[str, Any]:
        """
        Async version of pattern summary generation
        
        Args:
            patterns: Detected patterns
            
        Returns:
            Pattern summary dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.pattern_engine.get_pattern_summary,
            patterns
        )
    
    async def get_consultation_strategy_async(self, patterns: List[PatternMatch]) -> Dict[str, Any]:
        """
        Async version of consultation strategy generation
        
        Args:
            patterns: Detected patterns
            
        Returns:
            Consultation strategy dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.pattern_engine.get_consultation_strategy,
            patterns
        )
    
    def get_sensitivity_info(self) -> Dict[str, Any]:
        """Get current sensitivity configuration"""
        try:
            return self.pattern_engine.get_sensitivity_info()
        except Exception as e:
            logging.warning(f"Failed to get sensitivity info: {e}")
            return {
                "global_level": "medium",
                "confidence_threshold": 0.7,
                "context_multiplier": 1.0,
                "min_matches_for_consultation": 2,
                "severity_threshold": "medium",
                "category_overrides": {}
            }
    
    def update_sensitivity(self, 
                          global_level: Optional[str] = None,
                          category_overrides: Optional[Dict[str, str]] = None) -> bool:
        """Update sensitivity configuration"""
        try:
            return self.pattern_engine.update_sensitivity(global_level, category_overrides)
        except Exception as e:
            logging.warning(f"Failed to update sensitivity: {e}")
            return False
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        async with self._stats_lock:
            stats = dict(self.stats)
        
        # Add cache stats if available
        if self.cache:
            cache_metrics = self.cache.get_metrics()
            stats.update({
                "cache_metrics": cache_metrics,
                "cache_enabled": True
            })
        else:
            stats["cache_enabled"] = False
        
        # Calculate rates
        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["cache_hit_rate"] = (stats["cache_hits"] / total_requests) * 100
            stats["deduplication_rate"] = (stats["deduplication_saves"] / total_requests) * 100
        else:
            stats["cache_hit_rate"] = 0.0
            stats["deduplication_rate"] = 0.0
        
        return stats
    
    async def clear_cache(self):
        """Clear the cache"""
        if self.cache:
            await self.cache.clear()
    
    async def close(self):
        """Close and cleanup resources"""
        if self.cache and hasattr(self.cache, 'close'):
            await self.cache.close()


class AsyncPatternDetectionPipeline:
    """High-level async pipeline for pattern detection with caching"""
    
    def __init__(self,
                 engine: Optional[AsyncCachedPatternEngine] = None,
                 batch_size: int = 10,
                 max_concurrent: int = 5):
        """
        Initialize async pattern detection pipeline
        
        Args:
            engine: Async cached pattern engine
            batch_size: Maximum batch size for processing
            max_concurrent: Maximum concurrent operations
        """
        self.engine = engine or AsyncCachedPatternEngine()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Pipeline statistics
        self.pipeline_stats = {
            "texts_processed": 0,
            "batches_processed": 0,
            "total_patterns_found": 0,
            "avg_batch_time": 0.0
        }
    
    async def process_text(self, 
                          text: str,
                          sensitivity_level: Optional[str] = None) -> AsyncPatternResult:
        """
        Process a single text through the pipeline
        
        Args:
            text: Text to analyze
            sensitivity_level: Sensitivity level for detection
            
        Returns:
            AsyncPatternResult
        """
        async with self.semaphore:
            result = await self.engine.detect_patterns_async(text, sensitivity_level)
            
            # Update pipeline stats
            self.pipeline_stats["texts_processed"] += 1
            self.pipeline_stats["total_patterns_found"] += len(result.patterns)
            
            return result
    
    async def process_batch(self, 
                           texts: List[str],
                           sensitivity_level: Optional[str] = None) -> List[AsyncPatternResult]:
        """
        Process a batch of texts concurrently
        
        Args:
            texts: List of texts to analyze
            sensitivity_level: Sensitivity level for detection
            
        Returns:
            List of AsyncPatternResult objects
        """
        start_time = time.time()
        
        # Process texts concurrently
        tasks = [
            self.process_text(text, sensitivity_level)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Error processing text {i}: {result}")
                # Create error result
                error_result = AsyncPatternResult(
                    patterns=[],
                    summary={"error": str(result)},
                    strategy={},
                    processing_time=0.0,
                    was_cached=False,
                    sensitivity_level=sensitivity_level or "medium"
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Update batch stats
        batch_time = time.time() - start_time
        self.pipeline_stats["batches_processed"] += 1
        
        # Update average batch time
        total_batches = self.pipeline_stats["batches_processed"]
        current_avg = self.pipeline_stats["avg_batch_time"]
        self.pipeline_stats["avg_batch_time"] = (
            (current_avg * (total_batches - 1) + batch_time) / total_batches
        )
        
        return processed_results
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        engine_stats = await self.engine.get_performance_stats()
        
        combined_stats = {
            "pipeline": dict(self.pipeline_stats),
            "engine": engine_stats
        }
        
        return combined_stats
    
    async def close(self):
        """Close pipeline and cleanup resources"""
        await self.engine.close()


# Factory functions for easy creation
def create_async_cached_engine(cache_config: Optional[Dict[str, Any]] = None) -> AsyncCachedPatternEngine:
    """
    Create an async cached pattern engine with optional configuration
    
    Args:
        cache_config: Optional cache configuration
        
    Returns:
        AsyncCachedPatternEngine instance
    """
    if not PATTERN_DETECTION_AVAILABLE:
        raise ImportError("Pattern detection components not available")
    
    return AsyncCachedPatternEngine(cache_config=cache_config)

def create_async_pipeline(cache_config: Optional[Dict[str, Any]] = None,
                         batch_size: int = 10,
                         max_concurrent: int = 5) -> AsyncPatternDetectionPipeline:
    """
    Create an async pattern detection pipeline
    
    Args:
        cache_config: Optional cache configuration
        batch_size: Maximum batch size
        max_concurrent: Maximum concurrent operations
        
    Returns:
        AsyncPatternDetectionPipeline instance
    """
    engine = create_async_cached_engine(cache_config)
    return AsyncPatternDetectionPipeline(engine, batch_size, max_concurrent)


if __name__ == "__main__":
    # Test the async cached engine
    async def test_async_engine():
        print("Testing Async Cached Pattern Engine")
        print("=" * 50)
        
        try:
            # Create engine with test cache config
            cache_config = {
                "max_size": 100,
                "max_memory_mb": 10,
                "base_ttl_seconds": 300
            }
            
            engine = create_async_cached_engine(cache_config)
            
            # Test texts
            test_texts = [
                "TODO: Implement password hashing",
                "Fix the O(n^2) algorithm performance", 
                "Handle timezone conversion properly",
                "Should I use microservices architecture?",
                "Encrypt the API key before storage"
            ]
            
            print("\nTesting async pattern detection:")
            for i, text in enumerate(test_texts):
                result = await engine.detect_patterns_async(text)
                
                print(f"\nText {i}: {text[:30]}...")
                print(f"  Patterns found: {len(result.patterns)}")
                print(f"  Processing time: {result.processing_time:.3f}s")
                print(f"  Was cached: {result.was_cached}")
                print(f"  Was deduplicated: {result.was_deduplicated}")
            
            # Test caching by repeating first text
            print("\nTesting cache hit:")
            result = await engine.detect_patterns_async(test_texts[0])
            print(f"  Was cached: {result.was_cached}")
            print(f"  Cache age: {result.cache_age:.3f}s" if result.cache_age else "  No cache age")
            
            # Test batch processing
            print("\nTesting batch processing:")
            pipeline = AsyncPatternDetectionPipeline(engine)
            
            batch_results = await pipeline.process_batch(test_texts[:3])
            print(f"  Processed {len(batch_results)} texts in batch")
            
            # Show performance stats
            print("\nPerformance Statistics:")
            stats = await engine.get_performance_stats()
            for key, value in stats.items():
                if key != "cache_metrics":
                    print(f"  {key}: {value}")
            
            if "cache_metrics" in stats:
                print("\nCache Metrics:")
                cache_metrics = stats["cache_metrics"]
                for key, value in cache_metrics.items():
                    print(f"  {key}: {value}")
            
            await engine.close()
            print("\n✅ Async engine test completed successfully!")
            
        except ImportError as e:
            print(f"❌ Cannot run test: {e}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_async_engine())
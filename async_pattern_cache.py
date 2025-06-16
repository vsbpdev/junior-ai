#!/usr/bin/env python3
"""
Async Pattern Detection Cache for Junior AI Assistant
High-performance async caching with deduplication and dynamic TTL
"""

import asyncio
import hashlib
import time
import json
import weakref
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import Future
from enum import Enum

# Import pattern detection types
try:
    from pattern_detection import PatternMatch, PatternCategory, PatternSeverity, SensitivitySettings
    PATTERN_TYPES_AVAILABLE = True
except ImportError:
    PATTERN_TYPES_AVAILABLE = False


class CacheStrategy(Enum):
    """Cache strategies for different pattern types"""
    AGGRESSIVE = "aggressive"  # Long TTL, high priority
    BALANCED = "balanced"     # Medium TTL, normal priority
    CONSERVATIVE = "conservative"  # Short TTL, low priority
    SECURITY_FOCUSED = "security_focused"  # Very short TTL for security patterns


@dataclass
class AsyncCacheEntry:
    """Async cache entry with enhanced metadata"""
    key: str
    text_hash: str
    patterns: List[Dict[str, Any]]
    summary: Dict[str, Any]
    strategy: Dict[str, Any]
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    pattern_categories: Set[str] = field(default_factory=set)
    sensitivity_level: str = "medium"
    cache_strategy: CacheStrategy = CacheStrategy.BALANCED
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate entry size after initialization"""
        self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate memory size of this entry"""
        try:
            # Rough estimation of memory usage
            size = len(self.key) * 2  # Unicode chars
            size += len(self.text_hash) * 2
            size += len(json.dumps(self.patterns)) * 2
            size += len(json.dumps(self.summary)) * 2
            size += len(json.dumps(self.strategy)) * 2
            size += len(self.pattern_categories) * 20  # Set overhead
            size += 200  # Base object overhead
            return size
        except Exception:
            return 1000  # Default estimate
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.timestamp > self.ttl
    
    def refresh_access(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


class PendingRequest:
    """Manages a pending async request for deduplication"""
    
    def __init__(self, key: str):
        self.key = key
        self.future: asyncio.Future = asyncio.Future()
        self.waiters: List[asyncio.Future] = []
        self.created_at = time.time()
    
    def add_waiter(self) -> asyncio.Future:
        """Add a waiter for this request"""
        waiter = asyncio.Future()
        self.waiters.append(waiter)
        return waiter
    
    def set_result(self, result: Any):
        """Set result for all waiters"""
        if not self.future.done():
            self.future.set_result(result)
        
        for waiter in self.waiters:
            if not waiter.done():
                waiter.set_result(result)
    
    def set_exception(self, exc: Exception):
        """Set exception for all waiters"""
        if not self.future.done():
            self.future.set_exception(exc)
        
        for waiter in self.waiters:
            if not waiter.done():
                waiter.set_exception(exc)


@dataclass
class CacheMetrics:
    """Comprehensive cache metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    deduplication_saves: int = 0
    ttl_expirations: int = 0
    pattern_category_stats: Dict[str, int] = field(default_factory=dict)
    sensitivity_level_stats: Dict[str, int] = field(default_factory=dict)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def deduplication_rate(self) -> float:
        """Calculate deduplication efficiency"""
        total = self.hits + self.misses + self.deduplication_saves
        return (self.deduplication_saves / total * 100) if total > 0 else 0.0


class AsyncPatternCache:
    """High-performance async pattern detection cache"""
    
    def __init__(self,
                 max_size: int = 2000,
                 max_memory_mb: int = 100,
                 base_ttl_seconds: int = 300,
                 enable_deduplication: bool = True,
                 cleanup_interval: int = 60):
        """
        Initialize async pattern cache
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            base_ttl_seconds: Base TTL for cache entries
            enable_deduplication: Enable request deduplication
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.base_ttl = base_ttl_seconds
        self.enable_deduplication = enable_deduplication
        self.cleanup_interval = cleanup_interval
        
        # Cache storage (ordered for LRU)
        self.cache: OrderedDict[str, AsyncCacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        
        # Deduplication tracking
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.dedup_lock = asyncio.Lock()
        
        # Metrics and monitoring
        self.metrics = CacheMetrics()
        
        # TTL multipliers for different sensitivity levels
        self.ttl_multipliers = {
            "low": 2.0,      # Cache longer for low sensitivity
            "medium": 1.0,   # Base TTL
            "high": 0.5,     # Cache shorter for high sensitivity
            "maximum": 0.2   # Very short cache for maximum sensitivity
        }
        
        # Pattern category TTL adjustments
        self.category_ttl_multipliers = {
            "security": 0.3,      # Security patterns expire quickly
            "uncertainty": 0.5,   # Uncertainty patterns short-lived
            "algorithm": 1.5,     # Algorithm advice cached longer
            "architecture": 1.2,  # Architecture patterns moderately cached
            "gotcha": 1.0        # Standard caching for gotchas
        }
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        # Weak reference tracking for automatic cleanup
        self._instances = weakref.WeakSet()
        self._instances.add(self)
    
    async def get(self, text: str, sensitivity_level: str = "medium") -> Optional[Dict[str, Any]]:
        """
        Get cached pattern detection results
        
        Args:
            text: The text to look up
            sensitivity_level: Current sensitivity level
            
        Returns:
            Cached results or None if not found/expired
        """
        cache_key = self._generate_cache_key(text)
        
        async with self.lock:
            if cache_key not in self.cache:
                self.metrics.misses += 1
                return None
            
            entry = self.cache[cache_key]
            
            # Check expiration
            if entry.is_expired():
                await self._remove_entry(cache_key)
                self.metrics.misses += 1
                self.metrics.ttl_expirations += 1
                return None
            
            # Update access stats
            entry.refresh_access()
            
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            
            self.metrics.hits += 1
            
            # Update pattern category stats
            for category in entry.pattern_categories:
                self.metrics.pattern_category_stats[category] = \
                    self.metrics.pattern_category_stats.get(category, 0) + 1
            
            return {
                "patterns": entry.patterns,
                "summary": entry.summary,
                "strategy": entry.strategy,
                "cached": True,
                "cache_age": time.time() - entry.timestamp,
                "access_count": entry.access_count,
                "sensitivity_level": entry.sensitivity_level
            }
    
    async def set(self,
                  text: str,
                  patterns: List[Any],
                  summary: Dict[str, Any],
                  strategy: Dict[str, Any],
                  sensitivity_level: str = "medium") -> None:
        """
        Cache pattern detection results
        
        Args:
            text: The analyzed text
            patterns: Detected patterns
            summary: Pattern summary
            strategy: Consultation strategy
            sensitivity_level: Current sensitivity level
        """
        cache_key = self._generate_cache_key(text)
        text_hash = self._hash_text(text)
        
        # Serialize patterns
        serialized_patterns = await self._serialize_patterns(patterns)
        
        # Determine pattern categories for TTL calculation
        pattern_categories = set()
        if PATTERN_TYPES_AVAILABLE and patterns:
            for pattern in patterns:
                if hasattr(pattern, 'category'):
                    pattern_categories.add(pattern.category.value)
        
        # Calculate dynamic TTL
        ttl = self._calculate_dynamic_ttl(sensitivity_level, pattern_categories)
        
        # Determine cache strategy
        cache_strategy = self._determine_cache_strategy(pattern_categories, sensitivity_level)
        
        # Create cache entry
        entry = AsyncCacheEntry(
            key=cache_key,
            text_hash=text_hash,
            patterns=serialized_patterns,
            summary=summary,
            strategy=strategy,
            timestamp=time.time(),
            ttl=ttl,
            pattern_categories=pattern_categories,
            sensitivity_level=sensitivity_level,
            cache_strategy=cache_strategy
        )
        
        async with self.lock:
            # Ensure we have space
            await self._ensure_space_for_entry(entry)
            
            # Add to cache
            self.cache[cache_key] = entry
            
            # Update metrics
            self.metrics.memory_usage_bytes += entry.size_bytes
            self.metrics.sensitivity_level_stats[sensitivity_level] = \
                self.metrics.sensitivity_level_stats.get(sensitivity_level, 0) + 1
    
    async def get_with_deduplication(self,
                                   text: str,
                                   sensitivity_level: str = "medium",
                                   executor_func=None) -> Tuple[Dict[str, Any], bool]:
        """
        Get cached results with request deduplication
        
        Args:
            text: Text to analyze
            sensitivity_level: Current sensitivity level
            executor_func: Function to execute if cache miss
            
        Returns:
            Tuple of (result, was_deduplicated)
        """
        if not self.enable_deduplication or not executor_func:
            result = await self.get(text, sensitivity_level)
            return result, False
        
        cache_key = self._generate_cache_key(text)
        
        # Check cache first
        cached_result = await self.get(text, sensitivity_level)
        if cached_result:
            return cached_result, False
        
        async with self.dedup_lock:
            # Check if request is already pending
            if cache_key in self.pending_requests:
                pending = self.pending_requests[cache_key]
                waiter = pending.add_waiter()
                
                # Release lock before waiting to avoid deadlock
        
        # If we found an existing pending request, wait for it
        if 'waiter' in locals():
            try:
                result = await waiter
                self.metrics.deduplication_saves += 1
                return result, True
            except Exception as e:
                # Original request failed - the finally block of the original request will clean up
                raise e
        
        # No existing request, create new pending request
        async with self.dedup_lock:
            # Double-check after re-acquiring lock (race condition protection)
            if cache_key in self.pending_requests:
                pending = self.pending_requests[cache_key]
                waiter = pending.add_waiter()
                
        # Handle the race condition case
        if 'waiter' in locals():
            try:
                result = await waiter
                self.metrics.deduplication_saves += 1
                return result, True
            except Exception as e:
                raise e
        
        # Create new pending request (final attempt)
        async with self.dedup_lock:
            pending = PendingRequest(cache_key)
            self.pending_requests[cache_key] = pending
        
        try:
            # Execute the function
            result = await executor_func()
            
            # Set result for all waiters
            pending.set_result(result)
            
            return result, False
            
        except Exception as e:
            # Set exception for all waiters
            pending.set_exception(e)
            raise
        finally:
            # Clean up pending request
            async with self.dedup_lock:
                if cache_key in self.pending_requests:
                    del self.pending_requests[cache_key]
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key with SHA256 hash"""
        # Use first 1000 chars for key generation to balance uniqueness and performance
        text_sample = text[:1000] if len(text) > 1000 else text
        return hashlib.sha256(text_sample.encode('utf-8')).hexdigest()
    
    def _hash_text(self, text: str) -> str:
        """Generate full text hash"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _calculate_dynamic_ttl(self, sensitivity_level: str, pattern_categories: Set[str]) -> float:
        """Calculate dynamic TTL based on sensitivity and pattern types"""
        base_ttl = self.base_ttl
        
        # Apply sensitivity multiplier
        if sensitivity_level in self.ttl_multipliers:
            base_ttl *= self.ttl_multipliers[sensitivity_level]
        
        # Apply most restrictive category multiplier
        if pattern_categories:
            category_multipliers = [
                self.category_ttl_multipliers.get(cat, 1.0) 
                for cat in pattern_categories
            ]
            # Use the most restrictive (smallest) multiplier
            base_ttl *= min(category_multipliers)
        
        return base_ttl
    
    def _determine_cache_strategy(self, pattern_categories: Set[str], sensitivity_level: str) -> CacheStrategy:
        """Determine cache strategy based on patterns and sensitivity"""
        # Security patterns get special treatment
        if "security" in pattern_categories:
            return CacheStrategy.SECURITY_FOCUSED
        
        # High sensitivity gets conservative caching
        if sensitivity_level in ["high", "maximum"]:
            return CacheStrategy.CONSERVATIVE
        
        # Algorithm patterns can be cached aggressively
        if "algorithm" in pattern_categories and sensitivity_level == "low":
            return CacheStrategy.AGGRESSIVE
        
        return CacheStrategy.BALANCED
    
    async def _serialize_patterns(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Serialize pattern objects to dictionaries"""
        if not PATTERN_TYPES_AVAILABLE or not patterns:
            return []
        
        serialized = []
        for pattern in patterns:
            if hasattr(pattern, 'category'):
                serialized.append({
                    "category": pattern.category.value,
                    "severity": pattern.severity.value,
                    "keyword": pattern.keyword,
                    "context": pattern.context,
                    "start_pos": pattern.start_pos,
                    "end_pos": pattern.end_pos,
                    "confidence": pattern.confidence,
                    "requires_multi_ai": pattern.requires_multi_ai,
                    "line_number": getattr(pattern, 'line_number', None),
                    "full_line": getattr(pattern, 'full_line', None)
                })
        
        return serialized
    
    async def _ensure_space_for_entry(self, new_entry: AsyncCacheEntry):
        """Ensure there's space for a new entry"""
        # Check size limit
        while len(self.cache) >= self.max_size:
            await self._evict_lru()
        
        # Check memory limit
        while (self.metrics.memory_usage_bytes + new_entry.size_bytes) > self.max_memory_bytes:
            if not self.cache:
                break  # Cache is empty, can't evict more
            await self._evict_lru()
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Get LRU entry (first in OrderedDict)
        lru_key, lru_entry = next(iter(self.cache.items()))
        await self._remove_entry(lru_key)
        self.metrics.evictions += 1
    
    async def _remove_entry(self, key: str):
        """Remove entry and update metrics"""
        if key in self.cache:
            entry = self.cache[key]
            self.metrics.memory_usage_bytes -= entry.size_bytes
            del self.cache[key]
    
    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
            self.metrics = CacheMetrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_mb": self.metrics.memory_usage_bytes / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": f"{self.metrics.hit_rate():.1f}%",
            "evictions": self.metrics.evictions,
            "ttl_expirations": self.metrics.ttl_expirations,
            "deduplication_saves": self.metrics.deduplication_saves,
            "deduplication_rate": f"{self.metrics.deduplication_rate():.1f}%",
            "pattern_category_stats": dict(self.metrics.pattern_category_stats),
            "sensitivity_level_stats": dict(self.metrics.sensitivity_level_stats),
            "pending_requests": len(self.pending_requests),
            "base_ttl_seconds": self.base_ttl
        }
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_entries()
                await self._cleanup_stale_pending_requests()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Remove expired cache entries"""
        async with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._remove_entry(key)
                self.metrics.ttl_expirations += 1
    
    async def _cleanup_stale_pending_requests(self):
        """Remove stale pending requests"""
        async with self.dedup_lock:
            current_time = time.time()
            stale_keys = []
            
            for key, pending in self.pending_requests.items():
                # Remove requests older than 30 seconds
                if current_time - pending.created_at > 30:
                    stale_keys.append(key)
            
            for key in stale_keys:
                pending = self.pending_requests[key]
                if not pending.future.done():
                    pending.set_exception(TimeoutError("Request timed out"))
                del self.pending_requests[key]
    
    async def close(self):
        """Clean up resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear()


# Global cache instance
_global_cache: Optional[AsyncPatternCache] = None

def get_global_cache() -> AsyncPatternCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = AsyncPatternCache()
    return _global_cache

async def clear_global_cache():
    """Clear global cache"""
    global _global_cache
    if _global_cache:
        await _global_cache.clear()

async def get_global_cache_metrics() -> Dict[str, Any]:
    """Get global cache metrics"""
    cache = get_global_cache()
    return cache.get_metrics()


if __name__ == "__main__":
    # Test the async cache
    async def test_async_cache():
        print("Testing Async Pattern Detection Cache")
        print("=" * 60)
        
        # Create cache with test settings
        cache = AsyncPatternCache(
            max_size=5,
            max_memory_mb=1,
            base_ttl_seconds=60,
            enable_deduplication=True
        )
        
        # Test data
        test_texts = [
            "TODO: Implement password hashing",
            "Fix the O(n^2) algorithm performance",
            "Handle timezone conversion properly",
            "Should I use microservices architecture?",
            "Encrypt the API key before storage"
        ]
        
        print("\nTesting cache operations:")
        for i, text in enumerate(test_texts):
            # Mock pattern data
            mock_patterns = []
            mock_summary = {"total_matches": i + 1, "categories": {}}
            mock_strategy = {"strategy": "single_ai", "reason": f"Test {i}"}
            
            # First access (miss)
            result = await cache.get(text)
            print(f"\nText {i}: First access - {'HIT' if result else 'MISS'}")
            
            # Store in cache
            await cache.set(text, mock_patterns, mock_summary, mock_strategy)
            
            # Second access (hit)
            result = await cache.get(text)
            print(f"Text {i}: Second access - {'HIT' if result else 'MISS'}")
            if result:
                print(f"  Cache age: {result['cache_age']:.2f}s")
        
        # Test deduplication
        print("\nTesting deduplication:")
        
        async def mock_executor():
            await asyncio.sleep(0.1)  # Simulate processing
            return {"result": "executed", "timestamp": time.time()}
        
        # Start multiple concurrent requests for same text
        tasks = []
        for _ in range(3):
            task = cache.get_with_deduplication(
                "concurrent test text", 
                "medium", 
                mock_executor
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        deduplicated_count = sum(1 for _, was_dedup in results if was_dedup)
        print(f"Deduplication saves: {deduplicated_count}/3")
        
        # Show metrics
        print("\nCache Metrics:")
        metrics = cache.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Cleanup
        await cache.close()
        print("\nCache closed and cleaned up")
    
    # Run the test
    asyncio.run(test_async_cache())
#!/usr/bin/env python3
"""
Unit tests for Async Pattern Detection Cache
Comprehensive test suite for async caching functionality
"""

import asyncio
import pytest
import time
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

# Import the cache system
from async_pattern_cache import (
    AsyncPatternCache, 
    AsyncCacheEntry, 
    PendingRequest,
    CacheStrategy,
    CacheMetrics,
    get_global_cache,
    clear_global_cache,
    get_global_cache_metrics
)

# Mock pattern detection types for testing
class MockPatternCategory:
    def __init__(self, value):
        self.value = value

class MockPatternSeverity:
    def __init__(self, value):
        self.value = value

class MockPatternMatch:
    def __init__(self, category, severity, keyword="test", context="test context"):
        self.category = MockPatternCategory(category)
        self.severity = MockPatternSeverity(severity)
        self.keyword = keyword
        self.context = context
        self.start_pos = 0
        self.end_pos = 10
        self.confidence = 0.8
        self.requires_multi_ai = False
        self.line_number = 1
        self.full_line = "test line"


class TestAsyncCacheEntry:
    """Test AsyncCacheEntry functionality"""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation and size calculation"""
        entry = AsyncCacheEntry(
            key="test_key",
            text_hash="abc123",
            patterns=[],
            summary={},
            strategy={},
            timestamp=time.time(),
            ttl=300
        )
        
        assert entry.key == "test_key"
        assert entry.text_hash == "abc123"
        assert entry.size_bytes > 0
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic"""
        # Create expired entry
        old_timestamp = time.time() - 400  # 400 seconds ago
        entry = AsyncCacheEntry(
            key="test_key",
            text_hash="abc123",
            patterns=[],
            summary={},
            strategy={},
            timestamp=old_timestamp,
            ttl=300  # 5 minutes
        )
        
        assert entry.is_expired()
    
    def test_refresh_access(self):
        """Test access count and timestamp updates"""
        entry = AsyncCacheEntry(
            key="test_key",
            text_hash="abc123",
            patterns=[],
            summary={},
            strategy={},
            timestamp=time.time(),
            ttl=300
        )
        
        initial_count = entry.access_count
        initial_access_time = entry.last_accessed
        
        time.sleep(0.01)  # Small delay
        entry.refresh_access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_access_time


class TestPendingRequest:
    """Test PendingRequest functionality"""
    
    @pytest.mark.asyncio
    async def test_pending_request_creation(self):
        """Test pending request creation"""
        pending = PendingRequest("test_key")
        
        assert pending.key == "test_key"
        assert not pending.future.done()
        assert len(pending.waiters) == 0
    
    @pytest.mark.asyncio
    async def test_add_waiter(self):
        """Test adding waiters to pending request"""
        pending = PendingRequest("test_key")
        
        waiter = pending.add_waiter()
        assert len(pending.waiters) == 1
        assert not waiter.done()
    
    @pytest.mark.asyncio
    async def test_set_result(self):
        """Test setting result for all waiters"""
        pending = PendingRequest("test_key")
        
        waiter1 = pending.add_waiter()
        waiter2 = pending.add_waiter()
        
        result = {"test": "result"}
        pending.set_result(result)
        
        assert pending.future.done()
        assert await pending.future == result
        assert await waiter1 == result
        assert await waiter2 == result
    
    @pytest.mark.asyncio
    async def test_set_exception(self):
        """Test setting exception for all waiters"""
        pending = PendingRequest("test_key")
        
        waiter = pending.add_waiter()
        
        exc = ValueError("test error")
        pending.set_exception(exc)
        
        assert pending.future.done()
        
        with pytest.raises(ValueError):
            await pending.future
        
        with pytest.raises(ValueError):
            await waiter


class TestAsyncPatternCache:
    """Test AsyncPatternCache functionality"""
    
    @pytest.fixture
    async def cache(self):
        """Create a test cache instance"""
        cache = AsyncPatternCache(
            max_size=10,
            max_memory_mb=1,
            base_ttl_seconds=60,
            enable_deduplication=True,
            cleanup_interval=30
        )
        yield cache
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_creation(self, cache):
        """Test cache creation with proper settings"""
        assert cache.max_size == 10
        assert cache.max_memory_bytes == 1024 * 1024
        assert cache.base_ttl == 60
        assert cache.enable_deduplication
        assert len(cache.cache) == 0
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache):
        """Test cache key generation with SHA256"""
        text1 = "This is a test text"
        text2 = "This is a different test text"
        
        key1 = cache._generate_cache_key(text1)
        key2 = cache._generate_cache_key(text2)
        key3 = cache._generate_cache_key(text1)  # Same as text1
        
        assert key1 != key2
        assert key1 == key3
        assert len(key1) == 64  # SHA256 hex length
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache):
        """Test basic get/set operations"""
        text = "test text for caching"
        patterns = [MockPatternMatch("security", "high")]
        summary = {"total": 1}
        strategy = {"method": "single"}
        
        # Cache miss
        result = await cache.get(text)
        assert result is None
        
        # Store in cache
        await cache.set(text, patterns, summary, strategy)
        
        # Cache hit
        result = await cache.get(text)
        assert result is not None
        assert result["cached"] is True
        assert result["summary"] == summary
        assert result["strategy"] == strategy
        assert len(result["patterns"]) == 1
    
    @pytest.mark.asyncio
    async def test_sensitivity_level_ttl(self, cache):
        """Test dynamic TTL based on sensitivity levels"""
        base_text = "test text"
        
        # Test different sensitivity levels
        sensitivity_levels = ["low", "medium", "high", "maximum"]
        ttls = []
        
        for level in sensitivity_levels:
            ttl = cache._calculate_dynamic_ttl(level, set())
            ttls.append(ttl)
        
        # TTL should decrease with higher sensitivity
        assert ttls[0] > ttls[1]  # low > medium
        assert ttls[1] > ttls[2]  # medium > high
        assert ttls[2] > ttls[3]  # high > maximum
    
    @pytest.mark.asyncio
    async def test_pattern_category_ttl(self, cache):
        """Test TTL adjustments based on pattern categories"""
        # Security patterns should have shorter TTL
        security_ttl = cache._calculate_dynamic_ttl("medium", {"security"})
        algorithm_ttl = cache._calculate_dynamic_ttl("medium", {"algorithm"})
        
        assert security_ttl < algorithm_ttl
    
    @pytest.mark.asyncio
    async def test_cache_strategy_determination(self, cache):
        """Test cache strategy selection"""
        # Security patterns get security-focused strategy
        strategy = cache._determine_cache_strategy({"security"}, "medium")
        assert strategy == CacheStrategy.SECURITY_FOCUSED
        
        # High sensitivity gets conservative strategy
        strategy = cache._determine_cache_strategy(set(), "high")
        assert strategy == CacheStrategy.CONSERVATIVE
        
        # Algorithm patterns with low sensitivity get aggressive strategy
        strategy = cache._determine_cache_strategy({"algorithm"}, "low")
        assert strategy == CacheStrategy.AGGRESSIVE
        
        # Default case gets balanced strategy
        strategy = cache._determine_cache_strategy(set(), "medium")
        assert strategy == CacheStrategy.BALANCED
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full"""
        # Fill cache to capacity
        for i in range(cache.max_size):
            text = f"test text {i}"
            await cache.set(text, [], {}, {})
        
        assert len(cache.cache) == cache.max_size
        
        # Add one more entry to trigger eviction
        await cache.set("new text", [], {}, {})
        
        assert len(cache.cache) == cache.max_size
        assert cache.metrics.evictions > 0
        
        # First entry should be evicted
        first_key = cache._generate_cache_key("test text 0")
        assert first_key not in cache.cache
    
    @pytest.mark.asyncio
    async def test_memory_based_eviction(self):
        """Test eviction based on memory limits"""
        # Create cache with very small memory limit
        cache = AsyncPatternCache(
            max_size=100,
            max_memory_mb=0.001,  # Very small limit
            base_ttl_seconds=60
        )
        
        try:
            # Add entries that exceed memory limit
            large_data = {"large": "x" * 1000}  # Large summary
            
            await cache.set("text1", [], large_data, {})
            await cache.set("text2", [], large_data, {})
            
            # Should trigger memory-based eviction
            assert cache.metrics.evictions > 0
            
        finally:
            await cache.close()
    
    @pytest.mark.asyncio
    async def test_deduplication(self, cache):
        """Test request deduplication functionality"""
        call_count = 0
        
        async def mock_executor():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"result": f"executed {call_count}", "timestamp": time.time()}
        
        # Start multiple concurrent requests for same text
        tasks = []
        for _ in range(5):
            task = cache.get_with_deduplication(
                "dedup test text",
                "medium",
                mock_executor
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Should only execute once
        assert call_count == 1
        
        # Check deduplication results
        deduplicated_count = sum(1 for _, was_dedup in results if was_dedup)
        assert deduplicated_count == 4  # 4 out of 5 should be deduplicated
        
        # All results should be the same
        result_data = [result for result, _ in results]
        assert all(r["result"] == result_data[0]["result"] for r in result_data)
    
    @pytest.mark.asyncio
    async def test_expiration_cleanup(self):
        """Test automatic cleanup of expired entries"""
        cache = AsyncPatternCache(
            max_size=10,
            base_ttl_seconds=0.1,  # Very short TTL
            cleanup_interval=0.2
        )
        
        try:
            # Add entries
            await cache.set("text1", [], {}, {})
            await cache.set("text2", [], {}, {})
            
            assert len(cache.cache) == 2
            
            # Wait for expiration
            await asyncio.sleep(0.15)
            
            # Trigger cleanup
            await cache._cleanup_expired_entries()
            
            assert len(cache.cache) == 0
            assert cache.metrics.ttl_expirations > 0
            
        finally:
            await cache.close()
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, cache):
        """Test comprehensive metrics tracking"""
        # Perform various cache operations
        await cache.set("text1", [MockPatternMatch("security", "high")], {}, {}, "high")
        await cache.set("text2", [MockPatternMatch("algorithm", "medium")], {}, {}, "low")
        
        # Cache hits
        await cache.get("text1", "high")
        await cache.get("text1", "high")
        
        # Cache miss
        await cache.get("nonexistent", "medium")
        
        metrics = cache.get_metrics()
        
        assert metrics["cache_size"] == 2
        assert metrics["hits"] == 2
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == "66.7%"
        assert "security" in metrics["pattern_category_stats"]
        assert "algorithm" in metrics["pattern_category_stats"]
        assert "high" in metrics["sensitivity_level_stats"]
        assert "low" in metrics["sensitivity_level_stats"]
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, cache):
        """Test cache clearing"""
        # Add some entries
        await cache.set("text1", [], {}, {})
        await cache.set("text2", [], {}, {})
        
        assert len(cache.cache) == 2
        
        # Clear cache
        await cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.metrics.memory_usage_bytes == 0


class TestGlobalCache:
    """Test global cache functionality"""
    
    @pytest.mark.asyncio
    async def test_global_cache_singleton(self):
        """Test global cache singleton behavior"""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        
        assert cache1 is cache2
    
    @pytest.mark.asyncio
    async def test_global_cache_operations(self):
        """Test global cache operations"""
        cache = get_global_cache()
        
        # Add some data
        await cache.set("global test", [], {"test": "data"}, {})
        
        # Get metrics
        metrics = await get_global_cache_metrics()
        assert metrics["cache_size"] >= 1
        
        # Clear cache
        await clear_global_cache()
        
        metrics = await get_global_cache_metrics()
        assert metrics["cache_size"] == 0


class TestCacheIntegration:
    """Integration tests for cache with mock pattern detection"""
    
    @pytest.mark.asyncio
    async def test_pattern_serialization(self):
        """Test pattern object serialization and deserialization"""
        cache = AsyncPatternCache()
        
        try:
            # Create mock patterns
            patterns = [
                MockPatternMatch("security", "high", "password", "password context"),
                MockPatternMatch("algorithm", "medium", "O(n^2)", "algorithm context")
            ]
            
            # Serialize patterns
            serialized = await cache._serialize_patterns(patterns)
            
            assert len(serialized) == 2
            assert serialized[0]["category"] == "security"
            assert serialized[0]["severity"] == "high"
            assert serialized[0]["keyword"] == "password"
            assert serialized[1]["category"] == "algorithm"
            
        finally:
            await cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_with_pattern_categories(self):
        """Test cache behavior with different pattern categories"""
        cache = AsyncPatternCache()
        
        try:
            # Security patterns
            security_patterns = [MockPatternMatch("security", "critical")]
            await cache.set("security test", security_patterns, {}, {}, "high")
            
            # Algorithm patterns
            algorithm_patterns = [MockPatternMatch("algorithm", "medium")]
            await cache.set("algorithm test", algorithm_patterns, {}, {}, "low")
            
            # Check that entries have correct metadata
            security_key = cache._generate_cache_key("security test")
            algorithm_key = cache._generate_cache_key("algorithm test")
            
            security_entry = cache.cache[security_key]
            algorithm_entry = cache.cache[algorithm_key]
            
            assert "security" in security_entry.pattern_categories
            assert "algorithm" in algorithm_entry.pattern_categories
            assert security_entry.cache_strategy == CacheStrategy.SECURITY_FOCUSED
            assert algorithm_entry.cache_strategy == CacheStrategy.AGGRESSIVE
            
        finally:
            await cache.close()


if __name__ == "__main__":
    # Run tests manually if not using pytest
    import asyncio
    
    async def run_manual_tests():
        """Run basic tests manually"""
        print("Running Async Pattern Cache Tests")
        print("=" * 50)
        
        # Test cache entry
        print("\n1. Testing AsyncCacheEntry...")
        entry = AsyncCacheEntry(
            key="test",
            text_hash="abc123",
            patterns=[],
            summary={},
            strategy={},
            timestamp=time.time(),
            ttl=300
        )
        assert not entry.is_expired()
        print("✓ Cache entry creation and expiration test passed")
        
        # Test basic cache operations
        print("\n2. Testing basic cache operations...")
        cache = AsyncPatternCache(max_size=5, base_ttl_seconds=60)
        
        await cache.set("test text", [], {"test": True}, {"strategy": "test"})
        result = await cache.get("test text")
        
        assert result is not None
        assert result["cached"] is True
        print("✓ Basic cache operations test passed")
        
        # Test deduplication
        print("\n3. Testing deduplication...")
        call_count = 0
        
        async def mock_executor():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {"executed": True}
        
        tasks = [
            cache.get_with_deduplication("dedup test", "medium", mock_executor)
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        assert call_count == 1
        print("✓ Deduplication test passed")
        
        # Test metrics
        print("\n4. Testing metrics...")
        metrics = cache.get_metrics()
        assert metrics["cache_size"] >= 0
        assert "hit_rate" in metrics
        print("✓ Metrics test passed")
        
        await cache.close()
        print("\n✅ All manual tests passed!")
    
    # Run manual tests
    asyncio.run(run_manual_tests())
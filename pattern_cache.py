#!/usr/bin/env python3
"""
Pattern Detection Cache for Junior AI Assistant
Caches pattern detection results to avoid redundant processing
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from threading import Lock
from pathlib import Path


@dataclass
class CacheEntry:
    """Represents a cached pattern detection result"""
    text_hash: str
    patterns: List[Dict[str, Any]]
    summary: Dict[str, Any]
    strategy: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0


class PatternDetectionCache:
    """LRU cache for pattern detection results"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: int = 300,
                 persist_to_disk: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Initialize pattern detection cache
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            persist_to_disk: Whether to persist cache to disk
            cache_dir: Directory for persistent cache (if enabled)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_to_disk = persist_to_disk
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".junior_ai_cache"
        
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Load persistent cache if enabled
        if self.persist_to_disk:
            self._load_from_disk()
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash for the input text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached pattern detection results
        
        Args:
            text: The text to look up
            
        Returns:
            Cached results or None if not found/expired
        """
        text_hash = self._hash_text(text)
        
        with self.lock:
            if text_hash not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[text_hash]
            
            # Check if entry has expired
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self.cache[text_hash]
                self.misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (LRU)
            del self.cache[text_hash]
            self.cache[text_hash] = entry
            
            self.hits += 1
            
            return {
                "patterns": entry.patterns,
                "summary": entry.summary,
                "strategy": entry.strategy,
                "cached": True,
                "cache_age": time.time() - entry.timestamp
            }
    
    def put(self, text: str, patterns: List[Any], summary: Dict[str, Any], strategy: Dict[str, Any]):
        """
        Cache pattern detection results
        
        Args:
            text: The analyzed text
            patterns: Detected patterns (will be serialized)
            summary: Pattern summary
            strategy: Consultation strategy
        """
        text_hash = self._hash_text(text)
        
        # Serialize patterns (PatternMatch objects to dicts)
        serialized_patterns = []
        for pattern in patterns:
            serialized_patterns.append({
                "category": pattern.category.value,
                "severity": pattern.severity.value,
                "keyword": pattern.keyword,
                "context": pattern.context,
                "start_pos": pattern.start_pos,
                "end_pos": pattern.end_pos,
                "confidence": pattern.confidence,
                "requires_multi_ai": pattern.requires_multi_ai,
                "line_number": pattern.line_number,
                "full_line": pattern.full_line
            })
        
        entry = CacheEntry(
            text_hash=text_hash,
            patterns=serialized_patterns,
            summary=summary,
            strategy=strategy,
            timestamp=time.time(),
            last_accessed=time.time()
        )
        
        with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[text_hash] = entry
            
            # Persist to disk if enabled
            if self.persist_to_disk:
                self._save_entry_to_disk(entry)
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Find least recently accessed entry
        lru_hash = min(self.cache.keys(), 
                      key=lambda k: self.cache[k].last_accessed)
        
        del self.cache[lru_hash]
        self.evictions += 1
        
        # Remove from disk if persisting
        if self.persist_to_disk:
            cache_file = self.cache_dir / f"{lru_hash}.json"
            if cache_file.exists():
                cache_file.unlink()
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            
            # Clear disk cache if persisting
            if self.persist_to_disk and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": self.ttl_seconds,
            "persist_enabled": self.persist_to_disk
        }
    
    def _save_entry_to_disk(self, entry: CacheEntry):
        """Save a cache entry to disk"""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self.cache_dir / f"{entry.text_hash}.json"
        
        # Convert entry to dict for JSON serialization
        entry_dict = {
            "text_hash": entry.text_hash,
            "patterns": entry.patterns,
            "summary": entry.summary,
            "strategy": entry.strategy,
            "timestamp": entry.timestamp,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry_dict, f, indent=2)
        except Exception as e:
            # Log error but don't fail
            print(f"Failed to save cache entry to {cache_file}: {e}")
    
    def _load_from_disk(self):
        """Load cache entries from disk"""
        if not self.cache_dir.exists():
            return
        
        loaded = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    entry_dict = json.load(f)
                
                # Check if entry is still valid
                if current_time - entry_dict["timestamp"] > self.ttl_seconds:
                    cache_file.unlink()  # Remove expired entry
                    continue
                
                # Recreate CacheEntry
                entry = CacheEntry(**entry_dict)
                self.cache[entry.text_hash] = entry
                loaded += 1
                
                # Stop if cache is full
                if len(self.cache) >= self.max_size:
                    break
                    
            except Exception as e:
                # Remove corrupted cache file
                cache_file.unlink()
                continue
        
        # Sort by last accessed time (LRU order)
        if self.cache:
            sorted_entries = sorted(self.cache.items(), 
                                  key=lambda x: x[1].last_accessed)
            self.cache = dict(sorted_entries)


class CachedPatternDetectionEngine:
    """Wrapper for pattern detection engine with caching"""
    
    def __init__(self, pattern_engine, cache: Optional[PatternDetectionCache] = None):
        """
        Initialize cached pattern detection engine
        
        Args:
            pattern_engine: The underlying pattern detection engine
            cache: Optional cache instance (creates default if not provided)
        """
        self.pattern_engine = pattern_engine
        self.cache = cache or PatternDetectionCache()
        self.cache_enabled = True
    
    def detect_patterns(self, text: str):
        """Detect patterns with caching"""
        # Check cache first if enabled
        if self.cache_enabled:
            cached_result = self.cache.get(text)
            if cached_result:
                # Reconstruct PatternMatch objects from cached data
                from pattern_detection import PatternMatch, PatternCategory, PatternSeverity
                
                patterns = []
                for pattern_data in cached_result["patterns"]:
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
                
                return patterns
        
        # Not in cache, detect patterns
        patterns = self.pattern_engine.detect_patterns(text)
        
        # Cache the results if enabled
        if self.cache_enabled and patterns:
            summary = self.pattern_engine.get_pattern_summary(patterns)
            strategy = self.pattern_engine.get_consultation_strategy(patterns)
            self.cache.put(text, patterns, summary, strategy)
        
        return patterns
    
    def get_pattern_summary(self, patterns):
        """Get pattern summary (may use cached data)"""
        # If patterns have cache metadata, we might have cached summary
        if hasattr(patterns, '_cached') and patterns._cached:
            # For now, just regenerate
            # In future, could store summary in pattern metadata
            pass
        
        return self.pattern_engine.get_pattern_summary(patterns)
    
    def get_consultation_strategy(self, patterns):
        """Get consultation strategy (may use cached data)"""
        return self.pattern_engine.get_consultation_strategy(patterns)
    
    def should_trigger_consultation(self, patterns, threshold=None):
        """Determine if consultation should be triggered"""
        if threshold:
            return self.pattern_engine.should_trigger_consultation(patterns, threshold)
        return self.pattern_engine.should_trigger_consultation(patterns)
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self.cache_enabled = enabled
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()


if __name__ == "__main__":
    # Test the cache
    print("Testing Pattern Detection Cache")
    print("=" * 60)
    
    # Create cache with short TTL for testing
    cache = PatternDetectionCache(max_size=5, ttl_seconds=60, persist_to_disk=True)
    
    # Test data
    test_texts = [
        "TODO: Implement password hashing",
        "Fix the O(n^2) algorithm performance",
        "Handle timezone conversion properly",
        "Should I use microservices architecture?",
        "Encrypt the API key before storage"
    ]
    
    # Simulate pattern detection results
    print("\nTesting cache operations:")
    for i, text in enumerate(test_texts):
        # Mock patterns
        mock_patterns = []
        mock_summary = {"total_matches": i + 1, "categories": {}}
        mock_strategy = {"strategy": "single_ai", "reason": f"Test {i}"}
        
        # First access (miss)
        result = cache.get(text)
        print(f"\nText {i}: First access - {'HIT' if result else 'MISS'}")
        
        # Store in cache
        cache.put(text, mock_patterns, mock_summary, mock_strategy)
        
        # Second access (hit)
        result = cache.get(text)
        print(f"Text {i}: Second access - {'HIT' if result else 'MISS'}")
        if result:
            print(f"  Cache age: {result['cache_age']:.2f}s")
    
    # Test eviction
    print("\nTesting LRU eviction:")
    extra_text = "This should evict the least recently used entry"
    cache.put(extra_text, [], {}, {})
    
    # Check which entry was evicted (should be test_texts[0])
    result = cache.get(test_texts[0])
    print(f"First entry after eviction: {'HIT' if result else 'MISS (evicted)'}")
    
    # Show statistics
    print("\nCache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test persistence
    if cache.persist_to_disk:
        print(f"\nCache files saved to: {cache.cache_dir}")
        
        # Create new cache instance to test loading
        cache2 = PatternDetectionCache(max_size=5, ttl_seconds=60, 
                                     persist_to_disk=True, 
                                     cache_dir=str(cache.cache_dir))
        
        print(f"Loaded {len(cache2.cache)} entries from disk")
        
        # Verify loaded entries
        for text in test_texts[1:]:  # First was evicted
            result = cache2.get(text)
            if result:
                print(f"  Loaded: {text[:30]}...")
    
    # Cleanup
    cache.clear()
    print("\nCache cleared")
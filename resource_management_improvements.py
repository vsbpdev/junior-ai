#!/usr/bin/env python3
"""
Resource Management Improvements for AI Consultation Manager
Memory management, cleanup, and monitoring enhancements
"""

import psutil
import os
import gc
import weakref
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import threading
import time
import logging


@dataclass
class ResourceMetrics:
    """Track resource usage metrics"""
    memory_usage_mb: float
    audit_trail_size: int
    cache_size: int
    active_consultations: int
    total_ai_calls: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ResourceMonitor:
    """Monitor and manage resource usage"""
    
    def __init__(self, 
                 warning_threshold_mb: int = 500,
                 critical_threshold_mb: int = 1000):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.logger = logging.getLogger(__name__)
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def check_resources(self, 
                       audit_size: int, 
                       cache_size: int,
                       active_consultations: int) -> Optional[str]:
        """Check resource usage and return warning if needed"""
        memory_mb = self.get_memory_usage()
        
        metrics = ResourceMetrics(
            memory_usage_mb=memory_mb,
            audit_trail_size=audit_size,
            cache_size=cache_size,
            active_consultations=active_consultations,
            total_ai_calls=0  # Set by caller
        )
        
        self.metrics_history.append(metrics)
        
        if memory_mb > self.critical_threshold_mb:
            return f"CRITICAL: Memory usage ({memory_mb:.1f}MB) exceeds limit"
        if memory_mb > self.warning_threshold_mb:
            return f"WARNING: High memory usage ({memory_mb:.1f}MB)"
        
        return None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                memory_mb = self.get_memory_usage()
                if memory_mb > self.critical_threshold_mb:
                    self.logger.critical(f"Memory usage critical: {memory_mb:.1f}MB")
                    # Force garbage collection
                    gc.collect()
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self._monitoring = False


class BoundedAuditTrail:
    """Audit trail with automatic size management"""
    
    def __init__(self, 
                 max_entries: int = 1000,
                 max_age_hours: int = 24,
                 persist_callback: Optional[Callable] = None):
        self.max_entries = max_entries
        self.max_age_seconds = max_age_hours * 3600
        self.entries = deque(maxlen=max_entries)
        self.persist_callback = persist_callback
        self.lock = threading.Lock()
        
        # Index for fast lookups
        self.category_index: Dict[str, List[weakref.ref]] = {}
        self.ai_index: Dict[str, List[weakref.ref]] = {}
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def add_entry(self, audit_entry: Any):
        """Add new audit entry with indexing"""
        with self.lock:
            # Remove old entry if at capacity
            if len(self.entries) >= self.max_entries:
                oldest = self.entries[0]
                if self.persist_callback:
                    # Persist to storage before removal
                    self.persist_callback(oldest)
            
            self.entries.append(audit_entry)
            
            # Update indexes
            entry_ref = weakref.ref(audit_entry)
            
            # Category index
            categories = audit_entry.pattern_summary.get('categories', [])
            for category in categories:
                if category not in self.category_index:
                    self.category_index[category] = []
                self.category_index[category].append(entry_ref)
            
            # AI index
            for ai in audit_entry.ais_consulted:
                if ai not in self.ai_index:
                    self.ai_index[ai] = []
                self.ai_index[ai].append(entry_ref)
    
    def get_recent_entries(self, 
                          limit: int = 10,
                          category: Optional[str] = None,
                          ai_name: Optional[str] = None) -> List[Any]:
        """Get recent entries with optional filtering"""
        with self.lock:
            entries = list(self.entries)
            
            # Filter by category if specified
            if category and category in self.category_index:
                category_refs = self.category_index[category]
                entries = [ref() for ref in category_refs if ref() is not None]
            
            # Filter by AI if specified
            if ai_name and ai_name in self.ai_index:
                ai_refs = self.ai_index[ai_name]
                if category:
                    # Intersection of both filters
                    ai_entries = {ref() for ref in ai_refs if ref() is not None}
                    entries = [e for e in entries if e in ai_entries]
                else:
                    entries = [ref() for ref in ai_refs if ref() is not None]
            
            # Sort by timestamp (newest first)
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            return entries[:limit]
    
    def _cleanup_loop(self):
        """Periodically clean up old entries"""
        while True:
            try:
                time.sleep(3600)  # Run hourly
                self._cleanup_old_entries()
                self._cleanup_indexes()
            except Exception as e:
                logging.error(f"Audit cleanup error: {e}")
    
    def _cleanup_old_entries(self):
        """Remove entries older than max age"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.max_age_seconds
            
            # Find entries to remove
            to_remove = []
            for i, entry in enumerate(self.entries):
                if hasattr(entry, 'timestamp'):
                    entry_time = entry.timestamp.timestamp()
                    if entry_time < cutoff_time:
                        to_remove.append(i)
            
            # Remove from newest to oldest to maintain indexes
            for i in reversed(to_remove):
                removed = self.entries[i]
                if self.persist_callback:
                    self.persist_callback(removed)
                del self.entries[i]
    
    def _cleanup_indexes(self):
        """Clean up dead references in indexes"""
        with self.lock:
            # Clean category index
            for category in list(self.category_index.keys()):
                self.category_index[category] = [
                    ref for ref in self.category_index[category]
                    if ref() is not None
                ]
                if not self.category_index[category]:
                    del self.category_index[category]
            
            # Clean AI index
            for ai in list(self.ai_index.keys()):
                self.ai_index[ai] = [
                    ref for ref in self.ai_index[ai]
                    if ref() is not None
                ]
                if not self.ai_index[ai]:
                    del self.ai_index[ai]


class SmartCacheManager:
    """Intelligent cache with eviction policies"""
    
    def __init__(self,
                 max_memory_mb: int = 100,
                 max_entries: int = 1000,
                 ttl_seconds: int = 3600):
        self.max_memory_mb = max_memory_mb
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        
        # Multiple cache levels
        self.hot_cache: Dict[str, Tuple[Any, float, int]] = {}  # value, timestamp, hits
        self.warm_cache: Dict[str, Tuple[Any, float, int]] = {}
        self.lock = threading.Lock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU promotion"""
        with self.lock:
            current_time = time.time()
            
            # Check hot cache
            if key in self.hot_cache:
                value, timestamp, hits = self.hot_cache[key]
                if current_time - timestamp < self.ttl_seconds:
                    self.hot_cache[key] = (value, timestamp, hits + 1)
                    self.hits += 1
                    return value
                else:
                    del self.hot_cache[key]
            
            # Check warm cache
            if key in self.warm_cache:
                value, timestamp, hits = self.warm_cache[key]
                if current_time - timestamp < self.ttl_seconds:
                    # Promote to hot cache
                    self.warm_cache[key] = (value, timestamp, hits + 1)
                    if hits >= 3:  # Promote after 3 hits
                        self._promote_to_hot(key, value, timestamp, hits + 1)
                    self.hits += 1
                    return value
                else:
                    del self.warm_cache[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """Add item to cache with memory management"""
        with self.lock:
            # Check memory usage
            if self._estimate_memory_usage() > self.max_memory_mb:
                self._evict_entries()
            
            # Add to warm cache initially
            self.warm_cache[key] = (value, time.time(), 0)
            
            # Check total entries
            total_entries = len(self.hot_cache) + len(self.warm_cache)
            if total_entries > self.max_entries:
                self._evict_lru()
    
    def _promote_to_hot(self, key: str, value: Any, timestamp: float, hits: int):
        """Promote entry from warm to hot cache"""
        if key in self.warm_cache:
            del self.warm_cache[key]
        self.hot_cache[key] = (value, timestamp, hits)
        
        # Evict from hot cache if needed
        if len(self.hot_cache) > self.max_entries // 2:
            self._evict_from_hot()
    
    def _evict_from_hot(self):
        """Evict least recently used from hot cache"""
        if not self.hot_cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self.hot_cache.keys(),
            key=lambda k: self.hot_cache[k][1]  # timestamp
        )
        
        # Demote to warm cache
        value, timestamp, hits = self.hot_cache[lru_key]
        self.warm_cache[lru_key] = (value, timestamp, hits)
        del self.hot_cache[lru_key]
        self.evictions += 1
    
    def _evict_lru(self):
        """Evict least recently used entry from warm cache"""
        if not self.warm_cache:
            return
        
        lru_key = min(
            self.warm_cache.keys(),
            key=lambda k: self.warm_cache[k][1]
        )
        del self.warm_cache[lru_key]
        self.evictions += 1
    
    def _evict_entries(self):
        """Evict entries to free memory"""
        # Evict 10% of warm cache
        evict_count = max(1, len(self.warm_cache) // 10)
        
        sorted_keys = sorted(
            self.warm_cache.keys(),
            key=lambda k: self.warm_cache[k][1]  # Sort by timestamp
        )
        
        for key in sorted_keys[:evict_count]:
            del self.warm_cache[key]
            self.evictions += 1
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB"""
        # Rough estimation based on entry count
        # Assume average entry is 1KB
        total_entries = len(self.hot_cache) + len(self.warm_cache)
        return total_entries * 0.001  # Convert KB to MB
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'hot_cache_size': len(self.hot_cache),
                'warm_cache_size': len(self.warm_cache),
                'estimated_memory_mb': self._estimate_memory_usage()
            }
#!/usr/bin/env python3
"""
Performance Optimizations for AI Consultation Manager
Request deduplication, intelligent caching, and batch processing
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import weakref


@dataclass
class RequestSignature:
    """Unique signature for deduplication"""
    prompt_hash: str
    context_hash: str
    ai_names: Tuple[str, ...]
    options_hash: str
    
    @classmethod
    def from_request(cls, 
                    prompt: str,
                    context: str,
                    ai_names: List[str],
                    options: Dict[str, Any] = None) -> 'RequestSignature':
        """Create signature from request parameters"""
        # Hash prompt (first 1000 chars for performance)
        prompt_sample = prompt[:1000] if len(prompt) > 1000 else prompt
        prompt_hash = hashlib.md5(prompt_sample.encode()).hexdigest()[:8]
        
        # Hash context
        context_sample = context[:1000] if len(context) > 1000 else context
        context_hash = hashlib.md5(context_sample.encode()).hexdigest()[:8]
        
        # Sort AI names for consistent hashing
        ai_tuple = tuple(sorted(ai_names))
        
        # Hash options
        options_str = str(sorted(options.items())) if options else ""
        options_hash = hashlib.md5(options_str.encode()).hexdigest()[:8]
        
        return cls(prompt_hash, context_hash, ai_tuple, options_hash)
    
    def to_key(self) -> str:
        """Convert to cache key"""
        return f"{self.prompt_hash}:{self.context_hash}:{','.join(self.ai_names)}:{self.options_hash}"


class RequestDeduplicator:
    """Deduplicate concurrent identical requests"""
    
    def __init__(self):
        self.pending_requests: Dict[str, Future] = {}
        self.lock = threading.Lock()
        self.request_counts = defaultdict(int)
    
    def deduplicate(self,
                   signature: RequestSignature,
                   executor: Callable[[], Any]) -> Tuple[Any, bool]:
        """
        Execute request or wait for identical pending request.
        Returns (result, was_deduplicated)
        """
        key = signature.to_key()
        
        with self.lock:
            # Check if identical request is pending
            if key in self.pending_requests:
                future = self.pending_requests[key]
                self.request_counts[key] += 1
                # Release lock before waiting
                
        if key in self.pending_requests:
            # Wait for pending request
            try:
                result = future.result()
                return result, True
            except Exception as e:
                # Original request failed
                raise e
        
        # No pending request, create new one
        with self.lock:
            # Double-check after acquiring lock
            if key in self.pending_requests:
                future = self.pending_requests[key]
            else:
                # Create new future
                future = Future()
                self.pending_requests[key] = future
        
        # Execute request
        try:
            result = executor()
            future.set_result(result)
            return result, False
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            # Clean up
            with self.lock:
                if key in self.pending_requests:
                    del self.pending_requests[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        with self.lock:
            total_requests = sum(self.request_counts.values())
            deduplicated = sum(
                count - 1 for count in self.request_counts.values() 
                if count > 1
            )
            
            return {
                'total_requests': total_requests,
                'deduplicated_requests': deduplicated,
                'deduplication_rate': deduplicated / total_requests if total_requests > 0 else 0,
                'unique_patterns': len(self.request_counts)
            }


class BatchProcessor:
    """Batch similar requests for efficient processing"""
    
    def __init__(self,
                 batch_size: int = 10,
                 batch_timeout: float = 0.1,
                 processor: Callable[[List[Any]], List[Any]] = None):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.processor = processor
        
        self.pending_queue = queue.Queue()
        self.result_futures: Dict[str, Future] = {}
        self.processing = True
        
        self.worker_thread = threading.Thread(
            target=self._process_batches,
            daemon=True
        )
        self.worker_thread.start()
    
    def submit(self, request_id: str, request: Any) -> Future:
        """Submit request for batch processing"""
        future = Future()
        self.result_futures[request_id] = future
        self.pending_queue.put((request_id, request))
        return future
    
    def _process_batches(self):
        """Process requests in batches"""
        while self.processing:
            batch = []
            batch_ids = []
            deadline = time.time() + self.batch_timeout
            
            # Collect batch
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    request_id, request = self.pending_queue.get(timeout=timeout)
                    batch.append(request)
                    batch_ids.append(request_id)
                except queue.Empty:
                    break
            
            if batch and self.processor:
                try:
                    # Process batch
                    results = self.processor(batch)
                    
                    # Deliver results
                    for request_id, result in zip(batch_ids, results):
                        if request_id in self.result_futures:
                            self.result_futures[request_id].set_result(result)
                            del self.result_futures[request_id]
                            
                except Exception as e:
                    # Batch processing failed
                    for request_id in batch_ids:
                        if request_id in self.result_futures:
                            self.result_futures[request_id].set_exception(e)
                            del self.result_futures[request_id]
    
    def stop(self):
        """Stop batch processing"""
        self.processing = False
        self.worker_thread.join()


class IntelligentCache:
    """Smart caching with pattern-based TTL and preloading"""
    
    def __init__(self,
                 base_ttl: int = 3600,
                 max_size: int = 1000):
        self.base_ttl = base_ttl
        self.max_size = max_size
        
        # Multi-level cache
        self.l1_cache: Dict[str, Tuple[Any, float, Dict[str, Any]]] = {}  # Hot cache
        self.l2_cache: Dict[str, Tuple[Any, float, Dict[str, Any]]] = {}  # Warm cache
        
        # Pattern-based TTL adjustments
        self.ttl_multipliers = {
            'security': 0.5,      # Security patterns expire faster
            'algorithm': 2.0,     # Algorithm advice cached longer
            'architecture': 1.5,  # Architecture patterns moderately cached
            'uncertainty': 0.3,   # Uncertainty patterns expire quickly
            'gotcha': 1.0        # Standard caching for gotchas
        }
        
        # Access patterns for predictive caching
        self.access_history = defaultdict(list)
        self.lock = threading.Lock()
    
    def get(self, 
           key: str,
           pattern_category: Optional[str] = None) -> Optional[Any]:
        """Get from cache with pattern-aware logic"""
        with self.lock:
            current_time = time.time()
            
            # Check L1 cache
            if key in self.l1_cache:
                value, timestamp, metadata = self.l1_cache[key]
                ttl = self._calculate_ttl(pattern_category, metadata)
                
                if current_time - timestamp < ttl:
                    # Update access history
                    self.access_history[key].append(current_time)
                    metadata['hits'] = metadata.get('hits', 0) + 1
                    return value
                else:
                    del self.l1_cache[key]
            
            # Check L2 cache
            if key in self.l2_cache:
                value, timestamp, metadata = self.l2_cache[key]
                ttl = self._calculate_ttl(pattern_category, metadata)
                
                if current_time - timestamp < ttl:
                    # Promote to L1
                    self._promote_to_l1(key, value, timestamp, metadata)
                    return value
                else:
                    del self.l2_cache[key]
            
            return None
    
    def set(self,
           key: str,
           value: Any,
           pattern_category: Optional[str] = None,
           metadata: Dict[str, Any] = None):
        """Set in cache with intelligent placement"""
        with self.lock:
            current_time = time.time()
            cache_metadata = metadata or {}
            cache_metadata['pattern_category'] = pattern_category
            cache_metadata['hits'] = 0
            
            # Determine cache level based on value characteristics
            if self._should_cache_l1(value, pattern_category):
                self._add_to_l1(key, value, current_time, cache_metadata)
            else:
                self._add_to_l2(key, value, current_time, cache_metadata)
    
    def _calculate_ttl(self, 
                      pattern_category: Optional[str],
                      metadata: Dict[str, Any]) -> float:
        """Calculate dynamic TTL based on pattern and usage"""
        base_ttl = self.base_ttl
        
        # Apply pattern multiplier
        if pattern_category in self.ttl_multipliers:
            base_ttl *= self.ttl_multipliers[pattern_category]
        
        # Adjust based on hit rate
        hits = metadata.get('hits', 0)
        if hits > 10:
            base_ttl *= 1.5  # Popular items cached longer
        elif hits < 2:
            base_ttl *= 0.7  # Unpopular items expire faster
        
        return base_ttl
    
    def _should_cache_l1(self, 
                        value: Any,
                        pattern_category: Optional[str]) -> bool:
        """Determine if value should go to L1 cache"""
        # High-priority patterns go to L1
        if pattern_category in ['security', 'algorithm']:
            return True
        
        # Large responses go to L2
        if isinstance(value, dict) and len(str(value)) > 10000:
            return False
        
        return True
    
    def _add_to_l1(self, 
                  key: str,
                  value: Any,
                  timestamp: float,
                  metadata: Dict[str, Any]):
        """Add to L1 cache with eviction"""
        if len(self.l1_cache) >= self.max_size // 2:
            # Evict LRU from L1 to L2
            lru_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k][2].get('last_access', 0)
            )
            self._demote_to_l2(lru_key)
        
        metadata['last_access'] = timestamp
        self.l1_cache[key] = (value, timestamp, metadata)
    
    def _add_to_l2(self,
                  key: str,
                  value: Any,
                  timestamp: float,
                  metadata: Dict[str, Any]):
        """Add to L2 cache with eviction"""
        if len(self.l2_cache) >= self.max_size // 2:
            # Evict LRU from L2
            lru_key = min(
                self.l2_cache.keys(),
                key=lambda k: self.l2_cache[k][2].get('last_access', 0)
            )
            del self.l2_cache[lru_key]
        
        metadata['last_access'] = timestamp
        self.l2_cache[key] = (value, timestamp, metadata)
    
    def _promote_to_l1(self,
                      key: str,
                      value: Any,
                      timestamp: float,
                      metadata: Dict[str, Any]):
        """Promote entry from L2 to L1"""
        if key in self.l2_cache:
            del self.l2_cache[key]
        self._add_to_l1(key, value, timestamp, metadata)
    
    def _demote_to_l2(self, key: str):
        """Demote entry from L1 to L2"""
        if key in self.l1_cache:
            value, timestamp, metadata = self.l1_cache[key]
            del self.l1_cache[key]
            self._add_to_l2(key, value, timestamp, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_size = len(self.l1_cache) + len(self.l2_cache)
            
            # Calculate hit rates by pattern
            pattern_hits = defaultdict(lambda: {'hits': 0, 'total': 0})
            
            for cache in [self.l1_cache, self.l2_cache]:
                for _, (_, _, metadata) in cache.items():
                    pattern = metadata.get('pattern_category', 'unknown')
                    pattern_hits[pattern]['total'] += 1
                    pattern_hits[pattern]['hits'] += metadata.get('hits', 0)
            
            return {
                'total_entries': total_size,
                'l1_size': len(self.l1_cache),
                'l2_size': len(self.l2_cache),
                'memory_usage_estimate_mb': total_size * 0.001,  # Rough estimate
                'pattern_statistics': dict(pattern_hits)
            }


class PerformanceOptimizedConsultationManager:
    """Consultation manager with performance optimizations"""
    
    def __init__(self,
                 base_manager: Any,
                 enable_deduplication: bool = True,
                 enable_batching: bool = True,
                 enable_caching: bool = True):
        self.base_manager = base_manager
        self.deduplicator = RequestDeduplicator() if enable_deduplication else None
        self.cache = IntelligentCache() if enable_caching else None
        
        # Batch processor for similar patterns
        self.batch_processor = None
        if enable_batching:
            self.batch_processor = BatchProcessor(
                batch_size=5,
                batch_timeout=0.05,
                processor=self._process_batch
            )
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'deduplicated': 0,
            'batched': 0
        }
    
    def consult_optimized(self,
                         prompt: str,
                         context: str,
                         ai_names: List[str],
                         options: Dict[str, Any] = None) -> Any:
        """Optimized consultation with all performance features"""
        self.metrics['total_requests'] += 1
        
        # Create request signature
        signature = RequestSignature.from_request(
            prompt, context, ai_names, options
        )
        
        # Check cache first
        if self.cache:
            cache_key = signature.to_key()
            cached_result = self.cache.get(
                cache_key,
                options.get('pattern_category') if options else None
            )
            if cached_result:
                self.metrics['cache_hits'] += 1
                return cached_result
        
        # Deduplicate if enabled
        if self.deduplicator:
            def execute():
                return self._execute_consultation(
                    prompt, context, ai_names, options
                )
            
            result, was_deduplicated = self.deduplicator.deduplicate(
                signature, execute
            )
            
            if was_deduplicated:
                self.metrics['deduplicated'] += 1
        else:
            result = self._execute_consultation(
                prompt, context, ai_names, options
            )
        
        # Cache result
        if self.cache and result:
            self.cache.set(
                signature.to_key(),
                result,
                options.get('pattern_category') if options else None,
                {'ai_names': ai_names, 'options': options}
            )
        
        return result
    
    def _execute_consultation(self,
                            prompt: str,
                            context: str,
                            ai_names: List[str],
                            options: Dict[str, Any] = None) -> Any:
        """Execute actual consultation"""
        # Implementation would call base_manager methods
        # This is a placeholder
        return {
            'prompt': prompt,
            'context': context,
            'ai_names': ai_names,
            'timestamp': time.time()
        }
    
    def _process_batch(self, requests: List[Any]) -> List[Any]:
        """Process batch of similar requests"""
        # Group by pattern type for efficient processing
        grouped = defaultdict(list)
        for req in requests:
            pattern = req.get('pattern_category', 'unknown')
            grouped[pattern].append(req)
        
        results = []
        for pattern, group in grouped.items():
            # Process group together
            # This is a placeholder - actual implementation would
            # optimize AI calls for similar patterns
            for req in group:
                results.append({'processed': True, 'pattern': pattern})
        
        self.metrics['batched'] += len(requests)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = dict(self.metrics)
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        if self.deduplicator:
            stats['deduplication_stats'] = self.deduplicator.get_stats()
        
        # Calculate rates
        total = stats['total_requests']
        if total > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total
            stats['deduplication_rate'] = stats['deduplicated'] / total
            stats['batch_rate'] = stats['batched'] / total
        
        return stats
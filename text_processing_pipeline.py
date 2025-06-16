#!/usr/bin/env python3
"""
Real-Time Text Processing Pipeline for Junior AI Assistant
Handles streaming text analysis with low latency
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from pattern_detection import PatternDetectionEngine, PatternMatch, PatternCategory, PatternSeverity
from pattern_cache import PatternDetectionCache, CachedPatternDetectionEngine


@dataclass
class TextChunk:
    """Represents a chunk of text to be processed"""
    id: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Result of processing a text chunk"""
    chunk_id: str
    patterns: List[PatternMatch]
    processing_time: float
    timestamp: float
    consultation_triggered: bool
    metadata: Dict[str, Any]


class TextProcessingPipeline:
    """Real-time text processing pipeline with pattern detection"""
    
    def __init__(self, 
                 pattern_engine: Optional[PatternDetectionEngine] = None,
                 buffer_size: int = 1000,
                 max_workers: int = 4,
                 batch_timeout: float = 0.1,
                 enable_cache: bool = True,
                 cache_config: Optional[Dict] = None):
        """
        Initialize the text processing pipeline
        
        Args:
            pattern_engine: Pattern detection engine instance
            buffer_size: Maximum size of the text buffer
            max_workers: Number of worker threads for processing
            batch_timeout: Maximum time to wait for batch accumulation
        """
        base_engine = pattern_engine or PatternDetectionEngine()
        
        # Set up caching if enabled
        if enable_cache:
            cache_config = cache_config or {}
            cache = PatternDetectionCache(
                max_size=cache_config.get('max_size', 1000),
                ttl_seconds=cache_config.get('ttl_seconds', 300),
                persist_to_disk=cache_config.get('persist_to_disk', False),
                cache_dir=cache_config.get('cache_dir')
            )
            self.pattern_engine = CachedPatternDetectionEngine(base_engine, cache)
        else:
            self.pattern_engine = base_engine
            
        self.buffer_size = buffer_size
        self.batch_timeout = batch_timeout
        self.cache_enabled = enable_cache
        
        # Processing components
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # State management
        self.is_running = False
        self.processing_thread = None
        self.stats = {
            "chunks_processed": 0,
            "patterns_detected": 0,
            "consultations_triggered": 0,
            "total_processing_time": 0.0,
            "average_latency": 0.0
        }
        
        # Callbacks
        self.pattern_callbacks: List[Callable] = []
        self.result_callbacks: List[Callable] = []
        
    def start(self):
        """Start the processing pipeline"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
    
    def stop(self):
        """Stop the processing pipeline"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
    
    def process_text(self, text: str, chunk_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Process a text chunk through the pipeline
        
        Args:
            text: Text content to process
            chunk_id: Optional identifier for the chunk
            metadata: Optional metadata for the chunk
            
        Returns:
            Chunk ID for tracking
        """
        chunk_id = chunk_id or f"chunk_{time.time()}"
        chunk = TextChunk(
            id=chunk_id,
            content=text,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        try:
            self.input_queue.put(chunk, timeout=1.0)
            return chunk_id
        except queue.Full:
            raise RuntimeError("Processing pipeline is full")
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get a processing result from the output queue"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_pattern_callback(self, callback: Callable[[List[PatternMatch]], None]):
        """Add a callback for when patterns are detected"""
        self.pattern_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """Add a callback for processing results"""
        self.result_callbacks.append(callback)
    
    def _process_loop(self):
        """Main processing loop"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Collect chunks for batch processing
                timeout = max(0.01, self.batch_timeout - (time.time() - last_batch_time))
                chunk = self.input_queue.get(timeout=timeout)
                batch.append(chunk)
                
                # Process batch if timeout reached or batch is large enough
                if (time.time() - last_batch_time >= self.batch_timeout or 
                    len(batch) >= 10):
                    if batch:
                        self._process_batch(batch)
                        batch = []
                        last_batch_time = time.time()
                        
            except queue.Empty:
                # Process any remaining chunks in batch
                if batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
    
    def _process_batch(self, batch: List[TextChunk]):
        """Process a batch of text chunks"""
        futures = []
        
        for chunk in batch:
            future = self.executor.submit(self._process_chunk, chunk)
            futures.append((chunk, future))
        
        # Collect results
        for chunk, future in futures:
            try:
                result = future.result(timeout=5.0)
                self.output_queue.put(result)
                
                # Update statistics
                self._update_stats(result)
                
                # Trigger callbacks
                self._trigger_callbacks(result)
                
            except Exception as e:
                # Handle processing errors
                error_result = ProcessingResult(
                    chunk_id=chunk.id,
                    patterns=[],
                    processing_time=0.0,
                    timestamp=time.time(),
                    consultation_triggered=False,
                    metadata={"error": str(e)}
                )
                self.output_queue.put(error_result)
    
    def _process_chunk(self, chunk: TextChunk) -> ProcessingResult:
        """Process a single text chunk"""
        start_time = time.time()
        
        # Detect patterns
        patterns = self.pattern_engine.detect_patterns(chunk.content)
        
        # Determine if consultation should be triggered
        consultation_triggered = self.pattern_engine.should_trigger_consultation(patterns)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = ProcessingResult(
            chunk_id=chunk.id,
            patterns=patterns,
            processing_time=processing_time,
            timestamp=time.time(),
            consultation_triggered=consultation_triggered,
            metadata={
                "original_metadata": chunk.metadata,
                "pattern_summary": self.pattern_engine.get_pattern_summary(patterns),
                "consultation_strategy": self.pattern_engine.get_consultation_strategy(patterns)
            }
        )
        
        return result
    
    def _update_stats(self, result: ProcessingResult):
        """Update processing statistics"""
        self.stats["chunks_processed"] += 1
        self.stats["patterns_detected"] += len(result.patterns)
        if result.consultation_triggered:
            self.stats["consultations_triggered"] += 1
        self.stats["total_processing_time"] += result.processing_time
        self.stats["average_latency"] = (
            self.stats["total_processing_time"] / self.stats["chunks_processed"]
        )
    
    def _trigger_callbacks(self, result: ProcessingResult):
        """Trigger registered callbacks"""
        # Pattern callbacks
        if result.patterns:
            for callback in self.pattern_callbacks:
                try:
                    callback(result.patterns)
                except Exception:
                    pass  # Ignore callback errors
        
        # Result callbacks
        for callback in self.result_callbacks:
            try:
                callback(result)
            except Exception:
                pass  # Ignore callback errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        
        # Add cache statistics if available
        if self.cache_enabled and hasattr(self.pattern_engine, 'get_cache_stats'):
            stats['cache'] = self.pattern_engine.get_cache_stats()
        
        return stats


class StreamingTextProcessor:
    """Handles streaming text input with buffering and chunking"""
    
    def __init__(self, pipeline: TextProcessingPipeline, chunk_size: int = 500):
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.buffer = ""
        self.chunk_counter = 0
    
    def process_stream(self, text_stream: str) -> List[str]:
        """
        Process streaming text input
        
        Args:
            text_stream: Incoming text stream
            
        Returns:
            List of chunk IDs that were processed
        """
        chunk_ids = []
        self.buffer += text_stream
        
        # Process complete chunks
        while len(self.buffer) >= self.chunk_size:
            # Find a good break point (end of sentence/line)
            break_point = self._find_break_point(self.buffer[:self.chunk_size * 2])
            
            if break_point > 0:
                chunk = self.buffer[:break_point]
                self.buffer = self.buffer[break_point:].lstrip()
                
                # Process the chunk
                chunk_id = f"stream_chunk_{self.chunk_counter}"
                self.chunk_counter += 1
                self.pipeline.process_text(chunk, chunk_id=chunk_id)
                chunk_ids.append(chunk_id)
            else:
                # No good break point found, force chunk at size limit
                if len(self.buffer) > self.chunk_size * 2:
                    chunk = self.buffer[:self.chunk_size]
                    self.buffer = self.buffer[self.chunk_size:]
                    
                    chunk_id = f"stream_chunk_{self.chunk_counter}"
                    self.chunk_counter += 1
                    self.pipeline.process_text(chunk, chunk_id=chunk_id)
                    chunk_ids.append(chunk_id)
                break
        
        return chunk_ids
    
    def flush(self) -> Optional[str]:
        """Flush any remaining buffered text"""
        if self.buffer:
            chunk = self.buffer
            self.buffer = ""
            
            chunk_id = f"stream_chunk_{self.chunk_counter}"
            self.chunk_counter += 1
            self.pipeline.process_text(chunk, chunk_id=chunk_id)
            return chunk_id
        return None
    
    def _find_break_point(self, text: str) -> int:
        """Find a good break point in the text (end of sentence or paragraph)"""
        # Look for paragraph breaks first
        para_break = text.find("\n\n")
        if para_break > self.chunk_size // 2:
            return para_break + 2
        
        # Look for sentence endings
        sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
        best_pos = -1
        
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > self.chunk_size // 2 and pos > best_pos:
                best_pos = pos + len(ending)
        
        if best_pos > 0:
            return best_pos
        
        # Fall back to line break
        line_break = text.rfind("\n")
        if line_break > self.chunk_size // 2:
            return line_break + 1
        
        # Fall back to space
        space = text.rfind(" ")
        if space > self.chunk_size // 2:
            return space + 1
        
        return -1


# Async version for integration with async frameworks
class AsyncTextProcessingPipeline:
    """Asynchronous version of the text processing pipeline"""
    
    def __init__(self, pattern_engine: Optional[PatternDetectionEngine] = None):
        self.pattern_engine = pattern_engine or PatternDetectionEngine()
        self.processing_queue = asyncio.Queue()
        self.is_running = False
        self.tasks = []
    
    async def start(self):
        """Start the async processing pipeline"""
        self.is_running = True
        # Start worker tasks
        for i in range(4):  # 4 concurrent workers
            task = asyncio.create_task(self._worker(f"worker_{i}"))
            self.tasks.append(task)
    
    async def stop(self):
        """Stop the async processing pipeline"""
        self.is_running = False
        # Wait for all tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def process_text(self, text: str, chunk_id: Optional[str] = None) -> ProcessingResult:
        """Process text asynchronously"""
        chunk = TextChunk(
            id=chunk_id or f"async_chunk_{time.time()}",
            content=text,
            timestamp=time.time(),
            metadata={}
        )
        
        result_future = asyncio.Future()
        await self.processing_queue.put((chunk, result_future))
        return await result_future
    
    async def _worker(self, worker_id: str):
        """Async worker for processing chunks"""
        while self.is_running:
            try:
                chunk, result_future = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )
                
                # Process chunk
                result = await self._process_chunk_async(chunk)
                result_future.set_result(result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not result_future.done():
                    result_future.set_exception(e)
    
    async def _process_chunk_async(self, chunk: TextChunk) -> ProcessingResult:
        """Process a chunk asynchronously"""
        # Run pattern detection in executor to avoid blocking
        loop = asyncio.get_event_loop()
        patterns = await loop.run_in_executor(
            None, self.pattern_engine.detect_patterns, chunk.content
        )
        
        consultation_triggered = self.pattern_engine.should_trigger_consultation(patterns)
        
        return ProcessingResult(
            chunk_id=chunk.id,
            patterns=patterns,
            processing_time=0.0,  # Not tracked in async version
            timestamp=time.time(),
            consultation_triggered=consultation_triggered,
            metadata={
                "pattern_summary": self.pattern_engine.get_pattern_summary(patterns),
                "consultation_strategy": self.pattern_engine.get_consultation_strategy(patterns)
            }
        )


if __name__ == "__main__":
    # Test the pipeline
    print("Testing Text Processing Pipeline...")
    
    # Create pipeline
    pipeline = TextProcessingPipeline()
    
    # Add callbacks
    def pattern_callback(patterns):
        print(f"Patterns detected: {len(patterns)}")
        for pattern in patterns[:3]:  # Show first 3
            print(f"  - {pattern.category.value}: {pattern.keyword}")
    
    def result_callback(result):
        if result.consultation_triggered:
            print(f"Consultation triggered for chunk {result.chunk_id}")
            print(f"Strategy: {result.metadata['consultation_strategy']['strategy']}")
    
    pipeline.add_pattern_callback(pattern_callback)
    pipeline.add_result_callback(result_callback)
    
    # Start pipeline
    pipeline.start()
    
    # Test with sample texts
    test_texts = [
        "I need to implement password hashing for the login system.",
        "TODO: Optimize this sorting algorithm - it has O(n^2) complexity",
        "Working with datetime and timezone conversion",
        "Should I use microservices architecture?",
        "The API key should be encrypted"
    ]
    
    chunk_ids = []
    for text in test_texts:
        chunk_id = pipeline.process_text(text)
        chunk_ids.append(chunk_id)
        time.sleep(0.1)  # Simulate streaming
    
    # Wait for processing
    time.sleep(1.0)
    
    # Get results
    print("\nProcessing Results:")
    while True:
        result = pipeline.get_result(timeout=0.1)
        if not result:
            break
        print(f"Chunk {result.chunk_id}: {len(result.patterns)} patterns, "
              f"consultation: {result.consultation_triggered}")
    
    # Show stats
    print("\nPipeline Statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Stop pipeline
    pipeline.stop()
    
    # Test streaming processor
    print("\n\nTesting Streaming Processor...")
    pipeline2 = TextProcessingPipeline()
    pipeline2.start()
    
    streaming_processor = StreamingTextProcessor(pipeline2, chunk_size=50)
    
    # Simulate streaming input
    stream_text = "This is a long text that talks about password security. " \
                  "We need to implement proper authentication. " \
                  "TODO: Add encryption for sensitive data. " \
                  "The system should handle timezone conversions properly."
    
    # Process in chunks
    for i in range(0, len(stream_text), 20):
        chunk = stream_text[i:i+20]
        streaming_processor.process_stream(chunk)
        time.sleep(0.05)
    
    # Flush remaining
    streaming_processor.flush()
    
    time.sleep(1.0)
    pipeline2.stop()
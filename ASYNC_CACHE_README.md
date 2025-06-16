# Async Pattern Detection Cache

This document describes the async pattern detection caching system implemented for the Junior AI Assistant.

## Overview

The async caching system provides high-performance pattern detection with the following key features:

- **Async/Await Support**: Non-blocking cache operations using asyncio
- **SHA256 Key Generation**: Secure and collision-resistant cache keys
- **Memory-Bounded LRU Eviction**: Automatic memory management with size limits
- **Request Deduplication**: Prevents redundant processing of identical concurrent requests
- **Dynamic TTL**: Time-to-live adjusts based on pattern sensitivity and categories
- **Comprehensive Metrics**: Detailed performance and usage statistics

## Architecture

### Core Components

1. **AsyncPatternCache**: Main async cache implementation
2. **AsyncCachedPatternEngine**: Wrapper for pattern engines with async caching
3. **AsyncPatternDetectionPipeline**: High-level pipeline for batch processing
4. **Cache Metrics**: Comprehensive performance tracking

### Key Features

#### Dynamic TTL Calculation

Cache TTL is dynamically calculated based on:

- **Sensitivity Level**: 
  - `low`: 2.0x base TTL (cache longer)
  - `medium`: 1.0x base TTL (baseline)
  - `high`: 0.5x base TTL (cache shorter)
  - `maximum`: 0.2x base TTL (very short cache)

- **Pattern Categories**:
  - `security`: 0.3x multiplier (shortest cache)
  - `uncertainty`: 0.5x multiplier
  - `algorithm`: 1.5x multiplier (longer cache)
  - `architecture`: 1.2x multiplier
  - `gotcha`: 1.0x multiplier (baseline)

#### Memory Management

- **Size-based Eviction**: Maximum number of entries
- **Memory-based Eviction**: Maximum memory usage in MB
- **LRU Strategy**: Least recently used entries evicted first
- **Background Cleanup**: Automatic cleanup of expired entries

#### Request Deduplication

- **Concurrent Request Detection**: Identifies identical requests in flight
- **Future-based Waiting**: Waiters share results from primary request
- **Automatic Cleanup**: Stale pending requests are cleaned up

## Configuration

### Credentials.json Settings

```json
{
  "pattern_detection": {
    "async_cache_enabled": true,
    "cache_max_size": 2000,
    "cache_max_memory_mb": 100,
    "cache_ttl_seconds": 300,
    "cache_deduplication": true,
    "cache_cleanup_interval": 60,
    "batch_size": 10,
    "max_concurrent": 5
  }
}
```

### Configuration Options

- `async_cache_enabled`: Enable/disable async caching (default: true)
- `cache_max_size`: Maximum number of cache entries (default: 2000)
- `cache_max_memory_mb`: Maximum memory usage in MB (default: 100)
- `cache_ttl_seconds`: Base TTL in seconds (default: 300)
- `cache_deduplication`: Enable request deduplication (default: true)
- `cache_cleanup_interval`: Background cleanup interval (default: 60s)
- `batch_size`: Maximum batch size for pipeline (default: 10)
- `max_concurrent`: Maximum concurrent operations (default: 5)

## MCP Tools

### cache_stats

Get comprehensive cache statistics and performance metrics.

```bash
mcp__junior-ai__cache_stats
```

Returns:
- Engine performance metrics
- Cache size and memory usage
- Hit rates and deduplication statistics
- Pattern category breakdowns
- Sensitivity level statistics

### clear_cache

Clear the async pattern cache.

```bash
mcp__junior-ai__clear_cache
  confirm: true
```

Parameters:
- `confirm`: Must be `true` to proceed with cache clearing

### async_pattern_check

Async pattern detection with caching and deduplication.

```bash
mcp__junior-ai__async_pattern_check
  text: "TODO: implement password hashing"
  sensitivity_level: "high"
  auto_consult: true
```

Parameters:
- `text`: Text to analyze for patterns (required)
- `sensitivity_level`: Detection sensitivity (low/medium/high/maximum)
- `auto_consult`: Automatically consult AI if patterns detected

## Performance Benefits

### Before Async Cache

- **Synchronous blocking**: Pattern detection blocks other operations
- **No deduplication**: Identical requests processed multiple times
- **Fixed TTL**: Same cache duration regardless of pattern type
- **Basic metrics**: Limited performance tracking

### After Async Cache

- **Non-blocking operations**: Pattern detection runs asynchronously
- **Smart deduplication**: Concurrent identical requests share results
- **Dynamic TTL**: Cache duration optimized per pattern type
- **Rich metrics**: Comprehensive performance and usage statistics

### Expected Performance Improvements

- **60-80% faster response times** for cached patterns
- **Reduced CPU usage** through deduplication
- **Better memory management** with bounded caching
- **Improved scalability** with async operations

## Usage Examples

### Basic Usage

```python
from async_cached_pattern_engine import create_async_cached_engine

# Create async cached engine
engine = create_async_cached_engine({
    'max_size': 1000,
    'base_ttl_seconds': 300
})

# Async pattern detection
result = await engine.detect_patterns_async("TODO: fix this code")

# Check cache performance
stats = await engine.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
```

### Batch Processing

```python
from async_cached_pattern_engine import create_async_pipeline

# Create async pipeline
pipeline = create_async_pipeline(
    cache_config={'max_size': 2000},
    batch_size=20,
    max_concurrent=10
)

# Process multiple texts
texts = ["TODO: implement auth", "Fix O(n^2) algorithm", "Handle edge cases"]
results = await pipeline.process_batch(texts)

# Get pipeline statistics
stats = await pipeline.get_pipeline_stats()
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
python3 test_async_pattern_cache.py
```

Test coverage includes:
- Cache entry lifecycle
- Pending request management
- LRU eviction strategies
- Memory-based eviction
- TTL calculation
- Request deduplication
- Metrics tracking
- Pattern serialization

### Manual Testing

Quick functionality test:

```bash
python3 async_pattern_cache.py
```

Integration test with pattern engine:

```bash
python3 async_cached_pattern_engine.py
```

## Monitoring and Debugging

### Cache Metrics

Monitor cache performance using the `cache_stats` MCP tool:

- **Hit Rate**: Percentage of requests served from cache
- **Deduplication Rate**: Percentage of requests deduplicated
- **Memory Usage**: Current memory consumption
- **Eviction Count**: Number of entries evicted
- **TTL Expirations**: Entries expired due to TTL

### Common Issues

1. **High Memory Usage**: Reduce `cache_max_size` or `cache_max_memory_mb`
2. **Low Hit Rate**: Increase `cache_ttl_seconds` or check pattern variability
3. **High Deduplication**: Indicates good caching efficiency
4. **Cleanup Errors**: Check async resource management

### Performance Tuning

- **For Memory-Constrained Systems**: Reduce cache size and memory limits
- **For High-Throughput Systems**: Increase concurrent limits and batch sizes
- **For Security-Focused Applications**: Use shorter TTL for security patterns
- **For Algorithm-Heavy Workloads**: Enable longer caching for algorithm patterns

## Migration from Sync Cache

### Backward Compatibility

The async cache system maintains backward compatibility:

- **Legacy Cache**: Still available via `legacy_cache_enabled`
- **Sync Interface**: Original pattern detection tools continue to work
- **Gradual Migration**: Can enable async cache while keeping sync fallbacks

### Migration Steps

1. **Update Configuration**: Add async cache settings to credentials.json
2. **Test Async Tools**: Use new MCP tools to verify functionality
3. **Monitor Performance**: Compare cache metrics before/after
4. **Disable Legacy**: Set `legacy_cache_enabled: false` when confident

## Future Enhancements

### Planned Features

- **Distributed Caching**: Support for Redis/Memcached backends
- **Persistent Cache**: Optional disk persistence for cache entries
- **Smart Preloading**: Predictive caching based on usage patterns
- **Cache Warming**: Background preloading of common patterns
- **Multi-Level Cache**: L1/L2 cache hierarchy for optimal performance

### Performance Optimizations

- **Cache Partitioning**: Separate caches per pattern category
- **Compression**: Compress large cache entries
- **Bloom Filters**: Fast negative lookups for cache misses
- **Adaptive TTL**: Machine learning-based TTL optimization

## Troubleshooting

### Common Errors

1. **Import Error**: Pattern detection components not available
   - **Solution**: Ensure all pattern detection files are present

2. **Async Runtime Error**: Event loop issues
   - **Solution**: Use `asyncio.run()` for top-level async calls

3. **Memory Limit Exceeded**: Cache exceeds memory bounds
   - **Solution**: Reduce cache size or increase memory limit

4. **Cache Not Working**: Patterns not being cached
   - **Solution**: Check `async_cache_enabled` configuration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This provides detailed information about cache operations, hits/misses, and performance metrics.

## Contributing

When contributing to the async cache system:

1. **Add Tests**: Include comprehensive test coverage for new features
2. **Update Metrics**: Add relevant metrics for new functionality
3. **Document Configuration**: Update credentials template for new settings
4. **Maintain Compatibility**: Ensure backward compatibility with sync interface
5. **Performance Testing**: Benchmark changes against existing implementation

## License

This async caching system is part of the Junior AI Assistant project and follows the same MIT license terms.
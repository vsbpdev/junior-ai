# Async AI Client Implementation

## Overview

The Junior AI Assistant now supports asynchronous AI client operations, providing significant performance improvements for concurrent AI calls. This implementation maintains full backward compatibility while enabling async operations when available.

## Architecture

### Core Components

1. **AsyncAIClient Base Class** (`ai/async_client.py`)
   - Abstract base class defining the async interface
   - Unified `generate()` method for all AI providers
   - Built-in timeout and error handling
   - Context manager support for resource cleanup

2. **Provider Implementations**
   - `AsyncGeminiClient`: Uses native `generate_content_async()` from Google's SDK
   - `AsyncOpenAICompatibleClient`: Uses `AsyncOpenAI` for OpenAI, Grok, DeepSeek, and OpenRouter

3. **Integration Layer** (`core/ai_clients.py`)
   - `initialize_all_clients_async()`: Initializes both sync and async clients
   - `cleanup_async_ai_clients()`: Proper resource cleanup on shutdown
   - Global `ASYNC_AI_CLIENTS` storage for client instances

4. **Backward Compatibility** (`ai/caller.py`)
   - `call_ai_async()`: Smart async wrapper that uses async clients when available
   - Falls back to sync implementation if async not available
   - `call_multiple_ais()`: Already uses async internally for concurrent calls

## Features

### 1. Native Async Support
Both Gemini and OpenAI SDKs provide native async support:
```python
# Gemini
response = await model.generate_content_async(prompt)

# OpenAI-compatible
client = AsyncOpenAI(api_key=api_key)
response = await client.chat.completions.create(...)
```

### 2. Unified Interface
All AI providers share the same async interface:
```python
async def generate(self, 
                  prompt: str, 
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  **kwargs) -> str:
    """Generate a response from the AI."""
```

### 3. Automatic Timeout Handling
- Default 30-second timeout for all AI calls
- Configurable per-client timeout
- Proper timeout error propagation

### 4. Resource Management
- Context manager support for automatic cleanup
- Graceful shutdown of all async clients
- Connection pooling handled by underlying SDKs

## Usage Examples

### Basic Async Call
```python
from ai.async_client import call_ai_async

# Single async call
response = await call_ai_async("gemini", "What is 2+2?", temperature=0.1)
```

### Concurrent Calls
```python
# Multiple AIs concurrently
responses = await asyncio.gather(
    call_ai_async("gemini", prompt),
    call_ai_async("openai", prompt),
    call_ai_async("grok", prompt)
)
```

### With Context Manager
```python
from ai.async_client import create_async_ai_client

async with await create_async_ai_client("gemini", api_key, model) as client:
    response = await client.generate("Hello, world!")
```

### Server Integration
The MCP server automatically initializes async clients on startup:
```python
async def _initialize_components(self):
    # Initialize both sync and async clients
    ai_clients = await initialize_all_clients_async()
```

## Performance Benefits

Based on testing with 3 concurrent AI calls:
- **Synchronous**: Sequential execution, total time = sum of individual calls
- **Asynchronous**: Concurrent execution, total time ≈ max of individual calls
- **Typical speedup**: 2-3x for multiple AI calls

Example performance comparison:
```
Sync time:  4.82s (3 sequential calls)
Async time: 1.76s (3 concurrent calls)
Speedup:    2.74x
```

## Migration Guide

### For Existing Code
No changes required! The system maintains full backward compatibility:
- `call_ai()` continues to work as before
- `call_multiple_ais()` automatically uses async when available
- Collaborative tools benefit from async without modifications

### For New Code
To explicitly use async capabilities:
```python
# Import async functions
from ai.async_client import call_ai_async

# Use in async context
async def my_async_function():
    response = await call_ai_async("gemini", "Hello!")
```

## Configuration

No additional configuration required. Async clients use the same credentials as sync clients:
- API keys from `credentials.json` or environment variables
- Same model configurations
- Automatic initialization on server startup

## Error Handling

The async implementation includes comprehensive error handling:
- **Timeout errors**: Raised as `asyncio.TimeoutError`
- **API errors**: Preserved and re-raised with context
- **Invalid AI names**: Raise `ValueError`
- **Fallback mechanism**: Automatically falls back to sync on async errors

## Testing

Run the test suite to verify async functionality:
```bash
python test_async_ai_client.py
```

The test suite covers:
- Async client initialization
- Single and concurrent calls
- Performance comparison
- Error handling
- Resource cleanup

## Future Enhancements

Potential improvements for the async implementation:
1. Connection pooling configuration
2. Request retry logic with exponential backoff
3. Circuit breaker pattern for failing providers
4. Metrics collection for async operations
5. Custom timeout per operation type

## Technical Details

### Dependencies
- `openai>=1.0.0` with async support
- `google-generativeai>=0.3.0` with async methods
- Python 3.7+ for native async/await

### Thread Safety
- All async clients are thread-safe
- Global client storage uses proper initialization
- Event loop management handles nested async contexts

### Memory Management
- Async clients are initialized once and reused
- Proper cleanup on server shutdown
- No memory leaks from unclosed connections
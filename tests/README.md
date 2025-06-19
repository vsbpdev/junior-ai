# Junior AI Test Suite

## Overview

This directory contains the comprehensive test suite for the Junior AI MCP server. The tests are organized by type and follow a consistent structure to ensure maintainability and clarity.

## Test Structure

```text
tests/
├── unit/                   # Unit tests for individual components
│   ├── core/              # Core module tests
│   ├── ai/                # AI client and response tests
│   ├── handlers/          # Tool handler tests
│   └── pattern/           # Pattern detection tests
├── integration/           # Integration tests for system interactions
├── performance/           # Performance benchmarks and load tests
├── security/              # Security vulnerability tests
└── conftest.py            # Shared pytest fixtures and configuration
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov
```

### Run specific test categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Tests that don't require AI API keys
pytest -m "not ai_required"

# Fast tests only (exclude slow tests)
pytest -m "not slow"
```

### Run tests for specific modules
```bash
# Test only core modules
pytest tests/unit/core/

# Test only AI modules
pytest tests/unit/ai/

# Test only pattern detection
pytest tests/unit/pattern/
```

### Run with different verbosity levels
```bash
# Quiet mode
pytest -q

# Verbose mode
pytest -v

# Very verbose mode
pytest -vv
```

## Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests that test individual components
- `@pytest.mark.integration` - Integration tests that test component interactions
- `@pytest.mark.performance` - Performance benchmarks and load tests
- `@pytest.mark.security` - Security vulnerability tests
- `@pytest.mark.slow` - Tests that take more than 5 seconds
- `@pytest.mark.ai_required` - Tests that require AI API keys to be configured
- `@pytest.mark.mcp` - Tests for MCP protocol functionality

## Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, patch

from server_modules.core.config import ConfigManager

@pytest.mark.unit
class TestConfigManager:
    """Test suite for ConfigManager"""
    
    def test_load_config_success(self, temp_config_file):
        """Test successful config loading"""
        config = ConfigManager(temp_config_file)
        assert config.get("model") == "gpt-4"
        
    def test_load_config_missing_file(self):
        """Test config loading with missing file"""
        with pytest.raises(FileNotFoundError):
            ConfigManager("nonexistent.json")
```

### Integration Test Example

```python
import pytest
import asyncio

@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_protocol_flow():
    """Test complete MCP protocol flow"""
    # Test implementation
    pass
```

### Performance Test Example

```python
import pytest
import time

@pytest.mark.performance
@pytest.mark.slow
def test_pattern_detection_performance(benchmark):
    """Benchmark pattern detection performance"""
    result = benchmark(detect_patterns, large_text_sample)
    assert result.stats['mean'] < 0.1  # Should complete in under 100ms
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_config_file` - Creates a temporary configuration file
- `mock_ai_client` - Provides a mocked AI client
- `sample_patterns` - Provides sample pattern detection data
- `mcp_server` - Creates a test MCP server instance

## Coverage Requirements

- Overall coverage target: 80%
- Critical modules (core, handlers): 90%
- AI modules: 75% (due to external dependencies)
- Pattern detection: 85%

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Clear Naming**: Use descriptive test names that explain what is being tested
3. **Arrange-Act-Assert**: Follow the AAA pattern for test structure
4. **Mock External Dependencies**: Always mock AI API calls and external services
5. **Test Edge Cases**: Include tests for error conditions and edge cases
6. **Performance Awareness**: Mark slow tests appropriately
7. **Documentation**: Add docstrings to complex test cases

## Debugging Tests

### Run tests with debugging output
```bash
pytest -vv --log-cli-level=DEBUG
```

### Run tests with pdb on failure
```bash
pytest --pdb
```

### Run only the last failed tests
```bash
pytest --lf
```

### Run tests in parallel (requires pytest-xdist)
```bash
pytest -n auto
```

## CI/CD Integration

Tests are automatically run on:
- Every pull request
- Every push to main branch
- Scheduled weekly security scans

The CI pipeline enforces:
- All tests must pass
- Code coverage must be at least 80%
- No security vulnerabilities in dependencies
- All code quality checks pass (flake8, mypy, black)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running pytest from the project root
2. **API Key Errors**: Set test API keys or skip with `-m "not ai_required"`
3. **Timeout Errors**: Increase timeout in pytest.ini or mark test as slow
4. **Coverage Issues**: Check .coveragerc for excluded files

### Getting Help

- Check test output for detailed error messages
- Run with `-vv` for more verbose output
- Use `--tb=long` for full tracebacks
- Check CI logs for environment-specific issues
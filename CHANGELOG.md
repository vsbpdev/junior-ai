# Changelog

All notable changes to Junior AI Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Critical initialization order bug in pattern_detection.py that prevented configuration from loading
- Thread-safe credential manager caching to eliminate redundant instances and improve performance
- Test cleanup error handling to prevent AttributeError when setUp fails
- Missing logger initialization in _parse_sensitivity_config method
- EnvironmentProvider now returns complete sensitivity configuration

### Added
- Test dependencies file (requirements-test.txt) with pytest, pytest-asyncio, and pytest-cov
- Comprehensive test infrastructure status report
- CodeRabbit review documentation system

### Changed
- Test configuration now creates files in current directory to comply with security validation
- Removed unused imports from test files for cleaner code

### Performance
- Credential manager now uses singleton pattern, reducing initialization overhead by ~0.2-0.5ms per config reload
- Thread-safe implementation prevents race conditions in multi-threaded environments

## [2.1.0] - Previous Release

### Added
- Pattern detection system with 5 categories
- AI consultation manager
- Response synthesis system
- Manual override controls
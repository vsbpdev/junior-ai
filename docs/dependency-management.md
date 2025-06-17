# Dependency Management

## Overview

Junior AI uses Python dependencies for AI integrations and security features. Dependencies are managed through pip and separated into production and development requirements.

## Requirements Files

### `requirements.txt`
Production dependencies required for running the MCP server:
- **AI SDKs**: google-generativeai, openai
- **Security**: bcrypt, argon2-cffi, cryptography, PyJWT
- **Utilities**: python-dotenv, requests, psutil

### `requirements-dev.txt`
Development dependencies for testing and code quality:
- **Testing**: pytest, pytest-asyncio, pytest-cov, pytest-mock
- **Code Quality**: mypy, black, flake8, isort
- **Development**: ipython, ipdb
- **Documentation**: sphinx, sphinx-rtd-theme
- **Security Scanning**: bandit, safety

## Installation

### Production Setup
```bash
pip3 install -r requirements.txt
```

### Development Setup
```bash
pip3 install -r requirements-dev.txt
```

**Note**: The `requirements-dev.txt` file includes all production dependencies via `-r requirements.txt`, so you only need to run this single command to install both production and development dependencies.

Or use the setup script:
```bash
./setup.sh --dev
```

## Security Scanning

Dependencies are automatically scanned for vulnerabilities:

1. **GitHub Actions**: Weekly automated scans using Safety and Bandit
2. **Local Scanning**:
   ```bash
   # Basic usage
   safety check
   bandit -r .
   
   # Generate JSON reports (same as CI)
   safety check --json > safety-report.json
   bandit -r . -f json -o bandit-report.json
   ```

## Updating Dependencies

1. Update version in requirements file
2. Test thoroughly in development
3. Run security scans
4. Create PR with changes

## Version Strategy

- Use flexible versioning (>=) for better compatibility
- Pin major versions to avoid breaking changes
- Regular updates for security patches

## Troubleshooting

### Dependency Conflicts
If you encounter conflicts (like with cryptography versions):
1. Check which packages require the conflicting dependency
2. Find a compatible version range
3. Update requirements.txt accordingly

### Platform-Specific Issues
Some dependencies (psutil, bcrypt) have platform-specific builds. If installation fails:
1. Ensure you have build tools installed
2. Try upgrading pip: `pip3 install --upgrade pip`
3. Check platform-specific documentation
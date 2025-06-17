# Secure Credential Management

Junior AI Assistant now supports multiple secure storage backends for API keys and sensitive configuration, replacing the previous plain text `credentials.json` approach.

## Overview

The secure credential system provides:
- **Multiple storage backends** with automatic fallback
- **Encryption at rest** for sensitive data
- **OS integration** via system keyrings
- **Environment variable support** for containerized deployments
- **Backward compatibility** with existing `credentials.json` files
- **Automatic migration** tools

## Storage Backends

### 1. Environment Variables (Highest Priority)
Best for: Docker containers, CI/CD, cloud deployments

```bash
export JUNIOR_AI_GEMINI_API_KEY="your-api-key"
export JUNIOR_AI_GEMINI_ENABLED=true
export JUNIOR_AI_GEMINI_MODEL="gemini-2.0-flash"
```

**Advantages:**
- No files to manage
- Native support in cloud platforms
- Easy to rotate secrets
- Works with `.env` files

### 2. OS Keyring/Keychain
Best for: Desktop applications, maximum security

**Advantages:**
- Uses system's secure credential storage
- Encrypted by the OS
- Protected by user login
- No keys in memory

**Supported platforms:**
- macOS Keychain
- Windows Credential Manager
- Linux Secret Service (GNOME Keyring, KWallet)

### 3. Encrypted File Storage
Best for: Portable installations, shared systems

**Advantages:**
- AES encryption via Fernet
- Portable across systems
- Single file to backup
- Key derivation with PBKDF2

### 4. Plain JSON (Lowest Priority)
For: Backward compatibility only

**Warning:** Stores API keys in plain text. Migrate to secure storage ASAP.

## Quick Start

### New Installation

1. **Secure Setup (Recommended)**
   ```bash
   ./setup.sh --secure
   ```
   
   This will:
   - Install dependencies
   - Prompt for storage method
   - Guide you through configuration

2. **Manual Configuration**

   **Option A: Environment Variables**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your API keys
   nano .env
   ```

   **Option B: Direct Export**
   ```bash
   export JUNIOR_AI_GEMINI_API_KEY="your-key"
   export JUNIOR_AI_PATTERN_ENABLED=true
   ```

### Migrating Existing Credentials

If you have an existing `credentials.json`:

```bash
# Check current security status
python3 migrate_credentials.py --check-only

# Migrate to environment variables (recommended)
python3 migrate_credentials.py --target env

# Migrate to OS keyring
python3 migrate_credentials.py --target keyring

# Migrate to encrypted file
python3 migrate_credentials.py --target encrypted
```

## Environment Variable Reference

### AI Provider Configuration

Each AI provider uses the following pattern:
- `JUNIOR_AI_{PROVIDER}_API_KEY` - API key (required)
- `JUNIOR_AI_{PROVIDER}_ENABLED` - Enable/disable (true/false)
- `JUNIOR_AI_{PROVIDER}_MODEL` - Model selection
- `JUNIOR_AI_{PROVIDER}_BASE_URL` - Custom API endpoint (if applicable)

#### Gemini
```bash
JUNIOR_AI_GEMINI_API_KEY=your-gemini-key
JUNIOR_AI_GEMINI_ENABLED=true
JUNIOR_AI_GEMINI_MODEL=gemini-2.0-flash
```

#### Grok
```bash
JUNIOR_AI_GROK_API_KEY=your-grok-key
JUNIOR_AI_GROK_ENABLED=true
JUNIOR_AI_GROK_MODEL=grok-3
JUNIOR_AI_GROK_BASE_URL=https://api.x.ai/v1
```

#### OpenAI
```bash
JUNIOR_AI_OPENAI_API_KEY=your-openai-key
JUNIOR_AI_OPENAI_ENABLED=true
JUNIOR_AI_OPENAI_MODEL=gpt-4o
```

#### DeepSeek
```bash
JUNIOR_AI_DEEPSEEK_API_KEY=your-deepseek-key
JUNIOR_AI_DEEPSEEK_ENABLED=true
JUNIOR_AI_DEEPSEEK_MODEL=deepseek-chat
JUNIOR_AI_DEEPSEEK_BASE_URL=https://api.deepseek.com
```

#### OpenRouter
```bash
JUNIOR_AI_OPENROUTER_API_KEY=your-openrouter-key
JUNIOR_AI_OPENROUTER_ENABLED=true
JUNIOR_AI_OPENROUTER_MODEL=openai/gpt-4o
JUNIOR_AI_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### Pattern Detection Configuration

```bash
JUNIOR_AI_PATTERN_ENABLED=true
JUNIOR_AI_PATTERN_SENSITIVITY=medium  # low, medium, high, maximum
```

## Security Best Practices

### 1. Never Commit Secrets
Add to `.gitignore`:
```
.env
.env.local
credentials.json
*.enc
*.key
```

### 2. Use Appropriate Storage

| Deployment Type | Recommended Storage |
|----------------|-------------------|
| Local Development | Environment variables (.env) |
| Desktop App | OS Keyring |
| Docker/K8s | Environment variables |
| CI/CD | Secret management service |
| Shared Server | Encrypted file |

### 3. Rotate Keys Regularly
- Set reminders to rotate API keys
- Use different keys for dev/prod
- Monitor key usage

### 4. Principle of Least Privilege
- Only enable AIs you actually use
- Use read-only keys where possible
- Restrict key permissions by IP/domain

## Troubleshooting

### "No credentials found"
1. Check environment variables are set
2. Verify `.env` file is in the correct location
3. Run security check: `python3 migrate_credentials.py --check-only`

### "Keyring not available"
- Install keyring: `pip install keyring`
- On Linux, install: `sudo apt install python3-keyring`
- Fallback to environment variables if issues persist

### "Permission denied" on encrypted file
- Check file permissions: `ls -la ~/.junior-ai/`
- Ensure you own the files: `chown $USER ~/.junior-ai/*`

### Migration fails
1. Backup your `credentials.json`
2. Check you have write permissions
3. Try a different target backend
4. Report issue with error message

## Advanced Configuration

### Custom Credential Locations

```python
from secure_credentials import SecureCredentialManager

# Use custom encrypted file location
manager = SecureCredentialManager()
manager.providers[CredentialBackend.ENCRYPTED_FILE] = EncryptedFileProvider(
    Path.home() / ".config" / "junior-ai" / "creds.enc"
)
```

### Programmatic Credential Management

```python
from secure_credentials import SecureCredentialManager, CredentialBackend

# Create manager
manager = SecureCredentialManager()

# Load credentials
creds = manager.load_credentials()

# Save to specific backend
manager.save_credentials(
    {"gemini": {"api_key": "new-key"}},
    backend=CredentialBackend.KEYRING
)

# Check security
from secure_credentials import check_credential_security
assessment = check_credential_security()
print(f"Security level: {assessment['security_level']}")
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Use environment variables for credentials
ENV JUNIOR_AI_GEMINI_ENABLED=true
# Don't put actual keys in Dockerfile!

CMD ["python", "server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  junior-ai:
    build: .
    environment:
      - JUNIOR_AI_GEMINI_API_KEY=${GEMINI_API_KEY}
      - JUNIOR_AI_PATTERN_ENABLED=true
    env_file:
      - .env
```

## API Reference

### SecureCredentialManager

Main class for credential management:

```python
class SecureCredentialManager:
    def load_credentials(self) -> Dict[str, Any]
    def save_credentials(self, credentials: Dict[str, Any], 
                        backend: Optional[CredentialBackend] = None) -> bool
    def migrate_from_plain_json(self, 
                               target_backend: Optional[CredentialBackend] = None,
                               delete_original: bool = False) -> bool
    def get_available_backends(self) -> List[CredentialBackend]
    def get_active_backend(self) -> Optional[CredentialBackend]
```

### Credential Providers

All providers implement:

```python
class CredentialProvider(ABC):
    def load(self, key: str) -> Optional[Dict[str, Any]]
    def save(self, key: str, credentials: Dict[str, Any]) -> bool
    def delete(self, key: str) -> bool
    def is_available(self) -> bool
```

## Contributing

When adding new features:
1. Maintain backward compatibility
2. Add tests for new storage backends
3. Update migration script
4. Document environment variables
5. Consider security implications

## Security Reporting

Found a security issue? Please report privately to the maintainers rather than opening a public issue.
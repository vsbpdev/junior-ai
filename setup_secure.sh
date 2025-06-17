#!/bin/bash
# Junior AI Assistant Secure Setup Script

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Junior AI Assistant Secure Setup${NC}"
echo "Connect Claude Code with multiple AI assistants using secure credential storage!"
echo ""

# Check Python version
echo "📋 Checking requirements..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Check Claude Code
if ! command -v claude &> /dev/null; then
    echo -e "${RED}❌ Claude Code CLI not found. Please install it first:${NC}"
    echo "npm install -g @anthropic-ai/claude-code"
    exit 1
fi
echo "✅ Claude Code CLI found"

# Create directory
echo ""
echo "📁 Creating MCP server directory..."
mkdir -p ~/.claude-mcp-servers/junior-ai

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy server files
echo "📋 Installing server..."
cp "$SCRIPT_DIR/server.py" ~/.claude-mcp-servers/junior-ai/
cp "$SCRIPT_DIR/secure_credentials.py" ~/.claude-mcp-servers/junior-ai/

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
echo "   Installing production dependencies..."
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Install optional security dependencies
echo ""
echo "📦 Installing optional security dependencies..."
pip3 install keyring cryptography python-dotenv || echo "Some optional dependencies failed to install"

# Check if development mode
if [ "$1" = "--dev" ]; then
    echo "   Installing development dependencies..."
    pip3 install -r "$SCRIPT_DIR/requirements-dev.txt"
    echo -e "${GREEN}✅ Development dependencies installed${NC}"
fi

# Check for existing credentials
echo ""
echo "🔒 Checking credential security..."
SECURITY_CHECK=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from secure_credentials import check_credential_security
assessment = check_credential_security()
print(f\"Current backend: {assessment.get('active_backend', 'None')}|Security level: {assessment.get('security_level', 'None')}|Plain JSON exists: {assessment.get('plain_json_exists', False)}\")
")

IFS='|' read -r CURRENT_BACKEND SECURITY_LEVEL PLAIN_JSON_EXISTS <<< "$SECURITY_CHECK"

echo "   Current backend: $CURRENT_BACKEND"
echo "   Security level: $SECURITY_LEVEL"

# If plain JSON exists, offer migration
if [[ "$PLAIN_JSON_EXISTS" == *"True"* ]]; then
    echo ""
    echo -e "${YELLOW}⚠️  Found existing credentials.json with plain text API keys${NC}"
    echo ""
    echo "Choose a secure storage option:"
    echo "1) Environment variables (.env file) - Recommended for most users"
    echo "2) OS Keyring - Best security, requires keyring support"
    echo "3) Encrypted file - Good security, portable"
    echo "4) Keep using plain JSON (not recommended)"
    echo ""
    read -p "Select option (1-4): " STORAGE_CHOICE
    
    case $STORAGE_CHOICE in
        1)
            echo "🔄 Migrating to environment variables..."
            python3 "$SCRIPT_DIR/migrate_credentials.py" --target env
            echo -e "${GREEN}✅ Migration complete! Credentials now stored in .env file${NC}"
            ;;
        2)
            echo "🔄 Migrating to OS keyring..."
            python3 "$SCRIPT_DIR/migrate_credentials.py" --target keyring
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Migration complete! Credentials now stored in OS keyring${NC}"
            else
                echo -e "${YELLOW}⚠️  Keyring not available, falling back to .env file${NC}"
                python3 "$SCRIPT_DIR/migrate_credentials.py" --target env
            fi
            ;;
        3)
            echo "🔄 Migrating to encrypted file..."
            python3 "$SCRIPT_DIR/migrate_credentials.py" --target encrypted
            echo -e "${GREEN}✅ Migration complete! Credentials now encrypted${NC}"
            ;;
        4)
            echo -e "${YELLOW}⚠️  Continuing with plain JSON (not recommended)${NC}"
            ;;
    esac
else
    # No existing credentials, set up new ones
    echo ""
    echo "🔧 Setting up secure credentials..."
    echo ""
    echo "Choose credential storage method:"
    echo "1) Environment variables (.env file) - Recommended"
    echo "2) OS Keyring - Most secure"
    echo "3) Encrypted file - Portable and secure"
    echo ""
    read -p "Select option (1-3): " NEW_STORAGE_CHOICE
    
    # Create a template .env file or use other storage
    case $NEW_STORAGE_CHOICE in
        1|"")
            # Create .env template
            cat > "$SCRIPT_DIR/.env" << 'EOF'
# Junior AI Assistant Environment Variables

# Gemini Configuration
# Get free API key from https://aistudio.google.com/apikey
JUNIOR_AI_GEMINI_API_KEY=
JUNIOR_AI_GEMINI_ENABLED=false
JUNIOR_AI_GEMINI_MODEL=gemini-2.0-flash

# Grok Configuration
# Get API key from https://console.x.ai/
JUNIOR_AI_GROK_API_KEY=
JUNIOR_AI_GROK_ENABLED=false
JUNIOR_AI_GROK_MODEL=grok-3
JUNIOR_AI_GROK_BASE_URL=https://api.x.ai/v1

# OpenAI Configuration
# Get API key from https://platform.openai.com/api-keys
JUNIOR_AI_OPENAI_API_KEY=
JUNIOR_AI_OPENAI_ENABLED=false
JUNIOR_AI_OPENAI_MODEL=gpt-4o

# DeepSeek Configuration
# Get API key from https://platform.deepseek.com/
JUNIOR_AI_DEEPSEEK_API_KEY=
JUNIOR_AI_DEEPSEEK_ENABLED=false
JUNIOR_AI_DEEPSEEK_MODEL=deepseek-chat
JUNIOR_AI_DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenRouter Configuration
# Get API key from https://openrouter.ai/keys
JUNIOR_AI_OPENROUTER_API_KEY=
JUNIOR_AI_OPENROUTER_ENABLED=false
JUNIOR_AI_OPENROUTER_MODEL=openai/gpt-4o
JUNIOR_AI_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Pattern Detection Configuration
JUNIOR_AI_PATTERN_ENABLED=true
JUNIOR_AI_PATTERN_SENSITIVITY=medium
EOF
            echo -e "${GREEN}✅ Created .env template${NC}"
            echo ""
            echo "📝 Please edit .env file and add your API keys"
            echo "   Location: $SCRIPT_DIR/.env"
            ;;
        2)
            echo "📝 OS Keyring selected - API keys will be stored securely"
            echo "   You'll be prompted for credentials when you first run the server"
            ;;
        3)
            echo "📝 Encrypted storage selected - API keys will be encrypted"
            echo "   You'll be prompted for credentials when you first run the server"
            ;;
    esac
fi

# Copy all required Python files
echo ""
echo "📋 Installing all required Python modules..."
PYTHON_FILES=(
    "pattern_detection.py"
    "text_processing_pipeline.py"
    "response_handlers.py"
    "context_aware_matching.py"
    "ai_consultation_manager.py"
    "async_pattern_cache.py"
    "async_cached_pattern_engine.py"
    "response_synthesis.py"
    "pattern_cache.py"
    "performance_optimizations.py"
)

for file in "${PYTHON_FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        cp "$SCRIPT_DIR/$file" ~/.claude-mcp-servers/junior-ai/
        echo "   ✅ Installed $file"
    else
        echo "   ⚠️  Warning: $file not found, some features may not work"
    fi
done

# Also copy the migration script for future use
cp "$SCRIPT_DIR/migrate_credentials.py" ~/.claude-mcp-servers/junior-ai/

# Copy credentials template as fallback
if [ ! -f ~/.claude-mcp-servers/junior-ai/credentials.json ] && [ ! -f ~/.claude-mcp-servers/junior-ai/.env ]; then
    cp "$SCRIPT_DIR/credentials.template.json" ~/.claude-mcp-servers/junior-ai/credentials.json
    echo "📄 Created credentials.json from template (for backward compatibility)"
fi

# Register with Claude Code
echo ""
echo "📝 Registering Junior AI Assistant with Claude Code..."
claude mcp add --scope user junior-ai "python3" "$HOME/.claude-mcp-servers/junior-ai/server.py"

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "📌 Next steps:"
if [[ "$NEW_STORAGE_CHOICE" == "1" ]] || [[ "$STORAGE_CHOICE" == "1" ]]; then
    echo "   1. Edit .env file and add your API keys:"
    echo "      $SCRIPT_DIR/.env"
    echo "   2. Restart Claude Code to activate the server"
else
    echo "   1. Restart Claude Code to activate the server"
    echo "   2. The server will prompt for credentials on first run"
fi
echo ""
echo "💡 Security tips:"
echo "   - Never commit .env files to version control"
echo "   - Use 'python3 migrate_credentials.py --check-only' to check security"
echo "   - Consider using OS keyring for maximum security"
echo ""
echo "🔧 Test the connection after restarting Claude Code:"
echo "   In Claude Code, try: 'What AI models are available?'"
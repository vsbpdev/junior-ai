#!/bin/bash
# Multi-AI MCP Server Setup Script

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Multi-AI MCP Server Setup${NC}"
echo "Connect Claude Code with Gemini, Grok-3, ChatGPT, and DeepSeek!"
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
mkdir -p ~/.claude-mcp-servers/multi-ai-collab

# Copy server files
echo "📋 Installing server..."
cp server.py ~/.claude-mcp-servers/multi-ai-collab/

# Create credentials.json from template if it doesn't exist
if [ ! -f ~/.claude-mcp-servers/multi-ai-collab/credentials.json ]; then
    cp credentials.template.json ~/.claude-mcp-servers/multi-ai-collab/credentials.json
    echo "📄 Created credentials.json from template"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt --quiet

# Function to prompt for API key
prompt_for_key() {
    local service_name="$1"
    local current_key="$2"
    local description="$3"
    
    if [[ "$current_key" == *"YOUR_"*"_KEY_HERE" ]]; then
        echo -e "${YELLOW}🔑 $service_name API Key needed${NC}"
        echo "   $description"
        read -p "Enter $service_name API key (or press Enter to skip): " new_key
        if [ ! -z "$new_key" ]; then
            # Update credentials.json with new key
            python3 -c "
import json
with open('$HOME/.claude-mcp-servers/multi-ai-collab/credentials.json', 'r') as f:
    creds = json.load(f)
creds['$(echo $service_name | tr '[:upper:]' '[:lower:]')']['api_key'] = '$new_key'
creds['$(echo $service_name | tr '[:upper:]' '[:lower:]')']['enabled'] = True
with open('$HOME/.claude-mcp-servers/multi-ai-collab/credentials.json', 'w') as f:
    json.dump(creds, f, indent=2)
"
            echo -e "${GREEN}✅ $service_name API key configured${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name skipped (can be configured later)${NC}"
        fi
    else
        echo -e "${GREEN}✅ $service_name API key already configured${NC}"
    fi
}

# Configure API keys
echo ""
echo "🔧 Configuring API keys..."

# Read current credentials
GEMINI_KEY=$(python3 -c "import json; f=open('$HOME/.claude-mcp-servers/multi-ai-collab/credentials.json'); print(json.load(f)['gemini']['api_key'])")
GROK_KEY=$(python3 -c "import json; f=open('$HOME/.claude-mcp-servers/multi-ai-collab/credentials.json'); print(json.load(f)['grok']['api_key'])")
OPENAI_KEY=$(python3 -c "import json; f=open('$HOME/.claude-mcp-servers/multi-ai-collab/credentials.json'); print(json.load(f)['openai']['api_key'])")

# Prompt for missing keys
prompt_for_key "Gemini" "$GEMINI_KEY" "Get free key from: https://aistudio.google.com/apikey"
prompt_for_key "Grok" "$GROK_KEY" "Get key from: https://console.x.ai/"
prompt_for_key "OpenAI" "$OPENAI_KEY" "Get key from: https://platform.openai.com/api-keys"

echo ""
echo "🔧 Configuring Claude Code..."
# Remove any existing MCP configuration
claude mcp remove multi-ai-collab 2>/dev/null || true

# Add MCP server with global scope
claude mcp add --scope user multi-ai-collab python3 ~/.claude-mcp-servers/multi-ai-collab/server.py

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "🎉 Multi-AI MCP Server is ready!"
echo ""
echo "Available AIs:"
echo "  • 🤖 Gemini (Google)"
echo "  • 🚀 Grok-3 (xAI)"
[[ "$OPENAI_KEY" != *"YOUR_"*"_KEY_HERE" ]] && echo "  • 🧠 ChatGPT (OpenAI)"
echo "  • 🔮 DeepSeek (when available)"
echo ""
echo "Try it out:"
echo "  1. Run: claude"
echo "  2. Type: /mcp (should show multi-ai-collab connected)"
echo "  3. Use: mcp__multi-ai-collab__ask_gemini"
echo "         prompt: \"Hello from Claude!\""
echo ""
echo "Collaborative tools:"
echo "  • mcp__multi-ai-collab__ask_all_ais - Ask all AIs the same question"
echo "  • mcp__multi-ai-collab__ai_debate - Have two AIs debate a topic"
echo "  • mcp__multi-ai-collab__server_status - Check which AIs are available"
echo ""
echo "Individual AI tools:"
echo "  • mcp__multi-ai-collab__ask_[ai_name]"
echo "  • mcp__multi-ai-collab__[ai_name]_code_review"
echo ""
echo "🔧 To add more API keys later, edit:"
echo "   ~/.claude-mcp-servers/multi-ai-collab/credentials.json"
echo ""
echo "Enjoy the AI collaboration! 🚀"
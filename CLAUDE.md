# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Setup
```bash
# Initial setup - runs interactive configuration for API keys
chmod +x setup.sh
./setup.sh

# Install Python dependencies
pip3 install -r requirements.txt

# Manual MCP server configuration
claude mcp add --scope user junior-ai python3 ~/.claude-mcp-servers/junior-ai/server.py
```

### Running and Testing
```bash
# Check MCP server status
claude mcp list

# Test the server is working (from Claude Code)
mcp__junior-ai__server_status

# Run server manually for debugging
python3 ~/.claude-mcp-servers/junior-ai/server.py
```

## Architecture

This codebase implements an MCP (Model Context Protocol) server that enables Claude Code to communicate with multiple AI models:

### Core Components

1. **server.py** - Main MCP server implementation
   - Handles JSON-RPC protocol for MCP communication
   - Dynamically loads AI clients based on credentials configuration
   - Implements tools for individual AI interactions and collaborative features
   - Error handling and graceful degradation when AIs are unavailable

2. **credentials.json** - API configuration (not in repo, created from template)
   - Stores API keys and model selections for each AI service
   - Supports enabling/disabling individual AIs
   - Located at: `~/.claude-mcp-servers/junior-ai/credentials.json`

3. **setup.sh** - Automated installation script
   - Interactive API key configuration
   - Handles Python dependency installation
   - Configures Claude Code MCP integration

### AI Integration Pattern

The server uses a unified interface for different AI providers:
- **Gemini**: Uses `google.generativeai` SDK
- **Grok, OpenAI, DeepSeek, OpenRouter**: All use OpenAI-compatible API client
- Each AI has 6 specialized tools: ask, code_review, think_deep, brainstorm, debug, architecture
- Collaborative tools work with any combination of available AIs
- **OpenRouter** provides access to multiple models through a single API endpoint

### Tool Generation

Tools are dynamically generated based on available AI clients:
- Individual AI tools are created for each configured AI
- Collaborative tools (ask_all_ais, ai_debate, etc.) only appear when 2+ AIs are available
- Tool names follow pattern: `{ai_name}_{action}` for consistency

### Error Handling

The server implements multiple levels of error handling:
- Graceful initialization failures for individual AIs
- Per-tool error catching with informative messages
- JSON-RPC protocol compliance for Claude Code integration
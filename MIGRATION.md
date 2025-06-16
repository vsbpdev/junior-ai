# Migration Guide: From Multi-AI Collab to Junior AI Assistant

## Overview

The Multi-AI MCP Server has been rebranded to **Junior AI Assistant for Claude Code**. This guide helps you migrate your existing setup.

## What's Changed

### 1. Server Name
- **Old**: `multi-ai-collab`
- **New**: `junior-ai`

### 2. Tool Prefixes
All tool names have changed:
- **Old**: `mcp__multi-ai-collab__<tool_name>`
- **New**: `mcp__junior-ai__<tool_name>`

### 3. Installation Path
- **Old**: `~/.claude-mcp-servers/multi-ai-collab/`
- **New**: `~/.claude-mcp-servers/junior-ai/`

## Migration Steps

### Step 1: Remove Old Installation
```bash
# Remove old MCP configuration
claude mcp remove multi-ai-collab

# Optional: Backup your credentials
cp ~/.claude-mcp-servers/multi-ai-collab/credentials.json ~/credentials_backup.json
```

### Step 2: Install Junior AI Assistant
```bash
# Clone the new repository
git clone https://github.com/RaiAnsar/claude_code-multi-AI-MCP.git
cd claude_code-multi-AI-MCP

# Run setup
chmod +x setup.sh
./setup.sh
```

### Step 3: Restore Credentials (if backed up)
```bash
cp ~/credentials_backup.json ~/.claude-mcp-servers/junior-ai/credentials.json
```

### Step 4: Update Your Workflows

Replace all tool calls in your scripts or documentation:

| Old Tool Name | New Tool Name |
|--------------|---------------|
| `mcp__multi-ai-collab__server_status` | `mcp__junior-ai__server_status` |
| `mcp__multi-ai-collab__ask_gemini` | `mcp__junior-ai__ask_gemini` |
| `mcp__multi-ai-collab__ask_grok` | `mcp__junior-ai__ask_grok` |
| `mcp__multi-ai-collab__ask_openai` | `mcp__junior-ai__ask_openai` |
| `mcp__multi-ai-collab__ask_deepseek` | `mcp__junior-ai__ask_deepseek` |
| `mcp__multi-ai-collab__ask_openrouter` | `mcp__junior-ai__ask_openrouter` |
| `mcp__multi-ai-collab__ask_all_ais` | `mcp__junior-ai__ask_all_ais` |
| `mcp__multi-ai-collab__ai_debate` | `mcp__junior-ai__ai_debate` |
| `mcp__multi-ai-collab__collaborative_solve` | `mcp__junior-ai__collaborative_solve` |
| `mcp__multi-ai-collab__ai_consensus` | `mcp__junior-ai__ai_consensus` |
| `mcp__multi-ai-collab__<ai>_code_review` | `mcp__junior-ai__<ai>_code_review` |
| `mcp__multi-ai-collab__<ai>_think_deep` | `mcp__junior-ai__<ai>_think_deep` |
| `mcp__multi-ai-collab__<ai>_brainstorm` | `mcp__junior-ai__<ai>_brainstorm` |
| `mcp__multi-ai-collab__<ai>_debug` | `mcp__junior-ai__<ai>_debug` |
| `mcp__multi-ai-collab__<ai>_architecture` | `mcp__junior-ai__<ai>_architecture` |

### Step 5: Verify Installation
```bash
# Check MCP list
claude mcp list
# Should show "junior-ai"

# Test the new server
# In Claude Code, run:
mcp__junior-ai__server_status
```

## Quick Migration Script

For automated find-and-replace in your files:

```bash
#!/bin/bash
# Save this as migrate_tools.sh

# Find and replace in all .md, .txt, and script files
# macOS:
find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" \) -exec sed -i '' 's/mcp__multi-ai-collab__/mcp__junior-ai__/g' {} +

# Linux:
# find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" \) -exec sed -i 's/mcp__multi-ai-collab__/mcp__junior-ai__/g' {} +

echo "Migration complete! Old tool names have been updated."
```

## Troubleshooting

### "Server not found" error
- Make sure you've run the new setup.sh script
- Verify with `claude mcp list`

### Old tools still appearing
- Restart Claude Code after installation
- Make sure you removed the old server with `claude mcp remove multi-ai-collab`

### Credentials not working
- Check that credentials.json was properly copied to the new location
- Verify API keys are still valid

## New Features Coming Soon

Junior AI Assistant will soon include:
- Automatic pattern detection for proactive assistance
- Smart AI selection based on query type
- Enhanced security and code review capabilities

## Need Help?

If you encounter issues during migration:
1. Check this guide again
2. Review the main README.md
3. Open an issue on GitHub

---

**Note**: There is no automatic backward compatibility for the old tool names due to MCP protocol limitations. You must update all references to use the new `mcp__junior-ai__` prefix.
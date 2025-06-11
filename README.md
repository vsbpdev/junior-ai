# Multi-AI MCP Server for Claude Code

🚀 **Connect Claude Code with multiple AI models simultaneously!**

Get the best insights by collaborating with Gemini, Grok-3, ChatGPT, and DeepSeek all within Claude Code. Ask multiple AIs the same question, have them debate topics, or use each AI's unique strengths.

## 🤖 Supported AI Models

- **🧠 Gemini** (Google) - Free API available ✅
- **🚀 Grok-3** (xAI) - Paid API ✅  
- **💬 ChatGPT** (OpenAI) - Paid API ✅
- **🔮 DeepSeek** - Coming soon to more regions

**💡 Flexible Setup**: You can use any combination! Have only Gemini? Perfect! Only Grok? Works great! All of them? Even better!

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Claude Code CLI installed
- API keys for the AIs you want to use

### Installation

1. **Clone this repo:**
```bash
git clone https://github.com/RaiAnsar/claude_code-multi-AI-MCP.git
cd claude_code-multi-AI-MCP
```

2. **Get API keys (for the AIs you want):**
   - **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) (Free)
   - **Grok**: [xAI Console](https://console.x.ai/) (Paid)  
   - **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys) (Paid)
   
   **Note**: You don't need all of them! Configure only the AIs you have keys for.

3. **Run setup:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup will:
- Install Python dependencies  
- Ask for your API keys (skip any you don't have)
- Configure only the AIs you want to use
- Store credentials securely locally
- Add to Claude Code globally
- Work with any combination of AIs!

## 🛠️ Usage Examples

### Ask Individual AIs
```bash
# Ask Gemini
mcp__multi-ai-collab__ask_gemini
  prompt: "Explain quantum computing"

# Ask Grok-3 
mcp__multi-ai-collab__ask_grok
  prompt: "What's the best programming language in 2025?"

# Ask ChatGPT
mcp__multi-ai-collab__ask_openai
  prompt: "Debug this Python code"
```

### Collaborative Features

**Ask All AIs (Get Multiple Perspectives):**
```bash
mcp__multi-ai-collab__ask_all_ais
  prompt: "What are the pros and cons of microservices?"
```

**AI Debate:**
```bash
mcp__multi-ai-collab__ai_debate
  topic: "Is Python or JavaScript better for beginners?"
  ai1: "gemini"
  ai2: "grok"
```

**Code Reviews:**
```bash
mcp__multi-ai-collab__gemini_code_review
  code: "def auth(user): return user.password == 'admin'"
  focus: "security"
```

**Server Status:**
```bash
mcp__multi-ai-collab__server_status
# Shows which AIs are available and configured
```

## 🔧 Configuration

### API Keys
Edit `~/.claude-mcp-servers/multi-ai-collab/credentials.json`:

```json
{
  "gemini": {
    "api_key": "your-gemini-key",
    "enabled": true
  },
  "grok": {
    "api_key": "your-grok-key", 
    "enabled": true
  },
  "openai": {
    "api_key": "your-openai-key",
    "enabled": true
  }
}
```

### Getting API Keys
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) (Free)
- **Grok**: [xAI Console](https://console.x.ai/) (Paid)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys) (Paid)

## 🌟 Why Use Multiple AIs?

- **Different Strengths**: Each AI excels in different areas
- **Diverse Perspectives**: Get varied approaches to problems
- **Quality Assurance**: Cross-check answers for accuracy
- **Specialized Tasks**: Use the best AI for each specific task
- **Learning**: Compare different AI reasoning styles

## 🔧 Partial Configurations

**Don't have all the API keys? No problem!**

- **Only Gemini?** You'll have access to Google's powerful free AI
- **Only Grok?** Get xAI's unique perspective and humor
- **Only OpenAI?** Use the well-known ChatGPT models
- **Have 2 out of 3?** Perfect for AI debates and comparisons!

The server automatically adapts to your available AIs. Tools for unavailable AIs simply won't appear in Claude Code.

## 🔒 Security Notes

- **API keys stored locally**: All credentials are in `~/.claude-mcp-servers/multi-ai-collab/credentials.json`
- **Never committed to git**: The `.gitignore` file excludes all credential files
- **Optional AIs**: Only AIs with valid keys are loaded
- **Graceful failures**: Failed AI connections don't break the server
- **Input validation**: All requests are validated before processing

⚠️ **Important**: Never share your `credentials.json` file or commit it to version control!

## 🐛 Troubleshooting

**MCP not showing up?**
```bash
claude mcp list
# Should show "multi-ai-collab"
```

**AI not responding?**
```bash
mcp__multi-ai-collab__server_status
# Check which AIs are enabled and working
```

**Connection errors?**
- Verify API keys in `credentials.json`
- Check if the AI service is down
- Ensure you have sufficient API credits

**Reinstall:**
```bash
claude mcp remove multi-ai-collab
./setup.sh
```

## 📁 File Structure

```
~/.claude-mcp-servers/multi-ai-collab/
├── server.py           # Main MCP server
├── credentials.json    # API keys and configuration
└── requirements.txt    # Python dependencies
```

## 🚀 Advanced Usage

### Custom Temperature
```bash
mcp__multi-ai-collab__ask_gemini
  prompt: "Write a creative story"
  temperature: 0.9  # Higher = more creative
```

### Focused Code Reviews
```bash
mcp__multi-ai-collab__grok_code_review
  code: "[your code]"
  focus: "performance"  # or security, readability, etc.
```

## 🤝 Contributing

This is designed to be simple and extensible. To add new AI providers:

1. Add credentials to `credentials.json`
2. Add client initialization in `server.py`
3. Test with the existing tools

## 📜 License

MIT - Use freely!

---

**🎉 Enjoy having multiple AI assistants working together with Claude Code!**
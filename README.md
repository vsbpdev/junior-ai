# Multi-AI MCP Server for Claude Code

🚀 **Connect Claude Code with multiple AI models simultaneously!**

Get the best insights by collaborating with Gemini, Grok-3, ChatGPT, and DeepSeek all within Claude Code. Ask multiple AIs the same question, have them debate topics, or use each AI's unique strengths.

## 🤖 Supported AI Models

- **🧠 Gemini** (Google) - Free API available ✅
- **🚀 Grok-3** (xAI) - Paid API ✅  
- **💬 ChatGPT** (OpenAI) - Paid API ✅
- **🔮 DeepSeek** - Coming soon to more regions

**💡 Flexible Setup**: You can use any combination! Have only Gemini? Perfect! Only Grok? Works great! All of them? Even better!

**🎛️ Model Selection**: Choose your preferred model for each AI:
- **Gemini**: `gemini-2.0-flash` (default), `gemini-2.0-flash-exp`, `gemini-1.5-pro`
- **Grok**: `grok-3` (default), `grok-2`  
- **OpenAI**: `gpt-4o` (default), `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

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
- Let you choose preferred models for each AI
- Configure only the AIs you want to use
- Store credentials securely locally
- Add to Claude Code globally
- Work with any combination of AIs!

## 🛠️ Usage Examples

### Ask Individual AIs
```bash
# Ask Gemini (Google)
mcp__multi-ai-collab__ask_gemini
  prompt: "Explain quantum computing"

# Ask Grok-3 (xAI)
mcp__multi-ai-collab__ask_grok
  prompt: "What's the best programming language in 2025?"

# Ask ChatGPT (OpenAI)
mcp__multi-ai-collab__ask_openai
  prompt: "Debug this Python code"
```

### Get Different AI Perspectives
```bash
# Compare all three AI responses
mcp__multi-ai-collab__ask_all_ais
  prompt: "What are the pros and cons of microservices architecture?"

# Result: You'll see responses from Gemini, Grok, AND ChatGPT side-by-side!
```

### AI Collaboration Features

**🤝 AI Debates - Get Multiple Perspectives:**
```bash
# Have Gemini vs ChatGPT debate
mcp__multi-ai-collab__ai_debate
  topic: "Is Python or JavaScript better for beginners?"
  ai1: "gemini"
  ai2: "openai"

# Or try Grok vs Gemini
mcp__multi-ai-collab__ai_debate
  topic: "Which is better: REST APIs or GraphQL?"
  ai1: "grok"
  ai2: "gemini"
```

**🔍 Multi-AI Code Reviews:**
```bash
# Get Gemini's perspective
mcp__multi-ai-collab__gemini_code_review
  code: "def auth(user): return user.password == 'admin'"
  focus: "security"

# Get ChatGPT's analysis
mcp__multi-ai-collab__openai_code_review
  code: "def auth(user): return user.password == 'admin'"
  focus: "security"

# Get Grok's review
mcp__multi-ai-collab__grok_code_review
  code: "def auth(user): return user.password == 'admin'"
  focus: "security"
```

**📊 Server Management:**
```bash
mcp__multi-ai-collab__server_status
# Shows: Gemini ✅, Grok ✅, ChatGPT ✅ and their models
```

## 🔧 Configuration

### API Keys & Models
Edit `~/.claude-mcp-servers/multi-ai-collab/credentials.json`:

```json
{
  "gemini": {
    "api_key": "your-gemini-key",
    "model": "gemini-2.0-flash",
    "enabled": true
  },
  "grok": {
    "api_key": "your-grok-key",
    "model": "grok-3",
    "enabled": true
  },
  "openai": {
    "api_key": "your-openai-key",
    "model": "gpt-4o",
    "enabled": true
  }
}
```

**💡 Pro Tip**: You can mix and match any combination:
- Only Gemini? Works perfectly!
- Gemini + ChatGPT? Great for comparing Google vs OpenAI perspectives!
- All three? Maximum AI collaboration power!

### Getting API Keys
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) (Free)
- **Grok**: [xAI Console](https://console.x.ai/) (Paid)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys) (Paid)

## 🌟 Why Use Multiple AIs?

- **Different Strengths**: Each AI excels in different areas
  - **Gemini**: Excellent technical accuracy and detailed explanations
  - **Grok**: Unique perspective with humor and creative solutions
  - **ChatGPT**: Balanced analysis and comprehensive code examples
- **Diverse Perspectives**: Get varied approaches to problems
- **Quality Assurance**: Cross-check answers for accuracy across all three
- **Specialized Tasks**: Use the best AI for each specific task
- **Learning**: Compare different AI reasoning styles and approaches
- **Debate Features**: Have AIs argue different sides to explore all angles

## 🔧 Partial Configurations

**Don't have all the API keys? No problem!**

- **Only Gemini?** You'll have access to Google's powerful free AI
- **Only Grok?** Get xAI's unique perspective and humor  
- **Only ChatGPT?** Use OpenAI's well-established models
- **Gemini + ChatGPT?** Compare Google vs OpenAI approaches!
- **Grok + ChatGPT?** Get both creative and analytical perspectives!
- **Have all 3?** Ultimate AI collaboration with debates and comparisons!

The server automatically adapts to your available AIs. Tools for unavailable AIs simply won't appear in Claude Code.

**💰 Cost-Effective Options:**
- Start with **free Gemini** to test the system
- Add **ChatGPT** for proven OpenAI capabilities  
- Include **Grok** for unique xAI insights

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

### Temperature Control for All AIs
```bash
# Creative writing with high temperature
mcp__multi-ai-collab__ask_gemini
  prompt: "Write a creative story about AI collaboration"
  temperature: 0.9

mcp__multi-ai-collab__ask_openai
  prompt: "Write a creative story about AI collaboration" 
  temperature: 0.9

# Technical explanations with low temperature
mcp__multi-ai-collab__ask_grok
  prompt: "Explain how TCP/IP works"
  temperature: 0.2
```

### Specialized Code Reviews by AI
```bash
# Gemini: Technical accuracy focus
mcp__multi-ai-collab__gemini_code_review
  code: "[your code]"
  focus: "technical accuracy"

# ChatGPT: Best practices focus  
mcp__multi-ai-collab__openai_code_review
  code: "[your code]"
  focus: "best practices"

# Grok: Creative solutions focus
mcp__multi-ai-collab__grok_code_review
  code: "[your code]"
  focus: "alternative approaches"
```

### Multi-AI Workflows
```bash
# Step 1: Get all perspectives
mcp__multi-ai-collab__ask_all_ais
  prompt: "How should I structure a microservices architecture?"

# Step 2: Have top AIs debate specifics
mcp__multi-ai-collab__ai_debate
  topic: "Event-driven vs REST for microservices communication"
  ai1: "gemini"
  ai2: "openai"

# Step 3: Get Grok's creative alternative
mcp__multi-ai-collab__ask_grok
  prompt: "What's a creative alternative to traditional microservices?"
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
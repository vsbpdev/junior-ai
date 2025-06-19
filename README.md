# Junior AI Assistant for Claude Code

🚀 **Your intelligent assistant that proactively helps Claude Code with pattern detection and multi-AI consultations!**

**⚡ You can use any combination - just the ones you have API keys for!**

Instead of switching between different AI websites, now you can simply ask Claude Code to get help from other AIs. Just say things like:

> "Hey Claude, ask all the AIs to help debug this code"  
> "Claude, get Grok's opinion on this architecture"  
> "Have Gemini and ChatGPT debate this technical decision"

## 🤖 Which AIs Are Included?

- **🧠 Gemini** (Google) - Free API ✅
- **🚀 Grok-3** (xAI) - Paid API ✅  
- **💬 ChatGPT** (OpenAI) - Paid API ✅
- **🔮 DeepSeek** - Paid API ✅
- **🌐 OpenRouter** - Multi-Model Gateway ✅

**💡 You don't need all of them!** Start with just Gemini (it's free), then add others if you want.

## 📋 Migrating from Multi-AI Collab?

If you're upgrading from the previous Multi-AI MCP Server, please see our **[Migration Guide](MIGRATION.md)** for step-by-step instructions.

## 🚀 5-Minute Setup

### What You Need
- Claude Code installed
- At least one API key (Gemini is free!)

### Installation (Copy & Paste These 3 Commands)

```bash
# 1. Download the code
git clone https://github.com/vsbpdev/junior-ai.git
cd junior-ai

# 2. Run the automatic setup
chmod +x setup.sh
./setup.sh

# 3. That's it! Start using it right away
```

**During setup, you'll be asked for API keys:**
- **Gemini** (Free): [Get key here](https://aistudio.google.com/apikey) 
- **Grok** (Paid): [Get key here](https://console.x.ai/) 
- **OpenAI** (Paid): [Get key here](https://platform.openai.com/api-keys)
- **DeepSeek** (Paid): [Get key here](https://platform.deepseek.com/)
- **OpenRouter** (Paid): [Get key here](https://openrouter.ai/keys)

**💡 Pro tip:** Start with just Gemini (it's free), then add others later if you want.

### Test It Works
After setup, just ask Claude naturally:

> "Hey Claude, ask Gemini what the capital of France is"

If you see a response from Gemini, you're all set! 🎉

## 💬 How to Use It (Super Simple!)

Once installed, you just talk to Claude Code normally and ask it to use the other AIs. Here are real examples:

### Ask Claude to Get Multiple Opinions
> **You:** "Hey Claude, ask all the AIs what they think about using microservices vs monolith architecture"
> 
> **Claude:** I'll ask all available AIs for their perspectives on this...
> 
> *(Claude will use the `ask_all_ais` tool and show you all available AI responses)*

### Get Help Debugging Code
> **You:** "Claude, can you have Grok help debug this Python function that's running slowly?"
> 
> **Claude:** Let me ask Grok to analyze your code for performance issues...
> 
> *(Claude will use the `grok_debug` tool)*

### Compare Different AI Opinions
> **You:** "Have Gemini and ChatGPT debate whether to use React or Vue for my frontend"
> 
> **Claude:** I'll set up a debate between Gemini and ChatGPT on this topic...
> 
> *(Claude will use the `ai_debate` tool)*

### Get Code Reviews from Multiple AIs
> **You:** "Can you ask all the AIs to review this authentication function for security issues?"
> 
> **Claude:** I'll have all available AIs review your code...
> 
> *(Claude will use multiple code_review tools)*

### Brainstorm Creative Solutions
> **You:** "Ask Grok to brainstorm some creative features for my todo app"
> 
> **Claude:** Let me get Grok's creative input on your todo app...
> 
> *(Claude will use the `grok_brainstorm` tool)*

### Get Architecture Advice
> **You:** "Claude, have Gemini help design the database schema for my e-commerce site"
> 
> **Claude:** I'll ask Gemini to provide architecture recommendations...
> 
> *(Claude will use the `gemini_architecture` tool)*

**🎉 The beauty is you don't need to remember any commands - just ask Claude naturally!**

## 🚀 Advanced Features

### 🧠 Intelligent Pattern Detection
Junior AI automatically detects patterns in your code and context:
- **Security Issues**: Passwords, API keys, encryption concerns
- **Uncertainty**: TODOs, unclear implementations, questions
- **Algorithm Complexity**: Performance bottlenecks, optimization needs
- **Common Gotchas**: Edge cases, timezone issues, async problems
- **Architecture Decisions**: Design patterns, system structure

When patterns are detected, Junior AI automatically consults the most appropriate AIs!

### 🎨 Response Synthesis System (NEW!)
Junior AI doesn't just collect AI responses - it intelligently synthesizes them using different strategies:

- **Consensus Strategy**: Finds common agreements and best practices (great for security)
- **Debate Strategy**: Highlights different perspectives and trade-offs (perfect for architecture)
- **Expert-Weighted Strategy**: Prioritizes responses based on AI expertise (ideal for algorithms)
- **Comprehensive Strategy**: Includes all perspectives thoroughly (best for uncertainty)

**Example:**
> "Ask all AIs about password security"
> 
> Junior AI will:
> 1. Detect the security pattern
> 2. Consult multiple AIs automatically
> 3. Synthesize responses using consensus strategy
> 4. Present unified recommendations with confidence scores

### 🎯 Smart AI Selection
Based on detected patterns, Junior AI automatically selects the best AIs:
- **Security → Gemini, OpenAI, Grok** (consensus approach)
- **Algorithms → DeepSeek, Gemini, OpenAI** (expert weighting)
- **Architecture → All AIs** (debate different approaches)
- **Gotchas → Gemini, Grok** (experience-based insights)

### 📊 Pattern Detection Tools
- `pattern_check`: Analyze text for patterns
- `junior_consult`: Smart AI consultation with automatic synthesis
- `pattern_stats`: View detection statistics
- `set_synthesis_strategy`: Choose how responses are combined
- `get_sensitivity_config`: View pattern detection sensitivity

## 🔧 Configuration

### API Keys & Models
Edit `~/.claude-mcp-servers/junior-ai/credentials.json`:

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
  },
  "openrouter": {
    "api_key": "your-openrouter-key",
    "model": "openai/gpt-4o",
    "enabled": true
  }
}
```

**💡 Pro Tip**: You can mix and match any combination:
- Only Gemini? Works perfectly!
- Gemini + ChatGPT? Great for comparing Google vs OpenAI perspectives!
- Any combination? Maximum AI collaboration power!

### Getting API Keys
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) (Free)
- **Grok**: [xAI Console](https://console.x.ai/) (Paid)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys) (Paid)
- **DeepSeek**: [DeepSeek Platform](https://platform.deepseek.com/) (Paid)
- **OpenRouter**: [OpenRouter Keys](https://openrouter.ai/keys) (Paid)

## 🌟 Why Have Multiple AIs?

Think of it like having a team of experts with different personalities:

- **🧠 Gemini** (Google): The technical expert
  - Great for detailed explanations and accuracy
  - Best for: Complex technical questions, code analysis
  
- **🚀 Grok** (xAI): The creative problem solver  
  - Brings unique perspectives and humor
  - Best for: Creative solutions, brainstorming, alternative approaches
  
- **💬 ChatGPT** (OpenAI): The balanced advisor
  - Comprehensive analysis and practical examples
  - Best for: General advice, code examples, balanced perspectives

- **🔮 DeepSeek**: The reasoning specialist
  - Strong in math, coding, and logical reasoning
  - Best for: Complex algorithms, mathematical problems, code optimization

- **🌐 OpenRouter**: The multi-model gateway
  - Access to multiple models through one API
  - Best for: Model comparison, fallback options, cost optimization

**Real Benefits:**
- **Better Decisions**: Get 2-3 opinions before making important choices
- **Learn Faster**: See how different AIs approach the same problem  
- **Catch Mistakes**: If one AI misses something, another might catch it
- **Save Time**: Get multiple expert opinions without switching apps

## 🔧 Partial Configurations

**Don't have all the API keys? No problem!**

- **Only Gemini?** You'll have access to Google's powerful free AI
- **Only Grok?** Get xAI's unique perspective and humor  
- **Only ChatGPT?** Use OpenAI's well-established models
- **Only DeepSeek?** Get specialized reasoning and coding help
- **Only OpenRouter?** Access multiple models through one gateway
- **Gemini + ChatGPT?** Compare Google vs OpenAI approaches!
- **Grok + DeepSeek?** Creative thinking meets logical reasoning!
- **OpenRouter + Any AI?** Fallback options and model diversity!
- **Any 3 AIs?** Excellent multi-perspective collaboration!
- **Have all 5?** Ultimate AI collaboration with maximum diversity!

The server automatically adapts to your available AIs. Tools for unavailable AIs simply won't appear in Claude Code.

**💰 Cost-Effective Options:**
- Start with **free Gemini** to test the system
- Add **ChatGPT** for proven OpenAI capabilities  
- Include **Grok** for unique xAI insights
- Add **DeepSeek** for specialized reasoning tasks
- Use **OpenRouter** for flexible model access and cost optimization

## 🔒 Security & Credential Storage

Junior AI now supports **secure credential storage** with multiple options:

### Quick Secure Setup
```bash
./setup.sh --secure  # Recommended: Uses secure storage instead of plain text
```

### Storage Options (in order of security)
1. **🔐 OS Keyring** - Most secure, uses your system's credential manager
2. **🔑 Environment Variables** - Great for Docker/cloud deployments  
3. **🔒 Encrypted File** - Portable encrypted storage
4. **⚠️ Plain JSON** - Legacy support only (not recommended)

### Migrating Existing Credentials
If you already have a `credentials.json`:
```bash
# Check your current security level
python3 migrate_credentials.py --check-only

# Migrate to secure storage (recommended)
python3 migrate_credentials.py --target env
```

### Security Best Practices
- **Never commit credentials**: The `.gitignore` excludes all credential files
- **Use environment variables** for production deployments
- **Rotate API keys** regularly
- **Enable only needed AIs**: Reduces attack surface
- **Monitor API usage**: Check your provider dashboards

### Environment Variable Setup
Create a `.env` file (see `.env.example`):
```bash
JUNIOR_AI_GEMINI_API_KEY=your-key-here
JUNIOR_AI_GEMINI_ENABLED=true
JUNIOR_AI_PATTERN_ENABLED=true
```

For detailed security documentation, see [Secure Credentials Guide](docs/secure-credentials.md).

## 🧪 Testing

Junior AI Assistant includes a comprehensive test suite to ensure reliability:

### Running Tests
```bash
# Install test dependencies
pip3 install -r requirements-test.txt

# Run all tests
pytest -v

# Run specific test file
python3 test_context_aware_matching.py

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Test Coverage
- **79% test coverage** across 91 tests
- Pattern detection engine fully tested
- AI consultation manager tested
- Response synthesis tested
- Manual override controls tested

### Test Infrastructure
Tests use local configuration files to avoid security validation issues. All test files are automatically cleaned up after test runs.

## 🐛 Troubleshooting

**MCP not showing up?**
```bash
claude mcp list
# Should show "junior-ai"
```

**AI not responding?**
```bash
mcp__junior-ai__server_status
# Check which AIs are enabled and working
```

**Connection errors?**
- Verify API keys in `credentials.json`
- Check if the AI service is down
- Ensure you have sufficient API credits

**Reinstall:**
```bash
claude mcp remove junior-ai
./setup.sh
```

## 📁 File Structure

```text
~/.claude-mcp-servers/junior-ai/
├── server.py               # Main MCP server
├── secure_credentials.py   # Secure credential manager
├── credentials.json        # Legacy API keys (migrate to secure storage)
├── .env                    # Environment variables (recommended)
├── migrate_credentials.py  # Migration tool
└── requirements.txt        # Python dependencies
```

## 🚀 Advanced Usage

### Temperature Control for All AIs
```bash
# Creative writing with high temperature
mcp__junior-ai__ask_gemini
  prompt: "Write a creative story about AI collaboration"
  temperature: 0.9

mcp__junior-ai__ask_openai
  prompt: "Write a creative story about AI collaboration" 
  temperature: 0.9

# Technical explanations with low temperature
mcp__junior-ai__ask_grok
  prompt: "Explain how TCP/IP works"
  temperature: 0.2
```

### Specialized Code Reviews by AI
```bash
# Gemini: Technical accuracy focus
mcp__junior-ai__gemini_code_review
  code: "[your code]"
  focus: "technical accuracy"

# ChatGPT: Best practices focus  
mcp__junior-ai__openai_code_review
  code: "[your code]"
  focus: "best practices"

# Grok: Creative solutions focus
mcp__junior-ai__grok_code_review
  code: "[your code]"
  focus: "alternative approaches"
```

### Multi-AI Workflows
```bash
# Step 1: Get all perspectives
mcp__junior-ai__ask_all_ais
  prompt: "How should I structure a microservices architecture?"

# Step 2: Have top AIs debate specifics
mcp__junior-ai__ai_debate
  topic: "Event-driven vs REST for microservices communication"
  ai1: "gemini"
  ai2: "openai"

# Step 3: Get Grok's creative alternative
mcp__junior-ai__ask_grok
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
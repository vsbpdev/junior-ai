# Multi-AI MCP Usage Examples

## Basic AI Interactions

### Individual AI Conversations
```bash
# Have a conversation with Gemini (Google)
mcp__multi-ai-collab__ask_gemini
  prompt: "What's the capital of France?"
  temperature: 0.3

# Ask Grok-3 (xAI) for creative content  
mcp__multi-ai-collab__ask_grok
  prompt: "Write a haiku about coding"
  temperature: 0.8

# Get factual info from ChatGPT (OpenAI)
mcp__multi-ai-collab__ask_openai
  prompt: "Explain REST APIs"
  temperature: 0.2
```

### Compare AI Personalities
```bash
# Ask the same question to all three
mcp__multi-ai-collab__ask_all_ais
  prompt: "What makes a good software engineer?"

# You'll see:
# - Gemini: Technical, detailed analysis
# - Grok: Creative, humorous insights  
# - ChatGPT: Balanced, comprehensive response
```

## Collaborative AI Features

### Multi-AI Comparison
Get different perspectives on the same question:

```bash
mcp__multi-ai-collab__ask_all_ais
  prompt: "What are the best practices for password security?"
  temperature: 0.4
```

**Result**: You'll see responses from all available AIs side-by-side, letting you compare their approaches and find the most comprehensive answer.

### AI Debates
Have two AIs debate different perspectives:

```bash
# Google vs xAI perspective
mcp__multi-ai-collab__ai_debate
  topic: "Should we use TypeScript or JavaScript for new projects?"
  ai1: "gemini"
  ai2: "grok"

# Google vs OpenAI perspective  
mcp__multi-ai-collab__ai_debate
  topic: "Microservices vs Monolith architecture"
  ai1: "gemini"
  ai2: "openai"

# OpenAI vs xAI perspective
mcp__multi-ai-collab__ai_debate
  topic: "React vs Vue.js for frontend development"
  ai1: "openai"
  ai2: "grok"
```

**Result**: You'll see different AI companies' perspectives debate technical decisions, giving you insights from Google, OpenAI, and xAI approaches!

## Code Review Examples

### Multi-AI Code Review Comparison
Get different AI perspectives on the same code:

```bash
# Gemini: Technical accuracy and best practices
mcp__multi-ai-collab__gemini_code_review
  code: |
    def login(username, password):
        if username == "admin" and password == "password123":
            return {"token": "abc123", "role": "admin"}
        return None
  focus: "security"

# ChatGPT: Comprehensive analysis and suggestions
mcp__multi-ai-collab__openai_code_review
  code: |
    def login(username, password):
        if username == "admin" and password == "password123":
            return {"token": "abc123", "role": "admin"}
        return None
  focus: "security"

# Grok: Creative solutions and alternatives
mcp__multi-ai-collab__grok_code_review
  code: |
    def login(username, password):
        if username == "admin" and password == "password123":
            return {"token": "abc123", "role": "admin"}
        return None
  focus: "security"
```

**Result**: You'll get Google's technical precision, OpenAI's balanced analysis, and xAI's creative alternatives!

## Real-World Workflow Examples

### Architecture Decision
1. **Define the problem:**
```bash
mcp__multi-ai-collab__ask_all_ais
  prompt: "I'm building a chat app that needs to handle 100k concurrent users. What architecture patterns should I consider?"
```

2. **Debate specific approaches:**
```bash
mcp__multi-ai-collab__ai_debate
  topic: "Microservices vs Monolith for a high-traffic chat application"
  ai1: "gemini"
  ai2: "openai"
```

3. **Get implementation details:**
```bash
mcp__multi-ai-collab__ask_grok
  prompt: "Show me a WebSocket implementation for real-time chat using Node.js"
```

### Debugging Session
1. **Get multiple diagnostic approaches:**
```bash
mcp__multi-ai-collab__ask_all_ais
  prompt: "My React app is rendering slowly. What are the most common causes and how do I debug them?"
```

2. **Review specific code:**
```bash
mcp__multi-ai-collab__gemini_code_review
  code: "[paste your React component]"
  focus: "performance"
```

3. **Get optimization suggestions:**
```bash
mcp__multi-ai-collab__ask_openai
  prompt: "Given the performance issues identified, show me optimized versions of the problematic code"
```

### Learning New Technology
1. **Get overview from multiple perspectives:**
```bash
mcp__multi-ai-collab__ask_all_ais
  prompt: "Explain Kubernetes and when I should use it"
```

2. **Get hands-on examples:**
```bash
mcp__multi-ai-collab__ask_grok
  prompt: "Show me a simple Kubernetes deployment configuration for a Node.js app"
```

3. **Understand best practices:**
```bash
mcp__multi-ai-collab__ask_gemini
  prompt: "What are the security best practices for Kubernetes in production?"
```

## Advanced Usage Patterns

### Temperature Control for Different Tasks
```bash
# Factual/Technical (low temperature)
mcp__multi-ai-collab__ask_gemini
  prompt: "Explain how TCP handshake works"
  temperature: 0.1

# Creative/Brainstorming (high temperature)
mcp__multi-ai-collab__ask_grok
  prompt: "Suggest innovative features for a productivity app"
  temperature: 0.9

# Balanced/Analysis (medium temperature)
mcp__multi-ai-collab__ask_openai
  prompt: "Analyze the pros and cons of using GraphQL vs REST"
  temperature: 0.5
```

### Cross-AI Validation
Use multiple AIs to validate important decisions:

```bash
# Step 1: Get initial solution
mcp__multi-ai-collab__ask_gemini
  prompt: "Design a database schema for an e-commerce platform"

# Step 2: Have another AI review it
mcp__multi-ai-collab__grok_code_review
  code: "[paste the schema from Gemini]"
  focus: "scalability"

# Step 3: Get alternative perspective
mcp__multi-ai-collab__ask_openai
  prompt: "Review this e-commerce database schema and suggest improvements: [paste schema]"
```

## System Status and Management

### Check Available AIs
```bash
mcp__multi-ai-collab__server_status
```

This shows:
- Which AIs are configured and working
- Model versions being used
- Connection status
- Any error messages

## Tips for Best Results

1. **Use Specific Prompts**: More specific questions get better answers
2. **Leverage Each AI's Strengths**: 
   - Gemini: Technical accuracy, detailed explanations
   - Grok: Creative solutions, concise answers
   - ChatGPT: Balanced analysis, code examples
3. **Compare Responses**: When making important decisions, ask multiple AIs
4. **Adjust Temperature**: Lower for facts, higher for creativity
5. **Use Debates**: Great for exploring different approaches to problems

## Troubleshooting Examples

### API Key Issues
```bash
# Check which AIs are working
mcp__multi-ai-collab__server_status

# If an AI is showing errors, check credentials.json
# Edit: ~/.claude-mcp-servers/multi-ai-collab/credentials.json
```

### Rate Limiting
If you hit rate limits:
- Use different AIs for different tasks
- Lower temperature for more consistent responses (uses fewer tokens)
- Be more specific in prompts to get focused answers

This multi-AI setup gives you unprecedented flexibility in getting the best AI assistance for any task!
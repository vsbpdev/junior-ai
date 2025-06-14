# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task Management

**IMPORTANT**: Always check TaskMaster tasks before determining next steps:
- **Task List**: `/Users/denni1/Documents/GitHub/junior-ai/.taskmaster/tasks/tasks.json` - Machine-readable task definitions
- **Human-Readable Tasks**: `/Users/denni1/Documents/GitHub/junior-ai/.taskmaster/tasks/task_*.txt` - Human-readable task descriptions
- **Priority**: Check TaskMaster tasks FIRST before suggesting new work or asking what to do next

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

### Testing and Development
```bash
# Run pattern detection tests
python3 test_pattern_detection.py

# Run integration tests  
python3 test_integration.py

# Check MCP server status
claude mcp list

# Test the server is working (from Claude Code)
mcp__junior-ai__server_status

# Run server manually for debugging
python3 ~/.claude-mcp-servers/junior-ai/server.py
```

## Architecture

This codebase implements an MCP (Model Context Protocol) server with intelligent pattern detection that enables Claude Code to communicate with multiple AI models and proactively detect code patterns requiring consultation.

### Core System Components

1. **server.py** - Main MCP server implementation (v2.1.0)
   - JSON-RPC protocol handler for MCP communication
   - Dynamic AI client initialization based on credentials
   - Tool registration and execution framework
   - Pattern detection integration with auto-consultation
   - Graceful degradation when AIs are unavailable

2. **Pattern Detection System**
   - **pattern_detection.py**: Enhanced pattern detection engine with 5 categories:
     - Security (passwords, API keys, encryption)
     - Uncertainty (TODOs, FIXMEs, unclear implementations)
     - Algorithm (complexity, optimization needs)
     - Gotcha (common pitfalls, edge cases)
     - Architecture (design patterns, structural decisions)
   - **text_processing_pipeline.py**: Asynchronous text processing with caching
   - **response_handlers.py**: AI consultation orchestration
   - **pattern_cache.py**: LRU caching for pattern detection results

3. **Configuration System**
   - **credentials.json**: Runtime configuration (created from template)
     - API keys and model selections
     - Pattern detection settings
     - Cache configuration
   - **credentials.template.json**: Template with default configuration

### AI Integration Architecture

The system supports 5 AI providers through a unified interface:
- **Gemini** (Google): Uses `google.generativeai` SDK
- **Grok** (xAI): OpenAI-compatible client with custom base URL
- **OpenAI**: Standard OpenAI client
- **DeepSeek**: OpenAI-compatible with custom base URL
- **OpenRouter**: Multi-model gateway with OpenAI-compatible API

Each AI exposes 6 specialized tools:
1. `ask_{ai_name}` - General questions
2. `{ai_name}_code_review` - Code analysis
3. `{ai_name}_think_deep` - Deep analysis
4. `{ai_name}_brainstorm` - Creative solutions
5. `{ai_name}_debug` - Debugging assistance
6. `{ai_name}_architecture` - Design advice

### Pattern Detection Flow

1. Text input → Pattern Detection Engine
2. Pattern matching using keywords and regex
3. Context extraction with line numbers
4. Severity assessment (LOW/MEDIUM/HIGH/CRITICAL)
5. Consultation strategy determination
6. AI selection based on pattern type
7. Response synthesis and formatting

### Tool Categories

**Pattern Detection Tools:**
- `pattern_check` - Analyze text for patterns
- `junior_consult` - Smart AI consultation
- `pattern_stats` - Detection statistics

**Collaborative Tools (2+ AIs):**
- `ask_all_ais` - Multi-AI responses
- `ai_debate` - Two AI debate
- `collaborative_solve` - Sequential/parallel solving
- `ai_consensus` - Consensus building

### Key Design Decisions

1. **Asynchronous Processing**: Pattern detection runs in background threads to avoid blocking
2. **Graceful Degradation**: Server works with any subset of configured AIs
3. **Context-Aware Detection**: Enhanced context extraction for better AI consultations
4. **Caching Strategy**: LRU cache with configurable TTL for performance
5. **Multi-AI Triggering**: Critical patterns automatically trigger multiple AI consultations

### Error Handling Strategy

- Individual AI failures don't crash the server
- Pattern detection failures fall back to direct AI consultation
- JSON-RPC errors are properly formatted for Claude Code
- Comprehensive logging for debugging (to stderr)
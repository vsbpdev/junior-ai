# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL SECURITY RULES

**NEVER EVER CREATE PULL REQUESTS TO EXTERNAL REPOSITORIES**
- **FORBIDDEN**: Creating PRs to RaiAnsar/claude_code-multi-AI-MCP or ANY repository we don't own
- **FORBIDDEN**: Using `gh pr create` without `--repo vsbpdev/junior-ai` flag
- **REQUIRED**: ALWAYS use explicit repository: `gh pr create --repo vsbpdev/junior-ai`
- **REQUIRED**: ALWAYS verify with `gh repo view --json nameWithOwner --jq .nameWithOwner` before ANY GitHub operation
- **REQUIRED**: Only create PRs to vsbpdev/junior-ai - NO OTHER REPOSITORY
- **WARNING**: This is a FORK - GitHub defaults to parent repo for PRs if not explicit
- **MANDATORY**: Read GITHUB_SAFETY_CHECKLIST.md before ANY GitHub operation

## Task Management

**IMPORTANT**: Always check TaskMaster tasks before determining next steps:
- **Task List**: `./.taskmaster/tasks/tasks.json` - Machine-readable task definitions
- **Human-Readable Tasks**: `./.taskmaster/tasks/task_*.txt` - Human-readable task descriptions
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
# Run async pattern cache tests
python3 test_async_pattern_cache.py

# Run context-aware matching tests
python3 test_context_aware_matching.py

# Run response synthesis tests
python3 test_response_synthesis.py

# Check MCP server status
claude mcp list

# Test the server is working (from Claude Code)
mcp__junior-ai__server_status

# Run server manually for debugging
python3 ~/.claude-mcp-servers/junior-ai/server.py

# Run demo scripts
python3 examples/ai_consultation_demo.py
python3 examples/context_aware_demo.py
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
   - AI Consultation Manager integration

2. **Pattern Detection System**
   - **pattern_detection.py**: Enhanced pattern detection engine with 5 categories:
     - Security (passwords, API keys, encryption)
     - Uncertainty (TODOs, FIXMEs, unclear implementations)
     - Algorithm (complexity, optimization needs)
     - Gotcha (common pitfalls, edge cases)
     - Architecture (design patterns, structural decisions)
   - **context_aware_matching.py**: Context-aware pattern matching (NEW):
     - Language-specific syntax analysis
     - Test code detection to reduce false positives
     - Comment vs. active code differentiation
     - Semantic role understanding (parameters, returns, conditions)
     - Scope-aware pattern detection
   - **text_processing_pipeline.py**: Asynchronous text processing with caching
   - **response_handlers.py**: AI consultation orchestration
   - **pattern_cache.py**: LRU caching for pattern detection results

3. **AI Consultation Manager** (NEW)
   - **ai_consultation_manager.py**: Intelligent AI selection and coordination
     - AI expertise profiling (security, algorithms, architecture, etc.)
     - Smart AI selection based on pattern types and severity
     - Multiple consultation modes: single, multi, consensus, debate
     - Transparent audit trail and metrics tracking
     - Governance and compliance reporting
     - Performance optimization (cost vs. speed vs. accuracy)

4. **Response Synthesis System** (NEW)
   - **response_synthesis.py**: Intelligently combines multiple AI responses
     - Multiple synthesis strategies: consensus, debate, expert-weighted, comprehensive, summary, hierarchical
     - Confidence scoring and agreement/disagreement analysis
     - Key insight extraction and prioritization
     - Markdown formatting with proper structure
   - **docs/response_synthesis.md**: Detailed documentation

5. **Async Caching System** (NEW)
   - **async_pattern_cache.py**: High-performance async caching with deduplication
     - LRU cache with configurable TTL and size limits
     - Request deduplication for concurrent identical requests
     - Dynamic TTL based on pattern types and severity
     - Memory-efficient storage with automatic cleanup
   - **async_cached_pattern_engine.py**: Integrates caching with pattern detection
     - Transparent caching layer for pattern detection
     - Performance metrics and cache hit/miss tracking
     - Configurable cache strategies per pattern type

6. **Configuration System**
   - **credentials.json**: Runtime configuration (created from template)
     - API keys and model selections
     - Pattern detection settings with sensitivity levels (low, medium, high, maximum)
     - Async cache configuration (size, TTL, deduplication settings)
     - AI consultation preferences
     - Pattern category customization and overrides
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
3. `{ai_name}_think_deep` - Profound analysis with extended reasoning
4. `{ai_name}_brainstorm` - Creative solutions
5. `{ai_name}_debug` - Debugging assistance
6. `{ai_name}_architecture` - Design advice

### Pattern Detection Flow

1. Text input → Pattern Detection Engine
2. Pattern matching using keywords and regex
3. **Context-aware analysis** (NEW):
   - Language detection (Python, JavaScript, TypeScript, Java, etc.)
   - Syntax-aware scope extraction (function/class context)
   - Test code detection (reduces false positives)
   - Comment vs. active code differentiation
   - Import analysis for enhanced understanding
4. Context extraction with line numbers
5. Severity assessment (LOW/MEDIUM/HIGH/CRITICAL)
6. Consultation strategy determination
7. AI selection based on pattern type
8. Response synthesis with context insights

### Tool Categories

**Pattern Detection Tools:**
- `pattern_check` - Analyze text for patterns
- `junior_consult` - Smart AI consultation
- `pattern_stats` - Detection statistics
- `get_sensitivity_config` - View sensitivity settings
- `update_sensitivity` - Modify detection sensitivity

**AI Consultation Manager Tools:** (NEW)
- `ai_consultation_strategy` - Get recommended AI strategy
- `ai_consultation_metrics` - View performance metrics
- `ai_consultation_audit` - Access consultation history
- `ai_governance_report` - Export compliance report

**Collaborative Tools (2+ AIs):**
- `ask_all_ais` - Multi-AI responses
- `ai_debate` - Two AI debate
- `collaborative_solve` - Sequential/parallel solving
- `ai_consensus` - Consensus building

### Key Design Decisions

1. **Asynchronous Processing**: Pattern detection runs in background threads to avoid blocking
2. **Graceful Degradation**: Server works with any subset of configured AIs
3. **Context-Aware Detection**: Enhanced context extraction for better AI consultations
4. **Caching Strategy**: Dual-layer caching with async pattern cache and request deduplication
5. **Multi-AI Triggering**: Critical patterns automatically trigger multiple AI consultations
6. **Response Synthesis**: Intelligent combination of multiple AI responses for comprehensive insights
7. **Performance Optimization**: Batch processing and concurrent AI consultations

### Error Handling Strategy

- Individual AI failures don't crash the server
- Pattern detection failures fall back to direct AI consultation
- JSON-RPC errors are properly formatted for Claude Code
- Comprehensive logging for debugging (to stderr)
- Async operations with proper error propagation

### Demo Scripts

The `examples/` directory contains demonstration scripts:
- **ai_consultation_demo.py**: Shows AI consultation manager features
- **context_aware_demo.py**: Demonstrates context-aware pattern matching

### Pattern Detection Sensitivity

The system supports configurable sensitivity levels:
- **low**: Conservative detection - only obvious patterns (confidence threshold: 0.9)
- **medium**: Balanced detection - standard sensitivity (confidence threshold: 0.7)
- **high**: Aggressive detection - catch potential issues (confidence threshold: 0.5)
- **maximum**: Maximum detection - catch everything possible (confidence threshold: 0.3)

Each level affects:
- Confidence thresholds for pattern matching
- Context extraction size
- Minimum matches required for consultation
- Severity threshold for triggering consultations

Category-specific overrides can be set (e.g., always use "high" for security patterns)
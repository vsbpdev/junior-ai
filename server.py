#!/usr/bin/env python3
"""
Junior AI Assistant for Claude Code
Intelligent assistant with pattern detection and multi-AI consultations
"""

import json
import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import threading

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Import pattern detection components
try:
    from pattern_detection import PatternDetectionEngine, EnhancedPatternDetectionEngine, PatternCategory, PatternSeverity
    from text_processing_pipeline import TextProcessingPipeline, ProcessingResult
    from response_handlers import PatternResponseManager, ConsultationResponse
    from context_aware_matching import ContextAwarePatternMatcher
    from ai_consultation_manager import AIConsultationManager
    from async_pattern_cache import AsyncPatternCache, get_global_cache, get_global_cache_metrics
    from async_cached_pattern_engine import AsyncCachedPatternEngine, AsyncPatternDetectionPipeline
    PATTERN_DETECTION_AVAILABLE = True
    CONTEXT_AWARE_AVAILABLE = True
    AI_CONSULTATION_AVAILABLE = True
    ASYNC_CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pattern detection not available: {e}", file=sys.stderr)
    PATTERN_DETECTION_AVAILABLE = False
    CONTEXT_AWARE_AVAILABLE = False
    AI_CONSULTATION_AVAILABLE = False
    ASYNC_CACHE_AVAILABLE = False

# Server version
__version__ = "2.1.0"  # Updated version with pattern detection

# Load credentials
SCRIPT_DIR = Path(__file__).parent
CREDENTIALS_FILE = SCRIPT_DIR / "credentials.json"

try:
    with open(CREDENTIALS_FILE, 'r') as f:
        CREDENTIALS = json.load(f)
except Exception as e:
    print(json.dumps({
        "jsonrpc": "2.0",
        "error": {
            "code": -32603,
            "message": f"Failed to load credentials.json: {str(e)}"
        }
    }), file=sys.stdout, flush=True)
    sys.exit(1)

# Initialize AI clients
AI_CLIENTS = {}

# Gemini
if CREDENTIALS.get("gemini", {}).get("enabled", False):
    try:
        import google.generativeai as genai
        genai.configure(api_key=CREDENTIALS["gemini"]["api_key"])
        AI_CLIENTS["gemini"] = {
            "client": genai.GenerativeModel(CREDENTIALS["gemini"]["model"]),
            "type": "gemini"
        }
    except Exception as e:
        print(f"Warning: Gemini initialization failed: {e}", file=sys.stderr)

# Grok-3 and OpenAI (both use OpenAI client)
if CREDENTIALS.get("grok", {}).get("enabled", False) or CREDENTIALS.get("openai", {}).get("enabled", False):
    try:
        from openai import OpenAI
        
        # Grok-3
        if CREDENTIALS.get("grok", {}).get("enabled", False):
            AI_CLIENTS["grok"] = {
                "client": OpenAI(
                    api_key=CREDENTIALS["grok"]["api_key"],
                    base_url=CREDENTIALS["grok"]["base_url"]
                ),
                "model": CREDENTIALS["grok"]["model"],
                "type": "openai"
            }
        
        # OpenAI
        if CREDENTIALS.get("openai", {}).get("enabled", False):
            AI_CLIENTS["openai"] = {
                "client": OpenAI(api_key=CREDENTIALS["openai"]["api_key"]),
                "model": CREDENTIALS["openai"]["model"],
                "type": "openai"
            }
    except Exception as e:
        print(f"Warning: OpenAI client initialization failed: {e}", file=sys.stderr)

# DeepSeek
if CREDENTIALS.get("deepseek", {}).get("enabled", False):
    try:
        from openai import OpenAI
        AI_CLIENTS["deepseek"] = {
            "client": OpenAI(
                api_key=CREDENTIALS["deepseek"]["api_key"],
                base_url=CREDENTIALS["deepseek"]["base_url"]
            ),
            "model": CREDENTIALS["deepseek"]["model"],
            "type": "openai"
        }
    except Exception as e:
        print(f"Warning: DeepSeek initialization failed: {e}", file=sys.stderr)

# OpenRouter
if CREDENTIALS.get("openrouter", {}).get("enabled", False):
    try:
        from openai import OpenAI
        AI_CLIENTS["openrouter"] = {
            "client": OpenAI(
                api_key=CREDENTIALS["openrouter"]["api_key"],
                base_url=CREDENTIALS["openrouter"]["base_url"]
            ),
            "model": CREDENTIALS["openrouter"]["model"],
            "type": "openai"
        }
    except Exception as e:
        print(f"Warning: OpenRouter initialization failed: {e}", file=sys.stderr)

# Initialize pattern detection components if available
if PATTERN_DETECTION_AVAILABLE:
    # Get context window size from config
    context_window_size = CREDENTIALS.get("pattern_detection", {}).get("context_window_size", 150)
    
    # Use context-aware pattern matcher if available
    if CONTEXT_AWARE_AVAILABLE:
        base_engine = EnhancedPatternDetectionEngine(context_window_size=context_window_size)
        sync_pattern_engine = ContextAwarePatternMatcher(base_engine=base_engine)
    else:
        sync_pattern_engine = EnhancedPatternDetectionEngine(context_window_size=context_window_size)
    
    # Configure async caching
    async_cache_config = {
        'max_size': CREDENTIALS.get("pattern_detection", {}).get("cache_max_size", 2000),
        'max_memory_mb': CREDENTIALS.get("pattern_detection", {}).get("cache_max_memory_mb", 100),
        'base_ttl_seconds': CREDENTIALS.get("pattern_detection", {}).get("cache_ttl_seconds", 300),
        'enable_deduplication': CREDENTIALS.get("pattern_detection", {}).get("cache_deduplication", True),
        'cleanup_interval': CREDENTIALS.get("pattern_detection", {}).get("cache_cleanup_interval", 60)
    }
    
    # Create async cached pattern engine if async cache is available
    if ASYNC_CACHE_AVAILABLE and CREDENTIALS.get("pattern_detection", {}).get("async_cache_enabled", True):
        pattern_engine = AsyncCachedPatternEngine(
            pattern_engine=sync_pattern_engine,
            cache_config=async_cache_config,
            enable_cache=True,
            enable_deduplication=True
        )
        # Also create async pipeline for batch processing
        async_pipeline = AsyncPatternDetectionPipeline(
            engine=pattern_engine,
            batch_size=CREDENTIALS.get("pattern_detection", {}).get("batch_size", 10),
            max_concurrent=CREDENTIALS.get("pattern_detection", {}).get("max_concurrent", 5)
        )
    else:
        # Fallback to sync pattern engine
        pattern_engine = sync_pattern_engine
        async_pipeline = None
    
    # Keep the original text pipeline for backward compatibility
    legacy_cache_config = {
        'max_size': 1000,
        'ttl_seconds': CREDENTIALS.get("pattern_detection", {}).get("cache_ttl_seconds", 300),
        'persist_to_disk': False,  # Don't persist for MCP server
        'cache_dir': None
    }
    
    text_pipeline = TextProcessingPipeline(
        pattern_engine=sync_pattern_engine,
        enable_cache=CREDENTIALS.get("pattern_detection", {}).get("legacy_cache_enabled", False),
        cache_config=legacy_cache_config
    )
    response_manager = PatternResponseManager()
    
    # Set AI callers for response manager
    def call_ai_wrapper(ai_name: str, prompt: str, temperature: float = 0.7) -> str:
        return call_ai(ai_name, prompt, temperature)
    
    def call_multiple_ais_wrapper(prompt: str, ai_list: List[str], temperature: float = 0.7) -> Dict[str, str]:
        responses = {}
        for ai_name in ai_list:
            if ai_name in AI_CLIENTS:
                response = call_ai(ai_name, prompt, temperature)
                if not response.startswith("❌"):
                    responses[ai_name] = response
        return responses
    
    response_manager.set_ai_callers(call_ai_wrapper, call_multiple_ais_wrapper)
    
    # Initialize AI Consultation Manager if available
    if AI_CONSULTATION_AVAILABLE:
        ai_consultation_manager = AIConsultationManager(
            ai_caller=call_ai_wrapper,
            config=CREDENTIALS.get("ai_consultation", {})
        )
    else:
        ai_consultation_manager = None
    
    # Start the text processing pipeline
    text_pipeline.start()
    
    # Pattern detection configuration
    pattern_config = CREDENTIALS.get("pattern_detection", {
        "enabled": True,
        "default_junior": "openrouter",
        "accuracy_mode": True,
        "auto_consult_threshold": "always_when_pattern_detected",
        "multi_ai_for_critical": True,
        "show_all_consultations": True
    })
else:
    pattern_engine = None
    sync_pattern_engine = None
    async_pipeline = None
    text_pipeline = None
    response_manager = None
    ai_consultation_manager = None
    pattern_config = {"enabled": False}

def send_response(response: Dict[str, Any]):
    """Send a JSON-RPC response"""
    print(json.dumps(response), flush=True)

def call_ai(ai_name: str, prompt: str, temperature: float = 0.7) -> str:
    """Call a specific AI and return response"""
    if ai_name not in AI_CLIENTS:
        return f"❌ {ai_name.upper()} is not available or not configured"
    
    try:
        client_info = AI_CLIENTS[ai_name]
        client = client_info["client"]
        
        if client_info["type"] == "gemini":
            import google.generativeai as genai
            response = client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=8192,
                )
            )
            return response.text
        
        elif client_info["type"] == "openai":
            response = client.chat.completions.create(
                model=client_info["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=8192
            )
            return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error calling {ai_name.upper()}: {str(e)}"

def call_multiple_ais(prompt: str, ai_list: List[str], temperature: float = 0.7) -> str:
    """Call multiple AIs and return combined responses"""
    results = []
    available_ais = [ai for ai in ai_list if ai in AI_CLIENTS]
    
    if not available_ais:
        return "❌ None of the requested AIs are available"
    
    for ai_name in available_ais:
        response = call_ai(ai_name, prompt, temperature)
        results.append(f"## 🤖 {ai_name.upper()} Response:\n\n{response}")
    
    separator = "\n\n" + "=" * 80 + "\n\n"
    return separator.join(results)

def handle_initialize(request_id: Any) -> Dict[str, Any]:
    """Handle initialization"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "junior-ai-assistant-mcp",
                "version": __version__
            }
        }
    }

def handle_tools_list(request_id: Any) -> Dict[str, Any]:
    """List available tools"""
    tools = [
        {
            "name": "server_status",
            "description": "Get server status and available AI models",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    
    # Pattern detection tools (if available)
    if PATTERN_DETECTION_AVAILABLE and pattern_config.get("enabled", True):
        tools.extend([
            {
                "name": "pattern_check",
                "description": "Check text for patterns that might benefit from AI consultation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze for patterns"
                        },
                        "auto_consult": {
                            "type": "boolean",
                            "description": "Automatically consult AI if patterns detected",
                            "default": True
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "junior_consult",
                "description": "Smart AI consultation based on detected patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context or code to analyze"
                        },
                        "force_multi_ai": {
                            "type": "boolean",
                            "description": "Force multi-AI consultation",
                            "default": False
                        }
                    },
                    "required": ["context"]
                }
            },
            {
                "name": "pattern_stats",
                "description": "Get pattern detection statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_sensitivity_config",
                "description": "Get current pattern detection sensitivity configuration",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "update_sensitivity",
                "description": "Update pattern detection sensitivity settings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "global_level": {
                            "type": "string",
                            "description": "Global sensitivity level",
                            "enum": ["low", "medium", "high", "maximum"]
                        },
                        "category_overrides": {
                            "type": "object",
                            "description": "Category-specific sensitivity overrides",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "maximum", "null"]
                            }
                        }
                    }
                }
            },
            {
                "name": "toggle_pattern_detection",
                "description": "Enable or disable pattern detection globally",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": "True to enable, False to disable pattern detection"
                        }
                    },
                    "required": ["enabled"]
                }
            },
            {
                "name": "toggle_category",
                "description": "Enable or disable a specific pattern category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Pattern category to toggle",
                            "enum": ["security", "uncertainty", "algorithm", "gotcha", "architecture"]
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "True to enable, False to disable the category"
                        }
                    },
                    "required": ["category", "enabled"]
                }
            },
            {
                "name": "add_pattern_keywords",
                "description": "Add custom keywords to a pattern category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Pattern category",
                            "enum": ["security", "uncertainty", "algorithm", "gotcha", "architecture"]
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to add to the category"
                        }
                    },
                    "required": ["category", "keywords"]
                }
            },
            {
                "name": "remove_pattern_keywords",
                "description": "Remove keywords from a pattern category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Pattern category",
                            "enum": ["security", "uncertainty", "algorithm", "gotcha", "architecture"]
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to remove from the category"
                        }
                    },
                    "required": ["category", "keywords"]
                }
            },
            {
                "name": "list_pattern_keywords",
                "description": "List all keywords for a pattern category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Pattern category",
                            "enum": ["security", "uncertainty", "algorithm", "gotcha", "architecture"]
                        }
                    },
                    "required": ["category"]
                }
            },
            {
                "name": "force_consultation",
                "description": "Force AI consultation regardless of pattern detection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context to analyze"
                        },
                        "category": {
                            "type": "string",
                            "description": "Pattern category to use for consultation",
                            "enum": ["security", "uncertainty", "algorithm", "gotcha", "architecture"]
                        },
                        "multi_ai": {
                            "type": "boolean",
                            "description": "Use multiple AIs for consultation",
                            "default": False
                        }
                    },
                    "required": ["context", "category"]
                }
            }
        ])
        
        # AI Consultation Manager tools
        if AI_CONSULTATION_AVAILABLE:
            tools.extend([
                {
                    "name": "ai_consultation_strategy",
                    "description": "Get recommended AI consultation strategy for given patterns",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Code context to analyze"
                            },
                            "priority": {
                                "type": "string",
                                "description": "Optimization priority",
                                "enum": ["speed", "accuracy", "cost"],
                                "default": "accuracy"
                            }
                        },
                        "required": ["context"]
                    }
                },
                {
                    "name": "ai_consultation_metrics",
                    "description": "Get AI consultation metrics and performance statistics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "ai_consultation_audit",
                    "description": "Get audit trail of recent AI consultations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of recent consultations to retrieve",
                                "default": 10
                            },
                            "pattern_category": {
                                "type": "string",
                                "description": "Filter by pattern category",
                                "enum": ["security", "uncertainty", "algorithm", "gotcha", "architecture"]
                            }
                        }
                    }
                },
                {
                    "name": "ai_governance_report",
                    "description": "Export governance and compliance report for AI consultations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ])
    
    # Individual AI tools - comprehensive set
    for ai_name in AI_CLIENTS.keys():
        tools.extend([
            # Basic interaction
            {
                "name": f"ask_{ai_name}",
                "description": f"Ask {ai_name.upper()} a question",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The question or prompt"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature for response (0.0-1.0)",
                            "default": 0.7
                        }
                    },
                    "required": ["prompt"]
                }
            },
            # Code review
            {
                "name": f"{ai_name}_code_review",
                "description": f"Have {ai_name.upper()} review code for issues and improvements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to review"
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus area (security, performance, readability, etc.)",
                            "default": "general"
                        }
                    },
                    "required": ["code"]
                }
            },
            # Deep thinking/analysis
            {
                "name": f"{ai_name}_think_deep",
                "description": f"Have {ai_name.upper()} do deep analysis with extended reasoning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic or problem for deep analysis"
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context or constraints",
                            "default": ""
                        }
                    },
                    "required": ["topic"]
                }
            },
            # Brainstorming
            {
                "name": f"{ai_name}_brainstorm",
                "description": f"Brainstorm creative solutions with {ai_name.upper()}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "challenge": {
                            "type": "string",
                            "description": "The challenge or problem to brainstorm about"
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Any constraints or limitations",
                            "default": ""
                        }
                    },
                    "required": ["challenge"]
                }
            },
            # Debug assistance
            {
                "name": f"{ai_name}_debug",
                "description": f"Get debugging help from {ai_name.upper()}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error message or description"
                        },
                        "code": {
                            "type": "string",
                            "description": "Related code that's causing issues",
                            "default": ""
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context about the environment/setup",
                            "default": ""
                        }
                    },
                    "required": ["error"]
                }
            },
            # Architecture advice
            {
                "name": f"{ai_name}_architecture",
                "description": f"Get architecture design advice from {ai_name.upper()}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "string",
                            "description": "System requirements and goals"
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Technical constraints, budget, timeline etc.",
                            "default": ""
                        },
                        "scale": {
                            "type": "string",
                            "description": "Expected scale (users, data, etc.)",
                            "default": ""
                        }
                    },
                    "required": ["requirements"]
                }
            }
        ])
    
    # Collaborative tools
    if len(AI_CLIENTS) > 1:
        tools.extend([
            {
                "name": "ask_all_ais",
                "description": "Ask all available AIs the same question and compare responses",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The question to ask all AIs"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature for responses",
                            "default": 0.7
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "ai_debate",
                "description": "Have two AIs debate a topic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The debate topic"
                        },
                        "ai1": {
                            "type": "string",
                            "description": "First AI (gemini, grok, openai, deepseek)",
                            "default": "gemini"
                        },
                        "ai2": {
                            "type": "string",
                            "description": "Second AI (gemini, grok, openai, deepseek)",
                            "default": "grok"
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "collaborative_solve",
                "description": "Have multiple AIs collaborate to solve a complex problem",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "The complex problem to solve"
                        },
                        "approach": {
                            "type": "string",
                            "description": "How to divide the work (sequential, parallel, debate)",
                            "default": "sequential"
                        }
                    },
                    "required": ["problem"]
                }
            },
            {
                "name": "ai_consensus",
                "description": "Get consensus opinion from all available AIs",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to get consensus on"
                        },
                        "options": {
                            "type": "string",
                            "description": "Available options or approaches to choose from",
                            "default": ""
                        }
                    },
                    "required": ["question"]
                }
            }
        ])
    
    # Async cache tools (if available)
    if ASYNC_CACHE_AVAILABLE and PATTERN_DETECTION_AVAILABLE:
        tools.extend([
            {
                "name": "cache_stats",
                "description": "Get async pattern cache statistics and performance metrics",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "clear_cache",
                "description": "Clear the async pattern cache",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirm cache clearing",
                            "default": False
                        }
                    }
                }
            },
            {
                "name": "async_pattern_check",
                "description": "Async pattern detection with caching and deduplication",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze for patterns"
                        },
                        "sensitivity_level": {
                            "type": "string",
                            "description": "Sensitivity level for detection",
                            "enum": ["low", "medium", "high", "maximum"],
                            "default": "medium"
                        },
                        "auto_consult": {
                            "type": "boolean",
                            "description": "Automatically consult AI if patterns detected",
                            "default": True
                        }
                    },
                    "required": ["text"]
                }
            }
        ])
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": tools
        }
    }

def handle_pattern_check(text: str, auto_consult: bool = True, filename: Optional[str] = None) -> str:
    """Handle pattern checking in text"""
    if not PATTERN_DETECTION_AVAILABLE or not pattern_config.get("enabled", True):
        return "❌ Pattern detection is not available or disabled"
    
    # Detect patterns using context-aware matching if available
    if CONTEXT_AWARE_AVAILABLE and hasattr(pattern_engine, 'detect_with_context'):
        contextual_patterns = pattern_engine.detect_with_context(text, filename)
        patterns = [cp.base_match for cp in contextual_patterns]
        # Store contextual patterns for later use
        context_data = {
            'contextual_patterns': contextual_patterns,
            'has_test_code': any(cp.is_test_code for cp in contextual_patterns),
            'has_comments': any(cp.is_commented_out for cp in contextual_patterns)
        }
    else:
        patterns = pattern_engine.detect_patterns(text)
        context_data = None
    
    if not patterns:
        return "✅ No patterns detected that require AI consultation."
    
    # Generate pattern summary
    if hasattr(pattern_engine, 'base_engine'):
        # Using context-aware matcher
        summary = pattern_engine.base_engine.get_pattern_summary(patterns)
        if context_data and 'contextual_patterns' in context_data:
            strategy = pattern_engine.get_enhanced_consultation_strategy(context_data['contextual_patterns'])
        else:
            strategy = pattern_engine.base_engine.get_consultation_strategy(patterns)
    else:
        summary = pattern_engine.get_pattern_summary(patterns)
        strategy = pattern_engine.get_consultation_strategy(patterns)
    
    result = f"""## 🔍 Pattern Detection Results

**Patterns Found:** {summary['total_matches']}
**Max Severity:** {summary['max_severity'].name if summary['max_severity'] else 'N/A'}
**Consultation Strategy:** {strategy['strategy']}
"""
    
    # Add context insights if available
    if 'context_insights' in strategy:
        result += "\n**Context Insights:** " + "; ".join(strategy['context_insights']) + "\n"
    
    result += "\n### Detected Categories:\n"
    
    for category, data in summary['categories'].items():
        result += f"\n**{category.upper()}** ({data['count']} matches):\n"
        result += f"  Keywords: {', '.join(data['keywords'])}\n"
        result += f"  Severity: {PatternSeverity(data['severity']).name}\n"
    
    result += f"\n### Recommendation:\n{strategy['reason']}\n"
    
    # Auto-consult if requested and patterns warrant it
    if auto_consult and pattern_engine.should_trigger_consultation(patterns):
        result += "\n---\n## 🤖 AI Consultation (Auto-triggered)\n\n"
        
        # Handle patterns through response manager
        consultation_response = response_manager.handle_patterns(patterns, text)
        
        if consultation_response:
            result += f"**Consultation ID:** {consultation_response.request_id}\n"
            result += f"**Confidence:** {consultation_response.confidence_score:.2%}\n\n"
            result += consultation_response.primary_recommendation
            
            if consultation_response.synthesis:
                result += f"\n\n{consultation_response.synthesis}"
    
    return result

def handle_junior_consult(context: str, force_multi_ai: bool = False, filename: Optional[str] = None) -> str:
    """Handle smart junior AI consultation"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    # Detect patterns using context-aware matching if available
    contextual_patterns = None
    if CONTEXT_AWARE_AVAILABLE and hasattr(pattern_engine, 'detect_with_context'):
        contextual_patterns = pattern_engine.detect_with_context(context, filename)
        patterns = [cp.base_match for cp in contextual_patterns]
    else:
        patterns = pattern_engine.detect_patterns(context)
    
    if not patterns and not force_multi_ai:
        # No patterns detected, provide general assistance
        default_ai = pattern_config.get("default_junior", "openrouter")
        prompt = f"Please provide guidance on the following:\n\n{context}"
        response = call_ai(default_ai, prompt, 0.7)
        return f"## 💡 Junior AI Assistance\n\n{response}"
    
    # Use AI Consultation Manager if available
    if AI_CONSULTATION_AVAILABLE and ai_consultation_manager:
        # Let the consultation manager handle everything
        preferences = {
            'force_multi_ai': force_multi_ai,
            'priority': 'accuracy' if pattern_config.get('accuracy_mode', True) else 'speed'
        }
        
        consultation_response = ai_consultation_manager.execute_consultation(
            patterns=patterns,
            context=context,
            contextual_patterns=contextual_patterns,
            strategy=None  # Let it auto-select strategy
        )
    else:
        # Fallback to original response manager
        if force_multi_ai and patterns:
            for pattern in patterns:
                pattern.requires_multi_ai = True
        
        consultation_response = response_manager.handle_patterns(patterns, context)
    
    if not consultation_response:
        return "❌ Unable to generate consultation response"
    
    # Format response
    result = f"""## 🤖 Junior AI Consultation

**Consultation ID:** {consultation_response.request_id}
**Confidence Score:** {consultation_response.confidence_score:.2%}
**AIs Consulted:** {', '.join(consultation_response.ai_responses.keys())}
"""
    
    result += "\n"
    
    result += consultation_response.primary_recommendation
    
    if consultation_response.synthesis:
        result += f"\n\n{consultation_response.synthesis}"
    
    # Show individual AI responses if configured
    if pattern_config.get("show_all_consultations", True) and len(consultation_response.ai_responses) > 1:
        result += "\n\n---\n### Individual AI Responses:\n"
        for ai_name, response in consultation_response.ai_responses.items():
            result += f"\n**{ai_name.upper()}:**\n{response[:500]}...\n"
    
    return result

def handle_pattern_stats() -> str:
    """Get pattern detection statistics"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    # Get pipeline stats
    pipeline_stats = text_pipeline.get_stats()
    
    # Get handler stats
    handler_stats = response_manager.get_handler_stats()
    
    result = f"""## 📊 Pattern Detection Statistics

### Pipeline Performance:
- Chunks Processed: {pipeline_stats['chunks_processed']}
- Patterns Detected: {pipeline_stats['patterns_detected']}
- Consultations Triggered: {pipeline_stats['consultations_triggered']}
- Average Latency: {pipeline_stats['average_latency']:.3f}s

### Handler Statistics:
"""
    
    for category, stats in handler_stats.items():
        result += f"\n**{category.upper()}:**\n"
        result += f"  - Consultations: {stats['consultations']}\n"
        result += f"  - Avg Processing Time: {stats['avg_processing_time']:.3f}s\n"
    
    result += "\n### Configuration:"
    result += f"\n- Pattern Detection: {'Enabled' if pattern_config.get('enabled', True) else 'Disabled'}"
    result += f"\n- Accuracy Mode: {'On' if pattern_config.get('accuracy_mode', True) else 'Off'}"
    result += f"\n- Default Junior AI: {pattern_config.get('default_junior', 'openrouter')}"
    result += f"\n- Multi-AI for Critical: {'Yes' if pattern_config.get('multi_ai_for_critical', True) else 'No'}"
    
    return result

def handle_get_sensitivity_config() -> str:
    """Get current sensitivity configuration"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        # Get current sensitivity info from the pattern engine
        sensitivity_info = pattern_engine.get_sensitivity_info()
        
        result = f"""## 🎛️ Pattern Detection Sensitivity Configuration

### Current Settings:
- **Global Level**: {sensitivity_info['global_level']}
- **Confidence Threshold**: {sensitivity_info['confidence_threshold']}
- **Context Multiplier**: {sensitivity_info['context_multiplier']}x
- **Min Matches for Consultation**: {sensitivity_info['min_matches_for_consultation']}
- **Severity Threshold**: {sensitivity_info['severity_threshold']}
- **Effective Context Window**: {sensitivity_info['effective_context_window']} chars

### Category Overrides:"""
        
        overrides = sensitivity_info['category_overrides']
        for category, override in overrides.items():
            if override:
                result += f"\n- **{category.upper()}**: {override}"
            else:
                result += f"\n- **{category.upper()}**: (using global level)"
        
        result += f"""

### Available Sensitivity Levels:
- **low**: Conservative detection - only obvious patterns
- **medium**: Balanced detection - standard sensitivity  
- **high**: Aggressive detection - catch potential issues
- **maximum**: Maximum detection - catch everything possible

Use `update_sensitivity` to modify these settings."""
        
        return result
        
    except Exception as e:
        return f"❌ Error getting sensitivity config: {str(e)}"

def handle_update_sensitivity(global_level: str = None, category_overrides: Dict[str, str] = None) -> str:
    """Update sensitivity configuration"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        # Update sensitivity settings
        success = pattern_engine.update_sensitivity(global_level, category_overrides)
        
        if success:
            result = "✅ Sensitivity configuration updated successfully!\n\n"
            
            # Show updated configuration
            sensitivity_info = pattern_engine.get_sensitivity_info()
            result += f"**New Global Level**: {sensitivity_info['global_level']}\n"
            result += f"**New Confidence Threshold**: {sensitivity_info['confidence_threshold']}\n"
            result += f"**New Context Window**: {sensitivity_info['effective_context_window']} chars\n"
            
            if category_overrides:
                result += "\n**Updated Category Overrides**:\n"
                for category, level in category_overrides.items():
                    result += f"- {category.upper()}: {level}\n"
            
            result += "\n💡 Changes take effect immediately for new pattern detections."
            return result
        else:
            return "❌ Failed to update sensitivity configuration. Check that the specified levels are valid."
            
    except Exception as e:
        return f"❌ Error updating sensitivity: {str(e)}"

def handle_toggle_pattern_detection(enabled: bool) -> str:
    """Toggle pattern detection globally"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        success = pattern_engine.set_pattern_detection_enabled(enabled)
        
        if success:
            status = "enabled" if enabled else "disabled"
            return f"✅ Pattern detection has been {status} globally.\n\n💡 This setting persists across sessions."
        else:
            return "❌ Failed to update pattern detection setting."
    
    except Exception as e:
        return f"❌ Error toggling pattern detection: {str(e)}"

def handle_toggle_category(category: str, enabled: bool) -> str:
    """Toggle a specific pattern category"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        success = pattern_engine.set_category_enabled(category, enabled)
        
        if success:
            status = "enabled" if enabled else "disabled"
            return f"✅ Pattern category '{category}' has been {status}.\n\n💡 This affects pattern detection for {category}-related keywords."
        else:
            return f"❌ Failed to update category setting. Invalid category: {category}"
    
    except Exception as e:
        return f"❌ Error toggling category: {str(e)}"

def handle_add_pattern_keywords(category: str, keywords: List[str]) -> str:
    """Add custom keywords to a pattern category"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        success = pattern_engine.add_custom_keywords(category, keywords)
        
        if success:
            result = f"✅ Added {len(keywords)} keyword(s) to '{category}' category:\n"
            for keyword in keywords:
                result += f"  - {keyword}\n"
            result += "\n💡 These keywords will now trigger pattern detection."
            return result
        else:
            return f"❌ Failed to add keywords. Check category name: {category}"
    
    except Exception as e:
        return f"❌ Error adding keywords: {str(e)}"

def handle_remove_pattern_keywords(category: str, keywords: List[str]) -> str:
    """Remove keywords from a pattern category"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        success = pattern_engine.remove_keywords(category, keywords)
        
        if success:
            result = f"✅ Removed keyword(s) from '{category}' category:\n"
            for keyword in keywords:
                result += f"  - {keyword}\n"
            result += "\n💡 These keywords will no longer trigger pattern detection."
            return result
        else:
            return f"❌ Failed to remove keywords. Check category name: {category}"
    
    except Exception as e:
        return f"❌ Error removing keywords: {str(e)}"

def handle_list_pattern_keywords(category: str) -> str:
    """List all keywords for a pattern category"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    try:
        keywords = pattern_engine.get_all_keywords(category)
        
        if keywords:
            result = f"## 📋 Keywords for '{category}' category\n\n"
            result += f"Total keywords: {len(keywords)}\n\n"
            
            # Group keywords by length for better readability
            short_keywords = [k for k in keywords if len(k) <= 10]
            medium_keywords = [k for k in keywords if 10 < len(k) <= 20]
            long_keywords = [k for k in keywords if len(k) > 20]
            
            if short_keywords:
                result += "### Short keywords:\n"
                result += ", ".join(sorted(short_keywords)) + "\n\n"
            
            if medium_keywords:
                result += "### Medium keywords:\n"
                result += ", ".join(sorted(medium_keywords)) + "\n\n"
            
            if long_keywords:
                result += "### Long keywords:\n"
                for kw in sorted(long_keywords):
                    result += f"- {kw}\n"
            
            return result
        else:
            return f"❌ No keywords found for category '{category}'. Check category name."
    
    except Exception as e:
        return f"❌ Error listing keywords: {str(e)}"

def handle_force_consultation(context: str, category: str, multi_ai: bool = False) -> str:
    """Force AI consultation regardless of pattern detection"""
    if not AI_CLIENTS:
        return "❌ No AI clients are configured"
    
    # Add validation for empty context
    if not context or not context.strip():
        return "❌ Context cannot be empty. Please provide text or code to analyze."
    
    try:
        # Create a synthetic pattern match for the specified category
        from pattern_detection import PatternCategory, PatternSeverity, PatternMatch
        
        try:
            pattern_category = PatternCategory(category)
        except ValueError:
            return f"❌ Invalid category: {category}. Must be one of: security, uncertainty, algorithm, gotcha, architecture"
        
        # Create a high-severity pattern match
        synthetic_match = PatternMatch(
            category=pattern_category,
            severity=PatternSeverity.HIGH,
            keyword=f"[Forced {category} consultation]",
            context=context,
            start_pos=0,
            end_pos=len(context),
            confidence=1.0,
            requires_multi_ai=multi_ai,
            line_number=1,
            full_line=context.split('\n')[0] if context else ""
        )
        
        # Use the response manager to handle the consultation
        if AI_CONSULTATION_AVAILABLE and consultation_manager:
            # Get consultation strategy
            strategy = consultation_manager.get_consultation_strategy(
                [synthetic_match],
                priority="accuracy"
            )
            
            # Execute consultation
            consultation_response = consultation_manager.execute_consultation(
                [synthetic_match],
                context,
                strategy
            )
        else:
            # Fallback to original response manager
            consultation_response = response_manager.handle_patterns([synthetic_match], context)
        
        if not consultation_response:
            return "❌ Unable to generate consultation response"
        
        # Format response
        result = f"""## 🤖 Forced AI Consultation - {category.upper()}
        
**Context analyzed**: {len(context)} characters
**Multi-AI mode**: {'Yes' if multi_ai else 'No'}
**AIs Consulted**: {', '.join(consultation_response.ai_responses.keys())}

"""
        
        result += consultation_response.primary_recommendation
        
        if consultation_response.synthesis:
            result += f"\n\n{consultation_response.synthesis}"
        
        return result
        
    except Exception as e:
        return f"❌ Error during forced consultation: {str(e)}"

async def handle_cache_stats() -> str:
    """Get async cache statistics"""
    if not ASYNC_CACHE_AVAILABLE or not (ASYNC_CACHE_AVAILABLE and hasattr(pattern_engine, 'get_performance_stats')):
        return "❌ Async cache is not available"
    
    try:
        # Get performance stats from the async engine
        stats = await pattern_engine.get_performance_stats()
        
        result = "## 📊 Async Pattern Cache Statistics\n\n"
        
        # Engine performance stats
        result += "### Engine Performance:\n"
        result += f"- Total Requests: {stats.get('total_requests', 0)}\n"
        result += f"- Cache Hits: {stats.get('cache_hits', 0)}\n"
        result += f"- Cache Misses: {stats.get('cache_misses', 0)}\n"
        result += f"- Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1f}%\n"
        result += f"- Deduplication Saves: {stats.get('deduplication_saves', 0)}\n"
        result += f"- Deduplication Rate: {stats.get('deduplication_rate', 0):.1f}%\n"
        result += f"- Average Processing Time: {stats.get('avg_processing_time', 0):.3f}s\n"
        
        # Cache metrics
        if 'cache_metrics' in stats:
            cache_metrics = stats['cache_metrics']
            result += "\n### Cache Metrics:\n"
            result += f"- Cache Size: {cache_metrics.get('cache_size', 0)}/{cache_metrics.get('max_size', 0)}\n"
            result += f"- Memory Usage: {cache_metrics.get('memory_usage_mb', 0):.1f}/{cache_metrics.get('max_memory_mb', 0)}MB\n"
            result += f"- Cache Hits: {cache_metrics.get('hits', 0)}\n"
            result += f"- Cache Misses: {cache_metrics.get('misses', 0)}\n"
            result += f"- Cache Hit Rate: {cache_metrics.get('hit_rate', '0%')}\n"
            result += f"- Evictions: {cache_metrics.get('evictions', 0)}\n"
            result += f"- TTL Expirations: {cache_metrics.get('ttl_expirations', 0)}\n"
            result += f"- Pending Requests: {cache_metrics.get('pending_requests', 0)}\n"
            result += f"- Base TTL: {cache_metrics.get('base_ttl_seconds', 0)}s\n"
            
            # Pattern category statistics
            if cache_metrics.get('pattern_category_stats'):
                result += "\n### Pattern Category Stats:\n"
                for category, count in cache_metrics['pattern_category_stats'].items():
                    result += f"- {category.upper()}: {count}\n"
            
            # Sensitivity level statistics
            if cache_metrics.get('sensitivity_level_stats'):
                result += "\n### Sensitivity Level Stats:\n"
                for level, count in cache_metrics['sensitivity_level_stats'].items():
                    result += f"- {level.upper()}: {count}\n"
        
        # Global cache metrics (if using global cache)
        try:
            global_metrics = await get_global_cache_metrics()
            if global_metrics != cache_metrics:  # Different from engine cache
                result += "\n### Global Cache Metrics:\n"
                result += f"- Global Cache Size: {global_metrics.get('cache_size', 0)}\n"
                result += f"- Global Memory Usage: {global_metrics.get('memory_usage_mb', 0):.1f}MB\n"
        except Exception:
            pass  # Global cache not available
        
        result += "\n💡 Use `clear_cache` to reset cache statistics and free memory."
        return result
        
    except Exception as e:
        return f"❌ Error getting cache stats: {str(e)}"

async def handle_clear_cache(confirm: bool = False) -> str:
    """Clear the async cache"""
    if not ASYNC_CACHE_AVAILABLE or not (ASYNC_CACHE_AVAILABLE and hasattr(pattern_engine, 'clear_cache')):
        return "❌ Async cache is not available"
    
    if not confirm:
        return """⚠️ **Cache Clear Confirmation Required**

This will clear all cached pattern detection results and reset statistics.

**Impact:**
- All cached patterns will be removed
- Cache statistics will be reset to zero
- Subsequent pattern detections will need to be recomputed
- Memory usage will be reduced

To proceed, call this function with `confirm: true`"""
    
    try:
        # Clear the cache
        await pattern_engine.clear_cache()
        
        # Get fresh stats to confirm clearing
        stats = await pattern_engine.get_performance_stats()
        
        result = "✅ **Async cache cleared successfully!**\n\n"
        result += "### Post-Clear Status:\n"
        if 'cache_metrics' in stats:
            cache_metrics = stats['cache_metrics']
            result += f"- Cache Size: {cache_metrics.get('cache_size', 0)}\n"
            result += f"- Memory Usage: {cache_metrics.get('memory_usage_mb', 0):.1f}MB\n"
            result += f"- Pending Requests: {cache_metrics.get('pending_requests', 0)}\n"
        
        result += "\n💡 Pattern detection performance may be slower until cache is repopulated."
        return result
        
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"

async def handle_async_pattern_check(text: str, sensitivity_level: str = "medium", auto_consult: bool = True) -> str:
    """Handle async pattern checking with caching"""
    if not ASYNC_CACHE_AVAILABLE or not (ASYNC_CACHE_AVAILABLE and hasattr(pattern_engine, 'detect_patterns_async')):
        return "❌ Async pattern detection is not available"
    
    try:
        # Use async pattern detection
        result = await pattern_engine.detect_patterns_async(text, sensitivity_level)
        
        # Build response
        response = f"## 🔍 Async Pattern Detection Results\n\n"
        
        # Performance info
        response += f"### Performance:\n"
        response += f"- Processing Time: {result.processing_time:.3f}s\n"
        response += f"- Was Cached: {'✅ Yes' if result.was_cached else '❌ No'}\n"
        response += f"- Was Deduplicated: {'✅ Yes' if result.was_deduplicated else '❌ No'}\n"
        if result.cache_age is not None:
            response += f"- Cache Age: {result.cache_age:.3f}s\n"
        response += f"- Sensitivity Level: {result.sensitivity_level}\n"
        
        # Patterns found
        response += f"\n### Patterns Found: {len(result.patterns)}\n"
        
        if not result.patterns:
            response += "✅ No patterns detected that require AI consultation.\n"
            return response
        
        # List patterns by category
        patterns_by_category = {}
        for pattern in result.patterns:
            category = pattern.category.value
            if category not in patterns_by_category:
                patterns_by_category[category] = []
            patterns_by_category[category].append(pattern)
        
        for category, patterns in patterns_by_category.items():
            response += f"\n**{category.upper()}:**\n"
            for pattern in patterns:
                severity_emoji = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                emoji = severity_emoji.get(pattern.severity.name.lower(), "⚪")
                response += f"- {emoji} {pattern.keyword} (confidence: {pattern.confidence:.1f})\n"
                if pattern.context:
                    context_preview = pattern.context[:100] + "..." if len(pattern.context) > 100 else pattern.context
                    response += f"  Context: `{context_preview}`\n"
        
        # Pattern summary
        if result.summary:
            response += f"\n### Summary:\n"
            for key, value in result.summary.items():
                if key != "categories":
                    response += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        # Consultation strategy
        if result.strategy:
            response += f"\n### Recommended Strategy:\n"
            strategy_name = result.strategy.get('strategy', 'unknown')
            response += f"- Strategy: {strategy_name}\n"
            if 'reason' in result.strategy:
                response += f"- Reason: {result.strategy['reason']}\n"
            if 'ai_selection' in result.strategy:
                response += f"- Recommended AIs: {', '.join(result.strategy['ai_selection'])}\n"
        
        # Auto-consult if requested
        if auto_consult and result.patterns:
            consultation_needed = await pattern_engine.should_trigger_consultation_async(result.patterns)
            if consultation_needed:
                response += "\n🤖 **Auto-consulting AIs based on detected patterns...**\n"
                
                # Use the existing junior_consult handler but make it work with async
                try:
                    # This would need to be adapted for async consultation
                    # For now, just indicate that consultation would happen
                    response += "\n💡 Consultation would be triggered here with detected patterns.\n"
                    response += "Use the `junior_consult` tool with this text for AI consultation.\n"
                except Exception as e:
                    response += f"\n❌ Auto-consultation failed: {str(e)}\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error in async pattern detection: {str(e)}"

def handle_tool_call(request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool execution"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    try:
        result = ""
        
        # Pattern detection tools
        if tool_name == "pattern_check":
            text = arguments.get("text", "")
            auto_consult = arguments.get("auto_consult", True)
            result = handle_pattern_check(text, auto_consult)
        
        elif tool_name == "junior_consult":
            context = arguments.get("context", "")
            force_multi_ai = arguments.get("force_multi_ai", False)
            result = handle_junior_consult(context, force_multi_ai)
        
        elif tool_name == "pattern_stats":
            result = handle_pattern_stats()
        
        elif tool_name == "get_sensitivity_config":
            result = handle_get_sensitivity_config()
        
        elif tool_name == "update_sensitivity":
            global_level = arguments.get("global_level")
            category_overrides = arguments.get("category_overrides")
            result = handle_update_sensitivity(global_level, category_overrides)
        
        elif tool_name == "toggle_pattern_detection":
            enabled = arguments.get("enabled", True)
            result = handle_toggle_pattern_detection(enabled)
        
        elif tool_name == "toggle_category":
            category = arguments.get("category", "")
            enabled = arguments.get("enabled", True)
            result = handle_toggle_category(category, enabled)
        
        elif tool_name == "add_pattern_keywords":
            category = arguments.get("category", "")
            keywords = arguments.get("keywords", [])
            result = handle_add_pattern_keywords(category, keywords)
        
        elif tool_name == "remove_pattern_keywords":
            category = arguments.get("category", "")
            keywords = arguments.get("keywords", [])
            result = handle_remove_pattern_keywords(category, keywords)
        
        elif tool_name == "list_pattern_keywords":
            category = arguments.get("category", "")
            result = handle_list_pattern_keywords(category)
        
        elif tool_name == "force_consultation":
            context = arguments.get("context", "")
            category = arguments.get("category", "")
            multi_ai = arguments.get("multi_ai", False)
            result = handle_force_consultation(context, category, multi_ai)
        
        # AI Consultation Manager tools
        elif tool_name == "ai_consultation_strategy":
            context = arguments.get("context", "")
            priority = arguments.get("priority", "accuracy")
            
            if not AI_CONSULTATION_AVAILABLE or not ai_consultation_manager:
                result = "❌ AI Consultation Manager is not available"
            else:
                # Detect patterns first
                if CONTEXT_AWARE_AVAILABLE and hasattr(pattern_engine, 'detect_with_context'):
                    contextual_patterns = pattern_engine.detect_with_context(context)
                    patterns = [cp.base_match for cp in contextual_patterns]
                else:
                    patterns = pattern_engine.detect_patterns(context)
                
                if not patterns:
                    result = "✅ No patterns detected requiring AI consultation."
                else:
                    strategy = ai_consultation_manager.select_ai_strategy(
                        patterns, context, {'priority': priority}
                    )
                    
                    result = f"""## 🎯 AI Consultation Strategy

**Mode**: {strategy.consultation_mode}
**Selected AIs**: {', '.join(strategy.ai_selection)}
**Priority**: {strategy.priority}
**Estimated Time**: {strategy.estimated_time:.1f} seconds
**Estimated Cost**: {strategy.estimated_cost:.3f} units

**Reasoning**: {strategy.reasoning}"""
        
        elif tool_name == "ai_consultation_metrics":
            if not AI_CONSULTATION_AVAILABLE or not ai_consultation_manager:
                result = "❌ AI Consultation Manager is not available"
            else:
                metrics = ai_consultation_manager.get_metrics_summary()
                result = f"""## 📊 AI Consultation Metrics

**Total Consultations**: {metrics['total_consultations']}
**Success Rate**: {metrics['success_rate']}
**Average Response Time**: {metrics['average_response_time']}
**Average Confidence**: {metrics['average_confidence']}
**Most Used AI**: {metrics['most_used_ai']}
**Most Common Pattern**: {metrics['most_common_pattern']}
**Last Updated**: {metrics['last_updated']}"""
        
        elif tool_name == "ai_consultation_audit":
            limit = arguments.get("limit", 10)
            pattern_category = arguments.get("pattern_category")
            
            if not AI_CONSULTATION_AVAILABLE or not ai_consultation_manager:
                result = "❌ AI Consultation Manager is not available"
            else:
                audits = ai_consultation_manager.get_audit_trail(limit, pattern_category)
                
                result = f"## 📋 AI Consultation Audit Trail\n\n"
                if not audits:
                    result += "No consultations found."
                else:
                    for audit in audits:
                        result += f"""### {audit['timestamp']}
**ID**: {audit['consultation_id'][:8]}...
**AIs**: {', '.join(audit['ais_consulted'])}
**Mode**: {audit['mode']}
**Confidence**: {audit['confidence']}
**Time**: {audit['processing_time']}
**Patterns**: {', '.join(audit['pattern_summary'].get('categories', []))}

---
"""
        
        elif tool_name == "ai_governance_report":
            if not AI_CONSULTATION_AVAILABLE or not ai_consultation_manager:
                result = "❌ AI Consultation Manager is not available"
            else:
                report = ai_consultation_manager.export_governance_report()
                
                result = f"""## 🏛️ AI Governance & Compliance Report

### Consultation Metrics
{json.dumps(report['consultation_metrics'], indent=2)}

### AI Usage Distribution
"""
                for ai, count in report['ai_usage_distribution'].items():
                    result += f"- **{ai}**: {count} consultations\n"
                
                result += f"\n### Pattern Distribution\n"
                for pattern, count in report['pattern_distribution'].items():
                    result += f"- **{pattern}**: {count} occurrences\n"
                
                result += f"\n### Compliance Notes\n"
                for key, value in report['compliance_notes'].items():
                    result += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        elif tool_name == "server_status":
            available_ais = list(AI_CLIENTS.keys())
            total_configured = len([ai for ai in CREDENTIALS.keys() if CREDENTIALS[ai].get("enabled", False)])
            result = f"""Junior AI Assistant for Claude Code v{__version__}

🤖 Available AIs: {', '.join([ai.upper() for ai in available_ais])}
📊 Status: {len(available_ais)}/{total_configured} AIs ready
🔍 Pattern Detection: {'Enabled' if PATTERN_DETECTION_AVAILABLE and pattern_config.get('enabled', True) else 'Disabled'}

Configured Models:
"""
            for ai_name, client_info in AI_CLIENTS.items():
                model = client_info.get("model", CREDENTIALS[ai_name]["model"])
                result += f"• {ai_name.upper()}: {model} ✅\n"
            
            # Show disabled AIs
            disabled = [ai for ai in CREDENTIALS.keys() if not CREDENTIALS[ai].get("enabled", False) or ai not in AI_CLIENTS]
            if disabled:
                result += f"\n🚫 Disabled/Unavailable: {', '.join([ai.upper() for ai in disabled])}"
            
            if PATTERN_DETECTION_AVAILABLE:
                result += "\n\n🎯 Pattern Detection Active - Monitoring for:"
                result += "\n• Security vulnerabilities"
                result += "\n• Code uncertainties"
                result += "\n• Algorithm optimizations"
                result += "\n• Programming gotchas"
                result += "\n• Architecture decisions"
        
        elif tool_name == "ask_all_ais":
            prompt = arguments.get("prompt", "")
            temperature = arguments.get("temperature", 0.7)
            result = call_multiple_ais(prompt, list(AI_CLIENTS.keys()), temperature)
        
        elif tool_name == "ai_debate":
            topic = arguments.get("topic", "")
            ai1 = arguments.get("ai1", "gemini")
            ai2 = arguments.get("ai2", "grok")
            
            prompt1 = f"You are debating the topic: '{topic}'. Present your argument in favor of your position. Be persuasive and use examples."
            prompt2 = f"You are debating the topic: '{topic}'. Present a counter-argument to the previous position. Be persuasive and use examples."
            
            response1 = call_ai(ai1, prompt1, 0.8)
            response2 = call_ai(ai2, prompt2, 0.8)
            
            result = f"""🥊 AI DEBATE: {topic}

## 🤖 {ai1.upper()}'s Opening Argument:
{response1}

## 🤖 {ai2.upper()}'s Counter-Argument:
{response2}

---
*Both AIs presented their best arguments!*"""
        
        elif tool_name == "collaborative_solve":
            problem = arguments.get("problem", "")
            approach = arguments.get("approach", "sequential")
            
            if approach == "sequential":
                result = "🧠 COLLABORATIVE PROBLEM SOLVING (Sequential)\n\n"
                for i, ai_name in enumerate(AI_CLIENTS.keys(), 1):
                    prompt = f"Step {i}: Analyze this problem: {problem}. Build on previous insights if any, and provide your unique perspective and solution approach."
                    response = call_ai(ai_name, prompt, 0.6)
                    result += f"## Step {i} - {ai_name.upper()} Analysis:\n{response}\n\n"
            else:  # parallel
                result = call_multiple_ais(f"Solve this complex problem: {problem}", list(AI_CLIENTS.keys()), 0.6)
        
        elif tool_name == "ai_consensus":
            question = arguments.get("question", "")
            options = arguments.get("options", "")
            
            prompt = f"Question: {question}"
            if options:
                prompt += f"\nAvailable options: {options}"
            prompt += "\nProvide your recommendation and reasoning. Be concise but thorough."
            
            responses = []
            for ai_name in AI_CLIENTS.keys():
                response = call_ai(ai_name, prompt, 0.4)
                responses.append(f"## {ai_name.upper()} Recommendation:\n{response}")
            
            result = "🤝 AI CONSENSUS ANALYSIS\n\n" + "\n\n".join(responses)
        
        # Individual AI calls
        elif tool_name.startswith("ask_"):
            ai_name = tool_name.replace("ask_", "")
            prompt = arguments.get("prompt", "")
            temperature = arguments.get("temperature", 0.7)
            result = call_ai(ai_name, prompt, temperature)
        
        # Code review calls
        elif tool_name.endswith("_code_review"):
            ai_name = tool_name.replace("_code_review", "")
            code = arguments.get("code", "")
            focus = arguments.get("focus", "general")
            
            prompt = f"""Please review this code with a focus on {focus}:

```
{code}
```

Provide specific, actionable feedback on:
1. Potential issues or bugs
2. Security concerns
3. Performance optimizations
4. Best practices
5. Code clarity and maintainability"""
            
            result = call_ai(ai_name, prompt, 0.3)
        
        # Deep thinking calls
        elif tool_name.endswith("_think_deep"):
            ai_name = tool_name.replace("_think_deep", "")
            topic = arguments.get("topic", "")
            context = arguments.get("context", "")
            
            prompt = f"Think deeply and analytically about: {topic}"
            if context:
                prompt += f"\n\nContext: {context}"
            prompt += "\n\nProvide comprehensive analysis with multiple perspectives, implications, and detailed reasoning."
            
            result = call_ai(ai_name, prompt, 0.4)
        
        # Brainstorming calls
        elif tool_name.endswith("_brainstorm"):
            ai_name = tool_name.replace("_brainstorm", "")
            challenge = arguments.get("challenge", "")
            constraints = arguments.get("constraints", "")
            
            prompt = f"Brainstorm creative solutions for: {challenge}"
            if constraints:
                prompt += f"\n\nConstraints: {constraints}"
            prompt += "\n\nGenerate multiple innovative ideas, alternatives, and out-of-the-box approaches."
            
            result = call_ai(ai_name, prompt, 0.8)
        
        # Debug assistance calls
        elif tool_name.endswith("_debug"):
            ai_name = tool_name.replace("_debug", "")
            error = arguments.get("error", "")
            code = arguments.get("code", "")
            context = arguments.get("context", "")
            
            prompt = f"Help debug this issue: {error}"
            if code:
                prompt += f"\n\nRelated code:\n```\n{code}\n```"
            if context:
                prompt += f"\n\nContext: {context}"
            prompt += "\n\nProvide debugging steps, potential causes, and specific solutions."
            
            result = call_ai(ai_name, prompt, 0.3)
        
        # Architecture advice calls
        elif tool_name.endswith("_architecture"):
            ai_name = tool_name.replace("_architecture", "")
            requirements = arguments.get("requirements", "")
            constraints = arguments.get("constraints", "")
            scale = arguments.get("scale", "")
            
            prompt = f"Design architecture for: {requirements}"
            if constraints:
                prompt += f"\n\nConstraints: {constraints}"
            if scale:
                prompt += f"\n\nScale requirements: {scale}"
            prompt += "\n\nProvide detailed architecture recommendations, patterns, and implementation guidance."
            
            result = call_ai(ai_name, prompt, 0.5)
        
        # Async cache tools
        elif tool_name == "cache_stats":
            import asyncio
            if ASYNC_CACHE_AVAILABLE:
                result = asyncio.run(handle_cache_stats())
            else:
                result = "❌ Async cache is not available"
        
        elif tool_name == "clear_cache":
            import asyncio
            confirm = arguments.get("confirm", False)
            if ASYNC_CACHE_AVAILABLE:
                result = asyncio.run(handle_clear_cache(confirm))
            else:
                result = "❌ Async cache is not available"
        
        elif tool_name == "async_pattern_check":
            import asyncio
            text = arguments.get("text", "")
            sensitivity_level = arguments.get("sensitivity_level", "medium")
            auto_consult = arguments.get("auto_consult", True)
            if ASYNC_CACHE_AVAILABLE:
                result = asyncio.run(handle_async_pattern_check(text, sensitivity_level, auto_consult))
            else:
                result = "❌ Async pattern detection is not available"
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result
                    }
                ]
            }
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }

def cleanup():
    """Cleanup resources on shutdown"""
    if PATTERN_DETECTION_AVAILABLE and text_pipeline:
        text_pipeline.stop()
    
    # Cleanup async components
    if ASYNC_CACHE_AVAILABLE:
        import asyncio
        try:
            # Close async pattern engine if available
            if ASYNC_CACHE_AVAILABLE and hasattr(pattern_engine, 'close'):
                asyncio.run(pattern_engine.close())
            
            # Close async pipeline if available
            if 'async_pipeline' in globals() and async_pipeline:
                asyncio.run(async_pipeline.close())
        except Exception as e:
            print(f"Warning: Error during async cleanup: {e}", file=sys.stderr)

def main():
    """Main server loop"""
    try:
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                method = request.get("method")
                request_id = request.get("id")
                params = request.get("params", {})
                
                if method == "initialize":
                    response = handle_initialize(request_id)
                elif method == "tools/list":
                    response = handle_tools_list(request_id)
                elif method == "tools/call":
                    response = handle_tool_call(request_id, params)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                
                send_response(response)
                
            except json.JSONDecodeError:
                continue
            except EOFError:
                break
            except Exception as e:
                if 'request_id' in locals():
                    send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    })
    finally:
        cleanup()

if __name__ == "__main__":
    main()
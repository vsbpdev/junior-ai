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
    from pattern_detection import PatternDetectionEngine, PatternCategory, PatternSeverity
    from text_processing_pipeline import TextProcessingPipeline, ProcessingResult
    from response_handlers import PatternResponseManager, ConsultationResponse
    PATTERN_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pattern detection not available: {e}", file=sys.stderr)
    PATTERN_DETECTION_AVAILABLE = False

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
    pattern_engine = PatternDetectionEngine()
    text_pipeline = TextProcessingPipeline(pattern_engine=pattern_engine)
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
    text_pipeline = None
    response_manager = None
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
    
    return "\n\n" + "="*80 + "\n\n".join(results)

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
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": tools
        }
    }

def handle_pattern_check(text: str, auto_consult: bool = True) -> str:
    """Handle pattern checking in text"""
    if not PATTERN_DETECTION_AVAILABLE or not pattern_config.get("enabled", True):
        return "❌ Pattern detection is not available or disabled"
    
    # Detect patterns
    patterns = pattern_engine.detect_patterns(text)
    
    if not patterns:
        return "✅ No patterns detected that require AI consultation."
    
    # Generate pattern summary
    summary = pattern_engine.get_pattern_summary(patterns)
    strategy = pattern_engine.get_consultation_strategy(patterns)
    
    result = f"""## 🔍 Pattern Detection Results

**Patterns Found:** {summary['total_matches']}
**Max Severity:** {summary['max_severity'].name if summary['max_severity'] else 'N/A'}
**Consultation Strategy:** {strategy['strategy']}

### Detected Categories:
"""
    
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

def handle_junior_consult(context: str, force_multi_ai: bool = False) -> str:
    """Handle smart junior AI consultation"""
    if not PATTERN_DETECTION_AVAILABLE:
        return "❌ Pattern detection is not available"
    
    # Detect patterns
    patterns = pattern_engine.detect_patterns(context)
    
    if not patterns and not force_multi_ai:
        # No patterns detected, provide general assistance
        default_ai = pattern_config.get("default_junior", "openrouter")
        prompt = f"Please provide guidance on the following:\n\n{context}"
        response = call_ai(default_ai, prompt, 0.7)
        return f"## 💡 Junior AI Assistance\n\n{response}"
    
    # Force multi-AI if requested
    if force_multi_ai and patterns:
        for pattern in patterns:
            pattern.requires_multi_ai = True
    
    # Handle patterns
    consultation_response = response_manager.handle_patterns(patterns, context)
    
    if not consultation_response:
        return "❌ Unable to generate consultation response"
    
    # Format response
    result = f"""## 🤖 Junior AI Consultation

**Consultation ID:** {consultation_response.request_id}
**Confidence Score:** {consultation_response.confidence_score:.2%}
**AIs Consulted:** {', '.join(consultation_response.ai_responses.keys())}

"""
    
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
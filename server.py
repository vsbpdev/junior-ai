#!/usr/bin/env python3
"""
Multi-AI MCP Server
Enables Claude Code to collaborate with Gemini, Grok-3, ChatGPT, and DeepSeek
"""

import json
import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Server version
__version__ = "1.0.0"

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
                "name": "multi-ai-mcp",
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
    
    # Individual AI tools
    for ai_name in AI_CLIENTS.keys():
        tools.extend([
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
            {
                "name": f"{ai_name}_code_review",
                "description": f"Have {ai_name.upper()} review code",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to review"
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus area (security, performance, etc.)",
                            "default": "general"
                        }
                    },
                    "required": ["code"]
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
            }
        ])
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": tools
        }
    }

def handle_tool_call(request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool execution"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    try:
        result = ""
        
        if tool_name == "server_status":
            available_ais = list(AI_CLIENTS.keys())
            total_configured = len([ai for ai in CREDENTIALS.keys() if CREDENTIALS[ai].get("enabled", False)])
            result = f"""Multi-AI MCP Server v{__version__}

🤖 Available AIs: {', '.join([ai.upper() for ai in available_ais])}
📊 Status: {len(available_ais)}/{total_configured} AIs ready

Configured Models:
"""
            for ai_name, client_info in AI_CLIENTS.items():
                model = client_info.get("model", CREDENTIALS[ai_name]["model"])
                result += f"• {ai_name.upper()}: {model} ✅\n"
            
            # Show disabled AIs
            disabled = [ai for ai in CREDENTIALS.keys() if not CREDENTIALS[ai].get("enabled", False) or ai not in AI_CLIENTS]
            if disabled:
                result += f"\n🚫 Disabled/Unavailable: {', '.join([ai.upper() for ai in disabled])}"
        
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

def main():
    """Main server loop"""
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

if __name__ == "__main__":
    main()
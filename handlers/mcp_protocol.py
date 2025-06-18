"""MCP protocol handlers for initialization and tool listing."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from core.config import __version__


@dataclass
class ServerInfo:
    """Server information for MCP protocol."""
    name: str
    version: str


@dataclass
class AIClientInfo:
    """Information about an AI client."""
    name: str
    client: Any
    type: str  # 'gemini' or 'openai'


@dataclass
class PatternDetectionConfig:
    """Pattern detection configuration."""
    enabled: bool = True
    default_junior: str = "openrouter"
    accuracy_mode: bool = True


class MCPProtocolHandler:
    """Handles MCP protocol initialization and tool listing."""
    
    def __init__(
        self,
        server_info: ServerInfo,
        ai_clients: Dict[str, AIClientInfo],
        pattern_detection_available: bool = False,
        ai_consultation_available: bool = False,
        async_cache_available: bool = False,
        pattern_config: Optional[PatternDetectionConfig] = None
    ):
        """Initialize the MCP protocol handler."""
        self.server_info = server_info
        self.ai_clients = ai_clients
        self.pattern_detection_available = pattern_detection_available
        self.ai_consultation_available = ai_consultation_available
        self.async_cache_available = async_cache_available
        self.pattern_config = pattern_config or PatternDetectionConfig()
    
    def handle_initialize(self, request_id: Any) -> Dict[str, Any]:
        """Handle MCP initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": self.server_info.name,
                    "version": self.server_info.version
                }
            }
        }
    
    def handle_tools_list(self, request_id: Any) -> Dict[str, Any]:
        """List all available tools."""
        tools = []
        
        # Add base tools
        tools.extend(self._get_base_tools())
        
        # Add pattern detection tools if available
        if self.pattern_detection_available and self.pattern_config.enabled:
            tools.extend(self._get_pattern_detection_tools())
            
            # Add AI consultation manager tools if available
            if self.ai_consultation_available:
                tools.extend(self._get_ai_consultation_tools())
        
        # Add individual AI tools
        tools.extend(self._get_individual_ai_tools())
        
        # Add collaborative tools if multiple AIs available
        if len(self.ai_clients) > 1:
            tools.extend(self._get_collaborative_tools())
        
        # Add async cache tools if available
        if self.async_cache_available and self.pattern_detection_available:
            tools.extend(self._get_async_cache_tools())
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools
            }
        }
    
    def _get_base_tools(self) -> List[Dict[str, Any]]:
        """Get base server tools."""
        return [
            {
                "name": "server_status",
                "description": "Get server status and available AI models",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def _get_pattern_detection_tools(self) -> List[Dict[str, Any]]:
        """Get pattern detection related tools."""
        return [
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
        ]
    
    def _get_ai_consultation_tools(self) -> List[Dict[str, Any]]:
        """Get AI consultation manager tools."""
        return [
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
        ]
    
    def _get_individual_ai_tools(self) -> List[Dict[str, Any]]:
        """Get individual AI tools for each configured AI."""
        tools = []
        
        for ai_name in self.ai_clients.keys():
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
        
        return tools
    
    def _get_collaborative_tools(self) -> List[Dict[str, Any]]:
        """Get collaborative AI tools."""
        return [
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
        ]
    
    def _get_async_cache_tools(self) -> List[Dict[str, Any]]:
        """Get async cache related tools."""
        return [
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
        ]
"""Handlers for individual AI tool calls.

This module implements handlers for AI-specific tools, providing specialized
interfaces for different types of AI interactions. Each configured AI gets
its own set of tools with specific capabilities.

Tool types provided for each AI:
- ask_{ai_name}: General purpose questions and prompts
- {ai_name}_code_review: Code analysis and review
- {ai_name}_think_deep: Extended reasoning and deep analysis
- {ai_name}_brainstorm: Creative problem solving
- {ai_name}_debug: Debugging assistance
- {ai_name}_architecture: System design and architecture advice

The handler dynamically generates tools based on available AI clients and
routes requests to the appropriate AI with specialized prompts.
"""

from typing import Dict, Any, List
from .base import BaseHandler
from ai.caller import call_ai


class AIToolsHandler(BaseHandler):
    """Handles individual AI tool calls."""
    
    def get_tool_names(self) -> List[str]:
        """Return list of tool names this handler supports."""
        tools = []
        for ai_name in self.ai_clients:
            tools.extend([
                f"ask_{ai_name}",
                f"{ai_name}_code_review",
                f"{ai_name}_think_deep",
                f"{ai_name}_brainstorm",
                f"{ai_name}_debug",
                f"{ai_name}_architecture"
            ])
        return tools
    
    def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle an AI tool call."""
        # Extract AI name from tool name
        ai_name = self._extract_ai_name(tool_name)
        if not ai_name:
            return f"❌ Could not determine AI from tool name: {tool_name}"
        
        # Dispatch table for tool routing
        tool_handlers = {
            lambda name: name.startswith("ask_"): self._handle_ask,
            lambda name: name.endswith("_code_review"): self._handle_code_review,
            lambda name: name.endswith("_think_deep"): self._handle_think_deep,
            lambda name: name.endswith("_brainstorm"): self._handle_brainstorm,
            lambda name: name.endswith("_debug"): self._handle_debug,
            lambda name: name.endswith("_architecture"): self._handle_architecture,
        }
        
        # Find matching handler
        for condition, handler in tool_handlers.items():
            if condition(tool_name):
                return handler(ai_name, arguments)
        
        return f"❌ Unknown tool type: {tool_name}"
    
    def _extract_ai_name(self, tool_name: str) -> str:
        """Extract AI name from tool name."""
        # Check all configured AI names
        for ai_name in self.ai_clients:
            if ai_name in tool_name:
                return ai_name
        return ""
    
    def _handle_ask(self, ai_name: str, arguments: Dict[str, Any]) -> str:
        """Handle basic AI question."""
        prompt = arguments.get('prompt', '')
        temperature = arguments.get('temperature', 0.7)
        
        if not prompt:
            return "❌ Missing required parameter: prompt"
        
        return call_ai(ai_name, prompt, temperature)
    
    def _handle_code_review(self, ai_name: str, arguments: Dict[str, Any]) -> str:
        """Handle code review request."""
        code = arguments.get('code', '')
        focus = arguments.get('focus', 'general')
        
        if not code:
            return "❌ Missing required parameter: code"
        
        security_section = "- Security vulnerabilities" if focus == "security" else ""
        performance_section = "- Performance optimizations" if focus == "performance" else ""
        readability_section = "- Readability and maintainability" if focus == "readability" else ""
        
        prompt = f"""Please review the following code and provide feedback focused on {focus}:

```code
{code}
```

Please analyze for:
- Code quality and best practices
- Potential bugs or issues
- Performance considerations
- Suggestions for improvement
{security_section}
{performance_section}
{readability_section}"""
        
        return call_ai(ai_name, prompt, 0.7)
    
    def _handle_think_deep(self, ai_name: str, arguments: Dict[str, Any]) -> str:
        """Handle deep analysis request."""
        topic = arguments.get('topic', '')
        context = arguments.get('context', '')
        
        if not topic:
            return "❌ Missing required parameter: topic"
        
        context_line = f"Context: {context}" if context else ""
        
        prompt = f"""Please provide a deep, thorough analysis of the following topic:

{topic}

{context_line}

Take your time to think through this carefully and provide:
- Multiple perspectives and viewpoints
- Detailed reasoning and evidence
- Potential implications and consequences
- Creative insights and connections
- Actionable recommendations where appropriate"""
        
        return call_ai(ai_name, prompt, 0.8)
    
    def _handle_brainstorm(self, ai_name: str, arguments: Dict[str, Any]) -> str:
        """Handle brainstorming request."""
        challenge = arguments.get('challenge', '')
        constraints = arguments.get('constraints', '')
        
        if not challenge:
            return "❌ Missing required parameter: challenge"
        
        constraints_line = f"Constraints: {constraints}" if constraints else ""
        
        prompt = f"""Let's brainstorm creative solutions for this challenge:

{challenge}

{constraints_line}

Please provide:
- Multiple creative approaches (aim for at least 5-7 different ideas)
- Both conventional and unconventional solutions
- Consider different angles and perspectives
- Explain the potential benefits and drawbacks of each approach
- Suggest which might be most promising and why"""
        
        return call_ai(ai_name, prompt, 0.9)
    
    def _handle_debug(self, ai_name: str, arguments: Dict[str, Any]) -> str:
        """Handle debug assistance request."""
        error = arguments.get('error', '')
        code = arguments.get('code', '')
        context = arguments.get('context', '')
        
        if not error:
            return "❌ Missing required parameter: error"
        
        code_section = f"Code:\n```\n{code}\n```" if code else ""
        context_section = f"Additional context: {context}" if context else ""
        
        prompt = f"""Help me debug this issue:

Error: {error}

{code_section}

{context_section}

Please provide:
- Likely causes of this error
- Step-by-step debugging approach
- Potential solutions or fixes
- How to prevent this issue in the future"""
        
        return call_ai(ai_name, prompt, 0.7)
    
    def _handle_architecture(self, ai_name: str, arguments: Dict[str, Any]) -> str:
        """Handle architecture design request."""
        requirements = arguments.get('requirements', '')
        constraints = arguments.get('constraints', '')
        scale = arguments.get('scale', '')
        
        if not requirements:
            return "❌ Missing required parameter: requirements"
        
        constraints_line = f"Constraints: {constraints}" if constraints else ""
        scale_line = f"Expected scale: {scale}" if scale else ""
        
        prompt = f"""Please provide architecture design recommendations for:

Requirements: {requirements}

{constraints_line}
{scale_line}

Please include:
- High-level architecture overview
- Key components and their responsibilities
- Technology stack recommendations
- Data flow and integration points
- Scalability and performance considerations
- Security and reliability measures
- Potential challenges and mitigation strategies"""
        
        return call_ai(ai_name, prompt, 0.7)
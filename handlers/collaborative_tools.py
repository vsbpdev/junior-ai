"""Handlers for collaborative AI tool calls.

This module implements multi-AI collaboration tools that leverage multiple
AI models working together. These tools enable different collaboration
patterns for enhanced problem-solving and decision-making.

Collaboration modes:
- ask_all_ais: Parallel consultation of all available AIs
- ai_debate: Two AIs presenting different perspectives on a topic
- collaborative_solve: Multiple approaches (sequential, parallel, debate)
  for complex problem solving
- ai_consensus: Building consensus among multiple AI opinions

The handler supports various collaboration strategies:
- Sequential: Each AI builds upon previous responses
- Parallel: All AIs work independently on the same problem
- Debate: AIs present contrasting viewpoints with optional synthesis
- Consensus: Aggregating multiple perspectives into unified recommendations
"""

from typing import Dict, Any, List
from .base import BaseHandler
from ai.caller import call_ai, call_multiple_ais
from ai.response_formatter import (
    format_multi_ai_response,
    format_debate_response,
    format_consensus_response,
    format_collaborative_response
)


class CollaborativeToolsHandler(BaseHandler):
    """Handles multi-AI collaborative tool calls."""
    
    def get_tool_names(self) -> List[str]:
        """Return list of tool names this handler supports."""
        # Only available if multiple AIs are configured
        if len(self.ai_clients) > 1:
            return [
                "ask_all_ais",
                "ai_debate", 
                "collaborative_solve",
                "ai_consensus"
            ]
        return []
    
    def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a collaborative tool call."""
        # Dispatch table for collaborative tools
        tool_handlers = {
            "ask_all_ais": self._handle_ask_all_ais,
            "ai_debate": self._handle_ai_debate,
            "collaborative_solve": self._handle_collaborative_solve,
            "ai_consensus": self._handle_ai_consensus,
        }
        
        handler = tool_handlers.get(tool_name)
        if handler:
            return handler(arguments)
        
        return f"❌ Unknown collaborative tool: {tool_name}"
    
    def _handle_ask_all_ais(self, arguments: Dict[str, Any]) -> str:
        """Handle asking all AIs the same question."""
        prompt = arguments.get('prompt', '')
        temperature = arguments.get('temperature', 0.7)
        
        if not prompt:
            return "❌ Missing required parameter: prompt"
        
        responses = call_multiple_ais(prompt, None, temperature)
        return format_multi_ai_response(responses)
    
    def _handle_ai_debate(self, arguments: Dict[str, Any]) -> str:
        """Handle AI debate request."""
        topic = arguments.get('topic', '')
        ai1 = arguments.get('ai1', 'gemini')
        ai2 = arguments.get('ai2', 'grok')
        
        if not topic:
            return "❌ Missing required parameter: topic"
        
        # Validate AI availability
        if ai1 not in self.ai_clients:
            return f"❌ {ai1.upper()} is not available"
        if ai2 not in self.ai_clients:
            return f"❌ {ai2.upper()} is not available"
        
        # Create debate prompts
        debate_prompt_1 = f"""You are participating in a debate about: {topic}

Please provide your position on this topic with:
- Clear arguments supporting your view
- Evidence and reasoning
- Consideration of potential counterarguments
- A compelling conclusion

Be thorough but concise."""
        
        debate_prompt_2 = f"""You are participating in a debate about: {topic}

Please provide a different perspective on this topic with:
- Clear arguments that might contrast with common views
- Evidence and reasoning from alternative angles
- Consideration of different stakeholder perspectives
- A thought-provoking conclusion

Be thorough but concise."""
        
        # Get responses
        response1 = call_ai(ai1, debate_prompt_1, 0.8)
        response2 = call_ai(ai2, debate_prompt_2, 0.8)
        
        return format_debate_response(topic, ai1, response1, ai2, response2)
    
    def _handle_collaborative_solve(self, arguments: Dict[str, Any]) -> str:
        """Handle collaborative problem solving."""
        problem = arguments.get('problem', '')
        approach = arguments.get('approach', 'sequential')
        
        if not problem:
            return "❌ Missing required parameter: problem"
        
        available_ais = list(self.ai_clients)
        
        # Dispatch table for solving approaches
        solve_handlers = {
            "sequential": self._solve_sequential,
            "parallel": self._solve_parallel,
            "debate": self._solve_debate,
        }
        
        handler = solve_handlers.get(approach)
        if handler:
            return handler(problem, available_ais)
        
        return f"❌ Unknown approach: {approach}"
    
    def _solve_sequential(self, problem: str, ai_list: List[str]) -> str:
        """Sequential problem solving - each AI builds on previous."""
        responses = []
        current_context = problem
        
        for i, ai_name in enumerate(ai_list):
            if i == 0:
                prompt = f"Please analyze this problem: {problem}\n\nProvide your initial analysis and approach."
            else:
                prompt = f"Building on the previous analysis:\n\n{current_context}\n\nPlease add your insights and extend the solution."
            
            response = call_ai(ai_name, prompt, 0.7)
            responses.append({ai_name: response})
            current_context = f"Problem: {problem}\n\nPrevious analysis:\n{response}"
        
        return format_collaborative_response(problem, responses, "sequential")
    
    def _solve_parallel(self, problem: str, ai_list: List[str]) -> str:
        """Parallel problem solving - all AIs work independently."""
        prompt = f"""Please provide a comprehensive solution to this problem:

{problem}

Include:
- Problem analysis
- Proposed approach
- Implementation steps
- Potential challenges
- Success criteria"""
        
        responses = call_multiple_ais(prompt, ai_list, 0.7)
        return format_collaborative_response(problem, [responses], "parallel")
    
    def _solve_debate(self, problem: str, ai_list: List[str]) -> str:
        """Debate-style problem solving."""
        if len(ai_list) < 2:
            return "❌ Need at least 2 AIs for debate approach"
        
        # Initial positions
        prompt1 = f"Propose a solution to: {problem}\n\nFocus on practicality and proven approaches."
        prompt2 = f"Propose a solution to: {problem}\n\nFocus on innovation and creative approaches."
        
        response1 = call_ai(ai_list[0], prompt1, 0.8)
        response2 = call_ai(ai_list[1], prompt2, 0.8)
        
        # Synthesis
        if len(ai_list) > 2:
            synthesis_prompt = f"""Given these two approaches to solving: {problem}

Approach 1: {response1}

Approach 2: {response2}

Please synthesize the best elements of both into a unified solution."""
            
            synthesis = call_ai(ai_list[2], synthesis_prompt, 0.7)
            
            responses = [
                {ai_list[0]: response1},
                {ai_list[1]: response2},
                {ai_list[2]: synthesis}
            ]
        else:
            responses = [
                {ai_list[0]: response1},
                {ai_list[1]: response2}
            ]
        
        return format_collaborative_response(problem, responses, "debate")
    
    def _handle_ai_consensus(self, arguments: Dict[str, Any]) -> str:
        """Handle consensus building request."""
        question = arguments.get('question', '')
        options = arguments.get('options', '')
        
        if not question:
            return "❌ Missing required parameter: question"
        
        # Build consensus prompt
        prompt = f"Please analyze this question: {question}\n\n"
        
        if options:
            prompt += f"Consider these options:\n{options}\n\n"
        
        prompt += """Provide:
1. Your recommendation or answer
2. Key reasoning points
3. Potential risks or downsides
4. Confidence level (high/medium/low)"""
        
        # Get all AI responses
        responses = call_multiple_ais(prompt, None, 0.7)
        
        # Parse options into list if provided
        option_list = []
        if options:
            # Simple split by newlines or commas
            if '\n' in options:
                option_list = [opt.strip() for opt in options.split('\n') if opt.strip()]
            else:
                option_list = [opt.strip() for opt in options.split(',') if opt.strip()]
        
        return format_consensus_response(question, responses, option_list)
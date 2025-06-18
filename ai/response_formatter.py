"""Response formatting utilities for AI responses.

This module provides formatting functions to structure AI responses
in a consistent and readable format. It handles various response types
including single AI responses, multi-AI comparisons, debates, and
collaborative problem-solving outputs.

Key formatting functions:
- format_ai_response: Format single AI response with headers
- format_multi_ai_response: Format responses from multiple AIs
- format_debate_response: Structure debate between two AIs
- format_consensus_response: Format consensus-building outputs
- format_collaborative_response: Format collaborative problem-solving

The formatters ensure:
- Consistent markdown formatting
- Clear separation between different AI responses
- Proper error response handling
- Structured output for different collaboration modes
"""

from typing import Dict, List


def format_ai_response(ai_name: str, response: str) -> str:
    """Format a single AI response with appropriate headers."""
    header = f"### {ai_name.upper()} Response\n\n"
    return header + response


def format_multi_ai_response(responses: Dict[str, str]) -> str:
    """Format responses from multiple AIs into a coherent output."""
    if not responses:
        return "❌ No AI responses received"
    
    formatted_parts = []
    
    for ai_name, response in responses.items():
        if response.startswith("❌"):
            # Error response - include as-is
            formatted_parts.append(f"### {ai_name.upper()}\n{response}")
        else:
            # Success response
            formatted_parts.append(format_ai_response(ai_name, response))
    
    return "\n\n---\n\n".join(formatted_parts)


def format_debate_response(topic: str, ai1: str, response1: str, ai2: str, response2: str) -> str:
    """Format a debate between two AIs."""
    output = f"# AI Debate: {topic}\n\n"
    output += f"## {ai1.upper()}'s Position:\n\n{response1}\n\n"
    output += f"## {ai2.upper()}'s Position:\n\n{response2}\n\n"
    return output


def format_consensus_response(question: str, responses: Dict[str, str], options: List[str]) -> str:
    """Format a consensus-building response."""
    output = f"# AI Consensus on: {question}\n\n"
    
    if options:
        output += "## Options considered:\n"
        for i, option in enumerate(options, 1):
            output += f"{i}. {option}\n"
        output += "\n"
    
    output += "## Individual Perspectives:\n\n"
    
    for ai_name, response in responses.items():
        if not response.startswith("❌"):
            output += f"### {ai_name.upper()}:\n{response}\n\n"
    
    return output


def format_collaborative_response(problem: str, responses: List[Dict[str, str]], approach: str) -> str:
    """Format a collaborative problem-solving response."""
    output = f"# Collaborative Problem Solving\n\n"
    output += f"**Problem:** {problem}\n"
    output += f"**Approach:** {approach}\n\n"
    
    if approach == "sequential":
        output += "## Sequential Analysis:\n\n"
        for i, resp_dict in enumerate(responses, 1):
            for ai_name, response in resp_dict.items():
                output += f"### Step {i} - {ai_name.upper()}:\n{response}\n\n"
    else:
        output += "## Parallel Analysis:\n\n"
        # Merge all responses by AI
        merged = {}
        for resp_dict in responses:
            for ai_name, response in resp_dict.items():
                if ai_name not in merged:
                    merged[ai_name] = []
                merged[ai_name].append(response)
        
        for ai_name, ai_responses in merged.items():
            output += f"### {ai_name.upper()}:\n"
            for resp in ai_responses:
                output += f"{resp}\n\n"
    
    return output
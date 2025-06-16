#!/usr/bin/env python3
"""
Integration patch for Response Synthesis in server.py
This shows the changes needed to integrate the new synthesis module
"""

# Add these imports at the top of server.py
"""
from response_synthesis import (
    ResponseSynthesizer, SynthesisStrategy, synthesize_responses
)
from response_handlers_enhanced import EnhancedPatternResponseManager
"""

# Replace the existing response_manager initialization with:
"""
# Initialize enhanced response manager with synthesis
response_manager = EnhancedPatternResponseManager()
"""

# Add a new tool for synthesis strategy selection:
def register_synthesis_tools():
    """Register response synthesis tools"""
    return [
        {
            "name": "set_synthesis_strategy",
            "description": "Set the default synthesis strategy for AI responses",
            "input_schema": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Synthesis strategy to use",
                        "enum": ["consensus", "debate", "expert", "comprehensive", "summary", "hierarchical"]
                    }
                },
                "required": ["strategy"]
            }
        },
        {
            "name": "get_synthesis_strategies",
            "description": "Get available synthesis strategies and their descriptions",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        }
    ]

# Add handler for synthesis strategy tools:
def handle_synthesis_tools(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Handle synthesis-related tools"""
    
    if tool_name == "set_synthesis_strategy":
        strategy_name = arguments.get("strategy", "consensus")
        try:
            strategy = SynthesisStrategy(strategy_name)
            response_manager.set_default_synthesis_strategy(strategy)
            
            return f"""✅ Synthesis strategy updated to: **{strategy_name}**

This strategy will be used for all multi-AI consultations. 

**{strategy_name.title()} Strategy**: {get_strategy_description(strategy)}

Use `get_synthesis_strategies` to see all available strategies."""
        
        except ValueError:
            return f"❌ Invalid strategy: {strategy_name}. Use get_synthesis_strategies to see valid options."
    
    elif tool_name == "get_synthesis_strategies":
        return """## 🎨 Available Synthesis Strategies

### 1. **consensus** (Default)
Finds common agreements and shared recommendations across all AI responses. Best for security and best practices where consensus is valuable.

### 2. **debate**
Highlights different perspectives and disagreements. Ideal for architecture decisions and design choices where multiple valid approaches exist.

### 3. **expert** (expert_weighted)
Weights responses based on each AI's expertise in the specific domain. Great for algorithm optimization and specialized technical questions.

### 4. **comprehensive**
Includes all perspectives without filtering. Useful for uncertainty resolution where you need complete information.

### 5. **summary**
Provides concise key points only. Good for quick overviews and time-sensitive consultations.

### 6. **hierarchical**
Organizes responses by importance and relevance. Suitable for complex topics with multiple layers of information.

Use `set_synthesis_strategy` to change the active strategy."""

def get_strategy_description(strategy: SynthesisStrategy) -> str:
    """Get description for a synthesis strategy"""
    descriptions = {
        SynthesisStrategy.CONSENSUS: "Finds common agreements and best practices across AI responses",
        SynthesisStrategy.DEBATE: "Highlights different approaches and perspectives",
        SynthesisStrategy.EXPERT_WEIGHTED: "Weights responses based on AI expertise in the domain",
        SynthesisStrategy.COMPREHENSIVE: "Includes all perspectives comprehensively",
        SynthesisStrategy.SUMMARY: "Provides concise key points only",
        SynthesisStrategy.HIERARCHICAL: "Organizes by importance and relevance"
    }
    return descriptions.get(strategy, "Custom synthesis strategy")

# Enhanced handle_junior_consult function with synthesis options:
def handle_junior_consult_enhanced(context: str, force_multi_ai: bool = False, 
                                 filename: Optional[str] = None,
                                 synthesis_strategy: Optional[str] = None) -> str:
    """Enhanced junior AI consultation with synthesis options"""
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
    
    # Determine synthesis strategy
    force_strategy = None
    if synthesis_strategy:
        try:
            force_strategy = SynthesisStrategy(synthesis_strategy)
        except ValueError:
            pass
    
    # Use enhanced response manager
    if force_multi_ai and patterns:
        for pattern in patterns:
            pattern.requires_multi_ai = True
    
    consultation_response = response_manager.handle_patterns(
        patterns, context, force_strategy
    )
    
    if not consultation_response:
        return "❌ Unable to generate consultation response"
    
    # Format enhanced response
    result = f"""## 🤖 Junior AI Consultation

**Consultation ID:** {consultation_response.request_id}
**Confidence Score:** {consultation_response.confidence_score:.2%}
**AIs Consulted:** {', '.join(consultation_response.ai_responses.keys())}
**Synthesis Strategy:** {consultation_response.metadata.get('synthesis_strategy', 'default')}
"""
    
    # Add synthesis details if available
    if consultation_response.synthesized_response:
        synth = consultation_response.synthesized_response
        result += f"""
**Synthesis Metrics:**
- Processing Time: {synth.synthesis_time:.3f}s
- Sections Generated: {len(synth.sections)}
- Key Insights: {len(synth.key_insights)}
- Points of Agreement: {len(synth.agreements)}
- Different Perspectives: {len(synth.disagreements)}
"""
    
    result += "\n"
    
    # Add the synthesized content
    result += consultation_response.synthesis or consultation_response.primary_recommendation
    
    # Optionally show individual responses in debug mode
    if pattern_config.get("debug_mode", False) and len(consultation_response.ai_responses) > 1:
        result += "\n\n---\n### 🔍 Debug: Individual AI Responses\n"
        for ai_name, response in consultation_response.ai_responses.items():
            result += f"\n**{ai_name.upper()}:**\n{response[:300]}...\n"
    
    return result

# Update the tool handling in handle_tool_call to include synthesis tools:
"""
# In handle_tool_call function, add:

elif tool_name == "set_synthesis_strategy":
    strategy = arguments.get("strategy")
    result = handle_synthesis_tools("set_synthesis_strategy", {"strategy": strategy})

elif tool_name == "get_synthesis_strategies":
    result = handle_synthesis_tools("get_synthesis_strategies", {})

# And update junior_consult to support synthesis strategy:
elif tool_name == "junior_consult":
    context = arguments.get("context", "")
    force_multi_ai = arguments.get("force_multi_ai", False)
    synthesis_strategy = arguments.get("synthesis_strategy")
    result = handle_junior_consult_enhanced(context, force_multi_ai, synthesis_strategy=synthesis_strategy)
"""

# Add to the tool list in main():
"""
# In the tools list, add:
{
    "name": "junior_consult",
    "description": "Smart AI consultation with pattern-based routing and response synthesis",
    "input_schema": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": "The context or question for consultation"
            },
            "force_multi_ai": {
                "type": "boolean",
                "description": "Force multi-AI consultation even for simple patterns",
                "default": False
            },
            "synthesis_strategy": {
                "type": "string",
                "description": "Response synthesis strategy",
                "enum": ["consensus", "debate", "expert", "comprehensive", "summary", "hierarchical"]
            }
        },
        "required": ["context"]
    }
},
"""

# Example usage in the server:
def demonstrate_synthesis():
    """Example of using different synthesis strategies"""
    
    # Security pattern - best with consensus
    security_context = "How should I implement password hashing securely?"
    security_response = handle_junior_consult_enhanced(
        security_context, 
        force_multi_ai=True,
        synthesis_strategy="consensus"
    )
    
    # Architecture pattern - best with debate
    arch_context = "Should I use microservices or monolithic architecture?"
    arch_response = handle_junior_consult_enhanced(
        arch_context,
        force_multi_ai=True,
        synthesis_strategy="debate"
    )
    
    # Algorithm pattern - best with expert weighting
    algo_context = "How can I optimize this O(n^2) sorting algorithm?"
    algo_response = handle_junior_consult_enhanced(
        algo_context,
        force_multi_ai=True,
        synthesis_strategy="expert"
    )
    
    return {
        "security": security_response,
        "architecture": arch_response,
        "algorithm": algo_response
    }


# Configuration updates for credentials.json:
SYNTHESIS_CONFIG = {
    "default_strategy": "consensus",
    "strategy_overrides": {
        "security": "consensus",
        "architecture": "debate",
        "algorithm": "expert",
        "uncertainty": "comprehensive",
        "gotcha": "debate"
    },
    "enable_debug_mode": False,
    "show_synthesis_metrics": True,
    "min_ais_for_synthesis": 2
}


if __name__ == "__main__":
    print("Response Synthesis Integration Guide")
    print("=====================================")
    print("\nThis file shows how to integrate the response synthesis module into server.py")
    print("\nKey changes:")
    print("1. Import synthesis modules")
    print("2. Replace response_manager with EnhancedPatternResponseManager")
    print("3. Add synthesis strategy tools")
    print("4. Update junior_consult to support synthesis strategies")
    print("5. Add synthesis configuration to credentials.json")
    print("\nSee the code above for detailed implementation.")
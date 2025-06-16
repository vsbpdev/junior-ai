#!/usr/bin/env python3
"""
Enhanced Pattern-Specific Response Handlers for Junior AI Assistant
Integrates with the new Response Synthesis module for better AI response handling
"""

import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from pattern_detection import PatternCategory, PatternMatch, PatternSeverity
from response_synthesis import (
    ResponseSynthesizer, SynthesisStrategy, ResponseSection,
    SynthesizedResponse, synthesize_responses
)


@dataclass
class ConsultationRequest:
    """Request for AI consultation"""
    patterns: List[PatternMatch]
    context: str
    primary_category: PatternCategory
    severity: PatternSeverity
    requires_multi_ai: bool
    metadata: Dict[str, Any]


@dataclass
class ConsultationResponse:
    """Enhanced response from AI consultation with synthesis"""
    request_id: str
    ai_responses: Dict[str, str]  # AI name -> response
    primary_recommendation: str
    synthesis: Optional[str]
    synthesized_response: Optional[SynthesizedResponse]  # New: full synthesis data
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class BasePatternHandler(ABC):
    """Enhanced base class for pattern-specific handlers"""
    
    def __init__(self, category: PatternCategory):
        self.category = category
        self.consultation_count = 0
        self.total_processing_time = 0.0
        self.synthesizer = ResponseSynthesizer()
    
    @abstractmethod
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate AI consultation prompt based on patterns"""
        pass
    
    @abstractmethod
    def get_synthesis_strategy(self) -> SynthesisStrategy:
        """Get the preferred synthesis strategy for this pattern type"""
        pass
    
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process AI responses using the synthesis module"""
        start_time = time.time()
        
        # Get synthesis strategy
        strategy = self.get_synthesis_strategy()
        
        # Prepare context for synthesis
        synthesis_context = {
            "pattern_category": self.category,
            "severity": request.severity,
            "pattern_count": len(request.patterns),
            "keywords": list(set([p.keyword for p in request.patterns]))
        }
        
        # Synthesize responses
        synthesized = self.synthesizer.synthesize(
            ai_responses,
            strategy,
            synthesis_context
        )
        
        # Format synthesis
        synthesis_markdown = self.synthesizer.format_response(synthesized, "markdown")
        
        # Extract primary recommendation from synthesis
        if synthesized.sections:
            primary_section = max(synthesized.sections, key=lambda s: s.priority)
            primary_recommendation = f"## {primary_section.title}\n\n{primary_section.content}"
        else:
            primary_recommendation = synthesis_markdown
        
        response = ConsultationResponse(
            request_id=f"{self.category.value}_{int(time.time())}",
            ai_responses=ai_responses,
            primary_recommendation=primary_recommendation,
            synthesis=synthesis_markdown,
            synthesized_response=synthesized,
            confidence_score=synthesized.confidence_score,
            processing_time=time.time() - start_time,
            metadata={
                "pattern_count": len(request.patterns),
                "severity": request.severity.name,
                "keywords": list(set([p.keyword for p in request.patterns])),
                "synthesis_strategy": strategy.value,
                "synthesis_metadata": synthesized.metadata
            }
        )
        
        self.consultation_count += 1
        self.total_processing_time += response.processing_time
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            "category": self.category.value,
            "consultations": self.consultation_count,
            "avg_processing_time": self.total_processing_time / max(1, self.consultation_count)
        }


class SecurityPatternHandler(BasePatternHandler):
    """Enhanced handler for security-related patterns"""
    
    def __init__(self):
        super().__init__(PatternCategory.SECURITY)
    
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate security-focused consultation prompt"""
        keywords = [match.keyword for match in request.patterns]
        unique_keywords = list(set(keywords))
        
        prompt = f"""Security Analysis Required - Critical Review

Context: {request.context}

Security concerns detected: {', '.join(unique_keywords)}

Please provide a comprehensive security analysis covering:

1. **Vulnerability Assessment**
   - Identify specific security vulnerabilities in the approach
   - Rate severity of each vulnerability (Critical/High/Medium/Low)
   - Explain potential attack vectors

2. **Best Practices**
   - Recommend industry-standard security practices
   - Provide specific implementation guidelines
   - Include relevant security libraries or frameworks

3. **Code Examples**
   - Show secure implementation examples
   - Highlight what NOT to do with examples
   - Include proper error handling

4. **Testing & Validation**
   - Suggest security testing approaches
   - Recommend tools for security scanning
   - Provide validation checklist

5. **Compliance Considerations**
   - Relevant security standards (OWASP, etc.)
   - Regulatory requirements if applicable
   - Documentation needs

Be specific, actionable, and thorough. This is a critical security review."""
        
        return prompt
    
    def get_synthesis_strategy(self) -> SynthesisStrategy:
        """Security uses consensus strategy to find agreed-upon best practices"""
        return SynthesisStrategy.CONSENSUS


class UncertaintyPatternHandler(BasePatternHandler):
    """Enhanced handler for uncertainty patterns"""
    
    def __init__(self):
        super().__init__(PatternCategory.UNCERTAINTY)
    
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate prompt for uncertainty resolution"""
        keywords = [match.keyword for match in request.patterns]
        
        prompt = f"""Clarification and Guidance Needed

Context: {request.context}

Uncertainty indicators: {', '.join(keywords)}

Please provide clear, comprehensive guidance:

1. **Clarification**
   - Address the specific uncertainty or confusion
   - Provide clear explanations of concepts
   - Resolve any ambiguity in the approach

2. **Recommended Approach**
   - Suggest the best way forward
   - Provide step-by-step guidance
   - Include alternative approaches if applicable

3. **Code Examples**
   - Show concrete implementation examples
   - Demonstrate best practices
   - Include edge case handling

4. **Common Pitfalls**
   - Warn about common mistakes
   - Explain how to avoid them
   - Provide debugging tips

5. **Resources**
   - Link to relevant documentation
   - Suggest helpful libraries or tools
   - Recommend learning resources

Be thorough and educational. The goal is to resolve uncertainty and build confidence."""
        
        return prompt
    
    def get_synthesis_strategy(self) -> SynthesisStrategy:
        """Uncertainty benefits from comprehensive strategy to cover all aspects"""
        return SynthesisStrategy.COMPREHENSIVE


class AlgorithmPatternHandler(BasePatternHandler):
    """Enhanced handler for algorithm optimization patterns"""
    
    def __init__(self):
        super().__init__(PatternCategory.ALGORITHM)
    
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate algorithm analysis prompt"""
        keywords = [match.keyword for match in request.patterns]
        
        prompt = f"""Algorithm Analysis and Optimization

Context: {request.context}

Algorithm concerns: {', '.join(keywords)}

Please provide comprehensive algorithm analysis:

1. **Complexity Analysis**
   - Time complexity (best, average, worst case)
   - Space complexity
   - Scalability considerations

2. **Optimization Strategies**
   - Identify performance bottlenecks
   - Suggest algorithmic improvements
   - Provide alternative algorithms

3. **Implementation**
   - Optimized code implementation
   - Data structure recommendations
   - Language-specific optimizations

4. **Benchmarking**
   - Performance comparison approaches
   - Metrics to track
   - Testing strategies

5. **Trade-offs**
   - Performance vs readability
   - Memory vs speed
   - Complexity vs maintainability

Focus on practical, implementable optimizations with clear performance benefits."""
        
        return prompt
    
    def get_synthesis_strategy(self) -> SynthesisStrategy:
        """Algorithm analysis uses expert-weighted strategy based on AI strengths"""
        return SynthesisStrategy.EXPERT_WEIGHTED


class GotchaPatternHandler(BasePatternHandler):
    """Enhanced handler for common programming gotchas"""
    
    def __init__(self):
        super().__init__(PatternCategory.GOTCHA)
    
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate gotcha prevention prompt"""
        keywords = [match.keyword for match in request.patterns]
        
        prompt = f"""Programming Gotcha Analysis - Comprehensive Warning

Context: {request.context}

Potential gotchas detected: {', '.join(keywords)}

Provide detailed gotcha prevention guidance:

1. **Specific Gotchas**
   - Explain each potential gotcha in detail
   - Show examples of what goes wrong
   - Demonstrate the unexpected behavior

2. **Root Causes**
   - Explain why these gotchas occur
   - Language/framework specific issues
   - Common misconceptions

3. **Prevention Strategies**
   - Best practices to avoid each gotcha
   - Defensive programming techniques
   - Validation and testing approaches

4. **Safe Implementation**
   - Correct code examples
   - Robust error handling
   - Edge case coverage

5. **Detection & Debugging**
   - How to identify when you've hit the gotcha
   - Debugging techniques
   - Testing strategies

Be extremely thorough - these gotchas cause real production issues!"""
        
        return prompt
    
    def get_synthesis_strategy(self) -> SynthesisStrategy:
        """Gotchas benefit from debate strategy to show different perspectives"""
        return SynthesisStrategy.DEBATE


class ArchitecturePatternHandler(BasePatternHandler):
    """Enhanced handler for architecture and design patterns"""
    
    def __init__(self):
        super().__init__(PatternCategory.ARCHITECTURE)
    
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate architecture consultation prompt"""
        keywords = [match.keyword for match in request.patterns]
        
        prompt = f"""Architecture and Design Pattern Analysis

Context: {request.context}

Architecture concerns: {', '.join(keywords)}

Provide comprehensive architectural guidance:

1. **Design Patterns**
   - Identify applicable design patterns
   - Explain pattern benefits and trade-offs
   - Show implementation examples

2. **Architecture Recommendations**
   - System structure and organization
   - Component relationships
   - Scalability considerations

3. **Best Practices**
   - SOLID principles application
   - Separation of concerns
   - Dependency management

4. **Implementation Strategy**
   - Step-by-step architecture implementation
   - Module organization
   - Interface design

5. **Future Considerations**
   - Extensibility and maintainability
   - Performance implications
   - Testing strategies

Provide multiple architectural perspectives and justify recommendations."""
        
        return prompt
    
    def get_synthesis_strategy(self) -> SynthesisStrategy:
        """Architecture benefits from debate to show different design approaches"""
        return SynthesisStrategy.DEBATE


class EnhancedPatternResponseManager:
    """Enhanced manager with synthesis capabilities"""
    
    def __init__(self):
        self.handlers: Dict[PatternCategory, BasePatternHandler] = {
            PatternCategory.SECURITY: SecurityPatternHandler(),
            PatternCategory.UNCERTAINTY: UncertaintyPatternHandler(),
            PatternCategory.ALGORITHM: AlgorithmPatternHandler(),
            PatternCategory.GOTCHA: GotchaPatternHandler(),
            PatternCategory.ARCHITECTURE: ArchitecturePatternHandler()
        }
        
        self.ai_caller: Optional[Callable] = None
        self.multi_ai_caller: Optional[Callable] = None
        self.default_synthesis_strategy = SynthesisStrategy.CONSENSUS
    
    def set_ai_callers(self, single_ai_caller: Callable, multi_ai_caller: Callable):
        """Set the AI calling functions"""
        self.ai_caller = single_ai_caller
        self.multi_ai_caller = multi_ai_caller
    
    def set_default_synthesis_strategy(self, strategy: SynthesisStrategy):
        """Set default synthesis strategy"""
        self.default_synthesis_strategy = strategy
    
    def handle_patterns(
        self, 
        patterns: List[PatternMatch], 
        context: str,
        force_strategy: Optional[SynthesisStrategy] = None
    ) -> Optional[ConsultationResponse]:
        """Handle detected patterns with enhanced synthesis"""
        if not patterns:
            return None
        
        # Validate AI callers are set
        if not self.ai_caller or not self.multi_ai_caller:
            raise RuntimeError("AI callers not set. Call set_ai_callers() first.")
        
        # Group patterns by category
        category_groups = {}
        for pattern in patterns:
            if pattern.category not in category_groups:
                category_groups[pattern.category] = []
            category_groups[pattern.category].append(pattern)
        
        # Determine primary category
        primary_category = self._determine_primary_category(category_groups)
        primary_patterns = category_groups[primary_category]
        
        # Create consultation request
        requires_multi_ai = any(p.requires_multi_ai for p in patterns)
        max_severity = max(patterns, key=lambda p: p.severity.value).severity
        
        request = ConsultationRequest(
            patterns=patterns,
            context=context,
            primary_category=primary_category,
            severity=max_severity,
            requires_multi_ai=requires_multi_ai,
            metadata={
                "all_categories": list(category_groups.keys()),
                "pattern_counts": {cat.value: len(pats) for cat, pats in category_groups.items()}
            }
        )
        
        # Get appropriate handler
        handler = self.handlers[primary_category]
        
        # Generate prompt
        prompt = handler.generate_prompt(request)
        
        # Call AI(s)
        ai_responses = self._call_ais(prompt, requires_multi_ai, primary_category)
        
        # Process response with synthesis
        response = handler.process_response(ai_responses, request)
        
        # Override synthesis strategy if requested
        if force_strategy and response.synthesized_response:
            synthesizer = ResponseSynthesizer()
            new_synthesized = synthesizer.synthesize(
                ai_responses,
                force_strategy,
                {"pattern_category": primary_category}
            )
            response.synthesis = synthesizer.format_response(new_synthesized, "markdown")
            response.synthesized_response = new_synthesized
            response.confidence_score = new_synthesized.confidence_score
        
        return response
    
    def _determine_primary_category(self, category_groups: Dict[PatternCategory, List[PatternMatch]]) -> PatternCategory:
        """Determine the primary pattern category"""
        # Sort by severity first, then by count
        sorted_categories = sorted(
            category_groups.items(),
            key=lambda x: (max(p.severity.value for p in x[1]), len(x[1])),
            reverse=True
        )
        return sorted_categories[0][0]
    
    def _call_ais(self, prompt: str, requires_multi_ai: bool, category: PatternCategory) -> Dict[str, str]:
        """Call appropriate AI(s) based on requirements"""
        if not self.ai_caller or not self.multi_ai_caller:
            # Fallback for testing
            return {"test_ai": "Test response for: " + prompt[:100]}
        
        if requires_multi_ai:
            # Determine which AIs to use based on category
            if category == PatternCategory.SECURITY:
                ai_list = ["gemini", "grok", "openai"]
            elif category == PatternCategory.ALGORITHM:
                ai_list = ["deepseek", "gemini", "openai"]
            elif category == PatternCategory.ARCHITECTURE:
                ai_list = ["gemini", "grok", "openai", "deepseek"]
            else:
                ai_list = ["gemini", "grok"]
            
            # Call multiple AIs
            responses = {}
            for ai_name in ai_list:
                response = self.ai_caller(ai_name, prompt, temperature=0.3)
                if not response.startswith("❌"):  # Check for error
                    responses[ai_name] = response
            
            return responses
        else:
            # Single AI consultation - choose best AI for category
            ai_preferences = {
                PatternCategory.SECURITY: "gemini",
                PatternCategory.UNCERTAINTY: "openai",
                PatternCategory.ALGORITHM: "deepseek",
                PatternCategory.GOTCHA: "gemini",
                PatternCategory.ARCHITECTURE: "openai"
            }
            
            ai_name = ai_preferences.get(category, "openrouter")
            response = self.ai_caller(ai_name, prompt, temperature=0.5)
            return {ai_name: response}
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get statistics for all handlers"""
        stats = {}
        for category, handler in self.handlers.items():
            stats[category.value] = handler.get_stats()
        return stats


if __name__ == "__main__":
    # Test the enhanced response handlers
    from pattern_detection import PatternDetectionEngine
    
    print("Testing Enhanced Response Handlers with Synthesis...")
    
    # Create pattern detection engine
    pattern_engine = PatternDetectionEngine()
    
    # Create enhanced response manager
    response_manager = EnhancedPatternResponseManager()
    
    # Test cases with expected synthesis strategies
    test_cases = [
        ("I need to store passwords securely in the database. Should I use MD5 hashing?",
         "Security - Consensus synthesis expected"),
        ("TODO: Not sure how to implement this sorting algorithm efficiently",
         "Uncertainty - Comprehensive synthesis expected"),
        ("This O(n^2) algorithm is too slow. Need to optimize the search",
         "Algorithm - Expert-weighted synthesis expected"),
        ("Working with datetime objects and timezone conversions",
         "Gotcha - Debate synthesis expected"),
        ("Should I use microservices or monolithic architecture for this project?",
         "Architecture - Debate synthesis expected")
    ]
    
    for text, expected in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: {text}")
        print(f"Expected: {expected}")
        
        # Detect patterns
        patterns = pattern_engine.detect_patterns(text)
        
        if patterns:
            print(f"\nDetected {len(patterns)} patterns:")
            for pattern in patterns:
                print(f"  - {pattern.category.value}: {pattern.keyword} (severity: {pattern.severity.name})")
            
            # Simulate multi-AI for testing
            for pattern in patterns:
                if pattern.severity.value >= PatternSeverity.HIGH.value:
                    pattern.requires_multi_ai = True
            
            # Handle patterns
            response = response_manager.handle_patterns(patterns, text)
            
            if response:
                print(f"\nConsultation Response:")
                print(f"  Request ID: {response.request_id}")
                print(f"  Confidence: {response.confidence_score:.2f}")
                print(f"  AIs consulted: {list(response.ai_responses.keys())}")
                print(f"  Synthesis Strategy: {response.metadata.get('synthesis_strategy', 'unknown')}")
                
                if response.synthesized_response:
                    synth = response.synthesized_response
                    print(f"\nSynthesis Details:")
                    print(f"  Sections: {len(synth.sections)}")
                    print(f"  Key Insights: {len(synth.key_insights)}")
                    print(f"  Agreements: {len(synth.agreements)}")
                    print(f"  Disagreements: {len(synth.disagreements)}")
                    print(f"  Processing Time: {synth.synthesis_time:.3f}s")
                
                print(f"\nResponse Preview:")
                print(response.synthesis[:300] + "..." if response.synthesis else "No synthesis available")
        else:
            print("No patterns detected")
    
    # Show handler statistics
    print(f"\n{'='*80}")
    print("Handler Statistics:")
    stats = response_manager.get_handler_stats()
    for category, cat_stats in stats.items():
        print(f"\n{category}:")
        for key, value in cat_stats.items():
            print(f"  {key}: {value}")
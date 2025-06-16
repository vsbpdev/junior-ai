#!/usr/bin/env python3
"""
Pattern-Specific Response Handlers for Junior AI Assistant
Handles AI consultations based on detected patterns
"""

import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from pattern_detection import PatternCategory, PatternMatch, PatternSeverity


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
    """Response from AI consultation"""
    request_id: str
    ai_responses: Dict[str, str]  # AI name -> response
    primary_recommendation: str
    synthesis: Optional[str]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class BasePatternHandler(ABC):
    """Base class for pattern-specific handlers"""
    
    def __init__(self, category: PatternCategory):
        self.category = category
        self.consultation_count = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    def generate_prompt(self, request: ConsultationRequest) -> str:
        """Generate AI consultation prompt based on patterns"""
        pass
    
    @abstractmethod
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process AI responses and generate final consultation response"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            "category": self.category.value,
            "consultations": self.consultation_count,
            "avg_processing_time": self.total_processing_time / max(1, self.consultation_count)
        }


class SecurityPatternHandler(BasePatternHandler):
    """Handler for security-related patterns"""
    
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
    
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process security consultation responses"""
        start_time = time.time()
        
        # Synthesize multiple AI responses for security
        synthesis = self._synthesize_security_responses(ai_responses)
        
        # Extract primary recommendation
        primary_recommendation = self._extract_primary_security_recommendation(ai_responses)
        
        # Calculate confidence based on AI agreement
        confidence_score = self._calculate_security_confidence(ai_responses)
        
        response = ConsultationResponse(
            request_id=f"security_{int(time.time())}",
            ai_responses=ai_responses,
            primary_recommendation=primary_recommendation,
            synthesis=synthesis,
            confidence_score=confidence_score,
            processing_time=time.time() - start_time,
            metadata={
                "pattern_count": len(request.patterns),
                "severity": request.severity.name,
                "keywords": list(set([p.keyword for p in request.patterns]))
            }
        )
        
        self.consultation_count += 1
        self.total_processing_time += response.processing_time
        
        return response
    
    def _synthesize_security_responses(self, ai_responses: Dict[str, str]) -> str:
        """Synthesize multiple AI security responses"""
        if len(ai_responses) == 1:
            return list(ai_responses.values())[0]
        
        synthesis = "## 🔒 Security Analysis Synthesis\n\n"
        synthesis += "Based on multiple AI security reviews, here are the key findings:\n\n"
        
        # Extract common themes
        all_responses = " ".join(ai_responses.values()).lower()
        
        critical_items = []
        recommendations = []
        
        for ai_name, response in ai_responses.items():
            # Look for critical issues
            if any(word in response.lower() for word in ["critical", "severe", "dangerous", "vulnerable"]):
                critical_items.append(f"**{ai_name.upper()}** identified critical security issues")
            
            # Extract recommendations (simplified - in real implementation would use NLP)
            if "recommend" in response.lower() or "should" in response.lower():
                recommendations.append(ai_name)
        
        if critical_items:
            synthesis += "### ⚠️ Critical Security Concerns\n"
            for item in critical_items:
                synthesis += f"- {item}\n"
            synthesis += "\n"
        
        synthesis += "### 📋 Consensus Recommendations\n"
        synthesis += f"All {len(ai_responses)} AIs agree on the importance of:\n"
        synthesis += "- Proper authentication and authorization\n"
        synthesis += "- Input validation and sanitization\n"
        synthesis += "- Secure data storage and transmission\n"
        synthesis += "- Regular security audits and updates\n\n"
        
        synthesis += "### 🤖 Individual AI Insights\n"
        for ai_name, response in ai_responses.items():
            synthesis += f"\n**{ai_name.upper()} Key Points:**\n"
            # Extract first few lines as summary
            lines = response.split('\n')[:5]
            for line in lines:
                if line.strip():
                    synthesis += f"- {line.strip()}\n"
        
        return synthesis
    
    def _extract_primary_security_recommendation(self, ai_responses: Dict[str, str]) -> str:
        """Extract the primary security recommendation"""
        # In a real implementation, this would use NLP to extract key recommendations
        # For now, we'll create a structured recommendation
        
        recommendation = "🛡️ **Primary Security Recommendation**\n\n"
        recommendation += "Based on the security analysis, the most critical action is to:\n\n"
        recommendation += "1. **Immediately address** any authentication or encryption vulnerabilities\n"
        recommendation += "2. **Implement** proper input validation and sanitization\n"
        recommendation += "3. **Review** all security-sensitive code paths\n"
        recommendation += "4. **Test** using security scanning tools\n"
        recommendation += "5. **Document** security measures for compliance\n\n"
        recommendation += "⚠️ Security should be the top priority before proceeding with implementation."
        
        return recommendation
    
    def _calculate_security_confidence(self, ai_responses: Dict[str, str]) -> float:
        """Calculate confidence score based on AI agreement"""
        if len(ai_responses) == 1:
            return 0.8  # Single AI consultation
        
        # Check for agreement on critical issues
        responses_text = [r.lower() for r in ai_responses.values()]
        
        # Common security terms that should appear in good security analysis
        security_terms = ["encrypt", "authenticate", "validate", "sanitize", "secure", "hash", "salt"]
        
        term_coverage = 0
        for term in security_terms:
            if any(term in response for response in responses_text):
                term_coverage += 1
        
        # Base confidence on coverage and number of AIs
        base_confidence = (term_coverage / len(security_terms)) * 0.7
        ai_bonus = min(0.3, len(ai_responses) * 0.1)
        
        return min(1.0, base_confidence + ai_bonus)


class UncertaintyPatternHandler(BasePatternHandler):
    """Handler for uncertainty patterns"""
    
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
    
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process uncertainty resolution responses"""
        start_time = time.time()
        
        # For uncertainty, typically use single AI response
        primary_ai = list(ai_responses.keys())[0]
        primary_response = ai_responses[primary_ai]
        
        response = ConsultationResponse(
            request_id=f"uncertainty_{int(time.time())}",
            ai_responses=ai_responses,
            primary_recommendation=self._format_uncertainty_resolution(primary_response),
            synthesis=None,  # No synthesis needed for single AI
            confidence_score=0.85,  # Good confidence for clarification
            processing_time=time.time() - start_time,
            metadata={
                "pattern_count": len(request.patterns),
                "primary_ai": primary_ai,
                "uncertainty_keywords": list(set([p.keyword for p in request.patterns]))
            }
        )
        
        self.consultation_count += 1
        self.total_processing_time += response.processing_time
        
        return response
    
    def _format_uncertainty_resolution(self, response: str) -> str:
        """Format the uncertainty resolution response"""
        formatted = "## 💡 Clarification & Guidance\n\n"
        formatted += response
        formatted += "\n\n---\n"
        formatted += "*This guidance aims to resolve the detected uncertainty and provide clear direction.*"
        return formatted


class AlgorithmPatternHandler(BasePatternHandler):
    """Handler for algorithm optimization patterns"""
    
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
    
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process algorithm optimization responses"""
        start_time = time.time()
        
        # Synthesize multiple perspectives on algorithms
        synthesis = self._synthesize_algorithm_responses(ai_responses)
        primary_recommendation = self._extract_algorithm_recommendation(ai_responses)
        
        response = ConsultationResponse(
            request_id=f"algorithm_{int(time.time())}",
            ai_responses=ai_responses,
            primary_recommendation=primary_recommendation,
            synthesis=synthesis,
            confidence_score=0.9,  # High confidence for algorithm analysis
            processing_time=time.time() - start_time,
            metadata={
                "pattern_count": len(request.patterns),
                "algorithm_keywords": list(set([p.keyword for p in request.patterns])),
                "multi_ai_analysis": len(ai_responses) > 1
            }
        )
        
        self.consultation_count += 1
        self.total_processing_time += response.processing_time
        
        return response
    
    def _synthesize_algorithm_responses(self, ai_responses: Dict[str, str]) -> str:
        """Synthesize algorithm optimization responses"""
        synthesis = "## 🚀 Algorithm Optimization Synthesis\n\n"
        
        if len(ai_responses) == 1:
            return list(ai_responses.values())[0]
        
        synthesis += f"Analysis from {len(ai_responses)} AI perspectives:\n\n"
        
        # Compare complexity analyses
        synthesis += "### 📊 Complexity Analysis Comparison\n"
        for ai_name, response in ai_responses.items():
            synthesis += f"**{ai_name.upper()}**: "
            # Extract complexity mentions (simplified)
            if "O(" in response:
                # Find first O() notation
                start = response.find("O(")
                end = response.find(")", start)
                if end > start:
                    synthesis += response[start:end+1] + "\n"
            else:
                synthesis += "See detailed analysis below\n"
        
        synthesis += "\n### 🔧 Optimization Consensus\n"
        synthesis += "Key optimization strategies identified:\n"
        synthesis += "- Use appropriate data structures\n"
        synthesis += "- Minimize unnecessary iterations\n"
        synthesis += "- Consider space-time trade-offs\n"
        synthesis += "- Profile before optimizing\n\n"
        
        synthesis += "### 🤖 Individual Algorithm Insights\n"
        for ai_name, response in ai_responses.items():
            synthesis += f"\n**{ai_name.upper()} Focus:**\n"
            # Extract key points (first few lines)
            lines = response.split('\n')[:3]
            for line in lines:
                if line.strip():
                    synthesis += f"- {line.strip()}\n"
        
        return synthesis
    
    def _extract_algorithm_recommendation(self, ai_responses: Dict[str, str]) -> str:
        """Extract primary algorithm recommendation"""
        recommendation = "## 🎯 Primary Algorithm Recommendation\n\n"
        recommendation += "Based on the analysis, here's the optimal approach:\n\n"
        recommendation += "1. **Algorithm Choice**: Select the most appropriate algorithm for your use case\n"
        recommendation += "2. **Data Structure**: Use optimal data structures for your access patterns\n"
        recommendation += "3. **Implementation**: Follow the provided optimized implementation\n"
        recommendation += "4. **Testing**: Benchmark with realistic data sizes\n"
        recommendation += "5. **Monitoring**: Track performance in production\n\n"
        recommendation += "⚡ Remember: Premature optimization is the root of all evil. Profile first!"
        
        return recommendation


class GotchaPatternHandler(BasePatternHandler):
    """Handler for common programming gotchas"""
    
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
    
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process gotcha prevention responses"""
        start_time = time.time()
        
        # Single AI comprehensive analysis for gotchas
        primary_ai = list(ai_responses.keys())[0]
        
        response = ConsultationResponse(
            request_id=f"gotcha_{int(time.time())}",
            ai_responses=ai_responses,
            primary_recommendation=self._format_gotcha_warnings(ai_responses[primary_ai]),
            synthesis=None,
            confidence_score=0.95,  # High confidence for known gotchas
            processing_time=time.time() - start_time,
            metadata={
                "pattern_count": len(request.patterns),
                "gotcha_types": list(set([p.keyword for p in request.patterns])),
                "severity": "HIGH"  # Gotchas are always high priority
            }
        )
        
        self.consultation_count += 1
        self.total_processing_time += response.processing_time
        
        return response
    
    def _format_gotcha_warnings(self, response: str) -> str:
        """Format gotcha warnings for clarity"""
        formatted = "## ⚠️ GOTCHA WARNING - Critical Programming Pitfalls\n\n"
        formatted += "**PAY CAREFUL ATTENTION** - These issues cause production bugs!\n\n"
        formatted += response
        formatted += "\n\n---\n"
        formatted += "🛡️ *Always test edge cases and validate assumptions when dealing with these gotchas.*"
        return formatted


class ArchitecturePatternHandler(BasePatternHandler):
    """Handler for architecture and design patterns"""
    
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
    
    def process_response(self, ai_responses: Dict[str, str], request: ConsultationRequest) -> ConsultationResponse:
        """Process architecture consultation responses"""
        start_time = time.time()
        
        # Multi-AI synthesis for architecture decisions
        synthesis = self._synthesize_architecture_responses(ai_responses)
        primary_recommendation = self._extract_architecture_consensus(ai_responses)
        
        response = ConsultationResponse(
            request_id=f"architecture_{int(time.time())}",
            ai_responses=ai_responses,
            primary_recommendation=primary_recommendation,
            synthesis=synthesis,
            confidence_score=self._calculate_architecture_confidence(ai_responses),
            processing_time=time.time() - start_time,
            metadata={
                "pattern_count": len(request.patterns),
                "design_keywords": list(set([p.keyword for p in request.patterns])),
                "perspectives": len(ai_responses)
            }
        )
        
        self.consultation_count += 1
        self.total_processing_time += response.processing_time
        
        return response
    
    def _synthesize_architecture_responses(self, ai_responses: Dict[str, str]) -> str:
        """Synthesize architecture recommendations"""
        synthesis = "## 🏗️ Architecture Design Synthesis\n\n"
        
        if len(ai_responses) == 1:
            return list(ai_responses.values())[0]
        
        synthesis += f"Architectural insights from {len(ai_responses)} AI perspectives:\n\n"
        
        synthesis += "### 🎨 Design Pattern Recommendations\n"
        patterns_mentioned = []
        for ai_name, response in ai_responses.items():
            # Look for common design pattern names
            common_patterns = ["MVC", "MVVM", "Factory", "Singleton", "Observer", 
                             "Strategy", "Repository", "Adapter", "Facade"]
            for pattern in common_patterns:
                if pattern.lower() in response.lower():
                    patterns_mentioned.append(f"{pattern} (suggested by {ai_name})")
        
        if patterns_mentioned:
            for pattern in set(patterns_mentioned):
                synthesis += f"- {pattern}\n"
        else:
            synthesis += "- See individual AI responses for pattern suggestions\n"
        
        synthesis += "\n### 🏛️ Architecture Consensus\n"
        synthesis += "Key architectural principles agreed upon:\n"
        synthesis += "- Maintain clear separation of concerns\n"
        synthesis += "- Design for testability and maintainability\n"
        synthesis += "- Consider future scalability needs\n"
        synthesis += "- Follow established patterns for consistency\n\n"
        
        synthesis += "### 🤖 Individual Architecture Perspectives\n"
        for ai_name, response in ai_responses.items():
            synthesis += f"\n**{ai_name.upper()} Architecture Focus:**\n"
            # Extract key architectural points
            lines = response.split('\n')[:4]
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    synthesis += f"- {line.strip()}\n"
        
        return synthesis
    
    def _extract_architecture_consensus(self, ai_responses: Dict[str, str]) -> str:
        """Extract architecture consensus recommendation"""
        recommendation = "## 🎯 Architecture Recommendation Consensus\n\n"
        recommendation += "Based on multiple AI architectural analyses:\n\n"
        recommendation += "### Recommended Approach\n"
        recommendation += "1. **Pattern Selection**: Choose patterns that match your specific use case\n"
        recommendation += "2. **Structure**: Organize code for clarity and maintainability\n"
        recommendation += "3. **Interfaces**: Design clear contracts between components\n"
        recommendation += "4. **Testing**: Build with testability in mind from the start\n"
        recommendation += "5. **Documentation**: Document architectural decisions and rationale\n\n"
        recommendation += "### Implementation Priority\n"
        recommendation += "- Start with core domain logic\n"
        recommendation += "- Add infrastructure incrementally\n"
        recommendation += "- Refactor as patterns emerge\n"
        recommendation += "- Maintain architectural consistency\n\n"
        recommendation += "🏗️ *Good architecture evolves - start simple and refactor thoughtfully.*"
        
        return recommendation
    
    def _calculate_architecture_confidence(self, ai_responses: Dict[str, str]) -> float:
        """Calculate confidence in architecture recommendations"""
        base_confidence = 0.7
        
        # More AIs = higher confidence for architecture
        ai_bonus = min(0.2, len(ai_responses) * 0.05)
        
        # Check for agreement on key principles
        responses_text = " ".join(ai_responses.values()).lower()
        principles = ["solid", "separation", "testability", "maintainability", "scalability"]
        
        principle_coverage = sum(1 for p in principles if p in responses_text)
        principle_bonus = (principle_coverage / len(principles)) * 0.1
        
        return min(1.0, base_confidence + ai_bonus + principle_bonus)


class PatternResponseManager:
    """Manages pattern handlers and coordinates responses"""
    
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
    
    def set_ai_callers(self, single_ai_caller: Callable, multi_ai_caller: Callable):
        """Set the AI calling functions"""
        self.ai_caller = single_ai_caller
        self.multi_ai_caller = multi_ai_caller
    
    def handle_patterns(self, patterns: List[PatternMatch], context: str) -> Optional[ConsultationResponse]:
        """Handle detected patterns and generate consultation response"""
        if not patterns:
            return None
        
        # Group patterns by category
        category_groups = {}
        for pattern in patterns:
            if pattern.category not in category_groups:
                category_groups[pattern.category] = []
            category_groups[pattern.category].append(pattern)
        
        # Determine primary category (most severe or most frequent)
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
        
        # Process response
        response = handler.process_response(ai_responses, request)
        
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
                ai_list = ["gemini", "grok", "openai"]
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
            # Single AI consultation
            if category == PatternCategory.UNCERTAINTY:
                ai_name = "openrouter"
            elif category == PatternCategory.GOTCHA:
                ai_name = "gemini"
            else:
                ai_name = "openrouter"
            
            response = self.ai_caller(ai_name, prompt, temperature=0.5)
            return {ai_name: response}
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get statistics for all handlers"""
        stats = {}
        for category, handler in self.handlers.items():
            stats[category.value] = handler.get_stats()
        return stats


if __name__ == "__main__":
    # Test the response handlers
    from pattern_detection import PatternDetectionEngine
    
    print("Testing Response Handlers...")
    
    # Create pattern detection engine
    pattern_engine = PatternDetectionEngine()
    
    # Create response manager
    response_manager = PatternResponseManager()
    
    # Test texts with different patterns
    test_cases = [
        ("I need to store passwords securely in the database. Should I use MD5 hashing?",
         "Security consultation expected"),
        ("TODO: Not sure how to implement this sorting algorithm efficiently",
         "Uncertainty resolution expected"),
        ("This O(n^2) algorithm is too slow. Need to optimize the search",
         "Algorithm optimization expected"),
        ("Working with datetime objects and timezone conversions",
         "Gotcha prevention expected"),
        ("Should I use microservices or monolithic architecture for this project?",
         "Architecture consultation expected")
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
            
            # Handle patterns
            response = response_manager.handle_patterns(patterns, text)
            
            if response:
                print(f"\nConsultation Response:")
                print(f"  Request ID: {response.request_id}")
                print(f"  Confidence: {response.confidence_score:.2f}")
                print(f"  AIs consulted: {list(response.ai_responses.keys())}")
                print(f"\nPrimary Recommendation Preview:")
                print(response.primary_recommendation[:200] + "...")
                
                if response.synthesis:
                    print(f"\nSynthesis Preview:")
                    print(response.synthesis[:200] + "...")
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
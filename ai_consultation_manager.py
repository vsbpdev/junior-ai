#!/usr/bin/env python3
"""
AI Consultation Manager for Junior AI Assistant
Intelligent AI selection, coordination, and consultation management
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from collections import defaultdict
import logging

from pattern_detection import PatternCategory, PatternMatch, PatternSeverity
from context_aware_matching import ContextualPattern
from response_handlers import ConsultationRequest, ConsultationResponse, PatternResponseManager


class AIExpertise(Enum):
    """AI model expertise areas"""
    SECURITY = "security"
    ALGORITHMS = "algorithms"
    ARCHITECTURE = "architecture"
    GENERAL = "general"
    DEBUGGING = "debugging"
    CREATIVE = "creative"


@dataclass
class AIProfile:
    """Profile of an AI model's capabilities"""
    name: str
    expertise: List[AIExpertise]
    strengths: List[str]
    weaknesses: List[str]
    cost_tier: int  # 1=low, 2=medium, 3=high
    speed_tier: int  # 1=slow, 2=medium, 3=fast
    max_context: int  # Maximum context length
    preferred_patterns: List[PatternCategory]


@dataclass
class ConsultationStrategy:
    """Strategy for AI consultation"""
    ai_selection: List[str]
    consultation_mode: str  # "single", "multi", "consensus", "debate"
    priority: str  # "speed", "accuracy", "cost"
    reasoning: str
    estimated_time: float
    estimated_cost: float


@dataclass
class ConsultationMetrics:
    """Metrics for consultation tracking"""
    total_consultations: int = 0
    successful_consultations: int = 0
    failed_consultations: int = 0
    total_response_time: float = 0.0
    ai_usage_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pattern_category_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConsultationAudit:
    """Audit trail for a consultation"""
    consultation_id: str
    timestamp: datetime
    pattern_summary: Dict[str, Any]
    strategy_used: ConsultationStrategy
    ais_consulted: List[str]
    prompts_sent: Dict[str, str]
    responses_received: Dict[str, str]
    synthesis_method: str
    final_confidence: float
    processing_time: float
    user_feedback: Optional[str] = None


class AIConsultationManager:
    """Enhanced AI Consultation Manager with intelligent selection and coordination"""
    
    # AI Profile definitions
    AI_PROFILES = {
        "gemini": AIProfile(
            name="gemini",
            expertise=[AIExpertise.GENERAL, AIExpertise.ARCHITECTURE, AIExpertise.ALGORITHMS],
            strengths=["comprehensive analysis", "code generation", "detailed explanations"],
            weaknesses=["can be verbose", "sometimes over-engineered solutions"],
            cost_tier=2,
            speed_tier=2,
            max_context=32000,
            preferred_patterns=[PatternCategory.ARCHITECTURE, PatternCategory.ALGORITHM]
        ),
        "grok": AIProfile(
            name="grok",
            expertise=[AIExpertise.SECURITY, AIExpertise.DEBUGGING, AIExpertise.ARCHITECTURE],
            strengths=["security analysis", "concise responses", "practical solutions"],
            weaknesses=["limited context window", "less detailed explanations"],
            cost_tier=2,
            speed_tier=3,
            max_context=8192,
            preferred_patterns=[PatternCategory.SECURITY, PatternCategory.GOTCHA]
        ),
        "openai": AIProfile(
            name="openai",
            expertise=[AIExpertise.GENERAL, AIExpertise.CREATIVE, AIExpertise.ALGORITHMS],
            strengths=["well-rounded", "good code quality", "clear explanations"],
            weaknesses=["can miss edge cases", "sometimes generic advice"],
            cost_tier=2,
            speed_tier=2,
            max_context=16000,
            preferred_patterns=[PatternCategory.UNCERTAINTY, PatternCategory.ALGORITHM]
        ),
        "deepseek": AIProfile(
            name="deepseek",
            expertise=[AIExpertise.ALGORITHMS, AIExpertise.DEBUGGING],
            strengths=["algorithm optimization", "performance analysis", "technical depth"],
            weaknesses=["less intuitive explanations", "focuses on technical details"],
            cost_tier=1,
            speed_tier=3,
            max_context=16000,
            preferred_patterns=[PatternCategory.ALGORITHM]
        ),
        "openrouter": AIProfile(
            name="openrouter",
            expertise=[AIExpertise.GENERAL],
            strengths=["cost-effective", "fast responses", "good for simple queries"],
            weaknesses=["less sophisticated analysis", "may miss nuances"],
            cost_tier=1,
            speed_tier=3,
            max_context=8192,
            preferred_patterns=[PatternCategory.UNCERTAINTY]
        )
    }
    
    def __init__(self, 
                 ai_caller: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the AI Consultation Manager"""
        self.ai_caller = ai_caller
        self.config = config or {}
        
        # Initialize components
        self.response_manager = PatternResponseManager()
        self.metrics = ConsultationMetrics()
        self.audit_trail: List[ConsultationAudit] = []
        self.consultation_cache: Dict[str, ConsultationResponse] = {}
        
        # Threading for async operations
        self.audit_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Set up logging
        self.logger = logging.getLogger('ai_consultation_manager')
        
    def select_ai_strategy(self, 
                          patterns: List[PatternMatch],
                          context: str,
                          preferences: Optional[Dict[str, Any]] = None) -> ConsultationStrategy:
        """Select optimal AI consultation strategy based on patterns and context"""
        
        # Analyze pattern characteristics
        pattern_analysis = self._analyze_patterns(patterns)
        
        # Determine consultation requirements
        requires_multi_ai = pattern_analysis['requires_multi_ai']
        primary_category = pattern_analysis['primary_category']
        severity = pattern_analysis['max_severity']
        
        # Get user preferences
        prefs = preferences or {}
        priority = prefs.get('priority', 'accuracy')
        force_multi_ai = prefs.get('force_multi_ai', False)
        excluded_ais = prefs.get('excluded_ais', [])
        
        # Select AIs based on expertise matching
        selected_ais = self._select_ais_by_expertise(
            primary_category,
            severity,
            requires_multi_ai or force_multi_ai,
            excluded_ais
        )
        
        # Determine consultation mode
        if len(selected_ais) > 1:
            if severity == PatternSeverity.CRITICAL:
                mode = "consensus"  # All AIs must agree
            elif primary_category == PatternCategory.ARCHITECTURE:
                mode = "debate"  # Present different perspectives
            else:
                mode = "multi"  # Standard multi-AI consultation
        else:
            mode = "single"
        
        # Calculate estimates
        estimated_time = self._estimate_consultation_time(selected_ais, mode)
        estimated_cost = self._estimate_consultation_cost(selected_ais, len(context))
        
        # Generate reasoning
        reasoning = self._generate_strategy_reasoning(
            pattern_analysis, selected_ais, mode, priority
        )
        
        return ConsultationStrategy(
            ai_selection=selected_ais,
            consultation_mode=mode,
            priority=priority,
            reasoning=reasoning,
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )
    
    def execute_consultation(self,
                           patterns: List[PatternMatch],
                           context: str,
                           strategy: Optional[ConsultationStrategy] = None,
                           contextual_patterns: Optional[List[ContextualPattern]] = None) -> ConsultationResponse:
        """Execute AI consultation with the given or auto-selected strategy"""
        
        consultation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Use provided strategy or auto-select
        if not strategy:
            strategy = self.select_ai_strategy(patterns, context)
        
        # Create audit entry
        audit = ConsultationAudit(
            consultation_id=consultation_id,
            timestamp=datetime.now(),
            pattern_summary=self._summarize_patterns(patterns),
            strategy_used=strategy,
            ais_consulted=[],
            prompts_sent={},
            responses_received={},
            synthesis_method=strategy.consultation_mode,
            final_confidence=0.0,
            processing_time=0.0
        )
        
        try:
            # Generate prompts based on pattern types
            prompts = self._generate_consultation_prompts(patterns, context, contextual_patterns)
            
            # Execute consultations based on mode
            if strategy.consultation_mode == "single":
                ai_responses = self._execute_single_consultation(
                    strategy.ai_selection[0], prompts['primary'], audit
                )
            elif strategy.consultation_mode == "multi":
                ai_responses = self._execute_multi_consultation(
                    strategy.ai_selection, prompts['primary'], audit
                )
            elif strategy.consultation_mode == "consensus":
                ai_responses = self._execute_consensus_consultation(
                    strategy.ai_selection, prompts, audit
                )
            elif strategy.consultation_mode == "debate":
                ai_responses = self._execute_debate_consultation(
                    strategy.ai_selection, prompts, audit
                )
            else:
                raise ValueError(f"Unknown consultation mode: {strategy.consultation_mode}")
            
            # Process responses through pattern handlers
            response = self._process_consultation_responses(
                patterns, context, ai_responses, audit
            )
            
            # Update audit
            audit.processing_time = time.time() - start_time
            audit.final_confidence = response.confidence_score
            
            # Update metrics
            self._update_metrics(audit, response)
            
            # Store audit trail
            with self.audit_lock:
                self.audit_trail.append(audit)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Consultation failed: {e}")
            # Create error response
            return ConsultationResponse(
                request_id=consultation_id,
                ai_responses={"error": str(e)},
                primary_recommendation=f"❌ Consultation failed: {str(e)}",
                synthesis=None,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": True, "strategy": strategy.ai_selection}
            )
    
    def _analyze_patterns(self, patterns: List[PatternMatch]) -> Dict[str, Any]:
        """Analyze pattern characteristics"""
        if not patterns:
            return {
                'requires_multi_ai': False,
                'primary_category': None,
                'max_severity': None,
                'category_distribution': {}
            }
        
        # Category distribution
        category_counts = defaultdict(int)
        for pattern in patterns:
            category_counts[pattern.category] += 1
        
        # Find primary category (most severe, then most frequent)
        primary_pattern = max(patterns, key=lambda p: (p.severity.value, p.confidence))
        
        return {
            'requires_multi_ai': any(p.requires_multi_ai for p in patterns),
            'primary_category': primary_pattern.category,
            'max_severity': primary_pattern.severity,
            'category_distribution': dict(category_counts),
            'total_patterns': len(patterns)
        }
    
    def _select_ais_by_expertise(self,
                                primary_category: PatternCategory,
                                severity: PatternSeverity,
                                requires_multi_ai: bool,
                                excluded_ais: List[str]) -> List[str]:
        """Select AIs based on expertise matching"""
        
        # Filter out excluded AIs
        available_profiles = {
            name: profile for name, profile in self.AI_PROFILES.items()
            if name not in excluded_ais
        }
        
        # Score each AI for the task
        ai_scores = {}
        for name, profile in available_profiles.items():
            score = 0
            
            # Primary category match
            if primary_category in profile.preferred_patterns:
                score += 3
            
            # Severity consideration
            if severity == PatternSeverity.CRITICAL:
                # Prefer more reliable AIs for critical issues
                if AIExpertise.SECURITY in profile.expertise:
                    score += 2
            
            # Cost/speed trade-offs
            if self.config.get('optimize_for_cost', False):
                score += (4 - profile.cost_tier)
            if self.config.get('optimize_for_speed', False):
                score += profile.speed_tier
            
            ai_scores[name] = score
        
        # Sort by score
        sorted_ais = sorted(ai_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select number of AIs
        if requires_multi_ai:
            # Use top 3 AIs for multi-AI consultation
            return [ai[0] for ai in sorted_ais[:3]]
        else:
            # Use best matching AI
            return [sorted_ais[0][0]] if sorted_ais else ["openrouter"]
    
    def _generate_consultation_prompts(self,
                                     patterns: List[PatternMatch],
                                     context: str,
                                     contextual_patterns: Optional[List[ContextualPattern]] = None) -> Dict[str, str]:
        """Generate specialized prompts for consultation"""
        
        # Group patterns by category
        category_groups = defaultdict(list)
        for pattern in patterns:
            category_groups[pattern.category].append(pattern)
        
        # Primary prompt
        primary_prompt = self._build_primary_prompt(patterns, context, contextual_patterns)
        
        # Specialized prompts for different consultation modes
        prompts = {
            'primary': primary_prompt,
            'consensus': self._build_consensus_prompt(patterns, context),
            'debate': self._build_debate_prompt(patterns, context)
        }
        
        return prompts
    
    def _build_primary_prompt(self,
                            patterns: List[PatternMatch],
                            context: str,
                            contextual_patterns: Optional[List[ContextualPattern]] = None) -> str:
        """Build the primary consultation prompt"""
        
        # Extract pattern details
        categories = list(set(p.category.value for p in patterns))
        keywords = list(set(p.keyword for p in patterns))
        max_severity = max(p.severity for p in patterns)
        
        prompt = f"""Expert Code Analysis Required

Context:
{context}

Detected Patterns:
- Categories: {', '.join(categories)}
- Keywords: {', '.join(keywords[:10])}  # Limit to first 10
- Maximum Severity: {max_severity.name}
"""
        
        # Add contextual information if available
        if contextual_patterns:
            languages = set(cp.language.value for cp in contextual_patterns)
            has_test_code = any(cp.is_test_code for cp in contextual_patterns)
            
            prompt += f"""
Additional Context:
- Language(s): {', '.join(languages)}
- Test Code: {'Yes' if has_test_code else 'No'}
- Pattern Count: {len(patterns)}
"""
        
        prompt += """
Please provide a comprehensive analysis covering:

1. **Issue Identification**
   - Specific problems in the code
   - Severity assessment
   - Potential impacts

2. **Best Practices**
   - Industry standards
   - Recommended approaches
   - What to avoid

3. **Implementation Guidance**
   - Step-by-step solutions
   - Code examples
   - Edge case handling

4. **Testing & Validation**
   - How to verify the solution
   - Test cases to consider
   - Monitoring recommendations

Be specific, practical, and actionable in your recommendations."""
        
        return prompt
    
    def _build_consensus_prompt(self, patterns: List[PatternMatch], context: str) -> str:
        """Build prompt for consensus-seeking consultation"""
        return f"""Critical Analysis Required - Consensus Building

{self._build_primary_prompt(patterns, context)}

IMPORTANT: Focus on providing clear, unambiguous recommendations that would be 
universally accepted as best practices. Avoid controversial or debatable approaches."""
    
    def _build_debate_prompt(self, patterns: List[PatternMatch], context: str) -> str:
        """Build prompt for debate-style consultation"""
        return f"""Architectural Analysis - Multiple Perspectives Welcome

{self._build_primary_prompt(patterns, context)}

IMPORTANT: Present your unique perspective on the best approach. Consider:
- Trade-offs between different solutions
- Contextual factors that might influence the decision
- Alternative approaches and their merits
- Your reasoning for the recommended approach"""
    
    def _execute_single_consultation(self, ai_name: str, prompt: str, audit: ConsultationAudit) -> Dict[str, str]:
        """Execute single AI consultation"""
        audit.prompts_sent[ai_name] = prompt
        
        if self.ai_caller:
            response = self.ai_caller(ai_name, prompt, temperature=0.3)
            audit.responses_received[ai_name] = response
            audit.ais_consulted.append(ai_name)
            return {ai_name: response}
        else:
            # Fallback for testing
            return {ai_name: f"Test response from {ai_name}"}
    
    def _execute_multi_consultation(self, ai_names: List[str], prompt: str, audit: ConsultationAudit) -> Dict[str, str]:
        """Execute multi-AI consultation in parallel"""
        responses = {}
        
        # In a real implementation, this would be parallelized
        for ai_name in ai_names:
            audit.prompts_sent[ai_name] = prompt
            
            if self.ai_caller:
                response = self.ai_caller(ai_name, prompt, temperature=0.3)
                if not response.startswith("❌"):  # Skip errors
                    responses[ai_name] = response
                    audit.responses_received[ai_name] = response
                    audit.ais_consulted.append(ai_name)
            else:
                responses[ai_name] = f"Test response from {ai_name}"
        
        return responses
    
    def _execute_consensus_consultation(self, 
                                      ai_names: List[str], 
                                      prompts: Dict[str, str], 
                                      audit: ConsultationAudit) -> Dict[str, str]:
        """Execute consensus-building consultation"""
        # First round: Get initial responses
        initial_responses = self._execute_multi_consultation(
            ai_names, prompts['consensus'], audit
        )
        
        # In a full implementation, we would:
        # 1. Analyze initial responses for agreement/disagreement
        # 2. Create follow-up prompts to resolve differences
        # 3. Iterate until consensus is reached or max rounds
        
        return initial_responses
    
    def _execute_debate_consultation(self,
                                   ai_names: List[str],
                                   prompts: Dict[str, str],
                                   audit: ConsultationAudit) -> Dict[str, str]:
        """Execute debate-style consultation"""
        # Get diverse perspectives
        responses = self._execute_multi_consultation(
            ai_names, prompts['debate'], audit
        )
        
        # In a full implementation, we would:
        # 1. Have AIs respond to each other's arguments
        # 2. Synthesize the different viewpoints
        # 3. Present a balanced analysis
        
        return responses
    
    def _process_consultation_responses(self,
                                      patterns: List[PatternMatch],
                                      context: str,
                                      ai_responses: Dict[str, str],
                                      audit: ConsultationAudit) -> ConsultationResponse:
        """Process AI responses through pattern handlers"""
        
        # Use the existing response manager
        if hasattr(self.response_manager, 'ai_caller'):
            # Temporarily override to use our responses
            original_caller = self.response_manager.ai_caller
            self.response_manager.ai_caller = lambda ai, prompt, temp: ai_responses.get(ai, "No response")
        
        # Create consultation request
        primary_category = max(patterns, key=lambda p: p.severity.value).category
        
        request = ConsultationRequest(
            patterns=patterns,
            context=context,
            primary_category=primary_category,
            severity=max(p.severity for p in patterns),
            requires_multi_ai=len(ai_responses) > 1,
            metadata={
                'consultation_id': audit.consultation_id,
                'ai_count': len(ai_responses)
            }
        )
        
        # Get appropriate handler and process
        handler = self.response_manager.handlers[primary_category]
        response = handler.process_response(ai_responses, request)
        
        # Restore original caller if changed
        if hasattr(self.response_manager, 'ai_caller') and 'original_caller' in locals():
            self.response_manager.ai_caller = original_caller
        
        return response
    
    def _summarize_patterns(self, patterns: List[PatternMatch]) -> Dict[str, Any]:
        """Create a summary of detected patterns"""
        return {
            'total_count': len(patterns),
            'categories': list(set(p.category.value for p in patterns)),
            'severities': list(set(p.severity.name for p in patterns)),
            'top_keywords': list(set(p.keyword for p in patterns))[:10]
        }
    
    def _estimate_consultation_time(self, ai_names: List[str], mode: str) -> float:
        """Estimate consultation time in seconds"""
        base_time = 2.0  # Base response time
        
        # Add time per AI
        ai_times = {
            'gemini': 3.0,
            'grok': 2.0,
            'openai': 2.5,
            'deepseek': 2.0,
            'openrouter': 1.5
        }
        
        total_time = sum(ai_times.get(ai, 2.0) for ai in ai_names)
        
        # Mode multipliers
        if mode == "consensus":
            total_time *= 1.5  # Extra time for consensus building
        elif mode == "debate":
            total_time *= 1.3  # Extra time for debate
        
        return total_time
    
    def _estimate_consultation_cost(self, ai_names: List[str], context_length: int) -> float:
        """Estimate consultation cost in arbitrary units"""
        # Simplified cost model
        cost_per_1k_tokens = {
            'gemini': 0.01,
            'grok': 0.008,
            'openai': 0.015,
            'deepseek': 0.005,
            'openrouter': 0.003
        }
        
        # Estimate tokens (rough approximation)
        estimated_tokens = context_length / 4  # ~4 chars per token
        
        total_cost = 0.0
        for ai in ai_names:
            cost_rate = cost_per_1k_tokens.get(ai, 0.01)
            total_cost += (estimated_tokens / 1000) * cost_rate
        
        return total_cost
    
    def _generate_strategy_reasoning(self,
                                   pattern_analysis: Dict[str, Any],
                                   selected_ais: List[str],
                                   mode: str,
                                   priority: str) -> str:
        """Generate human-readable reasoning for strategy selection"""
        
        reasoning = f"Selected {len(selected_ais)} AI(s) for {mode} consultation. "
        
        if pattern_analysis['requires_multi_ai']:
            reasoning += "Multiple AIs required due to pattern criticality. "
        
        if pattern_analysis['max_severity'] == PatternSeverity.CRITICAL:
            reasoning += "Critical severity patterns demand thorough analysis. "
        
        reasoning += f"Optimizing for {priority}. "
        
        # AI selection reasoning
        ai_reasons = []
        for ai in selected_ais:
            profile = self.AI_PROFILES.get(ai)
            if profile:
                if pattern_analysis['primary_category'] in profile.preferred_patterns:
                    ai_reasons.append(f"{ai} selected for {pattern_analysis['primary_category'].value} expertise")
        
        if ai_reasons:
            reasoning += "AI selection: " + "; ".join(ai_reasons) + "."
        
        return reasoning
    
    def _update_metrics(self, audit: ConsultationAudit, response: ConsultationResponse):
        """Update consultation metrics"""
        with self.metrics_lock:
            self.metrics.total_consultations += 1
            
            if response.confidence_score > 0.5:
                self.metrics.successful_consultations += 1
            else:
                self.metrics.failed_consultations += 1
            
            self.metrics.total_response_time += audit.processing_time
            
            # Update AI usage
            for ai in audit.ais_consulted:
                self.metrics.ai_usage_count[ai] += 1
            
            # Update pattern categories
            for category in audit.pattern_summary.get('categories', []):
                self.metrics.pattern_category_count[category] += 1
            
            # Update average confidence
            total_confidence = (
                self.metrics.average_confidence * (self.metrics.total_consultations - 1) +
                response.confidence_score
            )
            self.metrics.average_confidence = total_confidence / self.metrics.total_consultations
            
            self.metrics.last_updated = datetime.now()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of consultation metrics"""
        with self.metrics_lock:
            success_rate = (
                self.metrics.successful_consultations / max(1, self.metrics.total_consultations)
            )
            avg_response_time = (
                self.metrics.total_response_time / max(1, self.metrics.total_consultations)
            )
            
            return {
                'total_consultations': self.metrics.total_consultations,
                'success_rate': f"{success_rate:.2%}",
                'average_response_time': f"{avg_response_time:.2f}s",
                'average_confidence': f"{self.metrics.average_confidence:.2%}",
                'most_used_ai': max(self.metrics.ai_usage_count.items(), key=lambda x: x[1])[0]
                              if self.metrics.ai_usage_count else "none",
                'most_common_pattern': max(self.metrics.pattern_category_count.items(), key=lambda x: x[1])[0]
                                     if self.metrics.pattern_category_count else "none",
                'last_updated': self.metrics.last_updated.isoformat()
            }
    
    def get_audit_trail(self, 
                       limit: int = 10,
                       pattern_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent consultation audit trail"""
        with self.audit_lock:
            audits = self.audit_trail[-limit:]
            
            if pattern_category:
                audits = [
                    a for a in audits 
                    if pattern_category in a.pattern_summary.get('categories', [])
                ]
            
            return [
                {
                    'consultation_id': a.consultation_id,
                    'timestamp': a.timestamp.isoformat(),
                    'pattern_summary': a.pattern_summary,
                    'ais_consulted': a.ais_consulted,
                    'mode': a.strategy_used.consultation_mode,
                    'confidence': a.final_confidence,
                    'processing_time': f"{a.processing_time:.2f}s"
                }
                for a in audits
            ]
    
    def export_governance_report(self) -> Dict[str, Any]:
        """Export governance and compliance report"""
        return {
            'consultation_metrics': self.get_metrics_summary(),
            'ai_usage_distribution': dict(self.metrics.ai_usage_count),
            'pattern_distribution': dict(self.metrics.pattern_category_count),
            'recent_consultations': self.get_audit_trail(limit=20),
            'compliance_notes': {
                'data_retention': 'No persistent storage of consultation data',
                'ai_transparency': 'All AI consultations logged with full audit trail',
                'user_control': 'Users can exclude specific AIs and set preferences',
                'security': 'No sensitive data logged in audit trail'
            }
        }


# Example usage and testing
if __name__ == "__main__":
    from pattern_detection import EnhancedPatternDetectionEngine
    
    # Create manager
    manager = AIConsultationManager()
    
    # Test text with multiple patterns
    test_text = """
    def authenticate_user(username, password):
        # TODO: Add proper validation
        if password == "admin123":  # FIXME: Hardcoded password
            return {"token": "secret_token"}
        
        # O(n²) search through users - need to optimize
        for user in all_users:
            for permission in user.permissions:
                if check_permission(permission):
                    return user
    """
    
    # Detect patterns
    engine = EnhancedPatternDetectionEngine()
    patterns = engine.detect_patterns(test_text)
    
    print("AI Consultation Manager Test")
    print("=" * 60)
    
    # Test strategy selection
    strategy = manager.select_ai_strategy(patterns, test_text)
    print(f"\nSelected Strategy:")
    print(f"  AIs: {strategy.ai_selection}")
    print(f"  Mode: {strategy.consultation_mode}")
    print(f"  Reasoning: {strategy.reasoning}")
    print(f"  Est. Time: {strategy.estimated_time:.1f}s")
    
    # Test consultation execution (without actual AI calls)
    response = manager.execute_consultation(patterns, test_text, strategy)
    print(f"\nConsultation Results:")
    print(f"  ID: {response.request_id}")
    print(f"  Confidence: {response.confidence_score:.2%}")
    print(f"  AIs Used: {list(response.ai_responses.keys())}")
    
    # Show metrics
    print(f"\nMetrics Summary:")
    metrics = manager.get_metrics_summary()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Show audit trail
    print(f"\nRecent Consultations:")
    for audit in manager.get_audit_trail(limit=3):
        print(f"  {audit['timestamp']}: {audit['ais_consulted']} ({audit['confidence']:.2%})")
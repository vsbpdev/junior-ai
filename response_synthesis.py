#!/usr/bin/env python3
"""
Response Synthesis Module for Junior AI Assistant
Intelligently combines and presents multiple AI consultation responses
"""

import re
import json
import time
import string
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import difflib

from pattern_detection import PatternCategory, PatternSeverity


class SynthesisStrategy(Enum):
    """Available synthesis strategies"""
    CONSENSUS = "consensus"          # Find common agreements
    DEBATE = "debate"               # Highlight differences
    EXPERT_WEIGHTED = "expert"      # Weight by AI expertise
    COMPREHENSIVE = "comprehensive"  # Include all perspectives
    SUMMARY = "summary"             # Concise key points only
    HIERARCHICAL = "hierarchical"   # Organize by importance


@dataclass
class ResponseSection:
    """Represents a section of synthesized response"""
    title: str
    content: str
    confidence: float
    source_ais: List[str]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesizedResponse:
    """Complete synthesized response"""
    sections: List[ResponseSection]
    summary: str
    key_insights: List[str]
    agreements: List[str]
    disagreements: List[str]
    confidence_score: float
    synthesis_time: float
    metadata: Dict[str, Any]


class ResponseAnalyzer:
    """Analyzes AI responses to extract key information"""
    
    def __init__(self):
        self.code_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self.header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^\s*[-*+]\s+(.+)$', re.MULTILINE)
        self.recommendation_keywords = [
            'recommend', 'suggest', 'should', 'must', 'best practice',
            'avoid', 'don\'t', 'never', 'always', 'consider'
        ]
    
    def extract_code_blocks(self, response: str) -> List[Tuple[str, str]]:
        """Extract code blocks with their language"""
        matches = self.code_pattern.findall(response)
        results = []
        
        for lang, code in matches:
            # If language is specified in the code fence
            if lang:
                results.append((lang, code.strip()))
            else:
                # Guess language from content
                guessed_lang = self._guess_language(code)
                results.append((guessed_lang, code.strip()))
        
        return results
    
    def extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations and action items"""
        recommendations = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            # Check for recommendation keywords
            if any(keyword in sentence_lower for keyword in self.recommendation_keywords):
                clean_sentence = sentence.strip()
                if clean_sentence and len(clean_sentence) > 10:
                    recommendations.append(clean_sentence)
        
        return recommendations
    
    def extract_sections(self, response: str) -> Dict[str, str]:
        """Extract sections based on headers"""
        sections = {}
        current_section = "introduction"
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            header_match = self.header_pattern.match(line)
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                # Start new section
                current_section = header_match.group(1).lower().strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Normalize texts - remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        
        text1_clean = text1.translate(translator).lower()
        text2_clean = text2.translate(translator).lower()
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity with bonus for key word matches
        intersection = words1 & words2
        union = words1 | words2
        
        # Base Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for matching important words (longer words)
        important_matches = sum(1 for word in intersection if len(word) > 4)
        bonus = min(0.2, important_matches * 0.05)
        
        return min(1.0, jaccard + bonus)
    
    def _guess_language(self, code: str) -> str:
        """Guess programming language from code content"""
        # Simple heuristics
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function' in code or 'const ' in code or 'let ' in code:
            return 'javascript'
        elif 'public class' in code or 'private ' in code:
            return 'java'
        elif '#include' in code:
            return 'cpp'
        else:
            return 'text'


class BaseSynthesisStrategy(ABC):
    """Base class for synthesis strategies"""
    
    def __init__(self, analyzer: ResponseAnalyzer):
        self.analyzer = analyzer
    
    @abstractmethod
    def synthesize(self, ai_responses: Dict[str, str], context: Dict[str, Any]) -> SynthesizedResponse:
        """Synthesize multiple AI responses"""
        pass
    
    def extract_common_themes(self, responses: Dict[str, str]) -> List[str]:
        """Extract themes mentioned by multiple AIs"""
        all_words = []
        for response in responses.values():
            # Extract meaningful words (simple approach)
            words = re.findall(r'\b\w{4,}\b', response.lower())
            all_words.extend(words)
        
        # Count word frequency
        word_counts = Counter(all_words)
        
        # Filter common words and get top themes
        common_words = {'that', 'this', 'with', 'from', 'have', 'will', 'your', 'what', 'when', 'where'}
        themes = [word for word, count in word_counts.most_common(20) 
                 if word not in common_words and count >= len(responses) * 0.5]
        
        return themes[:10]


class ConsensusSynthesisStrategy(BaseSynthesisStrategy):
    """Find and highlight agreements between AIs"""
    
    def synthesize(self, ai_responses: Dict[str, str], context: Dict[str, Any]) -> SynthesizedResponse:
        start_time = time.time()
        sections = []
        agreements = []
        disagreements = []
        
        # Extract recommendations from each AI
        all_recommendations = {}
        for ai_name, response in ai_responses.items():
            all_recommendations[ai_name] = self.analyzer.extract_recommendations(response)
        
        # Find common recommendations
        recommendation_texts = []
        for ai_name, recs in all_recommendations.items():
            # Tag each recommendation with its source
            for rec in recs:
                recommendation_texts.append((rec, ai_name))
        
        # Group similar recommendations
        if recommendation_texts:
            grouped = self._group_similar_recommendations(recommendation_texts)
            
            # Identify agreements (mentioned by multiple AIs)
            for group_text, ai_sources in grouped:
                if len(ai_sources) >= max(2, len(ai_responses) * 0.5):  # At least 2 AIs or half
                    agreements.append(group_text)
        
        # Extract code blocks
        all_code_blocks = {}
        for ai_name, response in ai_responses.items():
            all_code_blocks[ai_name] = self.analyzer.extract_code_blocks(response)
        
        # Create sections
        # 1. Consensus Recommendations
        if agreements:
            consensus_content = "All AIs agree on the following key points:\n\n"
            for i, agreement in enumerate(agreements, 1):
                consensus_content += f"{i}. {agreement}\n"
            
            sections.append(ResponseSection(
                title="🤝 Consensus Recommendations",
                content=consensus_content,
                confidence=0.95,
                source_ais=list(ai_responses.keys()),
                priority=1
            ))
        
        # 2. Implementation Examples
        code_section_content = ""
        for ai_name, code_blocks in all_code_blocks.items():
            if code_blocks:
                code_section_content += f"### {ai_name.upper()} Implementation:\n\n"
                for lang, code in code_blocks[:1]:  # Just first code block
                    code_section_content += f"```{lang}\n{code}\n```\n\n"
        
        if code_section_content:
            sections.append(ResponseSection(
                title="💻 Implementation Examples",
                content=code_section_content.strip(),
                confidence=0.85,
                source_ais=[ai for ai, blocks in all_code_blocks.items() if blocks],
                priority=2
            ))
        
        # 3. Additional Insights
        themes = self.extract_common_themes(ai_responses)
        if themes:
            insights_content = "Key themes identified across all responses:\n\n"
            insights_content += "• " + "\n• ".join(themes[:5])
            
            sections.append(ResponseSection(
                title="🔍 Key Insights",
                content=insights_content,
                confidence=0.8,
                source_ais=list(ai_responses.keys()),
                priority=3
            ))
        
        # Generate summary
        summary = self._generate_consensus_summary(agreements, len(ai_responses))
        
        return SynthesizedResponse(
            sections=sections,
            summary=summary,
            key_insights=themes[:5],
            agreements=agreements,
            disagreements=disagreements,
            confidence_score=self._calculate_consensus_confidence(agreements, ai_responses),
            synthesis_time=time.time() - start_time,
            metadata={
                "strategy": "consensus",
                "ai_count": len(ai_responses),
                "agreement_ratio": len(agreements) / max(1, len(recommendation_texts))
            }
        )
    
    def _group_similar_recommendations(self, rec_tuples: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
        """Group similar recommendations and track their sources"""
        groups = []
        used = set()
        
        for i, (rec1, ai1) in enumerate(rec_tuples):
            if i in used:
                continue
            
            # Start new group
            group_text = rec1
            ai_sources = [ai1]
            used.add(i)
            
            # Find similar recommendations
            for j, (rec2, ai2) in enumerate(rec_tuples[i+1:], i+1):
                if j not in used:
                    similarity = self.analyzer.calculate_similarity(rec1, rec2)
                    if similarity >= 0.5:  # Lower threshold for recommendations
                        ai_sources.append(ai2)
                        used.add(j)
            
            groups.append((group_text, ai_sources))
        
        return groups
    
    def _group_similar_items(self, items: List[str], threshold: float = 0.6) -> List[List[str]]:
        """Group similar text items together"""
        groups = []
        used = set()
        
        for i, item1 in enumerate(items):
            if i in used:
                continue
            
            group = [item1]
            used.add(i)
            
            for j, item2 in enumerate(items[i+1:], i+1):
                if j not in used:
                    similarity = self.analyzer.calculate_similarity(item1, item2)
                    if similarity >= threshold:
                        group.append(item2)
                        used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _generate_consensus_summary(self, agreements: List[str], ai_count: int) -> str:
        """Generate a summary of the consensus"""
        if not agreements:
            return f"The {ai_count} AIs provided diverse perspectives without strong consensus."
        
        summary = f"Based on analysis from {ai_count} AIs, there is strong consensus on "
        summary += f"{len(agreements)} key recommendations. "
        summary += "All AIs emphasize the importance of the identified best practices."
        
        return summary
    
    def _calculate_consensus_confidence(self, agreements: List[str], ai_responses: Dict[str, str]) -> float:
        """Calculate confidence based on agreement level"""
        if not ai_responses:
            return 0.0
        
        base_confidence = 0.7
        agreement_bonus = min(0.3, len(agreements) * 0.05)
        ai_count_bonus = min(0.2, len(ai_responses) * 0.05)
        
        return min(1.0, base_confidence + agreement_bonus + ai_count_bonus)


class DebateSynthesisStrategy(BaseSynthesisStrategy):
    """Highlight different perspectives and disagreements"""
    
    def synthesize(self, ai_responses: Dict[str, str], context: Dict[str, Any]) -> SynthesizedResponse:
        start_time = time.time()
        sections = []
        disagreements = []
        
        # Extract different viewpoints
        viewpoints = {}
        for ai_name, response in ai_responses.items():
            sections_dict = self.analyzer.extract_sections(response)
            viewpoints[ai_name] = sections_dict
        
        # Find topics with different approaches
        all_topics = set()
        for sections_dict in viewpoints.values():
            all_topics.update(sections_dict.keys())
        
        # Compare viewpoints on each topic
        for topic in all_topics:
            topic_viewpoints = {}
            for ai_name, sections_dict in viewpoints.items():
                if topic in sections_dict:
                    topic_viewpoints[ai_name] = sections_dict[topic]
            
            if len(topic_viewpoints) >= 2:
                # Check for significant differences
                divergence = self._calculate_viewpoint_divergence(topic_viewpoints)
                if divergence > 0.4:  # Significant difference threshold
                    disagreements.append(f"Different approaches to {topic}")
                    
                    # Create comparison section
                    comparison_content = self._format_viewpoint_comparison(topic, topic_viewpoints)
                    sections.append(ResponseSection(
                        title=f"🤔 Different Perspectives: {topic.title()}",
                        content=comparison_content,
                        confidence=0.75,
                        source_ais=list(topic_viewpoints.keys()),
                        priority=2
                    ))
        
        # Add synthesis overview
        overview = self._create_debate_overview(ai_responses, disagreements)
        sections.insert(0, ResponseSection(
            title="📊 Analysis Overview",
            content=overview,
            confidence=0.85,
            source_ais=list(ai_responses.keys()),
            priority=1
        ))
        
        summary = f"Analysis reveals {len(disagreements)} areas with different approaches across {len(ai_responses)} AIs."
        
        return SynthesizedResponse(
            sections=sections,
            summary=summary,
            key_insights=self.extract_common_themes(ai_responses)[:5],
            agreements=[],
            disagreements=disagreements,
            confidence_score=0.8,
            synthesis_time=time.time() - start_time,
            metadata={
                "strategy": "debate",
                "divergence_count": len(disagreements),
                "topics_analyzed": len(all_topics)
            }
        )
    
    def _calculate_viewpoint_divergence(self, viewpoints: Dict[str, str]) -> float:
        """Calculate how different the viewpoints are"""
        if len(viewpoints) < 2:
            return 0.0
        
        similarities = []
        viewpoint_list = list(viewpoints.values())
        
        for i in range(len(viewpoint_list)):
            for j in range(i + 1, len(viewpoint_list)):
                sim = self.analyzer.calculate_similarity(viewpoint_list[i], viewpoint_list[j])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def _format_viewpoint_comparison(self, topic: str, viewpoints: Dict[str, str]) -> str:
        """Format different viewpoints for comparison"""
        content = f"Different approaches to **{topic}**:\n\n"
        
        for ai_name, viewpoint in viewpoints.items():
            content += f"### {ai_name.upper()} Perspective:\n"
            # Truncate long viewpoints
            truncated = viewpoint[:300] + "..." if len(viewpoint) > 300 else viewpoint
            content += f"{truncated}\n\n"
        
        return content
    
    def _create_debate_overview(self, ai_responses: Dict[str, str], disagreements: List[str]) -> str:
        """Create an overview of the debate analysis"""
        overview = f"Analyzed responses from {len(ai_responses)} AIs:\n\n"
        
        if disagreements:
            overview += "**Areas of Different Approaches:**\n"
            for disagreement in disagreements:
                overview += f"• {disagreement}\n"
            overview += "\n"
        
        overview += "**Analysis Method:**\n"
        overview += "• Extracted and compared sections from each response\n"
        overview += "• Identified topics with divergent viewpoints\n"
        overview += "• Highlighted significant differences in approach\n"
        
        return overview


class ExpertWeightedSynthesisStrategy(BaseSynthesisStrategy):
    """Weight responses based on AI expertise for the pattern type"""
    
    # AI expertise mapping
    AI_EXPERTISE = {
        PatternCategory.SECURITY: {
            "gemini": 0.9,
            "openai": 0.85,
            "grok": 0.8,
            "deepseek": 0.7,
            "openrouter": 0.75
        },
        PatternCategory.ALGORITHM: {
            "deepseek": 0.95,
            "openai": 0.85,
            "gemini": 0.85,
            "grok": 0.7,
            "openrouter": 0.75
        },
        PatternCategory.ARCHITECTURE: {
            "gemini": 0.9,
            "openai": 0.9,
            "grok": 0.85,
            "deepseek": 0.75,
            "openrouter": 0.8
        },
        PatternCategory.UNCERTAINTY: {
            "openai": 0.9,
            "gemini": 0.85,
            "openrouter": 0.85,
            "grok": 0.8,
            "deepseek": 0.75
        },
        PatternCategory.GOTCHA: {
            "gemini": 0.9,
            "openai": 0.85,
            "grok": 0.8,
            "deepseek": 0.8,
            "openrouter": 0.75
        }
    }
    
    def synthesize(self, ai_responses: Dict[str, str], context: Dict[str, Any]) -> SynthesizedResponse:
        start_time = time.time()
        sections = []
        
        # Get pattern category from context
        pattern_category = context.get('pattern_category', PatternCategory.UNCERTAINTY)
        
        # Calculate weights
        weights = self._calculate_weights(ai_responses.keys(), pattern_category)
        
        # Extract weighted recommendations
        weighted_recommendations = self._extract_weighted_recommendations(ai_responses, weights)
        
        # Create expert recommendation section
        if weighted_recommendations:
            expert_content = "Based on AI expertise weighting:\n\n"
            for i, (rec, weight) in enumerate(weighted_recommendations[:5], 1):
                expert_content += f"{i}. **[Weight: {weight:.2f}]** {rec}\n"
            
            sections.append(ResponseSection(
                title="🎯 Expert-Weighted Recommendations",
                content=expert_content,
                confidence=0.9,
                source_ais=list(ai_responses.keys()),
                priority=1,
                metadata={"weights": weights}
            ))
        
        # Add individual expert opinions
        expert_opinions = self._format_expert_opinions(ai_responses, weights, pattern_category)
        sections.append(ResponseSection(
            title="👨‍🔬 Expert Analysis by AI",
            content=expert_opinions,
            confidence=0.85,
            source_ais=list(ai_responses.keys()),
            priority=2
        ))
        
        summary = self._generate_expert_summary(weights, pattern_category)
        
        return SynthesizedResponse(
            sections=sections,
            summary=summary,
            key_insights=self.extract_common_themes(ai_responses)[:5],
            agreements=[],
            disagreements=[],
            confidence_score=self._calculate_expert_confidence(weights),
            synthesis_time=time.time() - start_time,
            metadata={
                "strategy": "expert_weighted",
                "pattern_category": pattern_category.value,
                "weights": weights
            }
        )
    
    def _calculate_weights(self, ai_names: List[str], category: PatternCategory) -> Dict[str, float]:
        """Calculate expertise weights for each AI"""
        weights = {}
        expertise_map = self.AI_EXPERTISE.get(category, {})
        
        for ai_name in ai_names:
            # Default weight if not in expertise map
            weights[ai_name] = expertise_map.get(ai_name, 0.7)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _extract_weighted_recommendations(self, ai_responses: Dict[str, str], weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """Extract recommendations with their weights"""
        weighted_recs = []
        
        for ai_name, response in ai_responses.items():
            weight = weights.get(ai_name, 0.5)
            recommendations = self.analyzer.extract_recommendations(response)
            
            for rec in recommendations:
                weighted_recs.append((rec, weight))
        
        # Sort by weight
        weighted_recs.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_recs
    
    def _format_expert_opinions(self, ai_responses: Dict[str, str], weights: Dict[str, float], category: PatternCategory) -> str:
        """Format expert opinions with weights"""
        content = f"For {category.value} patterns, AI expertise ranking:\n\n"
        
        # Sort AIs by weight
        sorted_ais = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for ai_name, weight in sorted_ais:
            content += f"### {ai_name.upper()} (Expertise: {weight:.2%})\n"
            
            # Extract key points from response
            response = ai_responses.get(ai_name, "")
            sections = self.analyzer.extract_sections(response)
            
            # Get first meaningful section
            for section_name, section_content in sections.items():
                if section_content and len(section_content) > 50:
                    truncated = section_content[:200] + "..." if len(section_content) > 200 else section_content
                    content += f"{truncated}\n\n"
                    break
        
        return content
    
    def _generate_expert_summary(self, weights: Dict[str, float], category: PatternCategory) -> str:
        """Generate summary based on expert weighting"""
        top_expert = max(weights.items(), key=lambda x: x[1])[0]
        return f"For {category.value} patterns, {top_expert.upper()} has the highest expertise ({weights[top_expert]:.2%}). Recommendations are weighted accordingly."
    
    def _calculate_expert_confidence(self, weights: Dict[str, float]) -> float:
        """Calculate confidence based on expert weights"""
        if not weights:
            return 0.5
        
        # Higher confidence if we have high-expertise AIs
        max_weight = max(weights.values())
        avg_weight = sum(weights.values()) / len(weights)
        
        return min(1.0, 0.5 + max_weight * 0.3 + avg_weight * 0.2)


class ResponseSynthesizer:
    """Main synthesizer that coordinates different strategies"""
    
    def __init__(self):
        self.analyzer = ResponseAnalyzer()
        self.strategies = {
            SynthesisStrategy.CONSENSUS: ConsensusSynthesisStrategy(self.analyzer),
            SynthesisStrategy.DEBATE: DebateSynthesisStrategy(self.analyzer),
            SynthesisStrategy.EXPERT_WEIGHTED: ExpertWeightedSynthesisStrategy(self.analyzer)
        }
        self.synthesis_history = []
    
    def synthesize(
        self,
        ai_responses: Dict[str, str],
        strategy: SynthesisStrategy = SynthesisStrategy.CONSENSUS,
        context: Optional[Dict[str, Any]] = None
    ) -> SynthesizedResponse:
        """Synthesize AI responses using specified strategy"""
        
        # Filter out empty responses
        valid_responses = {k: v for k, v in ai_responses.items() if v and v.strip()}
        
        if not valid_responses:
            return self._create_empty_response()
        
        if len(valid_responses) == 1:
            return self._create_single_ai_response(valid_responses)
        
        # Use specified strategy
        if strategy not in self.strategies:
            strategy = SynthesisStrategy.CONSENSUS
        
        synthesis_strategy = self.strategies[strategy]
        result = synthesis_strategy.synthesize(valid_responses, context or {})
        
        # Record in history
        self.synthesis_history.append({
            "timestamp": time.time(),
            "strategy": strategy.value,
            "ai_count": len(valid_responses),
            "confidence": result.confidence_score
        })
        
        return result
    
    def format_response(self, synthesized: SynthesizedResponse, format_type: str = "markdown") -> str:
        """Format synthesized response for presentation"""
        
        if format_type == "markdown":
            return self._format_markdown(synthesized)
        elif format_type == "json":
            return self._format_json(synthesized)
        else:
            return self._format_text(synthesized)
    
    def _create_empty_response(self) -> SynthesizedResponse:
        """Create empty response when no AI responses available"""
        return SynthesizedResponse(
            sections=[],
            summary="No AI responses available for synthesis.",
            key_insights=[],
            agreements=[],
            disagreements=[],
            confidence_score=0.0,
            synthesis_time=0.0,
            metadata={"empty": True}
        )
    
    def _create_single_ai_response(self, ai_responses: Dict[str, str]) -> SynthesizedResponse:
        """Create response for single AI consultation"""
        ai_name = list(ai_responses.keys())[0]
        response = ai_responses[ai_name]
        
        sections = [ResponseSection(
            title=f"Analysis by {ai_name.upper()}",
            content=response,
            confidence=0.85,
            source_ais=[ai_name],
            priority=1
        )]
        
        return SynthesizedResponse(
            sections=sections,
            summary=f"Single AI consultation from {ai_name}.",
            key_insights=self.analyzer.extract_recommendations(response)[:3],
            agreements=[],
            disagreements=[],
            confidence_score=0.85,
            synthesis_time=0.0,
            metadata={"single_ai": True, "ai_name": ai_name}
        )
    
    def _format_markdown(self, synthesized: SynthesizedResponse) -> str:
        """Format as markdown"""
        output = []
        
        # Summary
        output.append(f"**Summary**: {synthesized.summary}\n")
        
        # Confidence
        output.append(f"**Confidence**: {synthesized.confidence_score:.2%}\n")
        
        # Sections
        for section in sorted(synthesized.sections, key=lambda s: s.priority):
            output.append(f"\n## {section.title}\n")
            output.append(section.content)
            if section.confidence < 0.8:
                output.append(f"\n*Confidence: {section.confidence:.2%}*")
        
        # Key insights
        if synthesized.key_insights:
            output.append("\n## 💡 Key Insights\n")
            for insight in synthesized.key_insights:
                output.append(f"- {insight}")
        
        # Agreements/Disagreements
        if synthesized.agreements:
            output.append("\n## ✅ Points of Agreement\n")
            for agreement in synthesized.agreements:
                output.append(f"- {agreement}")
        
        if synthesized.disagreements:
            output.append("\n## ⚡ Different Perspectives\n")
            for disagreement in synthesized.disagreements:
                output.append(f"- {disagreement}")
        
        return '\n'.join(output)
    
    def _format_json(self, synthesized: SynthesizedResponse) -> str:
        """Format as JSON"""
        data = {
            "summary": synthesized.summary,
            "confidence": synthesized.confidence_score,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "confidence": section.confidence,
                    "sources": section.source_ais
                }
                for section in synthesized.sections
            ],
            "insights": synthesized.key_insights,
            "agreements": synthesized.agreements,
            "disagreements": synthesized.disagreements,
            "metadata": synthesized.metadata
        }
        return json.dumps(data, indent=2)
    
    def _format_text(self, synthesized: SynthesizedResponse) -> str:
        """Format as plain text"""
        lines = []
        lines.append(f"SUMMARY: {synthesized.summary}")
        lines.append(f"CONFIDENCE: {synthesized.confidence_score:.2%}")
        lines.append("")
        
        for section in synthesized.sections:
            lines.append(f"=== {section.title} ===")
            lines.append(section.content)
            lines.append("")
        
        return '\n'.join(lines)


# Convenience functions
def synthesize_responses(
    ai_responses: Dict[str, str],
    strategy: str = "consensus",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to synthesize responses"""
    synthesizer = ResponseSynthesizer()
    
    # Convert string strategy to enum
    strategy_enum = SynthesisStrategy(strategy) if isinstance(strategy, str) else strategy
    
    # Synthesize
    result = synthesizer.synthesize(ai_responses, strategy_enum, context)
    
    # Format as markdown
    return synthesizer.format_response(result, "markdown")


if __name__ == "__main__":
    # Test the synthesis system
    print("Testing Response Synthesis System...")
    
    # Mock AI responses
    test_responses = {
        "gemini": """## Security Analysis
        
The main security concern here is the password storage. You should never use MD5 for passwords.

### Recommendations:
- Use bcrypt or Argon2 for password hashing
- Implement proper salt generation
- Consider using a minimum password length of 12 characters

Here's a secure implementation:
```python
import bcrypt

def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)
```

Always validate input and use parameterized queries.""",
        
        "openai": """## Password Security Best Practices

MD5 is cryptographically broken and should not be used for passwords.

### Key Points:
1. Use bcrypt, scrypt, or Argon2id for password hashing
2. Generate unique salts for each password
3. Implement rate limiting on login attempts

Example with bcrypt:
```python
import bcrypt

# Hash password
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

# Verify password
bcrypt.checkpw(password.encode('utf-8'), hashed)
```

Additional security measures:
- Implement 2FA
- Use HTTPS only
- Regular security audits""",
        
        "grok": """## Security Recommendations

MD5 is completely insecure for passwords. Here's what you need:

1. **Modern Hashing**: Use Argon2id (preferred) or bcrypt
2. **Salting**: Always use random salts
3. **Key Stretching**: Make brute force attacks impractical

Never store passwords in plain text or with weak hashing."""
    }
    
    synthesizer = ResponseSynthesizer()
    
    # Test different strategies
    strategies = [SynthesisStrategy.CONSENSUS, SynthesisStrategy.DEBATE, SynthesisStrategy.EXPERT_WEIGHTED]
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing {strategy.value} strategy:")
        print('='*80)
        
        context = {"pattern_category": PatternCategory.SECURITY}
        result = synthesizer.synthesize(test_responses, strategy, context)
        formatted = synthesizer.format_response(result, "markdown")
        
        print(formatted)
        print(f"\nSynthesis took: {result.synthesis_time:.3f}s")
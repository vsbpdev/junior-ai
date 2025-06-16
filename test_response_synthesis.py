#!/usr/bin/env python3
"""
Tests for Response Synthesis Module
"""

import unittest
from unittest.mock import Mock, patch
import json
from response_synthesis import (
    ResponseSynthesizer, ResponseAnalyzer, SynthesisStrategy,
    ConsensusSynthesisStrategy, DebateSynthesisStrategy, ExpertWeightedSynthesisStrategy,
    ResponseSection, SynthesizedResponse, synthesize_responses
)
from pattern_detection import PatternCategory


class TestResponseAnalyzer(unittest.TestCase):
    """Test response analysis functionality"""
    
    def setUp(self):
        self.analyzer = ResponseAnalyzer()
    
    def test_extract_code_blocks(self):
        """Test code block extraction"""
        response = """Here's a solution:
```python
def hello():
    print("Hello, World!")
```

And another example:
```javascript
console.log("Hello");
```
"""
        code_blocks = self.analyzer.extract_code_blocks(response)
        
        self.assertEqual(len(code_blocks), 2)
        self.assertEqual(code_blocks[0][0], "python")
        self.assertIn("def hello():", code_blocks[0][1])
        self.assertEqual(code_blocks[1][0], "javascript")
        self.assertIn("console.log", code_blocks[1][1])
    
    def test_extract_recommendations(self):
        """Test recommendation extraction"""
        response = """You should use bcrypt for password hashing. 
        I recommend implementing rate limiting. 
        Never store passwords in plain text.
        Consider using 2FA for additional security."""
        
        recommendations = self.analyzer.extract_recommendations(response)
        
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("bcrypt" in rec for rec in recommendations))
        self.assertTrue(any("Never" in rec for rec in recommendations))
    
    def test_extract_sections(self):
        """Test section extraction"""
        response = """## Introduction
This is the intro.

### Security Analysis
Important security points here.

## Recommendations
- Use strong hashing
- Implement 2FA"""
        
        sections = self.analyzer.extract_sections(response)
        
        self.assertIn("introduction", sections)
        self.assertIn("security analysis", sections)
        self.assertIn("recommendations", sections)
        self.assertIn("Important security points", sections["security analysis"])
    
    def test_calculate_similarity(self):
        """Test text similarity calculation"""
        text1 = "Use bcrypt for password hashing with proper salts"
        text2 = "Password hashing should use bcrypt with salt generation"
        text3 = "The weather is nice today"
        
        sim1 = self.analyzer.calculate_similarity(text1, text2)
        sim2 = self.analyzer.calculate_similarity(text1, text3)
        
        self.assertGreater(sim1, 0.3)  # Similar texts
        self.assertLess(sim2, 0.1)     # Different texts
    
    def test_guess_language(self):
        """Test programming language detection"""
        python_code = "def function():\n    import os\n    return True"
        js_code = "function test() {\n    const x = 5;\n    console.log(x);\n}"
        java_code = "public class Main {\n    private String name;\n}"
        
        self.assertEqual(self.analyzer._guess_language(python_code), "python")
        self.assertEqual(self.analyzer._guess_language(js_code), "javascript")
        self.assertEqual(self.analyzer._guess_language(java_code), "java")


class TestConsensusSynthesisStrategy(unittest.TestCase):
    """Test consensus synthesis strategy"""
    
    def setUp(self):
        self.analyzer = ResponseAnalyzer()
        self.strategy = ConsensusSynthesisStrategy(self.analyzer)
    
    def test_synthesize_consensus(self):
        """Test consensus synthesis"""
        ai_responses = {
            "ai1": "You should use bcrypt for passwords. Never use MD5. Implement rate limiting for security.",
            "ai2": "I recommend using bcrypt for password hashing. Avoid MD5. You must add rate limiting.",
            "ai3": "Best practice: use bcrypt for passwords. Do not use MD5. Always implement rate limiting."
        }
        
        result = self.strategy.synthesize(ai_responses, {})
        
        self.assertIsInstance(result, SynthesizedResponse)
        self.assertGreater(len(result.sections), 0)
        self.assertGreater(len(result.agreements), 0)
        self.assertGreater(result.confidence_score, 0.7)
    
    def test_group_similar_items(self):
        """Test grouping of similar recommendations"""
        items = [
            "Use bcrypt for password hashing",
            "Password hashing should use bcrypt",
            "Implement rate limiting",
            "Add rate limiting to login"
        ]
        
        groups = self.strategy._group_similar_items(items, threshold=0.4)
        
        # Should have at least 2 groups (bcrypt and rate limiting)
        self.assertGreaterEqual(len(groups), 2)
        # Each group should have at least one item
        for group in groups:
            self.assertGreaterEqual(len(group), 1)


class TestDebateSynthesisStrategy(unittest.TestCase):
    """Test debate synthesis strategy"""
    
    def setUp(self):
        self.analyzer = ResponseAnalyzer()
        self.strategy = DebateSynthesisStrategy(self.analyzer)
    
    def test_synthesize_debate(self):
        """Test debate synthesis with different viewpoints"""
        ai_responses = {
            "ai1": "## Approach\nUse microservices for scalability.",
            "ai2": "## Approach\nMonolithic architecture is simpler and more suitable.",
            "ai3": "## Approach\nConsider serverless for this use case."
        }
        
        result = self.strategy.synthesize(ai_responses, {})
        
        self.assertIsInstance(result, SynthesizedResponse)
        self.assertGreater(len(result.disagreements), 0)
        self.assertTrue(any("Different Perspectives" in s.title for s in result.sections))
    
    def test_calculate_viewpoint_divergence(self):
        """Test divergence calculation"""
        similar_viewpoints = {
            "ai1": "Use bcrypt with salt for passwords",
            "ai2": "Bcrypt with proper salting is recommended"
        }
        
        different_viewpoints = {
            "ai1": "Use microservices architecture",
            "ai2": "Monolithic design is better for this project"
        }
        
        low_divergence = self.strategy._calculate_viewpoint_divergence(similar_viewpoints)
        high_divergence = self.strategy._calculate_viewpoint_divergence(different_viewpoints)
        
        # Similar viewpoints should have lower divergence than different ones
        self.assertLess(low_divergence, high_divergence)
        # Very different viewpoints should have high divergence
        self.assertGreater(high_divergence, 0.4)


class TestExpertWeightedSynthesisStrategy(unittest.TestCase):
    """Test expert-weighted synthesis strategy"""
    
    def setUp(self):
        self.analyzer = ResponseAnalyzer()
        self.strategy = ExpertWeightedSynthesisStrategy(self.analyzer)
    
    def test_calculate_weights(self):
        """Test weight calculation for different categories"""
        ai_names = ["gemini", "openai", "deepseek"]
        
        # Test security weights
        security_weights = self.strategy._calculate_weights(ai_names, PatternCategory.SECURITY)
        self.assertAlmostEqual(sum(security_weights.values()), 1.0, places=5)
        self.assertGreater(security_weights["gemini"], security_weights["deepseek"])
        
        # Test algorithm weights
        algo_weights = self.strategy._calculate_weights(ai_names, PatternCategory.ALGORITHM)
        self.assertGreater(algo_weights["deepseek"], algo_weights["gemini"])
    
    def test_synthesize_expert_weighted(self):
        """Test expert-weighted synthesis"""
        ai_responses = {
            "gemini": "For security, use Argon2id with proper configuration.",
            "deepseek": "Consider bcrypt for password hashing.",
            "openai": "Argon2id is the modern standard for password hashing."
        }
        
        context = {"pattern_category": PatternCategory.SECURITY}
        result = self.strategy.synthesize(ai_responses, context)
        
        self.assertIsInstance(result, SynthesizedResponse)
        self.assertIn("weights", result.metadata)
        self.assertTrue(any("Expert-Weighted" in s.title for s in result.sections))


class TestResponseSynthesizer(unittest.TestCase):
    """Test main response synthesizer"""
    
    def setUp(self):
        self.synthesizer = ResponseSynthesizer()
    
    def test_synthesize_empty_responses(self):
        """Test handling of empty responses"""
        result = self.synthesizer.synthesize({})
        
        self.assertEqual(result.confidence_score, 0.0)
        self.assertIn("No AI responses", result.summary)
    
    def test_synthesize_single_response(self):
        """Test handling of single AI response"""
        ai_responses = {"gemini": "This is the analysis."}
        
        result = self.synthesizer.synthesize(ai_responses)
        
        self.assertEqual(len(result.sections), 1)
        self.assertIn("gemini", result.metadata["ai_name"])
        self.assertTrue(result.metadata["single_ai"])
    
    def test_synthesize_multiple_responses(self):
        """Test synthesis of multiple responses"""
        ai_responses = {
            "ai1": "Recommendation: Use secure hashing.",
            "ai2": "I recommend secure password hashing.",
            "ai3": "Secure hashing is essential."
        }
        
        result = self.synthesizer.synthesize(
            ai_responses, 
            SynthesisStrategy.CONSENSUS
        )
        
        self.assertGreater(len(result.sections), 0)
        self.assertGreater(result.confidence_score, 0.5)
    
    def test_format_markdown(self):
        """Test markdown formatting"""
        sections = [
            ResponseSection(
                title="Test Section",
                content="Test content",
                confidence=0.9,
                source_ais=["ai1"],
                priority=1
            )
        ]
        
        synthesized = SynthesizedResponse(
            sections=sections,
            summary="Test summary",
            key_insights=["Insight 1", "Insight 2"],
            agreements=["Agreement 1"],
            disagreements=["Disagreement 1"],
            confidence_score=0.85,
            synthesis_time=0.1,
            metadata={}
        )
        
        formatted = self.synthesizer.format_response(synthesized, "markdown")
        
        self.assertIn("**Summary**: Test summary", formatted)
        self.assertIn("## Test Section", formatted)
        self.assertIn("Key Insights", formatted)
        self.assertIn("Points of Agreement", formatted)
    
    def test_format_json(self):
        """Test JSON formatting"""
        sections = [
            ResponseSection(
                title="Test",
                content="Content",
                confidence=0.9,
                source_ais=["ai1"],
                priority=1
            )
        ]
        
        synthesized = SynthesizedResponse(
            sections=sections,
            summary="Summary",
            key_insights=[],
            agreements=[],
            disagreements=[],
            confidence_score=0.9,
            synthesis_time=0.1,
            metadata={}
        )
        
        formatted = self.synthesizer.format_response(synthesized, "json")
        data = json.loads(formatted)
        
        self.assertEqual(data["summary"], "Summary")
        self.assertEqual(data["confidence"], 0.9)
        self.assertEqual(len(data["sections"]), 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_synthesize_responses_function(self):
        """Test the convenience synthesis function"""
        ai_responses = {
            "ai1": "Use secure methods.",
            "ai2": "Security is important."
        }
        
        result = synthesize_responses(ai_responses, "consensus")
        
        self.assertIsInstance(result, str)
        self.assertIn("**", result)  # Markdown formatting


class TestIntegration(unittest.TestCase):
    """Integration tests for response synthesis"""
    
    def test_full_synthesis_workflow(self):
        """Test complete synthesis workflow"""
        # Complex AI responses
        ai_responses = {
            "gemini": """## Security Analysis
            
The password storage approach needs improvement.

### Recommendations:
1. Use Argon2id for password hashing
2. Implement proper salt generation
3. Add rate limiting

```python
import argon2

hasher = argon2.PasswordHasher()
hash = hasher.hash(password)
```

This provides strong security against modern attacks.""",
            
            "openai": """## Password Security

Your current approach has vulnerabilities.

### Best Practices:
- Never use MD5 for passwords
- Implement Argon2id or bcrypt
- Use rate limiting to prevent brute force

```python
from argon2 import PasswordHasher

ph = PasswordHasher()
hash = ph.hash(password)
```

Consider adding 2FA for enhanced security.""",
            
            "deepseek": """## Security Recommendations

Password security requires:
1. Modern hashing algorithms (not MD5)
2. Proper implementation
3. Defense in depth

I suggest bcrypt as it's well-tested:
```python
import bcrypt
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
```"""
        }
        
        synthesizer = ResponseSynthesizer()
        
        # Test all strategies
        for strategy in SynthesisStrategy:
            context = {"pattern_category": PatternCategory.SECURITY}
            result = synthesizer.synthesize(ai_responses, strategy, context)
            
            self.assertIsInstance(result, SynthesizedResponse)
            self.assertGreater(result.confidence_score, 0)
            self.assertGreater(len(result.sections), 0)
            
            # Format and verify
            formatted = synthesizer.format_response(result, "markdown")
            self.assertIn("**Summary**", formatted)
            self.assertIn("**Confidence**", formatted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
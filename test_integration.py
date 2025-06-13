#!/usr/bin/env python3
"""
Integration tests for Pattern Detection Engine
Tests the complete flow from detection to consultation
"""

import sys
import time
import json
import unittest
from unittest.mock import Mock, patch
from io import StringIO

# Import components
from pattern_detection import PatternDetectionEngine, PatternCategory, PatternSeverity
from text_processing_pipeline import TextProcessingPipeline
from response_handlers import PatternResponseManager
from pattern_cache import PatternDetectionCache, CachedPatternDetectionEngine


class TestPatternDetectionIntegration(unittest.TestCase):
    """Integration tests for pattern detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = PatternDetectionEngine(context_window_size=200)
        self.cache = PatternDetectionCache(max_size=10, ttl_seconds=60)
        self.cached_engine = CachedPatternDetectionEngine(self.engine, self.cache)
        self.response_manager = PatternResponseManager()
        
        # Mock AI callers
        self.mock_ai_caller = Mock(return_value="Mock AI response")
        self.mock_multi_ai_caller = Mock(return_value={
            "gemini": "Gemini response",
            "openai": "OpenAI response"
        })
        self.response_manager.set_ai_callers(self.mock_ai_caller, self.mock_multi_ai_caller)
    
    def test_basic_pattern_detection(self):
        """Test basic pattern detection functionality"""
        test_text = "TODO: Implement password hashing for security"
        
        patterns = self.engine.detect_patterns(test_text)
        
        self.assertGreater(len(patterns), 0)
        
        # Check we found both uncertainty and security patterns
        categories = {p.category for p in patterns}
        self.assertIn(PatternCategory.UNCERTAINTY, categories)
        self.assertIn(PatternCategory.SECURITY, categories)
        
        # Check severity
        security_patterns = [p for p in patterns if p.category == PatternCategory.SECURITY]
        self.assertTrue(any(p.severity == PatternSeverity.CRITICAL for p in security_patterns))
    
    def test_enhanced_context_extraction(self):
        """Test that enhanced context extraction provides better context"""
        test_code = """
        def process_payment(card_number, amount):
            # TODO: Add encryption for card number
            api_key = "sk-test-1234567890"  # This needs to be secured
            
            # Process the payment
            result = charge_card(card_number, amount, api_key)
            return result
        """
        
        patterns = self.engine.detect_patterns(test_code)
        
        # Find the API key pattern
        api_key_pattern = next((p for p in patterns if "api_key" in p.keyword.lower()), None)
        self.assertIsNotNone(api_key_pattern)
        
        # Check context includes the full line
        self.assertIn("sk-test-1234567890", api_key_pattern.context)
        self.assertIn("This needs to be secured", api_key_pattern.context)
    
    def test_pattern_caching(self):
        """Test pattern detection caching"""
        test_text = "Implement O(n^2) sorting algorithm"
        
        # First call - should miss cache
        start_time = time.time()
        patterns1 = self.cached_engine.detect_patterns(test_text)
        first_call_time = time.time() - start_time
        
        # Second call - should hit cache
        start_time = time.time()
        patterns2 = self.cached_engine.detect_patterns(test_text)
        second_call_time = time.time() - start_time
        
        # Cache hit should be faster
        self.assertLess(second_call_time, first_call_time)
        
        # Results should be the same
        self.assertEqual(len(patterns1), len(patterns2))
        
        # Check cache stats
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
    
    def test_response_generation(self):
        """Test response generation for different pattern types"""
        test_cases = [
            ("Store password securely using bcrypt", PatternCategory.SECURITY),
            ("TODO: Not sure how to implement this", PatternCategory.UNCERTAINTY),
            ("Optimize O(n^2) algorithm", PatternCategory.ALGORITHM),
            ("Handle timezone conversion", PatternCategory.GOTCHA),
            ("Should I use microservices?", PatternCategory.ARCHITECTURE)
        ]
        
        for text, expected_category in test_cases:
            with self.subTest(text=text):
                patterns = self.engine.detect_patterns(text)
                self.assertGreater(len(patterns), 0)
                
                # Handle patterns
                response = self.response_manager.handle_patterns(patterns, text)
                self.assertIsNotNone(response)
                
                # Check response structure
                self.assertIn('request_id', response.__dict__)
                self.assertIn('ai_responses', response.__dict__)
                self.assertIn('primary_recommendation', response.__dict__)
                self.assertIn('confidence_score', response.__dict__)
                
                # Check primary category matches
                primary_category = patterns[0].category
                self.assertEqual(primary_category, expected_category)
    
    def test_multi_ai_consultation(self):
        """Test multi-AI consultation for critical patterns"""
        text = "Need to implement authentication with password storage and API key management"
        
        patterns = self.engine.detect_patterns(text)
        response = self.response_manager.handle_patterns(patterns, text)
        
        # Should trigger multi-AI for security
        self.mock_multi_ai_caller.assert_called()
        
        # Check multiple AIs were consulted
        self.assertGreater(len(response.ai_responses), 1)
    
    def test_text_processing_pipeline(self):
        """Test the text processing pipeline"""
        pipeline = TextProcessingPipeline(
            pattern_engine=self.engine,
            enable_cache=True,
            cache_config={'max_size': 10, 'ttl_seconds': 60}
        )
        
        # Track results
        results = []
        
        def result_callback(result):
            results.append(result)
        
        pipeline.add_result_callback(result_callback)
        pipeline.start()
        
        try:
            # Process some text
            test_texts = [
                "Implement secure password hashing",
                "TODO: Fix the algorithm performance",
                "Handle datetime timezone issues"
            ]
            
            chunk_ids = []
            for text in test_texts:
                chunk_id = pipeline.process_text(text)
                chunk_ids.append(chunk_id)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Check results
            self.assertEqual(len(results), len(test_texts))
            
            # Verify each result
            for result in results:
                self.assertIsNotNone(result.patterns)
                self.assertGreater(result.processing_time, 0)
                self.assertIn(result.chunk_id, chunk_ids)
            
            # Check pipeline stats
            stats = pipeline.get_stats()
            self.assertEqual(stats['chunks_processed'], len(test_texts))
            self.assertGreater(stats['patterns_detected'], 0)
            
            # Check cache stats if available
            if 'cache' in stats:
                self.assertIn('hits', stats['cache'])
                self.assertIn('misses', stats['cache'])
        
        finally:
            pipeline.stop()
    
    def test_pattern_severity_ordering(self):
        """Test that patterns are ordered by severity"""
        text = """
        # TODO: Review this code
        password = "admin123"  # CRITICAL: Hardcoded password
        if float_value == 0.1:  # Gotcha: float comparison
        """
        
        patterns = self.engine.detect_patterns(text)
        
        # Should be ordered by severity (CRITICAL > HIGH > MEDIUM)
        severities = [p.severity.value for p in patterns]
        self.assertEqual(severities, sorted(severities, reverse=True))
        
        # First pattern should be the password (CRITICAL)
        self.assertEqual(patterns[0].category, PatternCategory.SECURITY)
        self.assertEqual(patterns[0].severity, PatternSeverity.CRITICAL)
    
    def test_line_number_extraction(self):
        """Test that line numbers are correctly extracted"""
        test_code = """line 1
line 2 with TODO
line 3
line 4 with password
line 5"""
        
        patterns = self.engine.detect_patterns(test_code)
        
        # Find patterns and check line numbers
        todo_pattern = next((p for p in patterns if p.keyword.upper() == "TODO"), None)
        password_pattern = next((p for p in patterns if p.keyword == "password"), None)
        
        self.assertIsNotNone(todo_pattern)
        self.assertIsNotNone(password_pattern)
        
        self.assertEqual(todo_pattern.line_number, 2)
        self.assertEqual(password_pattern.line_number, 4)
    
    def test_custom_keywords_from_config(self):
        """Test that custom keywords can be added via configuration"""
        # This would test loading custom keywords from credentials.json
        # For now, just verify the pattern structure supports it
        
        definition = self.engine.pattern_definitions[PatternCategory.SECURITY]
        self.assertIsInstance(definition.keywords, list)
        self.assertGreater(len(definition.keywords), 20)  # Should have many keywords
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            "",  # Empty string
            " " * 100,  # Only whitespace
            "a" * 10000,  # Very long string without patterns
            "TODO " * 1000,  # Many repeated patterns
            "😀 TODO: Unicode handling 🔒",  # Unicode characters
        ]
        
        for text in edge_cases:
            with self.subTest(text=text[:50]):
                # Should not crash
                patterns = self.engine.detect_patterns(text)
                self.assertIsInstance(patterns, list)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        # Generate a large text
        large_text = "\n".join([
            f"Line {i}: TODO implement feature {i % 10}"
            for i in range(1000)
        ])
        
        start_time = time.time()
        patterns = self.engine.detect_patterns(large_text)
        detection_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(detection_time, 1.0)  # Less than 1 second
        
        # Should find patterns
        self.assertGreater(len(patterns), 0)
        
        print(f"\nPerformance: Detected {len(patterns)} patterns in {detection_time:.3f}s")
        print(f"Rate: {len(large_text) / detection_time / 1000:.1f} KB/s")


class TestPatternDetectionEndToEnd(unittest.TestCase):
    """End-to-end tests simulating real usage"""
    
    def test_mcp_server_integration(self):
        """Test integration with MCP server format"""
        # This would test the actual MCP protocol integration
        # For now, test the response format
        
        from server import handle_pattern_check
        
        # Mock the global variables that server.py expects
        with patch('server.PATTERN_DETECTION_AVAILABLE', True), \
             patch('server.pattern_config', {'enabled': True}), \
             patch('server.pattern_engine', PatternDetectionEngine()), \
             patch('server.response_manager', PatternResponseManager()):
            
            result = handle_pattern_check("TODO: Implement password hashing", auto_consult=False)
            
            self.assertIsInstance(result, str)
            self.assertIn("Pattern Detection Results", result)
            self.assertIn("SECURITY", result)
            self.assertIn("UNCERTAINTY", result)


def run_performance_suite():
    """Run performance benchmarks"""
    print("\n" + "="*60)
    print("Running Performance Benchmarks")
    print("="*60)
    
    engine = PatternDetectionEngine()
    
    # Test different text sizes
    sizes = [100, 1000, 10000, 100000]
    
    for size in sizes:
        # Generate text with patterns
        text = generate_test_text(size)
        
        # Measure detection time
        start_time = time.time()
        patterns = engine.detect_patterns(text)
        detection_time = time.time() - start_time
        
        # Calculate metrics
        patterns_per_kb = len(patterns) / (size / 1000)
        kb_per_second = (size / 1000) / detection_time
        
        print(f"\nText size: {size:,} chars")
        print(f"Detection time: {detection_time:.3f}s")
        print(f"Patterns found: {len(patterns)}")
        print(f"Rate: {kb_per_second:.1f} KB/s")
        print(f"Pattern density: {patterns_per_kb:.1f} patterns/KB")


def generate_test_text(size: int) -> str:
    """Generate test text with known pattern distribution"""
    patterns = [
        "TODO: Implement this feature",
        "Fix the password storage issue",
        "Optimize the O(n^2) algorithm",
        "Handle timezone conversion properly",
        "Should we use microservices architecture?",
        "The API key needs encryption",
        "Not sure how to handle this edge case",
        "Check for race conditions in async code",
        "Follow SOLID design principles",
        "Beware of float comparison issues"
    ]
    
    text_parts = []
    while len(" ".join(text_parts)) < size:
        # Add some normal text
        text_parts.append("This is normal text without any patterns. ")
        
        # Add a pattern
        import random
        text_parts.append(random.choice(patterns) + ". ")
    
    return " ".join(text_parts)[:size]


if __name__ == "__main__":
    # Run unit tests
    print("Running Integration Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmarks
    run_performance_suite()
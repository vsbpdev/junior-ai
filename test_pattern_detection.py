#!/usr/bin/env python3
"""
Test script for pattern detection engine
"""

import sys
import time
from pattern_detection import PatternDetectionEngine, PatternCategory, PatternSeverity
from text_processing_pipeline import TextProcessingPipeline, StreamingTextProcessor
from response_handlers import PatternResponseManager

def test_pattern_detection():
    """Test pattern detection functionality"""
    print("=== Testing Pattern Detection Engine ===\n")
    
    engine = PatternDetectionEngine()
    
    test_cases = [
        # Security patterns
        ("I need to store the api_key and password securely in the database",
         PatternCategory.SECURITY, PatternSeverity.CRITICAL),
        
        # Uncertainty patterns
        ("TODO: Not sure how to implement this feature, might be complex",
         PatternCategory.UNCERTAINTY, PatternSeverity.MEDIUM),
        
        # Algorithm patterns
        ("This sorting algorithm has O(n^2) time complexity, need to optimize",
         PatternCategory.ALGORITHM, PatternSeverity.HIGH),
        
        # Gotcha patterns
        ("Working with datetime and timezone conversions, handling async callbacks",
         PatternCategory.GOTCHA, PatternSeverity.HIGH),
        
        # Architecture patterns
        ("Should I use microservices architecture or follow SOLID design patterns?",
         PatternCategory.ARCHITECTURE, PatternSeverity.HIGH),
        
        # Mixed patterns
        ("Need to encrypt passwords for authentication. TODO: Not sure about the best approach",
         PatternCategory.SECURITY, PatternSeverity.CRITICAL),
    ]
    
    for text, expected_category, expected_severity in test_cases:
        print(f"Testing: {text}")
        matches = engine.detect_patterns(text)
        
        if matches:
            print(f"✓ Found {len(matches)} patterns")
            
            # Check if expected category was detected
            categories_found = set(match.category for match in matches)
            if expected_category in categories_found:
                print(f"✓ Expected category '{expected_category.value}' detected")
            else:
                print(f"✗ Expected category '{expected_category.value}' NOT detected")
            
            # Check severity
            max_severity = max(match.severity.value for match in matches)
            if max_severity == expected_severity.value:
                print(f"✓ Expected severity '{expected_severity.name}' matched")
            else:
                severity_name = PatternSeverity(max_severity).name
                print(f"✗ Expected severity '{expected_severity.name}', got '{severity_name}'")
            
            # Show all matches
            for match in matches:
                print(f"  - {match.category.value}: '{match.keyword}' "
                      f"(severity: {match.severity.name}, confidence: {match.confidence:.2f})")
            
            # Test consultation strategy
            strategy = engine.get_consultation_strategy(matches)
            print(f"  Consultation strategy: {strategy['strategy']}")
            print(f"  Recommended AIs: {', '.join(strategy['recommended_ais'])}")
        else:
            print("✗ No patterns detected!")
        
        print("-" * 50)


def test_text_pipeline():
    """Test text processing pipeline"""
    print("\n=== Testing Text Processing Pipeline ===\n")
    
    pipeline = TextProcessingPipeline()
    results_received = []
    
    def result_callback(result):
        results_received.append(result)
        print(f"Result received for chunk '{result.chunk_id}':")
        print(f"  - Patterns: {len(result.patterns)}")
        print(f"  - Processing time: {result.processing_time:.3f}s")
        print(f"  - Consultation triggered: {result.consultation_triggered}")
    
    pipeline.add_result_callback(result_callback)
    pipeline.start()
    
    # Test with sample texts
    test_texts = [
        ("chunk1", "Store password securely using proper encryption"),
        ("chunk2", "TODO: Implement the sorting algorithm"),
        ("chunk3", "Handle timezone conversion carefully"),
    ]
    
    print("Submitting text chunks...")
    for chunk_id, text in test_texts:
        pipeline.process_text(text, chunk_id=chunk_id)
    
    # Wait for processing
    time.sleep(1.0)
    
    # Check results
    print(f"\nProcessed {len(results_received)} chunks")
    
    # Test streaming processor
    print("\n--- Testing Streaming Processor ---")
    streaming = StreamingTextProcessor(pipeline, chunk_size=50)
    
    stream_text = ("This is a streaming test. We need to handle authentication "
                   "with proper password hashing. TODO: Optimize the algorithm "
                   "for better performance. Watch out for timezone issues!")
    
    # Simulate streaming
    for i in range(0, len(stream_text), 10):
        chunk = stream_text[i:i+10]
        chunk_ids = streaming.process_stream(chunk)
        if chunk_ids:
            print(f"Created chunks: {chunk_ids}")
        time.sleep(0.05)
    
    # Flush remaining
    final_chunk = streaming.flush()
    if final_chunk:
        print(f"Final chunk: {final_chunk}")
    
    time.sleep(1.0)
    
    # Show pipeline stats
    stats = pipeline.get_stats()
    print("\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    pipeline.stop()


def test_response_handlers():
    """Test response handlers"""
    print("\n=== Testing Response Handlers ===\n")
    
    engine = PatternDetectionEngine()
    manager = PatternResponseManager()
    
    # Mock AI caller for testing
    def mock_ai_caller(ai_name, prompt, temperature=0.7):
        return f"Mock response from {ai_name} for prompt: {prompt[:50]}..."
    
    def mock_multi_ai_caller(prompt, ai_list, temperature=0.7):
        return {ai: f"Mock response from {ai}" for ai in ai_list}
    
    manager.set_ai_callers(mock_ai_caller, mock_multi_ai_caller)
    
    test_contexts = [
        "I need to implement secure password storage with proper encryption",
        "TODO: Not sure how to implement this feature efficiently",
        "Optimize this O(n^2) sorting algorithm for better performance",
        "Handle datetime timezone conversion edge cases",
        "Should I use microservices or monolithic architecture?",
    ]
    
    for context in test_contexts:
        print(f"Context: {context}")
        
        # Detect patterns
        patterns = engine.detect_patterns(context)
        
        if patterns:
            # Handle patterns
            response = manager.handle_patterns(patterns, context)
            
            if response:
                print(f"✓ Consultation Response Generated")
                print(f"  - Request ID: {response.request_id}")
                print(f"  - Confidence: {response.confidence_score:.2%}")
                print(f"  - AIs consulted: {list(response.ai_responses.keys())}")
                print(f"  - Primary recommendation preview:")
                print(f"    {response.primary_recommendation[:100]}...")
        else:
            print("✗ No patterns detected")
        
        print("-" * 50)
    
    # Show handler statistics
    print("\nHandler Statistics:")
    stats = manager.get_handler_stats()
    for category, cat_stats in stats.items():
        print(f"\n{category}:")
        for key, value in cat_stats.items():
            print(f"  {key}: {value}")


def test_pattern_categories():
    """Test each pattern category in detail"""
    print("\n=== Testing Pattern Categories ===\n")
    
    engine = PatternDetectionEngine()
    
    # Test each category
    categories_test = {
        PatternCategory.SECURITY: [
            "password", "api_key", "encrypt", "authentication", "oauth",
            "Store the secret_key securely",
            "Implement JWT token validation",
            "Handle SSL certificate verification"
        ],
        PatternCategory.UNCERTAINTY: [
            "TODO", "FIXME", "not sure", "might be", "maybe",
            "I think this could work",
            "Possibly the wrong approach",
            "Help needed with this implementation"
        ],
        PatternCategory.ALGORITHM: [
            "O(n^2)", "optimize", "algorithm", "recursive", "dynamic programming",
            "Binary search implementation",
            "Need to improve time complexity",
            "Implement efficient sorting"
        ],
        PatternCategory.GOTCHA: [
            "timezone", "datetime", "async", "promise", "race condition",
            "Handle null and undefined",
            "Float precision issues",
            "Memory leak in closure"
        ],
        PatternCategory.ARCHITECTURE: [
            "design pattern", "architecture", "should I", "best practice",
            "SOLID principles",
            "Microservices vs monolith",
            "How to structure this module"
        ]
    }
    
    for category, test_phrases in categories_test.items():
        print(f"\nTesting {category.value} patterns:")
        correct_detections = 0
        
        for phrase in test_phrases:
            patterns = engine.detect_patterns(phrase)
            if patterns and any(p.category == category for p in patterns):
                correct_detections += 1
                print(f"  ✓ '{phrase}' - Correctly detected")
            else:
                print(f"  ✗ '{phrase}' - NOT detected")
        
        accuracy = (correct_detections / len(test_phrases)) * 100
        print(f"  Category accuracy: {accuracy:.1f}% ({correct_detections}/{len(test_phrases)})")


def main():
    """Run all tests"""
    print("Pattern Detection Engine Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_pattern_detection()
        test_pattern_categories()
        test_text_pipeline()
        test_response_handlers()
        
        print("\n✓ All tests completed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
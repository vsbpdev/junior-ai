# Response Synthesis System

## Overview

The Response Synthesis system is a sophisticated module that intelligently combines multiple AI responses into coherent, actionable insights. It supports multiple synthesis strategies optimized for different pattern types and use cases.

## Key Features

1. **Multiple Synthesis Strategies**
   - Consensus: Finds agreements and common recommendations
   - Debate: Highlights different perspectives and approaches
   - Expert-Weighted: Weights responses based on AI expertise
   - Comprehensive: Includes all perspectives without filtering
   - Summary: Provides concise key points only
   - Hierarchical: Organizes by importance and relevance

2. **Intelligent Response Analysis**
   - Code block extraction with language detection
   - Recommendation extraction and grouping
   - Section-based content analysis
   - Similarity calculation for grouping related content

3. **Pattern-Specific Optimization**
   - Security patterns → Consensus strategy (find best practices)
   - Architecture patterns → Debate strategy (explore options)
   - Algorithm patterns → Expert-weighted strategy (leverage specialization)
   - Uncertainty patterns → Comprehensive strategy (complete information)
   - Gotcha patterns → Debate strategy (understand edge cases)

## Architecture

```text
response_synthesis.py
├── ResponseAnalyzer
│   ├── extract_code_blocks()
│   ├── extract_recommendations()
│   ├── extract_sections()
│   └── calculate_similarity()
├── Synthesis Strategies
│   ├── ConsensusSynthesisStrategy
│   ├── DebateSynthesisStrategy
│   └── ExpertWeightedSynthesisStrategy
├── ResponseSynthesizer
│   ├── synthesize()
│   └── format_response()
└── Data Models
    ├── ResponseSection
    └── SynthesizedResponse
```

## Usage

### Basic Usage

```python
from response_synthesis import ResponseSynthesizer, SynthesisStrategy

# Initialize synthesizer
synthesizer = ResponseSynthesizer()

# AI responses to synthesize
ai_responses = {
    "gemini": "Response from Gemini...",
    "openai": "Response from OpenAI...",
    "grok": "Response from Grok..."
}

# Synthesize with consensus strategy
result = synthesizer.synthesize(
    ai_responses,
    SynthesisStrategy.CONSENSUS,
    context={"pattern_category": PatternCategory.SECURITY}
)

# Format as markdown
output = synthesizer.format_response(result, "markdown")
print(output)
```

### Integration with Response Handlers

```python
from response_handlers_enhanced import EnhancedPatternResponseManager

# Initialize enhanced response manager
response_manager = EnhancedPatternResponseManager()

# Handle patterns with automatic synthesis
response = response_manager.handle_patterns(
    patterns,
    context,
    force_strategy=SynthesisStrategy.DEBATE  # Optional
)

# Access synthesized response
print(response.synthesis)
print(f"Confidence: {response.confidence_score}")
print(f"Strategy used: {response.metadata['synthesis_strategy']}")
```

### Available Strategies

#### 1. Consensus Strategy
Best for: Security, best practices, standards
```python
result = synthesizer.synthesize(
    ai_responses,
    SynthesisStrategy.CONSENSUS
)
```
- Identifies common recommendations
- Groups similar suggestions
- Highlights agreed-upon best practices
- Calculates confidence based on agreement level

#### 2. Debate Strategy
Best for: Architecture decisions, design choices
```python
result = synthesizer.synthesize(
    ai_responses,
    SynthesisStrategy.DEBATE
)
```
- Highlights different approaches
- Compares viewpoints side-by-side
- Identifies areas of disagreement
- Helps in understanding trade-offs

#### 3. Expert-Weighted Strategy
Best for: Specialized domains, algorithm optimization
```python
result = synthesizer.synthesize(
    ai_responses,
    SynthesisStrategy.EXPERT_WEIGHTED,
    context={"pattern_category": PatternCategory.ALGORITHM}
)
```
- Weights responses based on AI expertise
- Prioritizes recommendations from domain experts
- Provides expertise rankings
- Adjusts confidence based on expert consensus

## Output Formats

### Markdown (Default)
```python
formatted = synthesizer.format_response(result, "markdown")
```
Produces human-readable markdown with:
- Summary and confidence score
- Organized sections by priority
- Key insights and agreements/disagreements
- Formatted code blocks and lists

### JSON
```python
formatted = synthesizer.format_response(result, "json")
```
Structured data format with:
- Complete synthesis metadata
- Section details with confidence scores
- Arrays of insights and recommendations
- Source attribution for each section

### Plain Text
```python
formatted = synthesizer.format_response(result, "text")
```
Simple text format for:
- CLI output
- Log files
- Text-only environments

## Response Structure

### SynthesizedResponse Object
```python
@dataclass
class SynthesizedResponse:
    sections: List[ResponseSection]      # Organized content sections
    summary: str                         # Executive summary
    key_insights: List[str]             # Main takeaways
    agreements: List[str]               # Points of consensus
    disagreements: List[str]            # Areas of divergence
    confidence_score: float             # 0.0 to 1.0
    synthesis_time: float               # Processing duration
    metadata: Dict[str, Any]            # Additional context
```

### ResponseSection Object
```python
@dataclass
class ResponseSection:
    title: str                          # Section heading
    content: str                        # Section content
    confidence: float                   # Section confidence
    source_ais: List[str]              # Contributing AIs
    priority: int                       # Display priority
    metadata: Dict[str, Any]           # Extra data
```

## Configuration

### Setting Default Strategy
```python
response_manager.set_default_synthesis_strategy(
    SynthesisStrategy.CONSENSUS
)
```

### Pattern-Specific Strategies
Configure in credentials.json:
```json
{
    "synthesis_config": {
        "default_strategy": "consensus",
        "strategy_overrides": {
            "security": "consensus",
            "architecture": "debate",
            "algorithm": "expert",
            "uncertainty": "comprehensive",
            "gotcha": "debate"
        }
    }
}
```

## Advanced Features

### Custom Context
```python
context = {
    "pattern_category": PatternCategory.SECURITY,
    "severity": PatternSeverity.CRITICAL,
    "user_preference": "detailed",
    "time_constraint": 5.0  # seconds
}

result = synthesizer.synthesize(
    ai_responses,
    strategy,
    context
)
```

### AI Expertise Configuration
Modify `ExpertWeightedSynthesisStrategy.AI_EXPERTISE`:
```python
AI_EXPERTISE = {
    PatternCategory.SECURITY: {
        "gemini": 0.9,      # High expertise
        "openai": 0.85,
        "grok": 0.8,
        "deepseek": 0.7,
        "openrouter": 0.75
    },
    # ... other categories
}
```

### Performance Optimization
- Responses are processed asynchronously where possible
- Synthesis strategies use efficient text processing
- Caching can be added for repeated synthesis requests
- Typical synthesis time: 0.1-0.5 seconds

## Best Practices

1. **Choose the Right Strategy**
   - Match strategy to pattern type
   - Consider user needs (consensus vs options)
   - Factor in time constraints

2. **Provide Context**
   - Include pattern category for better synthesis
   - Add severity for prioritization
   - Specify any user preferences

3. **Handle Edge Cases**
   - Single AI responses are handled gracefully
   - Empty responses return appropriate messages
   - Errors are caught and reported clearly

4. **Monitor Performance**
   - Check synthesis_time in responses
   - Track confidence scores
   - Review synthesis quality periodically

## Troubleshooting

### Low Confidence Scores
- Check if AIs are providing relevant responses
- Verify pattern detection is accurate
- Consider using different synthesis strategy

### Slow Synthesis
- Reduce response length before synthesis
- Use Summary strategy for quick results
- Check for performance bottlenecks in analysis

### Poor Agreement Detection
- Adjust similarity thresholds
- Improve recommendation extraction keywords
- Consider preprocessing responses

## Future Enhancements

1. **Machine Learning Integration**
   - Learn optimal strategies per pattern
   - Improve similarity detection
   - Personalize synthesis based on history

2. **Advanced Strategies**
   - Temporal synthesis (time-based)
   - Confidence-weighted synthesis
   - Interactive synthesis with follow-ups

3. **Performance Improvements**
   - Response caching
   - Parallel processing
   - Streaming synthesis

## Examples

See `demo_response_synthesis.py` for comprehensive examples of:
- Different synthesis strategies in action
- Various pattern types and responses
- Output format comparisons
- Performance benchmarks
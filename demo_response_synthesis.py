#!/usr/bin/env python3
"""
Demo of Response Synthesis System
Shows different synthesis strategies in action
"""

from response_synthesis import ResponseSynthesizer, SynthesisStrategy
from pattern_detection import PatternCategory


def demo_synthesis():
    """Demonstrate different synthesis strategies"""
    
    # Create mock AI responses for a security question
    security_responses = {
        "gemini": """## Security Analysis: Password Storage

Your approach has critical vulnerabilities. MD5 is completely broken for password hashing.

### Recommendations:
1. **Use modern hashing**: Implement Argon2id as the gold standard
2. **Salt generation**: Always use cryptographically secure random salts
3. **Key stretching**: Configure appropriate work factors
4. **Rate limiting**: Implement progressive delays on failed attempts

```python
from argon2 import PasswordHasher
ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=4
)
hashed = ph.hash(password)
```

### Security Checklist:
- Never store passwords in plain text
- Use constant-time comparison for verification
- Implement account lockout policies
- Monitor for brute force attempts
- Regular security audits

This is critical for GDPR compliance and user trust.""",

        "openai": """## Password Security Best Practices

MD5 should never be used for passwords - it's been compromised since the 1990s.

### Modern Approach:
- **Primary choice**: Argon2id (winner of Password Hashing Competition)
- **Alternative**: bcrypt (well-tested, widely supported)
- **Legacy option**: scrypt (good but less modern)

### Implementation Guide:
```python
import argon2

# Configure hasher
hasher = argon2.PasswordHasher()

# Hash password
password_hash = hasher.hash(user_password)

# Verify password
try:
    hasher.verify(password_hash, user_input)
    # Password is correct
except argon2.exceptions.VerifyMismatchError:
    # Password is incorrect
```

### Additional Security Measures:
1. Implement rate limiting (e.g., 5 attempts per 15 minutes)
2. Use CAPTCHA after failed attempts
3. Enable 2FA for sensitive accounts
4. Log security events for monitoring
5. Implement secure password reset flows

Remember: Security is only as strong as the weakest link.""",

        "grok": """## Critical Security Issue: Password Storage

Using MD5 for passwords is a severe security vulnerability that will lead to breaches.

### Why MD5 Fails:
- Extremely fast to compute (billions per second on GPUs)
- Rainbow tables exist for common passwords
- No salt by default
- Collision vulnerabilities

### Secure Alternative:
```python
# Using bcrypt (simple and effective)
import bcrypt

# Hash a password
password = b"super secret password"
hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))

# Verify a password
if bcrypt.checkpw(password, hashed):
    print("Password matches")
```

### Defense in Depth:
- Password hashing is just one layer
- Implement account lockouts
- Use fail2ban or similar for IP blocking
- Monitor for credential stuffing attacks
- Require strong passwords (but not ridiculous rules)

Fix this immediately - your users' security depends on it!"""
    }
    
    # Create architecture debate responses
    architecture_responses = {
        "gemini": """## Architecture Analysis: Microservices vs Monolith

For your use case, I'd lean towards a modular monolith initially.

### Reasoning:
- **Team size**: Small teams struggle with microservices complexity
- **Early stage**: Rapid iteration is easier in monoliths
- **Domain boundaries**: Often unclear until the product matures

### Modular Monolith Approach:
```
/app
  /modules
    /auth        # Authentication module
    /orders      # Order management
    /inventory   # Inventory tracking
  /shared        # Shared utilities
  /api          # API layer
```

### Benefits:
- Single deployment initially
- Easy refactoring as you learn
- Can extract services later
- Lower operational overhead

### When to Consider Microservices:
- Clear bounded contexts emerge
- Team grows beyond 20 developers
- Specific scaling requirements per service
- Need for technology diversity

Start simple, evolve as needed.""",

        "openai": """## Architecture Decision: Service Architecture

I recommend starting with microservices if you have the right foundations.

### Microservices Benefits:
1. **Independent scaling**: Scale only what needs scaling
2. **Technology flexibility**: Use the right tool for each service
3. **Team autonomy**: Teams own their services
4. **Fault isolation**: One service failure doesn't crash everything

### Implementation Strategy:
```yaml
# docker-compose.yml example
services:
  api-gateway:
    build: ./gateway
    ports:
      - "8080:8080"
  
  auth-service:
    build: ./services/auth
    environment:
      - DB_URL=postgres://auth-db:5432
  
  order-service:
    build: ./services/orders
    depends_on:
      - auth-service
```

### Prerequisites for Success:
- Strong DevOps practices
- Service mesh (Istio/Linkerd)
- Distributed tracing
- Centralized logging
- API versioning strategy

If you lack these, consider a monolith first.""",

        "deepseek": """## Architectural Consideration: Monolith First

Based on analysis, start with a well-structured monolith.

### The Monolith Advantage:
- **Simplicity**: One codebase, one deployment
- **Performance**: No network latency between components
- **Debugging**: Easier to trace issues
- **Cost**: Significantly lower infrastructure costs

### Proper Monolith Structure:
```python
# Layered architecture
project/
├── presentation/    # Controllers, views
├── application/     # Use cases, services
├── domain/         # Business logic, entities
├── infrastructure/ # Database, external services
└── tests/         # Comprehensive test suite
```

### Key Principles:
1. Maintain clear module boundaries
2. Use dependency injection
3. Keep domain logic pure
4. Prepare for eventual extraction

### Migration Path:
When you need microservices:
1. Identify service boundaries from modules
2. Extract one service at a time
3. Use strangler fig pattern
4. Maintain backwards compatibility

Most successful microservices architectures evolved from monoliths."""
    }
    
    # Algorithm optimization responses
    algorithm_responses = {
        "deepseek": """## Algorithm Optimization: O(n²) to O(n log n)

Your sorting algorithm can be dramatically improved.

### Current Issue:
```python
# Bubble sort - O(n²)
for i in range(n):
    for j in range(n-i-1):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
```

### Optimized Solution:
```python
# Merge sort - O(n log n)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Performance Analysis:
- Time: O(n log n) in all cases
- Space: O(n) for temporary arrays
- Stable: Maintains relative order
- Parallelizable: Can use multiple cores

For in-place sorting with O(1) space, consider heapsort.""",

        "gemini": """## Algorithm Analysis: Sorting Optimization

The O(n²) complexity is problematic for large datasets. Let's explore options.

### Quick Analysis:
- Current: ~1,000,000 operations for n=1,000
- Optimized: ~10,000 operations for n=1,000

### Best Options by Use Case:

**General Purpose - Quicksort**:
```python
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

**Nearly Sorted Data - Timsort** (Python's default):
```python
# Just use Python's built-in
sorted_arr = sorted(arr)  # Timsort O(n log n)
```

### Recommendations:
1. For most cases: Use language built-ins
2. For learning: Implement quicksort
3. For stability: Use mergesort
4. For space constraints: Use heapsort""",

        "openai": """## Sorting Algorithm Optimization

Moving from O(n²) to O(n log n) provides massive performance gains.

### Complexity Comparison:
| n       | O(n²)      | O(n log n) | Improvement |
|---------|------------|------------|-------------|
| 100     | 10,000     | 664        | 15x         |
| 1,000   | 1,000,000  | 9,965      | 100x        |
| 10,000  | 100,000,000| 132,877    | 752x        |

### Modern Approach - Introsort:
```python
import heapq

def introsort(arr):
    max_depth = 2 * math.floor(math.log2(len(arr)))
    return introsort_helper(arr, 0, len(arr), max_depth)

def introsort_helper(arr, start, end, max_depth):
    if end - start <= 16:
        # Insertion sort for small arrays
        insertion_sort(arr, start, end)
    elif max_depth == 0:
        # Heapsort to guarantee O(n log n)
        heapsort(arr, start, end)
    else:
        # Quicksort with depth limit
        pivot = partition(arr, start, end)
        introsort_helper(arr, start, pivot, max_depth - 1)
        introsort_helper(arr, pivot + 1, end, max_depth - 1)
```

### Practical Advice:
- Profile first - is sorting really the bottleneck?
- Consider data characteristics (nearly sorted? many duplicates?)
- For production: use standard library implementations
- Benchmark with realistic data sizes"""
    }
    
    # Initialize synthesizer
    synthesizer = ResponseSynthesizer()
    
    print("=" * 80)
    print("RESPONSE SYNTHESIS DEMO")
    print("=" * 80)
    
    # Demo 1: Security with Consensus
    print("\n1. SECURITY PATTERN - CONSENSUS STRATEGY")
    print("-" * 40)
    result = synthesizer.synthesize(
        security_responses,
        SynthesisStrategy.CONSENSUS,
        {"pattern_category": PatternCategory.SECURITY}
    )
    print(synthesizer.format_response(result, "markdown"))
    print(f"\nConfidence: {result.confidence_score:.2%}")
    print(f"Processing Time: {result.synthesis_time:.3f}s")
    
    # Demo 2: Architecture with Debate
    print("\n\n2. ARCHITECTURE PATTERN - DEBATE STRATEGY")
    print("-" * 40)
    result = synthesizer.synthesize(
        architecture_responses,
        SynthesisStrategy.DEBATE,
        {"pattern_category": PatternCategory.ARCHITECTURE}
    )
    print(synthesizer.format_response(result, "markdown"))
    print(f"\nConfidence: {result.confidence_score:.2%}")
    print(f"Processing Time: {result.synthesis_time:.3f}s")
    
    # Demo 3: Algorithm with Expert Weighting
    print("\n\n3. ALGORITHM PATTERN - EXPERT WEIGHTED STRATEGY")
    print("-" * 40)
    result = synthesizer.synthesize(
        algorithm_responses,
        SynthesisStrategy.EXPERT_WEIGHTED,
        {"pattern_category": PatternCategory.ALGORITHM}
    )
    print(synthesizer.format_response(result, "markdown"))
    print(f"\nConfidence: {result.confidence_score:.2%}")
    print(f"Processing Time: {result.synthesis_time:.3f}s")
    
    # Demo 4: Show JSON output
    print("\n\n4. JSON OUTPUT FORMAT")
    print("-" * 40)
    result = synthesizer.synthesize(
        {"ai1": "Quick response", "ai2": "Another quick response"},
        SynthesisStrategy.SUMMARY,
        {}
    )
    print(synthesizer.format_response(result, "json"))


if __name__ == "__main__":
    demo_synthesis()
#!/usr/bin/env python3
"""
AI Consultation Manager Demo
Shows intelligent AI selection and consultation management
"""

# Example 1: Critical Security Pattern - Multi-AI Consensus Required
def authenticate_user(username, password):
    """Critical security issue - will trigger multi-AI consensus consultation"""
    # Multiple security red flags here!
    if password == "admin123":  # Hardcoded password - CRITICAL
        api_key = "sk-prod-key-12345"  # Exposed API key - CRITICAL
        
        # Weak hashing algorithm
        import md5  # Deprecated and insecure
        password_hash = md5.new(password).hexdigest()
        
        return {"token": api_key, "hash": password_hash}


# Example 2: Complex Algorithm - Algorithm Experts Needed
def find_optimal_path(graph, start, end):
    """Complex algorithm requiring optimization - will select algorithm-focused AIs"""
    # O(n³) algorithm - needs optimization
    paths = []
    
    # Brute force approach - TODO: optimize this
    for i in range(len(graph)):
        for j in range(len(graph)):
            for k in range(len(graph)):
                # Inefficient nested loops
                if graph[i][j] and graph[j][k]:
                    paths.append((i, j, k))
    
    # Should use Dijkstra's or A* algorithm instead
    return paths


# Example 3: Architecture Decision - Debate Mode
class SystemArchitecture:
    """Architecture pattern requiring multiple perspectives"""
    def __init__(self):
        # Should I use microservices or monolithic architecture?
        # Need to consider scalability, complexity, team size
        self.architecture_type = "undecided"
    
    def design_system(self, requirements):
        """
        Design considerations:
        - High traffic expected (1M+ requests/day)
        - Small team (3 developers)
        - Rapid iteration needed
        - Multiple integrations required
        
        What's the best approach?
        """
        pass


# Example 4: Simple Uncertainty - Single AI Sufficient
def calculate_average(numbers):
    """Simple clarification needed - single AI consultation"""
    # TODO: Should I handle empty lists?
    # Not sure about edge cases
    return sum(numbers) / len(numbers)


# Example 5: Multiple Pattern Types - Smart AI Selection
def process_payment(card_number, amount):
    """Mixed patterns - AI Consultation Manager will intelligently select AIs"""
    # Security concern: storing card data
    stored_card = card_number  # Should be tokenized
    
    # Algorithm concern: O(n²) validation
    for digit in card_number:
        for other in card_number:
            # Inefficient validation
            pass
    
    # Gotcha: float arithmetic
    tax = amount * 0.1  # Floating point precision issues
    total = amount + tax
    
    # Architecture question: sync or async?
    # Should this be processed asynchronously?
    return {"total": total, "card": stored_card}


if __name__ == "__main__":
    print("AI Consultation Manager Demo")
    print("=" * 60)
    print("\nThis file demonstrates how the AI Consultation Manager:")
    print("1. Selects appropriate AIs based on pattern types")
    print("2. Uses multi-AI consensus for critical security issues")
    print("3. Chooses algorithm experts for optimization problems")
    print("4. Enables debate mode for architecture decisions")
    print("5. Optimizes for single AI when sufficient")
    print("\nThe manager considers:")
    print("- Pattern severity and type")
    print("- AI expertise profiles")
    print("- Cost vs. accuracy trade-offs")
    print("- Required consultation modes")
    print("\nRun pattern detection on this file to see intelligent AI selection!")
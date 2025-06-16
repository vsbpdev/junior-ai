#!/usr/bin/env python3
"""
Pattern Detection Engine for Junior AI Assistant
Detects patterns in text that indicate need for AI consultation
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class PatternCategory(Enum):
    """Pattern categories for detection"""
    SECURITY = "security"
    UNCERTAINTY = "uncertainty"
    ALGORITHM = "algorithm"
    GOTCHA = "gotcha"
    ARCHITECTURE = "architecture"


class PatternSeverity(Enum):
    """Severity levels for detected patterns"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PatternMatch:
    """Represents a detected pattern match"""
    category: PatternCategory
    severity: PatternSeverity
    keyword: str
    context: str
    start_pos: int
    end_pos: int
    confidence: float
    requires_multi_ai: bool


@dataclass
class PatternDefinition:
    """Defines a pattern category with keywords and rules"""
    category: PatternCategory
    keywords: List[str]
    regex_patterns: List[str]
    severity: PatternSeverity
    requires_multi_ai: bool
    description: str


class PatternDetectionEngine:
    """Real-time pattern detection engine for text analysis"""
    
    def __init__(self):
        self.pattern_definitions = self._initialize_patterns()
        self.compiled_patterns = self._compile_patterns()
        self.context_window = 50  # Characters to capture around match
        
    def _initialize_patterns(self) -> Dict[PatternCategory, PatternDefinition]:
        """Initialize pattern definitions for each category"""
        patterns = {
            PatternCategory.SECURITY: PatternDefinition(
                category=PatternCategory.SECURITY,
                keywords=[
                    "password", "auth", "token", "encrypt", "decrypt", "jwt", 
                    "hash", "api_key", "secret", "certificate", "ssl", "tls",
                    "oauth", "login", "permission", "credential", "private_key",
                    "public_key", "signature", "verify", "authenticate", "authorization",
                    "access_token", "refresh_token", "session", "cookie", "cors",
                    "xss", "csrf", "sql_injection", "injection", "vulnerability"
                ],
                regex_patterns=[
                    r'\bapi[_\s]?key\b',
                    r'\bprivate[_\s]?key\b',
                    r'\baccess[_\s]?token\b',
                    r'\bsecret[_\s]?key\b',
                    r'\bauth(?:entication|orization)?\b',
                    r'\bcredential[s]?\b',
                    r'\bpassword[s]?\b',
                    r'\bencrypt(?:ion|ed)?\b',
                    r'\bdecrypt(?:ion|ed)?\b'
                ],
                severity=PatternSeverity.CRITICAL,
                requires_multi_ai=True,
                description="Security-related patterns requiring comprehensive review"
            ),
            
            PatternCategory.UNCERTAINTY: PatternDefinition(
                category=PatternCategory.UNCERTAINTY,
                keywords=[
                    "todo", "fixme", "not sure", "might be", "complex", "help",
                    "unsure", "maybe", "possibly", "could be", "i think", "probably",
                    "uncertain", "confused", "clarify", "question", "unclear",
                    "confusing", "doubt", "wondering", "not certain", "perhaps"
                ],
                regex_patterns=[
                    r'\bTODO\b',
                    r'\bFIXME\b',
                    r'\bXXX\b',
                    r'\bnot\s+sure\b',
                    r'\bmight\s+be\b',
                    r'\bcould\s+be\b',
                    r'\bi\s+think\b',
                    r'\bprobably\b',
                    r'\bmaybe\b',
                    r'\?\?\?+',
                    r'\bhelp\s+needed\b'
                ],
                severity=PatternSeverity.MEDIUM,
                requires_multi_ai=False,
                description="Uncertainty patterns indicating need for clarification"
            ),
            
            PatternCategory.ALGORITHM: PatternDefinition(
                category=PatternCategory.ALGORITHM,
                keywords=[
                    "sort", "search", "optimize", "performance", "algorithm",
                    "efficient", "complexity", "recursive", "dynamic programming",
                    "binary search", "hash table", "tree", "graph", "bfs", "dfs",
                    "dijkstra", "a*", "greedy", "backtracking", "memoization",
                    "time complexity", "space complexity", "big o", "optimization"
                ],
                regex_patterns=[
                    r'\bO\([n^2logn]+\)',
                    r'\balgorithm[s]?\b',
                    r'\boptimiz(?:e|ation)\b',
                    r'\bperformance\b',
                    r'\befficient(?:ly)?\b',
                    r'\bcomplexity\b',
                    r'\brecurs(?:ive|ion)\b',
                    r'\bdynamic\s+programming\b',
                    r'\b(?:time|space)\s+complexity\b'
                ],
                severity=PatternSeverity.HIGH,
                requires_multi_ai=True,
                description="Algorithm-related patterns requiring optimization analysis"
            ),
            
            PatternCategory.GOTCHA: PatternDefinition(
                category=PatternCategory.GOTCHA,
                keywords=[
                    "regex", "timezone", "date", "datetime", "float", "encoding",
                    "unicode", "async", "promise", "callback", "race condition",
                    "null", "undefined", "nan", "infinity", "precision", "rounding",
                    "memory leak", "circular reference", "closure", "hoisting",
                    "type coercion", "falsy", "truthy", "edge case", "corner case"
                ],
                regex_patterns=[
                    r'\bregex(?:p)?\b',
                    r'\btimezone[s]?\b',
                    r'\bdate(?:time)?\b',
                    r'\bfloat(?:ing)?\s*point\b',
                    r'\bencoding\b',
                    r'\bunicode\b',
                    r'\basync(?:hronous)?\b',
                    r'\bpromise[s]?\b',
                    r'\bcallback[s]?\b',
                    r'\brace\s+condition[s]?\b',
                    r'\bnull\b|\bundefined\b',
                    r'\bNaN\b',
                    r'\bmemory\s+leak[s]?\b'
                ],
                severity=PatternSeverity.HIGH,
                requires_multi_ai=False,
                description="Common programming gotchas requiring careful handling"
            ),
            
            PatternCategory.ARCHITECTURE: PatternDefinition(
                category=PatternCategory.ARCHITECTURE,
                keywords=[
                    "design pattern", "architecture", "should i", "best practice",
                    "approach", "structure", "organize", "pattern", "solid",
                    "mvc", "mvvm", "microservice", "monolith", "scalability",
                    "modularity", "coupling", "cohesion", "abstraction", "interface",
                    "dependency", "composition", "inheritance", "polymorphism"
                ],
                regex_patterns=[
                    r'\bdesign\s+pattern[s]?\b',
                    r'\barchitecture\b',
                    r'\bshould\s+i\b',
                    r'\bbest\s+practice[s]?\b',
                    r'\bapproach(?:es)?\b',
                    r'\bstructure\b',
                    r'\borganiz(?:e|ation)\b',
                    r'\bpattern[s]?\b',
                    r'\bSOLID\b',
                    r'\b(?:micro)?service[s]?\b',
                    r'\bscal(?:able|ability)\b'
                ],
                severity=PatternSeverity.HIGH,
                requires_multi_ai=True,
                description="Architecture patterns requiring design recommendations"
            )
        }
        return patterns
    
    def _compile_patterns(self) -> Dict[PatternCategory, List[re.Pattern]]:
        """Compile regex patterns for efficient matching"""
        compiled = {}
        for category, definition in self.pattern_definitions.items():
            patterns = []
            
            # Compile keyword patterns (case-insensitive)
            for keyword in definition.keywords:
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                patterns.append(pattern)
            
            # Compile custom regex patterns
            for regex_pattern in definition.regex_patterns:
                pattern = re.compile(regex_pattern, re.IGNORECASE)
                patterns.append(pattern)
            
            compiled[category] = patterns
        
        return compiled
    
    def detect_patterns(self, text: str) -> List[PatternMatch]:
        """Detect all patterns in the given text"""
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            definition = self.pattern_definitions[category]
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context around the match
                    context_start = max(0, start_pos - self.context_window)
                    context_end = min(len(text), end_pos + self.context_window)
                    context = text[context_start:context_end]
                    
                    # Calculate confidence based on exact match vs fuzzy match
                    confidence = 1.0 if match.group(0).lower() in definition.keywords else 0.8
                    
                    pattern_match = PatternMatch(
                        category=category,
                        severity=definition.severity,
                        keyword=match.group(0),
                        context=context,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        requires_multi_ai=definition.requires_multi_ai
                    )
                    matches.append(pattern_match)
        
        # Remove duplicate matches (same position, different patterns)
        matches = self._deduplicate_matches(matches)
        
        # Sort by severity (highest first) and position
        matches.sort(key=lambda m: (-m.severity.value, m.start_pos))
        
        return matches
    
    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove duplicate matches at the same position"""
        position_map = {}
        
        for match in matches:
            position_key = (match.start_pos, match.end_pos)
            if position_key not in position_map or match.severity.value > position_map[position_key].severity.value:
                position_map[position_key] = match
        
        return list(position_map.values())
    
    def get_pattern_summary(self, matches: List[PatternMatch]) -> Dict[str, any]:
        """Generate a summary of detected patterns"""
        summary = {
            "total_matches": len(matches),
            "categories": {},
            "requires_multi_ai": False,
            "max_severity": PatternSeverity.LOW if matches else None
        }
        
        for match in matches:
            category_name = match.category.value
            if category_name not in summary["categories"]:
                summary["categories"][category_name] = {
                    "count": 0,
                    "keywords": set(),
                    "severity": match.severity.value,
                    "requires_multi_ai": match.requires_multi_ai
                }
            
            summary["categories"][category_name]["count"] += 1
            summary["categories"][category_name]["keywords"].add(match.keyword)
            
            if match.requires_multi_ai:
                summary["requires_multi_ai"] = True
            
            if match.severity.value > (summary["max_severity"].value if summary["max_severity"] else 0):
                summary["max_severity"] = match.severity
        
        # Convert sets to lists for JSON serialization
        for category in summary["categories"].values():
            category["keywords"] = list(category["keywords"])
        
        return summary
    
    def should_trigger_consultation(self, matches: List[PatternMatch], threshold: PatternSeverity = PatternSeverity.MEDIUM) -> bool:
        """Determine if pattern matches warrant AI consultation"""
        if not matches:
            return False
        
        # Check if any match exceeds threshold
        for match in matches:
            if match.severity.value >= threshold.value:
                return True
        
        # Check if multiple lower-severity matches combined warrant consultation
        return len(matches) >= 3
    
    def get_consultation_strategy(self, matches: List[PatternMatch]) -> Dict[str, any]:
        """Determine the consultation strategy based on detected patterns"""
        if not matches:
            return {"strategy": "none", "reason": "No patterns detected"}
        
        summary = self.get_pattern_summary(matches)
        
        # Determine primary category (most severe or most frequent)
        primary_category = None
        max_severity = 0
        max_count = 0
        
        for category_name, data in summary["categories"].items():
            severity = data["severity"]
            count = data["count"]
            
            if severity > max_severity or (severity == max_severity and count > max_count):
                primary_category = category_name
                max_severity = severity
                max_count = count
        
        # Build consultation strategy
        strategy = {
            "strategy": "multi_ai" if summary["requires_multi_ai"] else "single_ai",
            "primary_category": primary_category,
            "severity": summary["max_severity"].name,
            "categories": list(summary["categories"].keys()),
            "reason": self._generate_consultation_reason(summary),
            "recommended_ais": self._recommend_ais(summary)
        }
        
        return strategy
    
    def _generate_consultation_reason(self, summary: Dict[str, any]) -> str:
        """Generate a human-readable reason for consultation"""
        categories = summary["categories"]
        category_names = list(categories.keys())
        
        if len(category_names) == 1:
            category = category_names[0]
            count = categories[category]["count"]
            return f"Detected {count} {category} pattern{'s' if count > 1 else ''}"
        else:
            return f"Detected patterns in multiple categories: {', '.join(category_names)}"
    
    def _recommend_ais(self, summary: Dict[str, any]) -> List[str]:
        """Recommend which AIs to consult based on pattern summary"""
        if summary["requires_multi_ai"]:
            # For critical patterns, use multiple AIs
            return ["gemini", "grok", "openai"]
        
        # For single AI consultation, choose based on category
        primary_categories = list(summary["categories"].keys())
        if PatternCategory.SECURITY.value in primary_categories:
            return ["openai", "gemini"]  # Good for security analysis
        if PatternCategory.ALGORITHM.value in primary_categories:
            return ["deepseek", "gemini"]  # Good for algorithms
        if PatternCategory.ARCHITECTURE.value in primary_categories:
            return ["gemini", "grok"]  # Good for architecture
        
        return ["openrouter"]  # Default junior AI


if __name__ == "__main__":
    # Test the pattern detection engine
    engine = PatternDetectionEngine()
    
    test_texts = [
        "I need to implement password hashing for the login system. Not sure which algorithm to use.",
        "TODO: Optimize this sorting algorithm - it has O(n^2) complexity",
        "Should I use microservices architecture or keep it as a monolith?",
        "Working with datetime and timezone conversion - this might be tricky",
        "The API key should be encrypted before storing in the database"
    ]
    
    for text in test_texts:
        print(f"\nAnalyzing: {text}")
        matches = engine.detect_patterns(text)
        
        if matches:
            print(f"Found {len(matches)} pattern(s):")
            for match in matches:
                print(f"  - {match.category.value}: '{match.keyword}' (severity: {match.severity.name})")
            
            strategy = engine.get_consultation_strategy(matches)
            print(f"Consultation strategy: {strategy['strategy']}")
            print(f"Reason: {strategy['reason']}")
            print(f"Recommended AIs: {', '.join(strategy['recommended_ais'])}")
        else:
            print("No patterns detected")
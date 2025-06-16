#!/usr/bin/env python3
"""
Enhanced Pattern Detection Engine for Junior AI Assistant
Improved context extraction and pattern matching
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


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
    line_number: Optional[int] = None
    full_line: Optional[str] = None


@dataclass
class PatternDefinition:
    """Defines a pattern category with keywords and rules"""
    category: PatternCategory
    keywords: List[str]
    regex_patterns: List[str]
    severity: PatternSeverity
    requires_multi_ai: bool
    description: str


class EnhancedPatternDetectionEngine:
    """Enhanced pattern detection engine with better context extraction"""
    
    def __init__(self, context_window_size: int = 150):
        self.pattern_definitions = self._initialize_patterns()
        self.compiled_patterns = self._compile_patterns()
        self.context_window_size = context_window_size
        
    def _initialize_patterns(self) -> Dict[PatternCategory, PatternDefinition]:
        """Initialize enhanced pattern definitions for each category"""
        patterns = {
            PatternCategory.SECURITY: PatternDefinition(
                category=PatternCategory.SECURITY,
                keywords=[
                    "password", "auth", "token", "encrypt", "decrypt", "jwt", 
                    "hash", "api_key", "secret", "certificate", "ssl", "tls",
                    "oauth", "login", "permission", "credential", "private_key",
                    "public_key", "signature", "verify", "authenticate", "authorization",
                    "access_token", "refresh_token", "session", "cookie", "cors",
                    "xss", "csrf", "sql_injection", "injection", "vulnerability",
                    # New additions
                    "csp", "content-security-policy", "sanitize", "escape",
                    "2fa", "mfa", "totp", "saml", "ldap", "rbac", "acl"
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
                    r'\bdecrypt(?:ion|ed)?\b',
                    # New patterns
                    r'\b[A-Z0-9]{20,}\b',  # Potential API keys
                    r'Bearer\s+[A-Za-z0-9\-_]+',  # Bearer tokens
                    r'\bsha\d{3}\b',  # SHA algorithms
                    r'\bbcrypt\b',
                    r'\bargon2\b'
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
                    "confusing", "doubt", "wondering", "not certain", "perhaps",
                    # New additions
                    "hack", "workaround", "temporary", "revisit", "review",
                    "check this", "verify", "confirm", "investigate"
                ],
                regex_patterns=[
                    r'\bTODO\b',
                    r'\bFIXME\b',
                    r'\bXXX\b',
                    r'\bHACK\b',
                    r'\bNOTE\b',
                    r'\bWARNING\b',
                    r'\bnot\s+sure\b',
                    r'\bmight\s+be\b',
                    r'\bcould\s+be\b',
                    r'\bi\s+think\b',
                    r'\bprobably\b',
                    r'\bmaybe\b',
                    r'\?\?\?+',
                    r'\bhelp\s+needed\b',
                    # New patterns
                    r'//\s*[Tt]emp',
                    r'#\s*[Tt]emp',
                    r'\bneeds?\s+review\b'
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
                    "time complexity", "space complexity", "big o", "optimization",
                    # New additions
                    "cache", "index", "query optimization", "n+1", "lazy loading",
                    "eager loading", "batch processing", "parallel", "concurrent",
                    "heap", "trie", "segment tree", "fenwick tree"
                ],
                regex_patterns=[
                    r'\bO\([n^2logn\s\*\+]+\)',
                    r'\balgorithm[s]?\b',
                    r'\boptimiz(?:e|ation)\b',
                    r'\bperformance\b',
                    r'\befficient(?:ly)?\b',
                    r'\bcomplexity\b',
                    r'\brecurs(?:ive|ion)\b',
                    r'\bdynamic\s+programming\b',
                    r'\b(?:time|space)\s+complexity\b',
                    # New patterns
                    r'\bbig-?o\b',
                    r'\bn\s*log\s*n\b',
                    r'\bquadratic\b',
                    r'\blinear\b',
                    r'\blogarithmic\b',
                    r'\bcache\s+miss\b',
                    r'\bbottleneck\b'
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
                    "type coercion", "falsy", "truthy", "edge case", "corner case",
                    # New additions
                    "mutation", "immutable", "side effect", "pure function",
                    "event loop", "microtask", "macrotask", "prototype pollution",
                    "integer overflow", "buffer overflow", "off by one",
                    "locale", "i18n", "character encoding", "endianness"
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
                    r'\bmemory\s+leak[s]?\b',
                    # New patterns
                    r'\b0\.\d+\s*[=!]=\s*0\.\d+\b',  # Float comparison
                    r'==\s*(?:null|undefined)',  # Loose equality with null/undefined
                    r'\butf-?8\b',
                    r'\bsetTimeout\b.*\b0\b',  # setTimeout with 0
                    r'typeof\s+\w+\s*===?\s*["\']object["\']'  # typeof checks
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
                    "dependency", "composition", "inheritance", "polymorphism",
                    # New additions
                    "event driven", "domain driven", "ddd", "cqrs", "event sourcing",
                    "repository pattern", "factory", "singleton", "observer",
                    "strategy pattern", "adapter", "facade", "proxy", "decorator",
                    "clean architecture", "hexagonal", "onion architecture",
                    "rest", "graphql", "grpc", "message queue", "pub sub"
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
                    r'\bscal(?:able|ability)\b',
                    # New patterns
                    r'\bDRY\b',
                    r'\bYAGNI\b',
                    r'\bKISS\b',
                    r'\bAPI\s+design\b',
                    r'\blayer(?:ed|ing)\b',
                    r'\bmodular(?:ity)?\b',
                    r'\bseparation\s+of\s+concerns\b'
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
                pattern = re.compile(regex_pattern, re.IGNORECASE | re.MULTILINE)
                patterns.append(pattern)
            
            compiled[category] = patterns
        
        return compiled
    
    def detect_patterns(self, text: str) -> List[PatternMatch]:
        """Detect all patterns in the given text with enhanced context"""
        matches = []
        lines = text.split('\n')
        
        for category, patterns in self.compiled_patterns.items():
            definition = self.pattern_definitions[category]
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Find line number
                    line_start = text.rfind('\n', 0, start_pos) + 1
                    line_end = text.find('\n', end_pos)
                    if line_end == -1:
                        line_end = len(text)
                    
                    line_number = text[:start_pos].count('\n') + 1
                    full_line = text[line_start:line_end].strip()
                    
                    # Extract enhanced context
                    context = self._extract_enhanced_context(text, start_pos, end_pos, lines)
                    
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
                        requires_multi_ai=definition.requires_multi_ai,
                        line_number=line_number,
                        full_line=full_line
                    )
                    matches.append(pattern_match)
        
        # Remove duplicate matches
        matches = self._deduplicate_matches(matches)
        
        # Sort by severity (highest first) and position
        matches.sort(key=lambda m: (-m.severity.value, m.start_pos))
        
        return matches
    
    def _extract_enhanced_context(self, text: str, start_pos: int, end_pos: int, lines: List[str]) -> str:
        """Extract enhanced context including complete sentences or code blocks"""
        # Try to find sentence boundaries
        sentence_start = start_pos
        sentence_end = end_pos
        
        # Look backwards for sentence start
        for i in range(start_pos - 1, max(0, start_pos - self.context_window_size), -1):
            if text[i] in '.!?\n' and (i + 1 < len(text) and text[i + 1] == ' '):
                sentence_start = i + 2
                break
            elif text[i] == '\n' and i > 0 and text[i - 1] == '\n':
                # Paragraph break
                sentence_start = i + 1
                break
        else:
            sentence_start = max(0, start_pos - self.context_window_size // 2)
        
        # Look forward for sentence end
        for i in range(end_pos, min(len(text), end_pos + self.context_window_size)):
            if text[i] in '.!?\n':
                sentence_end = i + 1
                break
        else:
            sentence_end = min(len(text), end_pos + self.context_window_size // 2)
        
        # For code contexts, try to include complete code blocks
        context = text[sentence_start:sentence_end].strip()
        
        # If context is code, try to balance brackets
        if any(char in context for char in '{}()[]'):
            context = self._balance_code_context(text, start_pos, end_pos)
        
        return context
    
    def _balance_code_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Try to include balanced brackets in code context"""
        # Simple approach: expand to include balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        closing = {v: k for k, v in brackets.items()}
        
        context_start = max(0, start_pos - self.context_window_size // 2)
        context_end = min(len(text), end_pos + self.context_window_size // 2)
        
        # Expand to include complete lines
        while context_start > 0 and text[context_start - 1] != '\n':
            context_start -= 1
        while context_end < len(text) and text[context_end] != '\n':
            context_end += 1
        
        return text[context_start:context_end].strip()
    
    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove duplicate matches with enhanced logic"""
        seen_positions = {}
        unique_matches = []
        
        for match in matches:
            # Create a key that includes line number for better deduplication
            position_key = (match.start_pos, match.end_pos, match.line_number)
            
            if position_key not in seen_positions:
                seen_positions[position_key] = match
                unique_matches.append(match)
            else:
                # Keep the match with higher severity or confidence
                existing = seen_positions[position_key]
                if (match.severity.value > existing.severity.value or 
                    (match.severity.value == existing.severity.value and 
                     match.confidence > existing.confidence)):
                    # Replace with higher priority match
                    idx = unique_matches.index(existing)
                    unique_matches[idx] = match
                    seen_positions[position_key] = match
        
        return unique_matches
    
    def get_pattern_summary(self, matches: List[PatternMatch]) -> Dict[str, Any]:
        """Generate an enhanced summary of detected patterns"""
        summary = {
            "total_matches": len(matches),
            "categories": {},
            "requires_multi_ai": False,
            "max_severity": PatternSeverity.LOW if matches else None,
            "code_lines_affected": set(),
            "context_preview": ""
        }
        
        for match in matches:
            category_name = match.category.value
            if category_name not in summary["categories"]:
                summary["categories"][category_name] = {
                    "count": 0,
                    "keywords": set(),
                    "severity": match.severity.value,
                    "requires_multi_ai": match.requires_multi_ai,
                    "line_numbers": set()
                }
            
            summary["categories"][category_name]["count"] += 1
            summary["categories"][category_name]["keywords"].add(match.keyword)
            if match.line_number:
                summary["categories"][category_name]["line_numbers"].add(match.line_number)
                summary["code_lines_affected"].add(match.line_number)
            
            if match.requires_multi_ai:
                summary["requires_multi_ai"] = True
            
            if match.severity.value > (summary["max_severity"].value if summary["max_severity"] else 0):
                summary["max_severity"] = match.severity
                summary["context_preview"] = match.context[:200] + "..." if len(match.context) > 200 else match.context
        
        # Convert sets to lists for JSON serialization
        for category in summary["categories"].values():
            category["keywords"] = list(category["keywords"])
            category["line_numbers"] = sorted(list(category["line_numbers"]))
        
        summary["code_lines_affected"] = sorted(list(summary["code_lines_affected"]))
        
        return summary
    
    def should_trigger_consultation(self, matches: List[PatternMatch], threshold: PatternSeverity = PatternSeverity.MEDIUM) -> bool:
        """Enhanced logic to determine if pattern matches warrant AI consultation"""
        if not matches:
            return False
        
        # Check if any match exceeds threshold
        for match in matches:
            if match.severity.value >= threshold.value:
                return True
        
        # Check if multiple lower-severity matches combined warrant consultation
        if len(matches) >= 3:
            return True
        
        # Check if multiple categories are involved (complex scenario)
        categories = set(match.category for match in matches)
        return len(categories) >= 2
    
    def get_consultation_strategy(self, matches: List[PatternMatch]) -> Dict[str, Any]:
        """Enhanced consultation strategy based on detected patterns"""
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
            
            # Weight severity more heavily than count
            score = severity * 10 + count
            if score > max_severity * 10 + max_count:
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
            "recommended_ais": self._recommend_ais(summary),
            "focus_areas": self._determine_focus_areas(summary),
            "lines_to_review": summary["code_lines_affected"][:10]  # Top 10 lines
        }
        
        return strategy
    
    def _generate_consultation_reason(self, summary: Dict[str, Any]) -> str:
        """Generate a more detailed human-readable reason for consultation"""
        categories = summary["categories"]
        category_names = list(categories.keys())
        
        if len(category_names) == 1:
            category = category_names[0]
            count = categories[category]["count"]
            lines = len(categories[category]["line_numbers"])
            return f"Detected {count} {category} pattern{'s' if count > 1 else ''} across {lines} line{'s' if lines > 1 else ''}"
        else:
            total_patterns = sum(cat["count"] for cat in categories.values())
            return f"Detected {total_patterns} patterns across {len(category_names)} categories: {', '.join(category_names)}"
    
    def _recommend_ais(self, summary: Dict[str, any]) -> List[str]:
        """Enhanced AI recommendation based on pattern summary"""
        recommendations = []
        categories = list(summary["categories"].keys())
        
        if summary["requires_multi_ai"]:
            # For critical patterns, use multiple AIs
            if PatternCategory.SECURITY.value in categories:
                recommendations = ["openai", "gemini", "grok"]  # Best for security
            elif PatternCategory.ALGORITHM.value in categories:
                recommendations = ["deepseek", "gemini", "openai"]  # Best for algorithms
            elif PatternCategory.ARCHITECTURE.value in categories:
                recommendations = ["gemini", "grok", "openai"]  # Best for architecture
            else:
                recommendations = ["gemini", "grok", "openai"]  # Default multi-AI
        else:
            # For single AI consultation, choose based on category
            if PatternCategory.UNCERTAINTY.value in categories:
                recommendations = ["openrouter"]  # Good for general clarification
            elif PatternCategory.GOTCHA.value in categories:
                recommendations = ["gemini"]  # Good for detailed explanations
            else:
                recommendations = ["openrouter"]  # Default
        
        return recommendations
    
    def _determine_focus_areas(self, summary: Dict[str, any]) -> List[str]:
        """Determine specific focus areas for AI consultation"""
        focus_areas = []
        
        for category_name, data in summary["categories"].items():
            if category_name == PatternCategory.SECURITY.value:
                focus_areas.append("Security vulnerability assessment")
                focus_areas.append("Authentication and encryption best practices")
            elif category_name == PatternCategory.ALGORITHM.value:
                focus_areas.append("Performance optimization strategies")
                focus_areas.append("Algorithm complexity analysis")
            elif category_name == PatternCategory.ARCHITECTURE.value:
                focus_areas.append("Design pattern recommendations")
                focus_areas.append("Scalability and maintainability")
            elif category_name == PatternCategory.Gotcha.value:
                focus_areas.append("Common pitfall prevention")
                focus_areas.append("Edge case handling")
            elif category_name == PatternCategory.UNCERTAINTY.value:
                focus_areas.append("Implementation clarification")
                focus_areas.append("Best practice guidance")
        
        return focus_areas[:4]  # Return top 4 focus areas


if __name__ == "__main__":
    # Test the enhanced pattern detection engine
    engine = EnhancedPatternDetectionEngine(context_window_size=200)
    
    test_code = """
def authenticate_user(username, password):
    # TODO: Add proper password hashing - this is not secure!
    if password == "admin123":  # FIXME: hardcoded password
        api_key = "sk-1234567890abcdef"  # This should be encrypted
        return {"token": api_key, "expires": datetime.now() + timedelta(hours=1)}
    
    # Not sure if this O(n^2) algorithm is efficient enough
    for user in users:
        for permission in user.permissions:
            if check_permission(permission):  # Might have race condition
                return True
    
    # Should I use Repository pattern or just direct DB access?
    return None
"""
    
    print("Enhanced Pattern Detection Test")
    print("=" * 60)
    
    matches = engine.detect_patterns(test_code)
    
    if matches:
        print(f"\nFound {len(matches)} patterns:\n")
        for match in matches:
            print(f"Line {match.line_number}: {match.category.value.upper()}")
            print(f"  Keyword: '{match.keyword}'")
            print(f"  Severity: {match.severity.name}")
            print(f"  Full line: {match.full_line}")
            print(f"  Context: {match.context[:100]}...")
            print()
        
        summary = engine.get_pattern_summary(matches)
        print("\nPattern Summary:")
        print(f"Total matches: {summary['total_matches']}")
        print(f"Lines affected: {summary['code_lines_affected']}")
        print(f"Max severity: {summary['max_severity'].name}")
        
        strategy = engine.get_consultation_strategy(matches)
        print(f"\nConsultation Strategy:")
        print(f"Strategy: {strategy['strategy']}")
        print(f"Reason: {strategy['reason']}")
        print(f"Recommended AIs: {', '.join(strategy['recommended_ais'])}")
        print(f"Focus areas: {', '.join(strategy['focus_areas'])}")
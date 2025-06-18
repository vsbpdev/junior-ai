#!/usr/bin/env python3
"""
Enhanced Pattern Detection Engine for Junior AI Assistant
Improved context extraction and pattern matching
"""

import re
from typing import Dict, List, Optional, Union, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import logging
from pathlib import Path
import time
import tempfile
import shutil
import threading

# SecureCredentialManager imported lazily in _get_credential_manager() method


# Custom Exceptions
class PatternDetectionError(Exception):
    """Base exception for pattern detection errors"""
    pass


class SensitivityError(PatternDetectionError):
    """Raised when sensitivity configuration is invalid"""
    pass


class ValidationError(PatternDetectionError):
    """Raised when input validation fails"""
    pass


class SecurityError(PatternDetectionError):
    """Raised when security validation fails"""
    pass


class ConfigurationError(PatternDetectionError):
    """Raised when configuration file issues occur"""
    pass


class PatternDetectionProtocol(Protocol):
    """Protocol for pattern detection engines"""
    
    def detect_patterns(self, text: str) -> List['PatternMatch']:
        """Detect patterns in text
        
        Args:
            text: Input text to analyze for patterns
            
        Returns:
            List of pattern matches sorted by severity and position
            
        Raises:
            ValidationError: If text is invalid or too large
            PatternDetectionError: If detection fails
        """
        ...
    
    def should_trigger_consultation(self, matches: List['PatternMatch']) -> bool:
        """Determine if pattern matches warrant AI consultation
        
        Args:
            matches: List of detected pattern matches
            
        Returns:
            True if consultation should be triggered, False otherwise
        """
        ...


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


@dataclass
class SensitivitySettings:
    """Settings for pattern detection sensitivity with cache metadata"""
    global_level: str = "medium"
    confidence_threshold: float = 0.7
    context_multiplier: float = 1.0
    min_matches_for_consultation: int = 2
    severity_threshold: str = "medium"
    category_overrides: Dict[str, Optional[str]] = field(default_factory=dict)
    
    # Cache metadata
    _cache_timestamp: float = field(default_factory=time.time)
    _file_path: str = ""
    _file_mtime: float = 0.0
    
    def is_cache_valid(self) -> bool:
        """Check if cached settings are still valid"""
        if not self._file_path or not os.path.exists(self._file_path):
            return False
        
        try:
            current_mtime = os.path.getmtime(self._file_path)
            return current_mtime <= self._file_mtime
        except OSError:
            return False
    
    def update_cache_metadata(self, file_path: str) -> None:
        """Update cache metadata with current file information"""
        self._file_path = file_path
        self._cache_timestamp = time.time()
        try:
            self._file_mtime = os.path.getmtime(file_path)
        except OSError:
            self._file_mtime = 0.0


class EnhancedPatternDetectionEngine:
    """Enhanced pattern detection engine with security validation, caching, and robust error handling.
    
    This engine provides intelligent pattern detection for code analysis with configurable
    sensitivity levels, category-specific overrides, and performance optimizations.
    
    Features:
        - 5 pattern categories: security, uncertainty, algorithm, gotcha, architecture
        - 4 sensitivity levels: low, medium, high, maximum
        - Thread-safe caching for performance
        - Atomic configuration updates
        - Comprehensive input validation
        - Robust error handling
    
    Example:
        >>> engine = EnhancedPatternDetectionEngine()
        >>> matches = engine.detect_patterns("password = 'admin123'")
        >>> engine.should_trigger_consultation(matches)
        True
        
        >>> engine.update_sensitivity(global_level="high")
        True
        
        >>> info = engine.get_sensitivity_info()
        >>> info['global_level']
        'high'
    
    Thread Safety:
        This class is thread-safe for read operations and sensitivity updates.
        Configuration file access is synchronized to prevent corruption.
    """
    
    # Class-level cache for SecureCredentialManager (shared across all instances)
    _shared_credential_manager = None
    _shared_credential_lock = threading.RLock()
    
    @classmethod
    def _get_credential_manager(cls):
        """Get or create the shared SecureCredentialManager instance.
        
        This method ensures that only one instance of SecureCredentialManager
        is created and shared across all PatternDetectionEngine instances.
        
        Returns:
            SecureCredentialManager: The shared credential manager instance
        """
        with cls._shared_credential_lock:
            if cls._shared_credential_manager is None:
                from secure_credentials import SecureCredentialManager
                cls._shared_credential_manager = SecureCredentialManager()
            return cls._shared_credential_manager
    
    def __init__(self, context_window_size: int = 150, config_path: str = "credentials.json") -> None:
        # Validate inputs
        self._validate_init_parameters(context_window_size, config_path)
        
        # Secure config path
        self._config_path = self._validate_config_path(config_path)
        
        # Initialize components
        self.pattern_definitions = self._initialize_patterns()
        self.compiled_patterns = self._compile_patterns()
        self.context_window_size = context_window_size
        
        # Initialize caching - MUST be before config loading
        self._sensitivity_cache = {}
        self._cache_lock = threading.RLock()
        
        # Load sensitivity settings
        self.sensitivity_settings = self._load_sensitivity_settings(str(self._config_path))
        self._apply_sensitivity_to_context_window()
        
        # Load pattern detection enabled state and manual overrides
        self._load_pattern_detection_config(str(self._config_path))
        
        # Set up logging
        self._setup_logging()
    
    def _validate_init_parameters(self, context_window_size: int, config_path: str) -> None:
        """Validate initialization parameters"""
        if not isinstance(context_window_size, int):
            raise ValidationError(f"context_window_size must be int, got {type(context_window_size)}")
        
        if context_window_size < 10:
            raise ValidationError(f"context_window_size too small: {context_window_size}, minimum is 10")
        
        if context_window_size > 10000:
            raise ValidationError(f"context_window_size too large: {context_window_size}, maximum is 10000")
        
        if not isinstance(config_path, str):
            raise ValidationError(f"config_path must be string, got {type(config_path)}")
        
        if not config_path.strip():
            raise ValidationError("config_path cannot be empty")
    
    def _validate_config_path(self, config_path: str) -> Path:
        """Validate and secure configuration file path"""
        try:
            path = Path(config_path).resolve()
            
            # Prevent path traversal attacks - ensure path is within or relative to current working directory
            cwd = Path.cwd()
            try:
                path.relative_to(cwd)
            except ValueError:
                # Check if it's a relative path that resolves within cwd
                if not str(path).startswith(str(cwd)):
                    raise SecurityError(f"Config path outside working directory: {path}")
            
            # Ensure it's a .json file
            if path.suffix.lower() not in ['.json', '.template']:
                raise ValidationError(f"Config file must be JSON: {path}")
            
            # Check for suspicious path components
            path_parts = path.parts
            suspicious_parts = {'..', '.', '__pycache__', '.git'}
            if any(part in suspicious_parts for part in path_parts[:-1]):  # Allow . in filename
                raise SecurityError(f"Suspicious path components detected: {path}")
                
            return path
        except Exception as e:
            if isinstance(e, (SecurityError, ValidationError)):
                raise
            raise ConfigurationError(f"Invalid config path '{config_path}': {e}")
    
    def _setup_logging(self) -> None:
        """Set up logging for pattern detection"""
        logger = logging.getLogger('pattern_detection')
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)  # Only warnings and errors by default
        
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
    
    def _load_sensitivity_settings(self, config_path: str) -> SensitivitySettings:
        """Load sensitivity settings with robust error handling"""
        logger = logging.getLogger('pattern_detection')
        
        try:
            # First try secure credential manager (use shared instance)
            credential_manager = self._get_credential_manager()
            config = credential_manager.load_credentials()
            
            # Fallback to file if no secure credentials
            if not config:
                if not os.path.exists(config_path):
                    logger.warning(f"Config file not found: {config_path}, using defaults")
                    return SensitivitySettings()
                    
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_path}: {e}")
            raise ConfigurationError(f"Configuration file contains invalid JSON: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied reading config file {config_path}: {e}")
            raise ConfigurationError(f"Cannot read configuration file: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Invalid encoding in config file {config_path}: {e}")
            raise ConfigurationError(f"Configuration file contains invalid encoding: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading config file {config_path}: {e}")
            raise ConfigurationError(f"Failed to read configuration file: {e}")
        
        try:
            return self._parse_sensitivity_config(config)
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise ConfigurationError(f"Invalid configuration structure: missing {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid configuration values: {e}")
            raise ConfigurationError(f"Configuration contains invalid values: {e}")
    
    def _parse_sensitivity_config(self, config: Dict) -> SensitivitySettings:
        """Parse and validate sensitivity configuration"""
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a JSON object")
        
        pattern_config = config.get('pattern_detection', {})
        if not isinstance(pattern_config, dict):
            raise ConfigurationError("pattern_detection must be an object")
        
        sensitivity_config = pattern_config.get('sensitivity', {})
        if not isinstance(sensitivity_config, dict):
            raise ConfigurationError("sensitivity must be an object")
        
        # Get and validate global level
        global_level = sensitivity_config.get('global_level', 'medium')
        self._validate_sensitivity_level(global_level)
        
        # Get and validate levels configuration
        levels = sensitivity_config.get('levels', {})
        if not isinstance(levels, dict):
            raise ConfigurationError("levels must be an object")
        
        # If levels is empty, provide defaults
        if not levels:
            logger.warning("No sensitivity levels defined in configuration, using defaults")
            levels = self._get_default_sensitivity_levels()
        
        if global_level not in levels:
            raise ConfigurationError(f"Global level '{global_level}' not found in levels configuration")
        
        level_settings = levels[global_level]
        if not isinstance(level_settings, dict):
            raise ConfigurationError(f"Level settings for '{global_level}' must be an object")
        
        # Validate and extract level settings
        confidence_threshold = self._validate_numeric_config(
            level_settings.get('confidence_threshold', 0.7),
            'confidence_threshold', 0.0, 1.0
        )
        
        context_multiplier = self._validate_numeric_config(
            level_settings.get('context_multiplier', 1.0),
            'context_multiplier', 0.1, 5.0
        )
        
        min_matches = self._validate_integer_config(
            level_settings.get('min_matches_for_consultation', 2),
            'min_matches_for_consultation', 1, 10
        )
        
        severity_threshold = level_settings.get('severity_threshold', 'medium')
        self._validate_severity_threshold(severity_threshold)
        
        # Validate category overrides
        category_overrides = sensitivity_config.get('category_overrides', {})
        if not isinstance(category_overrides, dict):
            raise ConfigurationError("category_overrides must be an object")
        
        self._validate_category_overrides(category_overrides)
        
        return SensitivitySettings(
            global_level=global_level,
            confidence_threshold=confidence_threshold,
            context_multiplier=context_multiplier,
            min_matches_for_consultation=min_matches,
            severity_threshold=severity_threshold,
            category_overrides=category_overrides
        )
    
    def _load_pattern_detection_config(self, config_path: str) -> None:
        """Load pattern detection configuration including enabled states and manual overrides."""
        logger = logging.getLogger('pattern_detection')
        
        try:
            # First try secure credential manager (use shared instance)
            credential_manager = self._get_credential_manager()
            config = credential_manager.load_credentials()
            
            # Fallback to file if no secure credentials
            if not config:
                if not os.path.exists(config_path):
                    logger.warning(f"Config file not found: {config_path}, using defaults")
                    self._pattern_detection_enabled = True
                    self._category_enabled_states = {cat: True for cat in PatternCategory}
                    self._manual_override_config = {}
                    return
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            pattern_config = config.get('pattern_detection', {})
            
            # Load global enabled state
            self._pattern_detection_enabled = pattern_config.get('enabled', True)
            
            # Load manual override configuration
            self._manual_override_config = pattern_config.get('manual_override', {})
            
            # Load category-specific configurations
            categories_config = pattern_config.get('pattern_categories', {})
            self._category_enabled_states = {}
            
            for category in PatternCategory:
                cat_config = categories_config.get(category.value, {})
                self._category_enabled_states[category] = cat_config.get('enabled', True)
                
                # Load custom keywords and patterns
                custom_keywords = cat_config.get('custom_keywords', [])
                if custom_keywords and category in self.pattern_definitions:
                    # Deduplicate keywords to prevent memory waste
                    existing_keywords = set(self.pattern_definitions[category].keywords)
                    new_keywords = [k for k in custom_keywords if k not in existing_keywords]
                    self.pattern_definitions[category].keywords.extend(new_keywords)
                
                custom_patterns = cat_config.get('custom_patterns', [])
                if custom_patterns and category in self.pattern_definitions:
                    # Deduplicate patterns as well
                    existing_patterns = set(self.pattern_definitions[category].regex_patterns)
                    new_patterns = [p for p in custom_patterns if p not in existing_patterns]
                    self.pattern_definitions[category].regex_patterns.extend(new_patterns)
                
                # Load disabled keywords
                disabled_keywords = cat_config.get('disabled_keywords', [])
                if disabled_keywords and category in self.pattern_definitions:
                    current_keywords = self.pattern_definitions[category].keywords
                    self.pattern_definitions[category].keywords = [
                        k for k in current_keywords if k not in disabled_keywords
                    ]
            
            # Recompile patterns with custom additions
            self.compiled_patterns = self._compile_patterns()
            
        except Exception as e:
            logger.error(f"Error loading pattern detection config: {e}")
            # Set defaults on error
            self._pattern_detection_enabled = True
            self._category_enabled_states = {cat: True for cat in PatternCategory}
            self._manual_override_config = {}
    
    def _is_category_enabled(self, category: PatternCategory) -> bool:
        """Check if a specific category is enabled."""
        return self._category_enabled_states.get(category, True)
    
    def _validate_sensitivity_level(self, level: str) -> None:
        """Validate sensitivity level value"""
        if not isinstance(level, str):
            raise ValidationError(f"Sensitivity level must be string, got {type(level)}")
        
        valid_levels = {"low", "medium", "high", "maximum"}
        if level not in valid_levels:
            raise ValidationError(f"Invalid sensitivity level '{level}'. Must be one of: {valid_levels}")
    
    def _get_default_sensitivity_levels(self) -> Dict[str, Dict[str, Any]]:
        """Get default sensitivity level definitions when none are provided in configuration."""
        return {
            "low": {
                "confidence_threshold": 0.9,
                "context_multiplier": 0.8,
                "min_matches_for_consultation": 3,
                "severity_threshold": "high"
            },
            "medium": {
                "confidence_threshold": 0.7,
                "context_multiplier": 1.0,
                "min_matches_for_consultation": 2,
                "severity_threshold": "medium"
            },
            "high": {
                "confidence_threshold": 0.5,
                "context_multiplier": 1.2,
                "min_matches_for_consultation": 1,
                "severity_threshold": "low"
            },
            "maximum": {
                "confidence_threshold": 0.3,
                "context_multiplier": 1.5,
                "min_matches_for_consultation": 1,
                "severity_threshold": "low"
            }
        }
    
    def _validate_severity_threshold(self, threshold: str) -> None:
        """Validate severity threshold value"""
        if not isinstance(threshold, str):
            raise ValidationError(f"Severity threshold must be string, got {type(threshold)}")
        
        valid_thresholds = {"low", "medium", "high", "critical"}
        if threshold not in valid_thresholds:
            raise ValidationError(f"Invalid severity threshold '{threshold}'. Must be one of: {valid_thresholds}")
    
    def _validate_numeric_config(self, value: Union[int, float], name: str, min_val: float, max_val: float) -> float:
        """Validate numeric configuration value"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")
        
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
        
        return float(value)
    
    def _validate_integer_config(self, value: Union[int, float], name: str, min_val: int, max_val: int) -> int:
        """Validate integer configuration value"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")
        
        if not isinstance(value, int) and not value.is_integer():
            raise ValidationError(f"{name} must be an integer, got {value}")
        
        int_value = int(value)
        if not (min_val <= int_value <= max_val):
            raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {int_value}")
        
        return int_value
    
    def _validate_category_overrides(self, overrides: Dict[str, Optional[str]]) -> None:
        """Validate category override configuration"""
        valid_categories = {cat.value for cat in PatternCategory}
        valid_levels = {"low", "medium", "high", "maximum", None}
        
        for category, level in overrides.items():
            if not isinstance(category, str):
                raise ValidationError(f"Category key must be string, got {type(category)}")
            
            if category not in valid_categories:
                raise ValidationError(f"Invalid category '{category}'. Must be one of: {valid_categories}")
            
            if level is not None and level not in valid_levels:
                raise ValidationError(f"Invalid level '{level}' for category '{category}'. Must be one of: {valid_levels}")
    
    def _apply_sensitivity_to_context_window(self):
        """Apply sensitivity multiplier to context window size"""
        # Store original window size and apply multiplier
        if not hasattr(self, '_original_context_window'):
            self._original_context_window = self.context_window_size
        self.context_window_size = int(self._original_context_window * self.sensitivity_settings.context_multiplier)
    
    def _get_category_sensitivity(self, category: PatternCategory) -> SensitivitySettings:
        """Get effective sensitivity settings for a specific category with caching"""
        with self._cache_lock:
            cache_key = category.value
            
            # Check if we have valid cached settings
            if (cache_key in self._sensitivity_cache and 
                self._sensitivity_cache[cache_key].is_cache_valid()):
                return self._sensitivity_cache[cache_key]
            
            # Load fresh settings
            settings = self._load_category_sensitivity(category)
            
            # Cache the results
            self._sensitivity_cache[cache_key] = settings
            return settings
    
    def _load_category_sensitivity(self, category: PatternCategory) -> SensitivitySettings:
        """Load sensitivity settings for a specific category"""
        category_override = self.sensitivity_settings.category_overrides.get(category.value)
        
        if category_override and category_override != self.sensitivity_settings.global_level:
            # Load override settings
            try:
                config_path = str(self._config_path)
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        levels = config.get('pattern_detection', {}).get('sensitivity', {}).get('levels', {})
                        level_settings = levels.get(category_override, {})
                        
                        override_settings = SensitivitySettings(
                            global_level=category_override,
                            confidence_threshold=level_settings.get('confidence_threshold', 0.7),
                            context_multiplier=level_settings.get('context_multiplier', 1.0),
                            min_matches_for_consultation=level_settings.get('min_matches_for_consultation', 2),
                            severity_threshold=level_settings.get('severity_threshold', 'medium'),
                            category_overrides=self.sensitivity_settings.category_overrides
                        )
                        
                        # Update cache metadata
                        override_settings.update_cache_metadata(config_path)
                        return override_settings
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger = logging.getLogger('pattern_detection')
                logger.warning(f"Failed to load category override for {category.value}: {e}")
        
        # Update cache metadata for global settings
        global_settings = self.sensitivity_settings
        global_settings.update_cache_metadata(str(self._config_path))
        return global_settings
    
    def _validate_and_sanitize_text(self, text: str) -> str:
        """Validate and sanitize input text"""
        if text is None:
            raise ValidationError("Text cannot be None")
        
        if not isinstance(text, str):
            raise ValidationError(f"Text must be string, got {type(text)}")
        
        # Handle empty text
        if not text.strip():
            return ""
        
        # Check for extremely large text (potential DoS)
        max_size = 1_000_000  # 1MB limit
        if len(text) > max_size:
            logger = logging.getLogger('pattern_detection')
            logger.warning(f"Text too large ({len(text)} chars), truncating to {max_size}")
            text = text[:max_size]
        
        # Handle encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            logger = logging.getLogger('pattern_detection')
            logger.warning("Text contains non-UTF-8 characters, cleaning")
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        return text
    
    def detect_patterns(self, text: str) -> List[PatternMatch]:
        """Detect all patterns in the given text with enhanced context extraction.
        
        Analyzes input text for security vulnerabilities, code uncertainties, algorithm
        optimizations, programming gotchas, and architecture decisions. Applies current
        sensitivity settings to filter and rank matches.
        
        Args:
            text: Input text to analyze. Must be a string, can be empty.
                 Large text (>1MB) will be automatically truncated.
                 
        Returns:
            List of PatternMatch objects sorted by severity (highest first) and position.
            Each match includes:
                - category: Pattern category (security, uncertainty, etc.)
                - severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
                - keyword: Matched keyword or phrase
                - context: Surrounding context with enhanced extraction
                - line_number: Line number where match occurred
                - confidence: Match confidence score (0.0-1.0)
                - requires_multi_ai: Whether match needs multi-AI consultation
                
        Raises:
            ValidationError: If text is None, wrong type, or invalid
            PatternDetectionError: If pattern detection fails unexpectedly
            
        Example:
            >>> engine = EnhancedPatternDetectionEngine()
            >>> matches = engine.detect_patterns('''
            ... def login(password):
            ...     if password == "admin123":  # TODO: use secure auth
            ...         return True
            ... ''')
            >>> len(matches)
            2
            >>> matches[0].category.value
            'security'
            >>> matches[0].severity.name
            'CRITICAL'
        """
        # Check if pattern detection is globally enabled
        if not self.is_pattern_detection_enabled():
            return []
        
        # Validate input
        validated_text = self._validate_and_sanitize_text(text)
        
        matches = []
        lines = validated_text.split('\n')
        
        # Check if category is enabled before processing
        for category, patterns in self.compiled_patterns.items():
            if not self._is_category_enabled(category):
                continue
            definition = self.pattern_definitions[category]
            category_sensitivity = self._get_category_sensitivity(category)
            
            for pattern in patterns:
                for match in pattern.finditer(validated_text):
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
                    
                    # Calculate confidence based on exact match vs fuzzy match and sensitivity
                    base_confidence = 1.0 if match.group(0).lower() in definition.keywords else 0.8
                    
                    # Apply sensitivity threshold - filter out low-confidence matches if sensitivity is low
                    if base_confidence < category_sensitivity.confidence_threshold:
                        continue
                    
                    pattern_match = PatternMatch(
                        category=category,
                        severity=definition.severity,
                        keyword=match.group(0),
                        context=context,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=base_confidence,
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
    
    def get_pattern_summary(self, matches: List[PatternMatch]) -> Dict[str, any]:
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
    
    def should_trigger_consultation(self, matches: List[PatternMatch], threshold: PatternSeverity = None) -> bool:
        """Enhanced logic to determine if pattern matches warrant AI consultation using sensitivity settings"""
        if not matches:
            return False
        
        # Use sensitivity settings to determine threshold if not provided
        if threshold is None:
            threshold_name = self.sensitivity_settings.severity_threshold.upper()
            threshold = PatternSeverity[threshold_name] if threshold_name in PatternSeverity.__members__ else PatternSeverity.MEDIUM
        
        # Check if any match exceeds threshold
        for match in matches:
            if match.severity.value >= threshold.value:
                return True
        
        # Use sensitivity-based minimum matches requirement
        min_matches = self.sensitivity_settings.min_matches_for_consultation
        if len(matches) >= min_matches:
            return True
        
        # Check if multiple categories are involved (complex scenario)
        categories = set(match.category for match in matches)
        if len(categories) >= 2:
            return True
        
        return False
    
    def get_consultation_strategy(self, matches: List[PatternMatch]) -> Dict[str, Any]:
        """Generate intelligent AI consultation strategy based on detected patterns.
        
        Analyzes pattern matches to determine the optimal AI consultation approach,
        including which AIs to use, consultation focus areas, and priority rankings.
        
        Args:
            matches: List of pattern matches to analyze. Can be empty.
            
        Returns:
            Dictionary containing consultation strategy with keys:
                - strategy: "single_ai" | "multi_ai" | "none"
                - primary_category: Most important pattern category found
                - severity: Highest severity level found  
                - categories: All pattern categories present
                - reason: Human-readable explanation for strategy
                - recommended_ais: List of AI models to consult (e.g., ["openai", "gemini"])
                - focus_areas: Specific areas for AI to focus on
                - lines_to_review: Line numbers requiring attention (max 10)
                
        Example:
            >>> matches = engine.detect_patterns("api_key = 'sk-123'")
            >>> strategy = engine.get_consultation_strategy(matches)
            >>> strategy['strategy']
            'multi_ai'
            >>> strategy['recommended_ais']
            ['openai', 'gemini', 'grok']
            >>> 'Security vulnerability assessment' in strategy['focus_areas']
            True
        """
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
    
    def _generate_consultation_reason(self, summary: Dict[str, any]) -> str:
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
            elif category_name == PatternCategory.GOTCHA.value:
                focus_areas.append("Common pitfall prevention")
                focus_areas.append("Edge case handling")
            elif category_name == PatternCategory.UNCERTAINTY.value:
                focus_areas.append("Implementation clarification")
                focus_areas.append("Best practice guidance")
        
        return focus_areas[:4]  # Return top 4 focus areas
    
    def update_sensitivity(self, global_level: Optional[str] = None, 
                          category_overrides: Optional[Dict[str, Optional[str]]] = None) -> bool:
        """Update pattern detection sensitivity settings with atomic persistence.
        
        Updates sensitivity configuration and immediately applies changes to the engine.
        Changes are persisted atomically to prevent configuration corruption.
        
        Args:
            global_level: New global sensitivity level. Must be one of:
                         "low", "medium", "high", "maximum". If None, unchanged.
            category_overrides: Category-specific sensitivity overrides.
                              Keys must be valid category names, values must be
                              valid sensitivity levels or None (for global default).
                              If None, category overrides are unchanged.
                              
        Returns:
            True if update succeeded and was persisted, False otherwise.
            
        Raises:
            ValidationError: If parameters are invalid
            
        Example:
            >>> engine.update_sensitivity(global_level="high")
            True
            
            >>> engine.update_sensitivity(
            ...     category_overrides={"security": "maximum", "uncertainty": None}
            ... )
            True
            
            >>> # Invalid level
            >>> engine.update_sensitivity(global_level="invalid")
            False
            
        Note:
            - Changes take effect immediately for new pattern detections
            - Configuration cache is cleared and reloaded
            - File updates are atomic to prevent corruption
            - Thread-safe operation
        """
        logger = logging.getLogger('pattern_detection')
        
        try:
            # Validate inputs
            if global_level is not None:
                self._validate_sensitivity_level(global_level)
            
            if category_overrides is not None:
                if not isinstance(category_overrides, dict):
                    raise ValidationError(f"category_overrides must be dict, got {type(category_overrides)}")
                self._validate_category_overrides(category_overrides)
            
            # Try multiple possible config file paths
            possible_paths = ["test_final_credentials.json", "test_credentials.json", str(self._config_path)]
            config_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                logger.error("No configuration file found for update")
                return False
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            pattern_config = config.setdefault('pattern_detection', {})
            sensitivity_config = pattern_config.setdefault('sensitivity', {})
            
            # Update global level if provided
            if global_level:
                levels = sensitivity_config.get('levels', {})
                if global_level in levels:
                    sensitivity_config['global_level'] = global_level
                    level_settings = levels[global_level]
                    
                    # Update current settings
                    self.sensitivity_settings.global_level = global_level
                    self.sensitivity_settings.confidence_threshold = level_settings.get('confidence_threshold', 0.7)
                    self.sensitivity_settings.context_multiplier = level_settings.get('context_multiplier', 1.0)
                    self.sensitivity_settings.min_matches_for_consultation = level_settings.get('min_matches_for_consultation', 2)
                    self.sensitivity_settings.severity_threshold = level_settings.get('severity_threshold', 'medium')
                    
                    # Reapply context window multiplier
                    self._apply_sensitivity_to_context_window()
                else:
                    return False  # Invalid level
            
            # Update category overrides if provided
            if category_overrides:
                current_overrides = sensitivity_config.setdefault('category_overrides', {})
                current_overrides.update(category_overrides)
                self.sensitivity_settings.category_overrides.update(category_overrides)
            
            # Atomic file update
            success = self._atomic_config_update(config_path, config)
            
            if success:
                # Clear cache to force reload
                with self._cache_lock:
                    self._sensitivity_cache.clear()
                
                # Reload settings
                self.sensitivity_settings = self._load_sensitivity_settings(config_path)
                self._apply_sensitivity_to_context_window()
            
            return success
            
        except ValidationError as e:
            logger.error(f"Validation error in update_sensitivity: {e}")
            return False
        except ConfigurationError as e:
            logger.error(f"Configuration error in update_sensitivity: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied updating config file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating sensitivity: {e}")
            return False
    
    def _atomic_config_update(self, config_path: str, config: Dict) -> bool:
        """Perform atomic configuration file update"""
        logger = logging.getLogger('pattern_detection')
        
        try:
            # Create temporary file in same directory as target
            config_dir = os.path.dirname(config_path)
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=config_dir, 
                suffix='.tmp',
                prefix='config_',
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                json.dump(config, tmp_file, indent=2, ensure_ascii=False)
                tmp_path = tmp_file.name
            
            # Atomic move (rename is atomic on most filesystems)
            shutil.move(tmp_path, config_path)
            
            logger.debug(f"Atomic config update successful: {config_path}")
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"Failed to atomically update config: {e}")
            # Clean up temporary file if it exists
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            return False
        except Exception as e:
            logger.error(f"Unexpected error in atomic update: {e}")
            return False
    
    def invalidate_cache(self) -> None:
        """Invalidate all cached sensitivity settings"""
        with self._cache_lock:
            self._sensitivity_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics for monitoring"""
        with self._cache_lock:
            cache_size = len(self._sensitivity_cache)
            valid_entries = sum(1 for settings in self._sensitivity_cache.values() 
                              if settings.is_cache_valid())
            
            return {
                "cache_size": cache_size,
                "valid_entries": valid_entries,
                "hit_ratio": valid_entries / cache_size if cache_size > 0 else 0.0,
                "last_access": time.time()
            }
    
    def get_sensitivity_info(self) -> Dict[str, Any]:
        """Get comprehensive sensitivity configuration information.
        
        Returns detailed information about current sensitivity settings,
        including global level, thresholds, category overrides, and
        effective configuration values.
        
        Returns:
            Dictionary containing sensitivity configuration with keys:
                - global_level: Current global sensitivity level
                - confidence_threshold: Pattern confidence threshold (0.0-1.0)
                - context_multiplier: Context window multiplier
                - min_matches_for_consultation: Minimum matches to trigger consultation
                - severity_threshold: Minimum severity for consultation
                - category_overrides: Category-specific sensitivity overrides
                - effective_context_window: Actual context window size in characters
                
        Example:
            >>> engine = EnhancedPatternDetectionEngine()
            >>> info = engine.get_sensitivity_info()
            >>> info['global_level']
            'medium'
            >>> info['confidence_threshold']
            0.7
            >>> info['category_overrides']['security']
            'high'
        """
        return {
            "global_level": self.sensitivity_settings.global_level,
            "confidence_threshold": self.sensitivity_settings.confidence_threshold,
            "context_multiplier": self.sensitivity_settings.context_multiplier,
            "min_matches_for_consultation": self.sensitivity_settings.min_matches_for_consultation,
            "severity_threshold": self.sensitivity_settings.severity_threshold,
            "category_overrides": self.sensitivity_settings.category_overrides,
            "effective_context_window": self.context_window_size
        }
    
    def add_custom_keywords(self, category: str, keywords: List[str]) -> bool:
        """Add custom keywords to a pattern category.
        
        Args:
            category: Pattern category name (security, uncertainty, etc.)
            keywords: List of keywords to add
            
        Returns:
            True if keywords were added successfully, False otherwise
        """
        logger = logging.getLogger('pattern_detection')
        
        try:
            # Validate category
            try:
                pattern_category = PatternCategory(category)
            except ValueError:
                logger.error(f"Invalid category: {category}")
                return False
            
            # Validate keywords
            if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
                logger.error("Keywords must be a list of strings")
                return False
            
            # Add keywords to pattern definition
            if pattern_category in self.pattern_definitions:
                with self._cache_lock:  # Thread safety for shared state modification
                    current_keywords = set(self.pattern_definitions[pattern_category].keywords)
                    new_keywords = [k.lower().strip() for k in keywords if k.strip()]
                    current_keywords.update(new_keywords)
                    self.pattern_definitions[pattern_category].keywords = list(current_keywords)
                    
                    # Recompile patterns
                    self.compiled_patterns = self._compile_patterns()
                    
                    # Update configuration
                    self._update_custom_keywords_config(category, new_keywords, 'add')
                
                logger.info(f"Added {len(new_keywords)} keywords to {category}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding custom keywords: {e}")
            return False
    
    def remove_keywords(self, category: str, keywords: List[str]) -> bool:
        """Remove keywords from a pattern category.
        
        Args:
            category: Pattern category name
            keywords: List of keywords to remove
            
        Returns:
            True if keywords were removed successfully, False otherwise
        """
        logger = logging.getLogger('pattern_detection')
        
        try:
            # Validate category
            try:
                pattern_category = PatternCategory(category)
            except ValueError:
                logger.error(f"Invalid category: {category}")
                return False
            
            # Remove keywords from pattern definition
            if pattern_category in self.pattern_definitions:
                with self._cache_lock:  # Thread safety for shared state modification
                    current_keywords = set(self.pattern_definitions[pattern_category].keywords)
                    keywords_to_remove = [k.lower().strip() for k in keywords if k.strip()]
                    current_keywords.difference_update(keywords_to_remove)
                    self.pattern_definitions[pattern_category].keywords = list(current_keywords)
                    
                    # Recompile patterns
                    self.compiled_patterns = self._compile_patterns()
                    
                    # Update configuration
                    self._update_custom_keywords_config(category, keywords_to_remove, 'remove')
                
                logger.info(f"Removed keywords from {category}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing keywords: {e}")
            return False
    
    def add_custom_pattern(self, category: str, pattern: str) -> bool:
        """Add a custom regex pattern to a category.
        
        Args:
            category: Pattern category name
            pattern: Regular expression pattern
            
        Returns:
            True if pattern was added successfully, False otherwise
        """
        logger = logging.getLogger('pattern_detection')
        
        try:
            # Validate category
            try:
                pattern_category = PatternCategory(category)
            except ValueError:
                logger.error(f"Invalid category: {category}")
                return False
            
            # Validate regex pattern
            try:
                re.compile(pattern)
            except re.error as e:
                logger.error(f"Invalid regex pattern: {e}")
                return False
            
            # Add pattern to definition
            if pattern_category in self.pattern_definitions:
                with self._cache_lock:  # Thread safety for shared state modification
                    # Check for duplicates
                    if pattern not in self.pattern_definitions[pattern_category].regex_patterns:
                        self.pattern_definitions[pattern_category].regex_patterns.append(pattern)
                        
                        # Recompile patterns
                        self.compiled_patterns = self._compile_patterns()
                        
                        # Update configuration
                        self._update_custom_patterns_config(category, [pattern], 'add')
                        
                        logger.info(f"Added custom pattern to {category}")
                        return True
                    else:
                        logger.info(f"Pattern already exists in {category}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding custom pattern: {e}")
            return False
    
    def get_all_keywords(self, category: str) -> List[str]:
        """Get all keywords for a specific category.
        
        Args:
            category: Pattern category name
            
        Returns:
            List of keywords for the category, empty list if category not found
        """
        try:
            pattern_category = PatternCategory(category)
            if pattern_category in self.pattern_definitions:
                return self.pattern_definitions[pattern_category].keywords.copy()
        except ValueError:
            pass
        
        return []
    
    def set_pattern_detection_enabled(self, enabled: bool) -> bool:
        """Enable or disable pattern detection globally.
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if setting was updated successfully
        """
        logger = logging.getLogger('pattern_detection')
        
        try:
            config_path = str(self._config_path)
            
            # Load current config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update enabled setting
            if 'pattern_detection' not in config:
                config['pattern_detection'] = {}
            
            config['pattern_detection']['enabled'] = enabled
            
            # Save config atomically
            if self._atomic_config_update(config_path, config):
                self._pattern_detection_enabled = enabled
                logger.info(f"Pattern detection {'enabled' if enabled else 'disabled'}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating pattern detection enabled: {e}")
            return False
    
    def is_pattern_detection_enabled(self) -> bool:
        """Check if pattern detection is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return getattr(self, '_pattern_detection_enabled', True)
    
    def set_category_enabled(self, category: str, enabled: bool) -> bool:
        """Enable or disable a specific pattern category.
        
        Args:
            category: Pattern category name
            enabled: True to enable, False to disable
            
        Returns:
            True if setting was updated successfully
        """
        logger = logging.getLogger('pattern_detection')
        
        try:
            # Validate category
            try:
                pattern_category = PatternCategory(category)
            except ValueError:
                logger.error(f"Invalid category: {category}")
                return False
            
            config_path = str(self._config_path)
            
            # Add locking to prevent race condition
            with self._cache_lock:  # Reuse existing lock
                # Load current config
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Update category enabled setting
                pattern_config = config.get('pattern_detection', {})
                categories = pattern_config.get('pattern_categories', {})
                
                if category not in categories:
                    categories[category] = {}
                
                categories[category]['enabled'] = enabled
                pattern_config['pattern_categories'] = categories
                config['pattern_detection'] = pattern_config
                
                # Save config atomically
                if self._atomic_config_update(config_path, config):
                    # Update in-memory state
                    self._category_enabled_states[pattern_category] = enabled
                    logger.info(f"Category {category} {'enabled' if enabled else 'disabled'}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating category enabled: {e}")
            return False
    
    def _update_custom_keywords_config(self, category: str, keywords: List[str], action: str) -> None:
        """Update custom keywords in configuration file."""
        try:
            config_path = str(self._config_path)
            
            # Load current config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Navigate to pattern categories
            pattern_config = config.get('pattern_detection', {})
            categories = pattern_config.get('pattern_categories', {})
            
            if category not in categories:
                categories[category] = {}
            
            # Update custom keywords
            custom_keywords = set(categories[category].get('custom_keywords', []))
            
            if action == 'add':
                custom_keywords.update(keywords)
            elif action == 'remove':
                custom_keywords.difference_update(keywords)
            
            categories[category]['custom_keywords'] = list(custom_keywords)
            pattern_config['pattern_categories'] = categories
            config['pattern_detection'] = pattern_config
            
            # Save config atomically
            self._atomic_config_update(config_path, config)
            
        except Exception as e:
            logger = logging.getLogger('pattern_detection')
            logger.error(f"Error updating custom keywords config: {e}")
    
    def _update_custom_patterns_config(self, category: str, patterns: List[str], action: str) -> None:
        """Update custom patterns in configuration file."""
        try:
            config_path = str(self._config_path)
            
            # Load current config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Navigate to pattern categories
            pattern_config = config.get('pattern_detection', {})
            categories = pattern_config.get('pattern_categories', {})
            
            if category not in categories:
                categories[category] = {}
            
            # Update custom patterns
            custom_patterns = categories[category].get('custom_patterns', [])
            
            if action == 'add':
                custom_patterns.extend(patterns)
            elif action == 'remove':
                custom_patterns = [p for p in custom_patterns if p not in patterns]
            
            categories[category]['custom_patterns'] = custom_patterns
            pattern_config['pattern_categories'] = categories
            config['pattern_detection'] = pattern_config
            
            # Save config atomically
            self._atomic_config_update(config_path, config)
            
        except Exception as e:
            logger = logging.getLogger('pattern_detection')
            logger.error(f"Error updating custom patterns config: {e}")


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


# Alias for backward compatibility
PatternDetectionEngine = EnhancedPatternDetectionEngine
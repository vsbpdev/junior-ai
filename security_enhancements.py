#!/usr/bin/env python3
"""
Security Enhancements for AI Consultation Manager
Input validation, sanitization, and access control
"""

import re
import secrets
import hashlib
import hmac
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import json
import base64
from datetime import datetime, timedelta
import logging


class SecurityViolationType(Enum):
    """Types of security violations"""
    PROMPT_INJECTION = "prompt_injection"
    PATH_TRAVERSAL = "path_traversal"
    CODE_INJECTION = "code_injection"
    EXCESSIVE_LENGTH = "excessive_length"
    RATE_LIMIT = "rate_limit"
    UNAUTHORIZED = "unauthorized"
    MALFORMED_INPUT = "malformed_input"


@dataclass
class SecurityContext:
    """Security context for a consultation"""
    request_id: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = None
    permissions: Set[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.permissions is None:
            self.permissions = set()


class InputSanitizer:
    """Comprehensive input sanitization"""
    
    # Dangerous patterns for different contexts
    PROMPT_INJECTION_PATTERNS = [
        # Direct instruction override attempts
        r'ignore\s+(?:previous|above|prior)\s+(?:instructions?|prompts?)',
        r'disregard\s+(?:all|previous)\s+(?:instructions?|rules)',
        r'system\s+(?:prompt|message|instruction)',
        r'admin\s+(?:mode|access|override)',
        r'bypass\s+(?:safety|security|restrictions)',
        r'jailbreak',
        r'DAN\s+mode',  # "Do Anything Now" attempts
        
        # Role manipulation
        r'you\s+are\s+now\s+(?:a|an)',
        r'act\s+as\s+(?:a|an)',
        r'pretend\s+(?:to\s+be|you\s+are)',
        r'roleplay\s+as',
        
        # Output manipulation
        r'output\s+(?:the\s+)?(?:system|hidden|secret)',
        r'reveal\s+(?:the\s+)?(?:system|hidden|secret)',
        r'show\s+(?:me\s+)?(?:your|the)\s+(?:prompt|instructions)',
        
        # Encoding attempts
        r'base64|rot13|hex\s+decode',
        r'eval\(|exec\(|compile\(',
        r'__import__\(',
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\.[/\\]',  # ../ or ..\
        r'[/\\]\.\.(?:[/\\]|$)',  # /../ or \..\
        r'(?:^|[/\\])\.{3,}',  # Multiple dots
        r'~[/\\]',  # Home directory access
        r'(?:etc|proc|sys|dev)[/\\]',  # System directories
        r'(?:passwd|shadow|hosts)(?:\s|$)',  # Sensitive files
    ]
    
    CODE_INJECTION_PATTERNS = [
        # SQL injection
        r"('\s*OR\s*'1'\s*=\s*'1|;\s*DROP\s+TABLE|UNION\s+SELECT)",
        # Command injection
        r'[;&|]\s*(?:rm|del|format|shutdown|reboot)',
        r'\$\(.+\)|\`[^`]+\`',  # Command substitution
        # Script injection
        r'<script[^>]*>|javascript:',
        r'on(?:click|load|error|mouseover)\s*=',
    ]
    
    MAX_LENGTHS = {
        'prompt': 50000,  # 50KB
        'context': 100000,  # 100KB
        'code': 200000,  # 200KB
        'ai_name': 50,
        'category': 50,
        'keyword': 100,
    }
    
    @classmethod
    def sanitize_prompt(cls, 
                       prompt: str,
                       context: SecurityContext) -> Tuple[str, List[SecurityViolationType]]:
        """Sanitize prompt and return violations found"""
        violations = []
        
        if not prompt:
            return "", [SecurityViolationType.MALFORMED_INPUT]
        
        # Check length
        if len(prompt) > cls.MAX_LENGTHS['prompt']:
            violations.append(SecurityViolationType.EXCESSIVE_LENGTH)
            prompt = prompt[:cls.MAX_LENGTHS['prompt']] + "... [truncated]"
        
        # Check for prompt injection
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(SecurityViolationType.PROMPT_INJECTION)
                # Don't process potentially malicious prompts
                return "", violations
        
        # Check for code injection
        for pattern in cls.CODE_INJECTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(SecurityViolationType.CODE_INJECTION)
                return "", violations
        
        # Sanitize control characters
        prompt = cls._remove_control_chars(prompt)
        
        # Escape special characters for AI safety
        prompt = cls._escape_ai_special_chars(prompt)
        
        return prompt, violations
    
    @classmethod
    def sanitize_file_path(cls, path: str) -> Tuple[str, List[SecurityViolationType]]:
        """Sanitize file paths to prevent traversal attacks"""
        violations = []
        
        if not path:
            return "", [SecurityViolationType.MALFORMED_INPUT]
        
        # Check for path traversal
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path):
                violations.append(SecurityViolationType.PATH_TRAVERSAL)
                return "", violations
        
        # Normalize path
        path = path.replace('\\', '/')
        path = re.sub(r'/+', '/', path)  # Remove multiple slashes
        
        # Remove dangerous characters
        path = re.sub(r'[^\w\-_./]', '', path)
        
        return path, violations
    
    @classmethod
    def _remove_control_chars(cls, text: str) -> str:
        """Remove control characters except newlines and tabs"""
        # Keep \n (0x0A) and \t (0x09)
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    @classmethod
    def _escape_ai_special_chars(cls, text: str) -> str:
        """Escape characters that might affect AI behavior"""
        # Escape potential delimiter characters
        text = text.replace('"""', '\\"\\"\\"')
        text = text.replace("'''", "\\'\\'\\'")
        
        # Limit consecutive special characters
        text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
        
        return text


class AccessController:
    """Access control and authorization"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.permissions = {
            'basic': {'pattern_check', 'junior_consult'},
            'advanced': {'pattern_check', 'junior_consult', 'multi_ai', 'consensus'},
            'admin': {'*'}  # All permissions
        }
        self.rate_limits = {
            'basic': {'calls_per_minute': 10, 'burst': 3},
            'advanced': {'calls_per_minute': 30, 'burst': 10},
            'admin': {'calls_per_minute': 100, 'burst': 30}
        }
    
    def generate_token(self, 
                      user_id: str, 
                      role: str = 'basic',
                      expires_in: int = 3600) -> str:
        """Generate secure access token"""
        payload = {
            'user_id': user_id,
            'role': role,
            'expires': (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
            'nonce': secrets.token_urlsafe(16)
        }
        
        # Create signature
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine payload and signature
        token_data = {
            'payload': payload,
            'signature': signature
        }
        
        return base64.urlsafe_b64encode(
            json.dumps(token_data).encode()
        ).decode()
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify access token and return payload"""
        try:
            # Decode token
            token_data = json.loads(
                base64.urlsafe_b64decode(token.encode())
            )
            
            payload = token_data['payload']
            signature = token_data['signature']
            
            # Verify signature
            message = json.dumps(payload, sort_keys=True)
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return False, None
            
            # Check expiration
            expires = datetime.fromisoformat(payload['expires'])
            if datetime.now() > expires:
                return False, None
            
            return True, payload
            
        except Exception:
            return False, None
    
    def check_permission(self,
                        context: SecurityContext,
                        required_permission: str) -> bool:
        """Check if context has required permission"""
        if not context.permissions:
            return False
        
        # Admin has all permissions
        if '*' in context.permissions:
            return True
        
        return required_permission in context.permissions
    
    def get_rate_limit(self, role: str) -> Dict[str, int]:
        """Get rate limit for role"""
        return self.rate_limits.get(role, self.rate_limits['basic'])


class SecurityAuditor:
    """Security audit logging"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.logger = logging.getLogger("security_audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_violation(self,
                     violation_type: SecurityViolationType,
                     context: SecurityContext,
                     details: Dict[str, Any]):
        """Log security violation"""
        self.logger.warning(
            f"SECURITY_VIOLATION: {violation_type.value} | "
            f"User: {context.user_id} | "
            f"IP: {context.ip_address} | "
            f"Details: {json.dumps(details)}"
        )
    
    def log_access(self,
                  context: SecurityContext,
                  action: str,
                  result: str):
        """Log access attempt"""
        self.logger.info(
            f"ACCESS: {action} | "
            f"User: {context.user_id} | "
            f"Result: {result}"
        )
    
    def log_suspicious_pattern(self,
                              context: SecurityContext,
                              pattern: str,
                              input_text: str):
        """Log suspicious patterns detected"""
        self.logger.warning(
            f"SUSPICIOUS_PATTERN: {pattern} | "
            f"User: {context.user_id} | "
            f"Sample: {input_text[:100]}..."
        )


class SecureAIConsultationWrapper:
    """Secure wrapper for AI consultation with validation"""
    
    def __init__(self,
                 consultation_manager: Any,
                 access_controller: AccessController,
                 security_auditor: SecurityAuditor):
        self.consultation_manager = consultation_manager
        self.access_controller = access_controller
        self.security_auditor = security_auditor
    
    def consult_with_validation(self,
                               prompt: str,
                               context: str,
                               security_context: SecurityContext,
                               options: Dict[str, Any] = None) -> Any:
        """Consult with comprehensive security validation"""
        # Check permissions
        if not self.access_controller.check_permission(
            security_context, 'junior_consult'
        ):
            self.security_auditor.log_access(
                security_context, 'junior_consult', 'DENIED'
            )
            raise PermissionError("Insufficient permissions")
        
        # Sanitize inputs
        clean_prompt, prompt_violations = InputSanitizer.sanitize_prompt(
            prompt, security_context
        )
        clean_context, context_violations = InputSanitizer.sanitize_prompt(
            context, security_context
        )
        
        violations = prompt_violations + context_violations
        
        if violations:
            for violation in violations:
                self.security_auditor.log_violation(
                    violation,
                    security_context,
                    {'prompt_sample': prompt[:100], 'context_sample': context[:100]}
                )
            
            if any(v in [SecurityViolationType.PROMPT_INJECTION,
                        SecurityViolationType.CODE_INJECTION] for v in violations):
                raise SecurityError("Security violation detected")
        
        # Log successful access
        self.security_auditor.log_access(
            security_context, 'junior_consult', 'ALLOWED'
        )
        
        # Execute consultation with cleaned inputs
        return self.consultation_manager.consult_junior(
            clean_prompt,
            clean_context,
            options
        )


class SecurityError(Exception):
    """Security-related error"""
    pass


class PermissionError(Exception):
    """Permission-related error"""
    pass
"""AI client initialization and management."""

import sys
import asyncio  # Used in async functions below (initialize_all_clients_async, cleanup_async_ai_clients)
from typing import Dict, Any, Optional
from .config import get_credentials

# Global AI clients storage
AI_CLIENTS: Dict[str, Any] = {}

# Async support will be imported dynamically to avoid circular imports
HAS_ASYNC_SUPPORT = False
ASYNC_AI_CLIENTS = {}

def _import_async_support():
    """Dynamically import async support to avoid circular imports."""
    global HAS_ASYNC_SUPPORT, ASYNC_AI_CLIENTS
    try:
        import sys
        import os
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ai.async_client import ASYNC_AI_CLIENTS as _async_clients
        ASYNC_AI_CLIENTS = _async_clients
        HAS_ASYNC_SUPPORT = True
        return True
    except ImportError as e:
        print(f"Async client import failed: {e}", file=sys.stderr)
        HAS_ASYNC_SUPPORT = False
        ASYNC_AI_CLIENTS = {}
        return False


def initialize_gemini_client(api_key: str, model_name: str = 'gemini-2.0-flash') -> Optional[Any]:
    """Initialize Gemini client."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        # Store model name for later reference
        model.model_name = model_name
        print(f"✅ GEMINI initialized successfully with model {model_name}", file=sys.stderr)
        return model
    except Exception as e:
        print(f"❌ Failed to initialize GEMINI: {str(e)}", file=sys.stderr)
        return None


def initialize_openai_compatible_client(
    service_name: str,
    api_key: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[Any]:
    """Initialize OpenAI-compatible client (Grok, OpenAI, DeepSeek, OpenRouter)."""
    try:
        from openai import OpenAI
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        client = OpenAI(**client_kwargs)
        
        # Store model name if provided
        if model_name:
            client.model_name = model_name
            
        print(f"✅ {service_name.upper()} initialized successfully", file=sys.stderr)
        return client
    except Exception as e:
        print(f"❌ Failed to initialize {service_name.upper()}: {str(e)}", file=sys.stderr)
        return None


def initialize_all_clients() -> Dict[str, Any]:
    """Initialize all configured AI clients."""
    credentials = get_credentials()
    
    # Initialize GEMINI
    if 'gemini' in credentials and credentials['gemini'].get('api_key'):
        model_name = credentials['gemini'].get('model', 'gemini-2.0-flash')
        client = initialize_gemini_client(credentials['gemini']['api_key'], model_name)
        if client:
            AI_CLIENTS['gemini'] = client
    
    # Initialize GROK
    if 'grok' in credentials and credentials['grok'].get('api_key'):
        client = initialize_openai_compatible_client(
            'grok',
            credentials['grok']['api_key'],
            credentials['grok'].get('base_url'),
            model_name=credentials['grok'].get('model', 'grok-3')
        )
        if client:
            AI_CLIENTS['grok'] = client
    
    # Initialize OpenAI
    if 'openai' in credentials and credentials['openai'].get('api_key'):
        client = initialize_openai_compatible_client(
            'openai',
            credentials['openai']['api_key'],
            model_name=credentials['openai'].get('model', 'gpt-4o-mini')
        )
        if client:
            AI_CLIENTS['openai'] = client
    
    # Initialize DeepSeek
    if 'deepseek' in credentials and credentials['deepseek'].get('api_key'):
        client = initialize_openai_compatible_client(
            'deepseek',
            credentials['deepseek']['api_key'],
            credentials['deepseek'].get('base_url'),
            model_name=credentials['deepseek'].get('model', 'deepseek-chat')
        )
        if client:
            AI_CLIENTS['deepseek'] = client
    
    # Initialize OpenRouter
    if 'openrouter' in credentials and credentials['openrouter'].get('api_key'):
        client = initialize_openai_compatible_client(
            'openrouter',
            credentials['openrouter']['api_key'],
            'https://openrouter.ai/api/v1',
            model_name=credentials['openrouter'].get('model', 'anthropic/claude-3.5-sonnet')
        )
        if client:
            AI_CLIENTS['openrouter'] = client
    
    # Report initialization status
    if AI_CLIENTS:
        print(f"✅ Successfully initialized {len(AI_CLIENTS)} AI client(s): {', '.join(AI_CLIENTS)}", file=sys.stderr)
    else:
        print("⚠️ No AI clients were initialized. Please check your credentials.", file=sys.stderr)
    
    return AI_CLIENTS


def get_ai_client(ai_name: str) -> Optional[Any]:
    """Get a specific AI client by name."""
    return AI_CLIENTS.get(ai_name)


def get_available_ai_clients() -> Dict[str, Any]:
    """Get all available AI clients."""
    return AI_CLIENTS


async def initialize_all_clients_async() -> Dict[str, Any]:
    """Initialize all AI clients including async versions."""
    # First initialize sync clients
    sync_clients = initialize_all_clients()
    
    # Try to import and initialize async clients
    if _import_async_support():
        try:
            from ai.async_client import initialize_async_ai_clients
            credentials = get_credentials()
            # Filter only AI-related credentials
            ai_credentials = {
                k: v for k, v in credentials.items() 
                if k in ['gemini', 'grok', 'openai', 'deepseek', 'openrouter']
            }
            await initialize_async_ai_clients(ai_credentials)
            print(f"✅ Async AI clients initialized: {', '.join(ASYNC_AI_CLIENTS.keys())}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Failed to initialize async AI clients: {e}", file=sys.stderr)
    
    return sync_clients


def get_async_ai_client(ai_name: str) -> Optional[Any]:
    """Get a specific async AI client by name."""
    if not HAS_ASYNC_SUPPORT:
        _import_async_support()
    return ASYNC_AI_CLIENTS.get(ai_name)


def has_async_support() -> bool:
    """Check if async AI support is available."""
    if not HAS_ASYNC_SUPPORT:
        _import_async_support()
    return HAS_ASYNC_SUPPORT


async def cleanup_async_ai_clients():
    """Clean up all async AI clients."""
    if _import_async_support():
        try:
            from ai.async_client import cleanup_async_ai_clients as cleanup_func
            await cleanup_func()
        except Exception as e:
            print(f"Error cleaning up async AI clients: {e}", file=sys.stderr)
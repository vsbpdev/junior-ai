"""
Async AI client wrapper for Junior AI Assistant.

Provides async wrappers for all AI providers with a unified interface,
proper timeout handling, and graceful error management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from enum import Enum

# AI SDK imports
try:
    from openai import AsyncOpenAI
    HAS_ASYNC_OPENAI = True
except ImportError:
    HAS_ASYNC_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    GROK = "grok"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


class AsyncAIClient(ABC):
    """Abstract base class for async AI clients."""
    
    def __init__(self, provider: AIProvider, model: str, timeout: float = 30.0):
        """Initialize async AI client.
        
        Args:
            provider: The AI provider type
            model: The model name to use
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the async client (if needed)."""
        pass
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      **kwargs) -> str:
        """Generate a response from the AI.
        
        Args:
            prompt: The input prompt
            temperature: Response temperature (0-1)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text response
            
        Raises:
            asyncio.TimeoutError: If request times out
            Exception: For other API errors
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc)."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


class AsyncGeminiClient(AsyncAIClient):
    """Async wrapper for Google Gemini."""
    
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        """Initialize Gemini async client.
        
        Args:
            api_key: Gemini API key
            model: Model name (e.g., 'gemini-2.0-flash')
            timeout: Request timeout in seconds
        """
        super().__init__(AIProvider.GEMINI, model, timeout)
        self.api_key = api_key
        self._client = None
        self._init_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize Gemini client."""
        if not HAS_GEMINI:
            raise ImportError("google-generativeai is not installed")
        
        async with self._init_lock:
            if not self._initialized:
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
                self._initialized = True
    
    async def generate(self, 
                      prompt: str, 
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      **kwargs) -> str:
        """Generate response using Gemini."""
        if not self._initialized:
            await self.initialize()
        
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Add any additional kwargs to generation config
        generation_config.update(kwargs)
        
        try:
            # Use asyncio timeout wrapper
            async with asyncio.timeout(self.timeout):
                # Gemini's generate_content_async method
                response = await self._client.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
        except asyncio.TimeoutError:
            logger.error(f"Gemini request timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up Gemini resources."""
        # Gemini doesn't require explicit cleanup
        self._initialized = False


class AsyncOpenAICompatibleClient(AsyncAIClient):
    """Async wrapper for OpenAI-compatible APIs."""
    
    def __init__(self, 
                 api_key: str, 
                 model: str,
                 base_url: Optional[str] = None,
                 provider: AIProvider = AIProvider.OPENAI,
                 timeout: float = 30.0):
        """Initialize OpenAI-compatible async client.
        
        Args:
            api_key: API key
            model: Model name
            base_url: Optional base URL for API
            provider: The provider type
            timeout: Request timeout in seconds
        """
        super().__init__(provider, model, timeout)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
    
    async def initialize(self) -> None:
        """Initialize async OpenAI client."""
        if not HAS_ASYNC_OPENAI:
            raise ImportError("openai is not installed or doesn't support async")
        
        if not self._initialized:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self._initialized = True
    
    async def generate(self, 
                      prompt: str, 
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      **kwargs) -> str:
        """Generate response using OpenAI-compatible API."""
        if not self._initialized:
            await self.initialize()
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error(f"{self.provider.value} request timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"{self.provider.value} API error: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up client resources."""
        if self._client:
            # Check if client has close method before calling
            if hasattr(self._client, 'close') and callable(self._client.close):
                await self._client.close()
        self._initialized = False


# Factory functions
async def create_async_ai_client(
    provider: Union[str, AIProvider],
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    timeout: float = 30.0
) -> AsyncAIClient:
    """Create an async AI client for the specified provider.
    
    Args:
        provider: Provider name or enum
        api_key: API key
        model: Model name
        base_url: Optional base URL (for OpenAI-compatible)
        timeout: Request timeout in seconds
        
    Returns:
        Initialized async AI client
        
    Raises:
        ValueError: If provider is not supported
        ImportError: If required SDK is not installed
    """
    # Convert string to enum if needed
    if isinstance(provider, str):
        provider_map = {p.value: p for p in AIProvider}
        if provider.lower() not in provider_map:
            raise ValueError(f"Unsupported provider: {provider}")
        provider = provider_map[provider.lower()]
    
    if provider == AIProvider.GEMINI:
        client = AsyncGeminiClient(api_key, model, timeout)
    elif provider in (AIProvider.OPENAI, AIProvider.GROK, 
                     AIProvider.DEEPSEEK, AIProvider.OPENROUTER):
        client = AsyncOpenAICompatibleClient(
            api_key, model, base_url, provider, timeout
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    await client.initialize()
    return client


# Global async client storage
ASYNC_AI_CLIENTS: Dict[str, AsyncAIClient] = {}


async def initialize_async_ai_clients(credentials: Dict[str, Any]) -> Dict[str, AsyncAIClient]:
    """Initialize all async AI clients from credentials.
    
    Args:
        credentials: Dictionary of AI credentials
        
    Returns:
        Dictionary of initialized async clients
    """
    clients = {}
    
    for ai_name, config in credentials.items():
        if not isinstance(config, dict) or not config.get('enabled', True):
            continue
            
        api_key = config.get('api_key')
        if not api_key:
            continue
        
        try:
            model = config.get('model', '')
            base_url = config.get('base_url')
            
            client = await create_async_ai_client(
                provider=ai_name,
                api_key=api_key,
                model=model,
                base_url=base_url
            )
            
            clients[ai_name] = client
            logger.info(f"Initialized async {ai_name} client with model {model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize async {ai_name} client: {e}")
    
    # Update global storage
    ASYNC_AI_CLIENTS.update(clients)
    return clients


async def cleanup_async_ai_clients():
    """Clean up all async AI clients."""
    for name, client in ASYNC_AI_CLIENTS.items():
        try:
            await client.cleanup()
            logger.info(f"Cleaned up async {name} client")
        except Exception as e:
            logger.error(f"Error cleaning up async {name} client: {e}")
    
    ASYNC_AI_CLIENTS.clear()


# Backward-compatible async wrapper
async def call_ai_async(
    ai_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """Call an AI asynchronously with backward compatibility.
    
    Args:
        ai_name: Name of the AI provider
        prompt: The prompt to send
        temperature: Response temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional provider-specific parameters
        
    Returns:
        The AI response text
        
    Raises:
        ValueError: If AI client not found
        Exception: For API errors
    """
    client = ASYNC_AI_CLIENTS.get(ai_name.lower())
    if not client:
        raise ValueError(f"Async AI client '{ai_name}' not initialized")
    
    return await client.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
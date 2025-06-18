"""Core utilities and configuration for Junior AI Assistant."""

from .config import (
    __version__,
    load_credentials,
    get_credentials,
    CREDENTIALS
)

from .ai_clients import (
    initialize_all_clients,
    get_ai_client,
    AI_CLIENTS
)

from .utils import (
    format_error_response,
    validate_temperature
)

__all__ = [
    '__version__',
    'load_credentials',
    'get_credentials',
    'CREDENTIALS',
    'initialize_all_clients',
    'get_ai_client',
    'AI_CLIENTS',
    'format_error_response',
    'validate_temperature'
]
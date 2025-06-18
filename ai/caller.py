"""AI calling functionality."""

import asyncio
from typing import Optional, List, Dict, Any
from core.ai_clients import get_ai_client
from core.utils import validate_temperature, format_error_response


def call_ai(ai_name: str, prompt: str, temperature: float = 0.7) -> str:
    """Call a specific AI with a prompt and return the response."""
    client = get_ai_client(ai_name)
    if not client:
        return f"❌ {ai_name.upper()} is not available. Please check your API key configuration."
    
    try:
        # Validate temperature
        temperature = validate_temperature(temperature)
        
        if ai_name == 'gemini':
            # Gemini uses generate_content
            response = client.generate_content(
                prompt,
                generation_config={
                    'temperature': temperature,
                    'max_output_tokens': 8192,
                }
            )
            return response.text
        else:
            # OpenAI-compatible clients
            model = getattr(client, 'model_name', None)
            if ai_name == 'grok':
                model = 'grok-beta'
            elif ai_name == 'deepseek':
                model = 'deepseek-chat'
            elif ai_name == 'openai' and not model:
                model = 'gpt-4o-mini'
            elif ai_name == 'openrouter' and not model:
                model = 'anthropic/claude-3.5-sonnet'
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=8192
            )
            return response.choices[0].message.content
            
    except Exception as e:
        return format_error_response(e, f"Error calling {ai_name.upper()}")


async def call_ai_async(ai_name: str, prompt: str, temperature: float = 0.7) -> str:
    """Async wrapper for call_ai."""
    return call_ai(ai_name, prompt, temperature)


def call_multiple_ais(prompt: str, ai_list: Optional[List[str]] = None, temperature: float = 0.7) -> Dict[str, str]:
    """Call multiple AIs with the same prompt and return all responses."""
    from core.ai_clients import get_available_ai_clients
    
    if ai_list is None:
        ai_list = list(get_available_ai_clients().keys())
    
    responses = {}
    
    # Create async tasks for concurrent execution
    async def gather_responses():
        tasks = []
        for ai_name in ai_list:
            if get_ai_client(ai_name):
                tasks.append((ai_name, call_ai_async(ai_name, prompt, temperature)))
        
        # Execute all tasks concurrently
        for ai_name, task in tasks:
            try:
                responses[ai_name] = await task
            except Exception as e:
                responses[ai_name] = format_error_response(e, f"Error calling {ai_name}")
        
        return responses
    
    # Run async gathering
    try:
        # Try to get existing event loop
        loop = asyncio.get_running_loop()
        # We're already in async context, create a task
        return asyncio.run_coroutine_threadsafe(gather_responses(), loop).result()
    except RuntimeError:
        # No event loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(gather_responses())
        finally:
            loop.close()


def get_ai_model_info(ai_name: str) -> Dict[str, Any]:
    """Get information about an AI model."""
    client = get_ai_client(ai_name)
    if not client:
        return {"available": False, "error": "Not configured"}
    
    info = {
        "available": True,
        "provider": ai_name
    }
    
    if ai_name == 'gemini':
        info["model"] = "gemini-1.5-flash"
    elif hasattr(client, 'model_name'):
        info["model"] = client.model_name
    else:
        # Default models
        model_map = {
            'grok': 'grok-beta',
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4o-mini',
            'openrouter': 'anthropic/claude-3.5-sonnet'
        }
        info["model"] = model_map.get(ai_name, "unknown")
    
    return info
"""
LLM Utilities

Simple utility functions for the LLM system.
These are lightweight helpers, not complex business logic.
"""

import os
import time
import asyncio
from typing import Optional, List, Any


def get_api_key_for_provider(provider_name: str, prefer_numbered: bool = True) -> Optional[str]:
    """
    Get API key for a specific provider from environment variables.
    
    Args:
        provider_name: Name of the provider ("gemini", "openai", "claude")
        prefer_numbered: Whether to prefer numbered keys (for rotation)
        
    Returns:
        API key string or None if not found
    """
    key_mappings = {
        'gemini': {
            'numbered': ['GEMINI_API_KEY1', 'GEMINI_API_KEY2', 'GEMINI_API_KEY3', 
                        'GEMINI_API_KEY4', 'GEMINI_API_KEY5'],
            'single': ['GEMINI_API_KEY', 'GOOGLE_GEMINI_API_KEY']
        },
        'openai': {
            'numbered': ['OPENAI_API_KEY1', 'OPENAI_API_KEY2'],
            'single': ['OPENAI_API_KEY']
        },
        'claude': {
            'numbered': ['CLAUDE_API_KEY1', 'CLAUDE_API_KEY2'],
            'single': ['CLAUDE_API_KEY', 'ANTHROPIC_API_KEY']
        }
    }
    
    if provider_name not in key_mappings:
        return None
        
    keys_to_try = []
    provider_keys = key_mappings[provider_name]
    
    if prefer_numbered:
        keys_to_try.extend(provider_keys.get('numbered', []))
        keys_to_try.extend(provider_keys.get('single', []))
    else:
        keys_to_try.extend(provider_keys.get('single', []))
        keys_to_try.extend(provider_keys.get('numbered', []))
    
    # Try each key in order
    for key_name in keys_to_try:
        key_value = os.getenv(key_name)
        if key_value:
            return key_value
            
    return None


def validate_api_keys() -> dict:
    """
    Validate API keys for all supported providers.
    
    Returns:
        Dictionary with validation results for each provider
    """
    results = {}
    providers = ['gemini', 'openai', 'claude']
    
    for provider in providers:
        key = get_api_key_for_provider(provider)
        results[provider] = {
            'available': key is not None,
            'key_preview': f"...{key[-8:]}" if key else None
        }
    
    return results


def format_provider_model_name(provider: str, model: str) -> str:
    """
    Format provider and model name for display.
    
    Args:
        provider: Provider name
        model: Model name
        
    Returns:
        Formatted string
    """
    return f"{provider.upper()}:{model}"


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation for text.
    This is a simple approximation: ~4 characters per token.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


def truncate_for_logging(text: str, max_length: int = 100) -> str:
    """
    Truncate text for logging purposes.
    
    Args:
        text: Text to truncate
        max_length: Maximum length to keep
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text:
        return ""
        
    if len(text) <= max_length:
        return text
        
    return f"{text[:max_length]}..."


def calculate_request_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate approximate request cost.
    This is a rough estimation based on typical pricing.
    
    Args:
        provider: Provider name
        model: Model name  
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    # Rough pricing per 1M tokens (as of 2024)
    pricing = {
        'gemini': {
            'gemini-2.5-flash': {'input': 0.075, 'output': 0.30},
            'gemini-2.5-pro': {'input': 1.25, 'output': 5.00},
            'gemini-1.5-pro': {'input': 1.25, 'output': 5.00}
        },
        'openai': {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'o1-preview': {'input': 15.00, 'output': 60.00}
        },
        'claude': {
            'claude-3-5-haiku': {'input': 0.25, 'output': 1.25},
            'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00}
        }
    }
    
    if provider not in pricing or model not in pricing[provider]:
        return 0.0  # Unknown model
        
    model_pricing = pricing[provider][model]
    
    input_cost = (input_tokens / 1_000_000) * model_pricing['input']
    output_cost = (output_tokens / 1_000_000) * model_pricing['output']
    
    return round(input_cost + output_cost, 6)


class SimpleTimer:
    """Simple timer for measuring request duration."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return round(end_time - self.start_time, 3)
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def clean_response_for_logging(response: str, max_lines: int = 3) -> str:
    """
    Clean response text for logging (remove excessive whitespace, limit lines).
    
    Args:
        response: Response text
        max_lines: Maximum number of lines to show
        
    Returns:
        Cleaned response text
    """
    if not response:
        return ""
        
    # Split into lines and clean each line
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    
    # Limit number of lines
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    
    return ' | '.join(lines)


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error should be retried.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error should be retried
    """
    error_str = str(error).lower()
    
    # HTTP status codes that should be retried
    retryable_codes = ['429', '500', '502', '503', '504']
    
    # Error messages that indicate temporary issues
    retryable_messages = [
        'rate limit', 'quota exceeded', 'overloaded', 'unavailable',
        'timeout', 'connection', 'network', 'temporary', 'retry'
    ]
    
    # Check for HTTP status codes
    for code in retryable_codes:
        if code in error_str:
            return True
    
    # Check for retryable messages
    for message in retryable_messages:
        if message in error_str:
            return True
            
    return False


async def wait_with_jitter(base_delay: float, attempt: int, max_delay: float = 60.0) -> None:
    """
    Wait with exponential backoff and jitter.
    
    Args:
        base_delay: Base delay in seconds
        attempt: Current attempt number (0-based)
        max_delay: Maximum delay in seconds
    """
    import random
    
    # Exponential backoff: base_delay * (2 ^ attempt)
    delay = base_delay * (2 ** attempt)
    
    # Add jitter (±25% of the delay)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    delay = delay + jitter
    
    # Cap at max_delay
    delay = min(delay, max_delay)
    
    # Ensure minimum delay
    delay = max(delay, 0.1)
    
    await asyncio.sleep(delay)


def debug_print(message: str, enabled: bool = None) -> None:
    """
    Print debug message if debug mode is enabled.
    
    Args:
        message: Message to print
        enabled: Override debug mode setting
    """
    if enabled is None:
        enabled = os.getenv('LLM_DEBUG', 'false').lower() in ('true', '1', 'yes')
    
    if enabled:
        print(f"🐛 LLM: {message}")


# Simple usage examples for testing
if __name__ == "__main__":
    print("🔧 LLM Utils Test")
    print("=" * 40)
    
    # Test API key validation
    keys = validate_api_keys()
    print("\n🔑 API Key Status:")
    for provider, status in keys.items():
        print(f"  {provider}: {'✅' if status['available'] else '❌'} {status.get('key_preview', 'Not found')}")
    
    # Test cost calculation
    cost = calculate_request_cost('gemini', 'gemini-2.5-flash', 1000, 500)
    print(f"\n💰 Estimated cost: ${cost}")
    
    # Test timer
    with SimpleTimer() as timer:
        import time
        time.sleep(0.1)
    print(f"\n⏱️  Timer test: {timer.elapsed()}s")
    
    print("\n✅ Utils test completed")
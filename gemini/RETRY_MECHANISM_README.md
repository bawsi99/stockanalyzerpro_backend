# Retry Mechanism for Gemini API

This document describes the comprehensive retry mechanism implemented for the Gemini API client to handle Google server errors gracefully.

## Overview

The retry mechanism provides intelligent error handling with exponential backoff, jitter, and error-specific retry strategies for different types of Google API errors.

## Components

### 1. RetryConfig
Configuration class for retry behavior:
- `max_retries`: Maximum number of retry attempts (default: 3)
- `base_delay`: Initial delay in seconds (default: 1.0)
- `max_delay`: Maximum delay cap in seconds (default: 60.0)
- `exponential_base`: Base for exponential backoff (default: 2.0)
- `jitter`: Whether to add random jitter (default: True)

### 2. RetryMechanism
Core retry logic with intelligent error classification:
- `retry_with_backoff()`: Decorator for synchronous functions
- `async_retry_with_backoff()`: Decorator for asynchronous functions
- `is_retryable_error()`: Determines if an error should be retried
- `calculate_delay()`: Computes delay with exponential backoff and jitter

### 3. GoogleAPIErrorHandler
Specialized error classification for Google API errors:
- `classify_error()`: Categorizes errors into types
- `get_retry_config_for_error()`: Returns appropriate retry config for error type

## Error Classification

The system classifies errors into the following categories:

### Retryable Errors
1. **Quota Exceeded**: Rate limiting and quota errors
   - Keywords: 'quota', 'rate limit', 'too many requests'
   - Strategy: Longer delays (5-120 seconds), fewer retries (2)

2. **Server Errors**: Google server issues
   - Keywords: 'internal server error', 'service unavailable', 'bad gateway'
   - Strategy: Moderate delays (2-30 seconds), standard retries (3)

3. **Network Errors**: Connection and timeout issues
   - Keywords: 'connection', 'timeout', 'network', 'socket'
   - Strategy: Aggressive retry (1-60 seconds), more retries (5)

4. **Temporary Errors**: Transient issues
   - Keywords: 'temporary', 'transient', 'retry'
   - Strategy: Standard retry configuration

### Non-Retryable Errors
- **Authentication Errors**: Invalid API keys, unauthorized access
- **Request Errors**: Malformed requests, invalid parameters
- **Permanent Errors**: Issues that won't resolve with retries

## Implementation in GeminiCore

The `GeminiCore` class integrates retry mechanisms for all API calls:

### 1. Synchronous Calls
```python
def call_llm(self, prompt: str, model: str = "gemini-2.5-flash", enable_code_execution: bool = False):
    def api_call():
        # API call logic
        pass
    
    return self._make_api_call_with_retry(api_call, "LLM call")
```

### 2. Asynchronous Calls
```python
async def call_llm_with_image(self, prompt: str, image, model: str = "gemini-2.5-flash", enable_code_execution: bool = False):
    def api_call():
        # API call logic
        pass
    
    async def async_api_call():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._make_api_call_with_retry(api_call, "LLM call with image"))
    
    retry_func = RetryMechanism.async_retry_with_backoff(async_api_call, retry_config, "LLM call with image")
    return await retry_func()
```

## Retry Strategy

### Exponential Backoff
Delay increases exponentially: `delay = min(base_delay * (exponential_base ^ attempt), max_delay)`

### Jitter
Random jitter prevents thundering herd: `jitter = random.uniform(0, 0.1 * delay)`

### Error-Specific Configurations
- **Quota Errors**: 2 retries, 5-120s delays, 3x exponential base
- **Server Errors**: 3 retries, 2-30s delays, 2x exponential base  
- **Network Errors**: 5 retries, 1-60s delays, 2x exponential base
- **Default**: 3 retries, 1-30s delays, 2x exponential base

## Usage Examples

### Basic Usage
```python
from gemini.error_utils import RetryMechanism, RetryConfig

# Default retry configuration
retry_config = RetryConfig()

# Apply retry decorator
@RetryMechanism.retry_with_backoff(retry_config, "API call")
def my_api_function():
    # Your API call here
    pass
```

### Custom Configuration
```python
# Custom retry configuration
custom_config = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=3.0,
    jitter=True
)

@RetryMechanism.retry_with_backoff(custom_config, "Custom API call")
def my_custom_api_function():
    # Your API call here
    pass
```

### Async Usage
```python
@RetryMechanism.async_retry_with_backoff(retry_config, "Async API call")
async def my_async_api_function():
    # Your async API call here
    pass
```

## Testing

Run the test script to verify retry behavior:
```bash
cd backend
python test_retry_mechanism.py
```

The test script simulates different error scenarios:
1. Success (no retries)
2. Quota errors (longer delays)
3. Server errors (moderate delays)
4. Network errors (aggressive retry)
5. Permanent errors (no retry)
6. Error classification verification

## Benefits

1. **Improved Reliability**: Handles transient Google API errors automatically
2. **Intelligent Retry**: Different strategies for different error types
3. **Rate Limiting**: Respects Google API rate limits with exponential backoff
4. **Graceful Degradation**: Falls back gracefully when retries are exhausted
5. **Comprehensive Logging**: Detailed logging for debugging and monitoring
6. **Async Support**: Full support for both sync and async operations

## Monitoring

The retry mechanism provides detailed logging:
- `[RETRY]` prefix for all retry-related messages
- Error classification information
- Retry attempt counts and delays
- Final success/failure status

## Configuration

Retry behavior can be customized by modifying:
- `RetryConfig` parameters in `error_utils.py`
- Error classification keywords in `GoogleAPIErrorHandler`
- Retry strategies in `get_retry_config_for_error()`

## Best Practices

1. **Monitor Retry Patterns**: Watch for excessive retries indicating underlying issues
2. **Adjust Configurations**: Tune retry parameters based on your usage patterns
3. **Handle Failures Gracefully**: Always have fallback mechanisms for when retries are exhausted
4. **Log Appropriately**: Use the built-in logging for debugging and monitoring
5. **Test Error Scenarios**: Regularly test with different error types to ensure proper handling 
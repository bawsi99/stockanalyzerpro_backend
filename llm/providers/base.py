"""
Base LLM Provider Interface

This module defines the abstract base class that all LLM providers must implement.
Currently focused on Gemini, but designed to be extensible for future providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, Tuple
import asyncio


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Each provider must implement the core methods for text generation
    and image processing.
    """
    
    def __init__(self, model: str, api_key: str = None, **kwargs):
        """
        Initialize the provider with model and API key.
        
        Args:
            model: Specific model name (e.g., "gemini-2.5-flash", "gpt-4o")
            api_key: API key for the provider
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def generate_text(self, 
                           prompt: str, 
                           enable_code_execution: bool = True,
                           max_retries: int = 3,
                           track_tokens: bool = True,
                           **kwargs) -> Tuple[str, Optional[Any]]:
        """
        Generate text response from prompt only.
        
        Args:
            prompt: Input text prompt
            enable_code_execution: Whether to enable code execution tools
            max_retries: Maximum number of retry attempts
            track_tokens: Whether to track token usage
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Tuple of (generated_text, token_usage_data)
            
        Raises:
            Exception: If generation fails after all retries
        """
        pass
    
    @abstractmethod
    async def generate_with_images(self,
                                  prompt: str,
                                  images: List[Any],
                                  enable_code_execution: bool = True,
                                  max_retries: int = 3,
                                  track_tokens: bool = True,
                                  **kwargs) -> Tuple[str, Optional[Any]]:
        """
        Generate text response from prompt and images.
        
        Args:
            prompt: Input text prompt
            images: List of images (PIL Images, bytes, or paths)
            enable_code_execution: Whether to enable code execution tools
            max_retries: Maximum number of retry attempts
            track_tokens: Whether to track token usage
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Tuple of (generated_text, token_usage_data)
            
        Raises:
            Exception: If generation fails after all retries
        """
        pass
    
    @abstractmethod
    def extract_text(self, response: Any) -> str:
        """
        Extract text from provider-specific response object.
        
        Args:
            response: Raw response from the provider API
            
        Returns:
            Extracted text or empty string if no text found
        """
        pass
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an error is retryable (rate limits, server errors, etc.).
        
        Args:
            error: Exception to check
            
        Returns:
            True if the error is retryable, False otherwise
        """
        error_str = str(error).lower()
        retryable_patterns = [
            '503', '429', '500', '502', '504',
            'overloaded', 'unavailable', 'timeout', 
            'connection', 'network', 'rate limit'
        ]
        return any(pattern in error_str for pattern in retryable_patterns)
    
    async def _retry_with_backoff(self, 
                                 operation,
                                 max_retries: int = 3,
                                 base_delay: float = 1.0) -> Any:
        """
        Execute operation with exponential backoff retry logic.
        
        Args:
            operation: Async function to execute
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds (will be multiplied by 2^attempt)
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If operation fails after all retries
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)
                    print(f"🔄 Retry attempt {attempt + 1}/{max_retries} after {delay}s delay...")
                    await asyncio.sleep(delay)
                
                result = await operation()
                
                if attempt > 0:
                    print(f"✅ Retry successful on attempt {attempt + 1}/{max_retries}")
                
                return result
                
            except Exception as ex:
                last_exception = ex
                
                if not self._is_retryable_error(ex) or attempt == max_retries - 1:
                    if attempt == max_retries - 1:
                        print(f"❌ Max retries ({max_retries}) reached. Giving up.")
                    break
                else:
                    print(f"⚠️ Retryable error on attempt {attempt + 1}/{max_retries}: {type(ex).__name__}")
        
        # Re-raise the last exception if we get here
        if last_exception:
            raise last_exception
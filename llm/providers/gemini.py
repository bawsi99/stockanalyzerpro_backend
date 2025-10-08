"""
Gemini Provider Implementation

This module provides a clean implementation of Google's Gemini API.
Extracted and simplified from the legacy backend/gemini/gemini_core.py.
"""

import os
import time
import asyncio
import io
from typing import List, Any, Optional, Tuple, Dict
from google import genai
from google.genai import types

try:
    # Try relative imports first (when used as package)
    from .base import BaseLLMProvider
    from ..key_manager import get_key_for_agent, KeyStrategy
    from ..token_counter import track_llm_usage, TokenUsageData
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)
    
    from base import BaseLLMProvider
    from key_manager import get_key_for_agent, KeyStrategy
    from token_counter import track_llm_usage, TokenUsageData


class GeminiProvider(BaseLLMProvider):
    """
    Clean Gemini API provider implementation.
    
    Supports:
    - Text-only generation
    - Multi-modal generation (text + images)  
    - Code execution tools
    - Automatic retries with exponential backoff
    - Multiple Gemini models (flash, pro, etc.)
    """
    
    def __init__(self, 
                 model: str = "gemini-2.5-flash", 
                 api_key: str = None, 
                 agent_name: str = None,
                 api_key_strategy: str = "round_robin",
                 api_key_index: Optional[int] = None,
                 **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            model: Gemini model name ("gemini-2.5-flash", "gemini-2.5-pro", etc.)
            api_key: Explicit API key (if None, will use key manager)
            agent_name: Agent name for key rotation tracking
            api_key_strategy: Strategy for key selection ("round_robin", "agent_specific", "single")
            api_key_index: Specific key index for agent_specific strategy
            **kwargs: Additional configuration
        """
        super().__init__(model, api_key, **kwargs)
        self.agent_name = agent_name or "unknown_agent"
        
        # Get API key using the new key manager
        if api_key:
            self.api_key = api_key
            self.debug_info = f"[{self.agent_name}] Using EXPLICIT key for gemini: ...{api_key[-8:]}"
        else:
            # Use key manager with strategy
            strategy = KeyStrategy(api_key_strategy)
            self.api_key, self.debug_info = get_key_for_agent(
                provider="gemini",
                agent_name=self.agent_name,
                strategy=strategy,
                key_index=api_key_index
            )
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Provide it as a parameter or set "
                "GEMINI_API_KEY1-5 or GEMINI_API_KEY environment variable."
            )
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        print(f"🔑 Gemini provider initialized - Model: {self.model}")
        print(f"🔑 {self.debug_info}")
    
    
    async def generate_text(self, 
                           prompt: str, 
                           enable_code_execution: bool = True,
                           max_retries: int = 3,
                           track_tokens: bool = True,
                           **kwargs) -> Tuple[str, Optional[TokenUsageData]]:
        """
        Generate text response from prompt only.
        
        This is the clean version of the original call_llm method.
        
        Args:
            prompt: Input text prompt
            enable_code_execution: Whether to enable code execution tools
            max_retries: Maximum number of retry attempts  
            track_tokens: Whether to track token usage
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (generated_text, token_usage_data)
        """
        start_time = time.time()
        
        async def _make_request():
            """Inner function to make the actual API request."""
            # Build the request configuration
            contents = [prompt]
            
            if enable_code_execution:
                config = types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                )
            else:
                config = None
            
            # Make the actual API call (THE CORE REQUEST)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config
            )
            
            return response
        
        # Execute with retry logic
        response = await self._retry_with_backoff(_make_request, max_retries)
        
        # Track token usage if enabled
        token_usage = None
        if track_tokens:
            duration_ms = (time.time() - start_time) * 1000
            token_usage = track_llm_usage(
                response=response,
                agent_name=self.agent_name,
                provider="gemini",
                model=self.model,
                duration_ms=duration_ms,
                success=True
            )
        
        # Extract text from response
        text_response = self.extract_text(response)
        
        return text_response, token_usage
    
    async def generate_with_images(self,
                                  prompt: str,
                                  images: List[Any],
                                  enable_code_execution: bool = True,
                                  max_retries: int = 3,
                                  track_tokens: bool = True,
                                  **kwargs) -> Tuple[str, Optional[TokenUsageData]]:
        """
        Generate text response from prompt and images.
        
        This is the clean version of the original call_llm_with_image(s) methods.
        
        Args:
            prompt: Input text prompt
            images: List of images (PIL Images, bytes, etc.)
            enable_code_execution: Whether to enable code execution tools
            max_retries: Maximum number of retry attempts
            track_tokens: Whether to track token usage
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (generated_text, token_usage_data)
        """
        start_time = time.time()
        
        async def _make_request():
            """Inner function to make the actual API request."""
            # Build contents with prompt and images
            contents = [prompt]
            
            # Convert images to the format Gemini expects
            for image in images:
                image_part = await self._process_image(image)
                contents.append(image_part)
            
            # Build the request configuration
            if enable_code_execution:
                config = types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                )
            else:
                config = None
            
            # Make the actual API call (THE CORE REQUEST)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config
            )
            
            return response
        
        # Execute with retry logic
        response = await self._retry_with_backoff(_make_request, max_retries)
        
        # Track token usage if enabled
        token_usage = None
        if track_tokens:
            duration_ms = (time.time() - start_time) * 1000
            token_usage = track_llm_usage(
                response=response,
                agent_name=self.agent_name,
                provider="gemini",
                model=self.model,
                duration_ms=duration_ms,
                success=True,
                call_metadata={'with_images': True, 'image_count': len(images)}
            )
        
        # Extract text from response
        text_response = self.extract_text(response)
        
        return text_response, token_usage
    
    async def _process_image(self, image: Any) -> types.Part:
        """
        Process an image into the format Gemini expects.
        
        Args:
            image: PIL Image, bytes, or other image format
            
        Returns:
            Gemini Part object for the image
        """
        if hasattr(image, 'save'):  # PIL Image
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            return types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        
        elif isinstance(image, bytes):
            # Already bytes
            return types.Part.from_bytes(data=image, mime_type="image/png")
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def extract_text(self, response: Any) -> str:
        """
        Extract text from Gemini response, handling all possible response formats.
        
        The response can have text in multiple places:
        1. response.text - convenience property that concatenates all text parts
        2. response.candidates[0].content.parts[].text - individual text parts
        3. When code execution is enabled, parts may include executable_code and code_execution_result
        
        Args:
            response: Gemini API response object
            
        Returns:
            Extracted text content
        """
        text_response = ""
        
        # Method 1: Try the convenience .text property first
        # This property automatically concatenates all text parts
        try:
            if hasattr(response, 'text') and response.text:
                # The .text property might return None if there are non-text parts
                # So we check if it's actually populated
                text_response = response.text
                if text_response:
                    return text_response
        except Exception as e:
            # The .text property might throw if there are mixed content types
            print(f"⚠️ response.text accessor failed: {e}")
        
        # Method 2: Manually extract from candidates[].content.parts[]
        # This is more reliable when there are mixed content types
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Iterate through all parts and extract text
                        for part in candidate.content.parts:
                            # Each part can be of different types:
                            # - text: regular text content
                            # - executable_code: code to be executed
                            # - code_execution_result: result of code execution
                            
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                            
                            # Log other part types for debugging
                            if hasattr(part, 'executable_code') and part.executable_code:
                                print(f"🔧 Skipping executable_code part")
                            
                            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                                print(f"📊 Skipping code_execution_result part")
                    
                    # Method 3: Try direct content.text as fallback
                    if not text_response and hasattr(candidate.content, 'text'):
                        content_text = candidate.content.text
                        if content_text:
                            text_response = content_text
        
        # If we still don't have text, try to extract from dict representation
        if not text_response and isinstance(response, dict):
            # Handle dictionary response format
            candidates = response.get('candidates', [])
            if candidates:
                candidate = candidates[0]
                content = candidate.get('content', {})
                parts = content.get('parts', [])
                for part in parts:
                    if isinstance(part, dict) and 'text' in part:
                        text_response += part.get('text', '')
        
        # Post-process: Remove markdown code blocks if present
        if text_response:
            # Remove ```json or ```JSON blocks
            import re
            cleaned = re.sub(r'^```(?:json|JSON)?\s*\n', '', text_response)
            cleaned = re.sub(r'\n```\s*$', '', cleaned)
            return cleaned.strip()
        
        return text_response

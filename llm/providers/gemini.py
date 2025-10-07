"""
Gemini Provider Implementation

This module provides a clean implementation of Google's Gemini API.
Extracted and simplified from the legacy backend/gemini/gemini_core.py.
"""

import os
import time
import asyncio
import io
from typing import List, Any, Optional
from google import genai
from google.genai import types

from .base import BaseLLMProvider


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
                 **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            model: Gemini model name ("gemini-2.5-flash", "gemini-2.5-pro", etc.)
            api_key: Explicit API key (if None, will use key manager)
            agent_name: Agent name for key rotation tracking
            **kwargs: Additional configuration
        """
        super().__init__(model, api_key, **kwargs)
        self.agent_name = agent_name
        
        # Get API key (simplified from original key manager logic)
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._get_api_key_from_env()
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Provide it as a parameter or set "
                "GEMINI_API_KEY1-5 or GEMINI_API_KEY environment variable."
            )
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        print(f"🔑 Gemini provider initialized - Model: {self.model}, Key: ...{self.api_key[-8:]}")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """
        Get API key from environment variables.
        Tries numbered keys first (GEMINI_API_KEY1-5), then fallback.
        """
        # Try numbered keys first (for rotation)
        for i in range(1, 6):
            key = os.environ.get(f"GEMINI_API_KEY{i}")
            if key:
                return key
        
        # Fallback to single key
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY")
    
    async def generate_text(self, 
                           prompt: str, 
                           enable_code_execution: bool = True,
                           max_retries: int = 3,
                           **kwargs) -> str:
        """
        Generate text response from prompt only.
        
        This is the clean version of the original call_llm method.
        """
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
        
        # Extract text from response
        return self.extract_text(response)
    
    async def generate_with_images(self,
                                  prompt: str,
                                  images: List[Any],
                                  enable_code_execution: bool = True,
                                  max_retries: int = 3,
                                  **kwargs) -> str:
        """
        Generate text response from prompt and images.
        
        This is the clean version of the original call_llm_with_image(s) methods.
        """
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
        
        # Extract text from response
        return self.extract_text(response)
    
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
        Extract text from Gemini response.
        
        THIS IS THE EXISTING COMPLEX LOGIC - WE'LL SIMPLIFY THIS NEXT
        Based on the current logic from gemini_core.py lines 92-105, 247-260, 319-333
        """
        text_response = ""
        
        # 1. Try response.text convenience field first
        if hasattr(response, 'text') and response.text:
            text_response = response.text
        
        # 2. If empty, iterate candidates and concatenate all parts[].text
        if not text_response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_response += part.text
                # 3. Fallback to direct content.text if parts missing or empty
                if not text_response and hasattr(candidate.content, 'text'):
                    text_response = candidate.content.text or ""
        
        return text_response
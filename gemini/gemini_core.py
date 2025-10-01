import os
import time
from google import genai
from google.genai import types
import io
from .debug_logger import debug_logger
from .api_key_manager import get_api_key_manager

class GeminiCore:
    def __init__(self, api_key: str = None, agent_name: str = None):
        """Initialize GeminiCore with API key management.
        
        Args:
            api_key: Explicit API key (if provided, uses this instead of rotation)
            agent_name: Name of the agent/component for key rotation tracking
        """
        self.agent_name = agent_name
        
        if api_key:
            # Use explicitly provided key
            self.api_key = api_key
            self.use_key_rotation = False
        else:
            # Use key manager for rotation
            self.key_manager = get_api_key_manager()
            self.api_key = self.key_manager.get_key(agent_name)
            self.use_key_rotation = True
        
        if not self.api_key:
            raise ValueError("Gemini API key is required. Provide it as a parameter or set GEMINI_API_KEY1-5 environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.last_api_call = 0
        self.min_api_interval = 0.1  # 100ms between calls
        self.rate_limiting_enabled = False  # Disabled for maximum performance

    def rate_limit(self):
        if not self.rate_limiting_enabled:
            return
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_api_interval:
            time.sleep(self.min_api_interval - time_since_last_call)
        self.last_api_call = time.time()

    def disable_rate_limiting(self):
        """Disable rate limiting for parallel execution"""
        self.rate_limiting_enabled = False
        print("[ASYNC-OPTIMIZED] Rate limiting disabled for parallel execution")

    def enable_rate_limiting(self):
        """Enable rate limiting"""
        self.rate_limiting_enabled = True
        print("[ASYNC-OPTIMIZED] Rate limiting enabled")

    def call_llm(self, prompt: str, model: str = "gemini-2.5-flash", enable_code_execution: bool = True, return_full_response: bool = False):        
        start_time = time.time()
        
        # Log API request
        debug_logger.log_api_request(
            method="call_llm",
            model=model,
            prompt=prompt,
            enable_code_execution=enable_code_execution
        )
        
        self.rate_limit()
        try:
            if enable_code_execution:
                response = self.client.models.generate_content(
                    model=model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=[prompt]
                )
            
            response_time = time.time() - start_time
            
            # Log API response
            debug_logger.log_api_response(response, response_time, "call_llm")
            
            # Robust text extraction: prefer response.text, then parts[].text, then content.text
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
            
            if return_full_response:
                return response
            else:
                # Return text or empty string (no exception on missing text)
                return text_response
        except Exception as ex:
            response_time = time.time() - start_time
            debug_logger.log_error(ex, "call_llm", prompt)
            # Return empty string instead of raising
            return ""

    async def call_llm_with_code_execution(self, prompt: str, model: str = "gemini-2.5-flash", return_full_response: bool = False, max_retries: int = 3):
        """
        Call the LLM with code execution enabled and extract both text and code results.
        Includes retry logic with exponential backoff for handling temporary errors.
        
        Returns a tuple of (text_response, code_results, execution_results) or full response if return_full_response=True
        """
        start_time = time.time()
        
        # Log API request
        debug_logger.log_api_request(
            method="call_llm_with_code_execution",
            model=model,
            prompt=prompt,
            enable_code_execution=True
        )
        
        # Rate limiting removed for parallel execution
        import asyncio
        loop = asyncio.get_event_loop()
        
        def sync_call():
            if self.rate_limiting_enabled:
                self.rate_limit()
            return self.client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                )
            )
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2^attempt seconds (2s, 4s, 8s)
                    delay = 2 ** attempt
                    print(f"🔄 Retry attempt {attempt + 1}/{max_retries} after {delay}s delay...")
                    await asyncio.sleep(delay)
                
                response = await loop.run_in_executor(None, sync_call)
                response_time = time.time() - start_time
                
                # Log API response and extract results
                text_response, code_results, execution_results = debug_logger.log_api_response(
                    response, response_time, "call_llm_with_code_execution"
                )
                
                if attempt > 0:
                    print(f"✅ Retry successful on attempt {attempt + 1}/{max_retries}")
                
                if return_full_response:
                    return response, code_results, execution_results
                else:
                    return text_response, code_results, execution_results
                
            except Exception as ex:
                last_exception = ex
                response_time = time.time() - start_time
                
                # Check if it's a retryable error (503, 429, connection errors)
                error_str = str(ex).lower()
                is_retryable = any(code in error_str for code in ['503', '429', 'overloaded', 'unavailable', 'timeout', 'connection'])
                
                if is_retryable and attempt < max_retries - 1:
                    print(f"⚠️ Retryable error on attempt {attempt + 1}/{max_retries}: {type(ex).__name__}")
                    debug_logger.log_error(ex, f"call_llm_with_code_execution_attempt_{attempt + 1}", prompt)
                    continue  # Retry
                else:
                    # Non-retryable error or max retries reached
                    if attempt == max_retries - 1:
                        print(f"❌ Max retries ({max_retries}) reached. Giving up.")
                    debug_logger.log_error(ex, "call_llm_with_code_execution", prompt)
                    return "", [], []
        
        # Should not reach here, but just in case
        if last_exception:
            debug_logger.log_error(last_exception, "call_llm_with_code_execution_final", prompt)
        return "", [], []

    async def call_llm_with_image(self, prompt: str, image, model: str = "gemini-2.5-flash", enable_code_execution: bool = True):
        """
        Call the LLM with a prompt and a PIL Image (multi-modal input).
        """
        start_time = time.time()
        
        # Log API request
        debug_logger.log_api_request(
            method="call_llm_with_image",
            model=model,
            prompt=prompt,
            enable_code_execution=enable_code_execution,
            images=[image]
        )
        
        # Rate limiting removed for parallel execution
        import asyncio
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        parts = [prompt, types.Part.from_bytes(data=img_bytes, mime_type="image/png")]
        # If the client supports async, use await; else, run in thread
        loop = asyncio.get_event_loop()
        def sync_call():
            if enable_code_execution:
                return self.client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    )
                )
            else:
                return self.client.models.generate_content(
                    model=model,
                    contents=parts
                )
        try:
            response = await loop.run_in_executor(None, sync_call)
            response_time = time.time() - start_time
            
            # Log API response
            debug_logger.log_api_response(response, response_time, "call_llm_with_image")
            
            if response and hasattr(response, 'candidates') and response.candidates:
                # Extract text from response parts, ignoring executable_code parts
                text_response = ""
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                    else:
                        if hasattr(candidate.content, 'text'):
                            text_response = candidate.content.text
                
                if text_response:
                    return text_response
                else:
                    raise Exception("No text content found in LLM response")
            else:
                raise Exception("Empty or invalid response from LLM (multi-modal)")
        except Exception as ex:
            response_time = time.time() - start_time
            debug_logger.log_error(ex, "call_llm_with_image", prompt)
            raise

    async def call_llm_with_images(self, prompt: str, images: list, model: str = "gemini-2.5-flash", enable_code_execution: bool = True):
        """
        Call the LLM with a prompt and a list of PIL Images (multi-modal input).
        """
        start_time = time.time()
        
        # Log API request
        debug_logger.log_api_request(
            method="call_llm_with_images",
            model=model,
            prompt=prompt,
            enable_code_execution=enable_code_execution,
            images=images
        )
        
        # Rate limiting removed for parallel execution
        import asyncio
        parts = [prompt]
        for image in images:
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            from google.genai import types
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        loop = asyncio.get_event_loop()
        def sync_call():
            if enable_code_execution:
                return self.client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    )
                )
            else:
                return self.client.models.generate_content(
                    model=model,
                    contents=parts
                )
        try:
            response = await loop.run_in_executor(None, sync_call)
            response_time = time.time() - start_time
            
            # Log API response
            debug_logger.log_api_response(response, response_time, "call_llm_with_images")
            
            if response and hasattr(response, 'candidates') and response.candidates:
                # Extract text from response parts, ignoring executable_code parts
                text_response = ""
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                    else:
                        if hasattr(candidate.content, 'text'):
                            text_response = candidate.content.text
                
                if text_response:
                    return text_response
                else:
                    raise Exception("No text content found in LLM response")
            else:
                raise Exception("Empty or invalid response from LLM (multi-modal, multi-image)")
        except Exception as ex:
            response_time = time.time() - start_time
            debug_logger.log_error(ex, "call_llm_with_images", prompt)
            raise 
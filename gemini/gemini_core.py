import os
import time
from google import genai
from google.genai import types
import io

class GeminiCore:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Provide it as a parameter or set GEMINI_API_KEY environment variable.")
        self.client = genai.Client(api_key=self.api_key)
        self.last_api_call = 0
        self.min_api_interval = 0.1  # 100ms between calls

    def rate_limit(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_api_interval:
            time.sleep(self.min_api_interval - time_since_last_call)
        self.last_api_call = time.time()

    def call_llm(self, prompt: str, model: str = "gemini-2.5-flash", enable_code_execution: bool = False):        
        #print(f"[DEBUG] Entering call_llm with prompt: {prompt}")
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
            #print(f"[DEBUG] Leaving call_llm, response: {repr(response)}")
            if response and hasattr(response, 'text'):
                return response.text
            else:
                raise Exception("Empty or invalid response from LLM")
        except Exception as ex:
            #print(f"[DEBUG-ERROR] Exception in call_llm: {ex}")
            import traceback; traceback.print_exc()
            #print(f"[DEBUG-ERROR] Prompt that caused error: {prompt}")
            raise

    def call_llm_with_code_execution(self, prompt: str, model: str = "gemini-2.5-flash"):
        """
        Call the LLM with code execution enabled and extract both text and code results.
        Returns a tuple of (text_response, code_results, execution_results)
        """
        self.rate_limit()
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                )
            )
            
            # Extract text and code execution results
            text_response = ""
            code_results = []
            execution_results = []
            
            # Handle different response structures
            if response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if part.text is not None:
                                text_response += part.text
                            if part.executable_code is not None:
                                code_results.append(part.executable_code.code)
                            if part.code_execution_result is not None:
                                execution_results.append(part.code_execution_result.output)
                    else:
                        # Fallback: try to get text directly
                        if hasattr(candidate.content, 'text'):
                            text_response = candidate.content.text
                else:
                    # Fallback: try to get text from response
                    if hasattr(response, 'text'):
                        text_response = response.text
            else:
                # Fallback: try to get text from response
                if hasattr(response, 'text'):
                    text_response = response.text
            
            return text_response, code_results, execution_results
            
        except Exception as ex:
            import traceback; traceback.print_exc()
            raise

    async def call_llm_with_image(self, prompt: str, image, model: str = "gemini-2.5-flash", enable_code_execution: bool = False):
        """
        Call the LLM with a prompt and a PIL Image (multi-modal input).
        """
        self.rate_limit()
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
        response = await loop.run_in_executor(None, sync_call)
        if response and hasattr(response, 'text'):
            return response.text
        else:
            raise Exception("Empty or invalid response from LLM (multi-modal)")

    async def call_llm_with_images(self, prompt: str, images: list, model: str = "gemini-2.5-flash", enable_code_execution: bool = False):
        """
        Call the LLM with a prompt and a list of PIL Images (multi-modal input).
        """
        self.rate_limit()
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
        response = await loop.run_in_executor(None, sync_call)
        if response and hasattr(response, 'text'):
            return response.text
        else:
            raise Exception("Empty or invalid response from LLM (multi-modal, multi-image)") 
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

    def call_llm(self, prompt: str, model: str = "gemini-2.5-flash"):        
        #print(f"[DEBUG] Entering call_llm with prompt: {prompt}")
        self.rate_limit()
        try:
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

    async def call_llm_with_image(self, prompt: str, image, model: str = "gemini-2.5-flash"):
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
            return self.client.models.generate_content(
                model=model,
                contents=parts
            )
        response = await loop.run_in_executor(None, sync_call)
        if response and hasattr(response, 'text'):
            return response.text
        else:
            raise Exception("Empty or invalid response from LLM (multi-modal)")

    async def call_llm_with_images(self, prompt: str, images: list, model: str = "gemini-2.5-flash"):
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
            return self.client.models.generate_content(
                model=model,
                contents=parts
            )
        response = await loop.run_in_executor(None, sync_call)
        if response and hasattr(response, 'text'):
            return response.text
        else:
            raise Exception("Empty or invalid response from LLM (multi-modal, multi-image)") 
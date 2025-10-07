#!/usr/bin/env python3
"""
Example New Mechanism Using Robust Text Extraction

This demonstrates how to implement a new mechanism that uses the
standardized robust text extraction pattern.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_response_extractor import (
    LLMResponseExtractor, 
    LLMResponseExtractorAsync,
    extract_text_from_response,
    extract_text_safe
)

logger = logging.getLogger(__name__)


class NewAnalysisMechanism:
    """
    Example of a new analysis mechanism that uses robust text extraction.
    
    This shows different patterns for implementing the text extraction:
    1. Basic extraction
    2. Extraction with metadata and debugging
    3. Safe extraction with fallbacks
    4. Async extraction patterns
    """
    
    def __init__(self, llm_client: Any):
        """
        Initialize the mechanism with an LLM client.
        
        Args:
            llm_client: Any LLM client (Gemini, OpenAI, etc.)
        """
        self.llm_client = llm_client
        self.extractor = LLMResponseExtractor()
    
    def analyze_basic(self, prompt: str) -> str:
        """
        Basic analysis with simple text extraction.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The extracted analysis text
        """
        try:
            # Call your LLM (this is a placeholder - replace with actual LLM call)
            raw_response = self._call_llm(prompt)
            
            # Extract text using the robust 3-tier pattern
            analysis_text = extract_text_from_response(raw_response)
            
            return analysis_text or "No analysis generated"
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    def analyze_with_debugging(self, prompt: str) -> Dict[str, Any]:
        """
        Analysis with detailed debugging and metadata.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            dict: Analysis results with extraction metadata
        """
        try:
            # Call your LLM
            raw_response = self._call_llm(prompt)
            
            # Extract text with metadata for debugging
            analysis_text, extraction_metadata = self.extractor.extract_text_with_metadata(raw_response)
            
            # Validate response structure for debugging
            structure_analysis = self.extractor.validate_response_structure(raw_response)
            
            return {
                "analysis": analysis_text or "No analysis generated",
                "success": bool(analysis_text),
                "extraction_metadata": extraction_metadata,
                "response_structure": structure_analysis,
                "debug_info": {
                    "extraction_tier": extraction_metadata.get("extraction_tier"),
                    "extraction_method": extraction_metadata.get("method"),
                    "has_error": extraction_metadata.get("error") is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Debug analysis failed: {e}")
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def analyze_with_fallback(self, prompt: str, fallback_analysis: str = None) -> str:
        """
        Analysis with custom fallback text.
        
        Args:
            prompt: The prompt to send to the LLM
            fallback_analysis: Custom fallback text if extraction fails
            
        Returns:
            str: The extracted analysis or fallback text
        """
        try:
            # Call your LLM
            raw_response = self._call_llm(prompt)
            
            # Use safe extraction with custom fallback
            fallback_text = fallback_analysis or "Analysis temporarily unavailable. Please try again."
            analysis_text = extract_text_safe(raw_response, fallback_text)
            
            return analysis_text
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return fallback_analysis or f"Analysis system error: {str(e)}"
    
    async def analyze_async(self, prompt: str) -> str:
        """
        Async analysis using the async extractor.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The extracted analysis text
        """
        try:
            # Call your async LLM
            raw_response = await self._call_llm_async(prompt)
            
            # Extract text using async pattern
            analysis_text = await LLMResponseExtractorAsync.extract_text(raw_response)
            
            return analysis_text or "No analysis generated"
            
        except Exception as e:
            logger.error(f"Async analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def analyze_async_with_retry(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Async analysis with retry logic and detailed extraction.
        
        This shows the pattern similar to the optimized Gemini system.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retries
            
        Returns:
            dict: Analysis results with retry information
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Call your async LLM
                raw_response = await self._call_llm_async(prompt)
                
                # Extract text with metadata
                analysis_text, metadata = await LLMResponseExtractorAsync.extract_text_with_metadata(raw_response)
                
                # Check if extraction was successful
                if analysis_text and analysis_text.strip():
                    return {
                        "analysis": analysis_text,
                        "success": True,
                        "attempts": attempt + 1,
                        "extraction_metadata": metadata
                    }
                
                # Empty response - log and retry
                logger.warning(f"Empty response on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                last_error = e
                logger.error(f"Analysis attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        return {
            "analysis": "Analysis failed after multiple attempts",
            "success": False,
            "attempts": max_retries,
            "last_error": str(last_error) if last_error else "Unknown error"
        }
    
    def _call_llm(self, prompt: str) -> Any:
        """
        Placeholder for synchronous LLM call.
        Replace this with your actual LLM client call.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Raw LLM response object
        """
        # Example - replace with actual LLM client call
        # return self.llm_client.generate_content(prompt)
        
        # For demo purposes, return a mock response
        class MockResponse:
            def __init__(self):
                self.text = "This is a mock analysis response."
        
        return MockResponse()
    
    async def _call_llm_async(self, prompt: str) -> Any:
        """
        Placeholder for asynchronous LLM call.
        Replace this with your actual async LLM client call.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Raw LLM response object
        """
        # Example - replace with actual async LLM client call
        # return await self.llm_client.generate_content_async(prompt)
        
        # For demo purposes, return a mock response
        class MockAsyncResponse:
            def __init__(self):
                self.text = "This is a mock async analysis response."
        
        return MockAsyncResponse()


# Example usage patterns
async def example_usage():
    """
    Example usage of the new mechanism with robust text extraction.
    """
    # Initialize mechanism with your LLM client
    mechanism = NewAnalysisMechanism(llm_client=None)  # Replace with actual client
    
    prompt = "Analyze the market conditions for XYZ stock"
    
    print("=== Basic Analysis ===")
    result = mechanism.analyze_basic(prompt)
    print(f"Result: {result}")
    
    print("\n=== Analysis with Debugging ===")
    debug_result = mechanism.analyze_with_debugging(prompt)
    print(f"Analysis: {debug_result['analysis']}")
    print(f"Success: {debug_result['success']}")
    print(f"Extraction Tier: {debug_result['extraction_metadata'].get('extraction_tier')}")
    
    print("\n=== Analysis with Fallback ===")
    fallback_result = mechanism.analyze_with_fallback(
        prompt, 
        "Market analysis temporarily unavailable due to system maintenance."
    )
    print(f"Result: {fallback_result}")
    
    print("\n=== Async Analysis ===")
    async_result = await mechanism.analyze_async(prompt)
    print(f"Result: {async_result}")
    
    print("\n=== Async Analysis with Retry ===")
    retry_result = await mechanism.analyze_async_with_retry(prompt)
    print(f"Analysis: {retry_result['analysis']}")
    print(f"Success: {retry_result['success']}")
    print(f"Attempts: {retry_result['attempts']}")


if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())
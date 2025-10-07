#!/usr/bin/env python3
"""
LLM Response Text Extraction Utility

Provides a robust, standardized text extraction mechanism for LLM responses
using the proven 3-tier pattern from the optimized Gemini system.

This utility can be used by any new mechanism that needs to extract text
from various types of LLM API responses.
"""

import logging
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class LLMResponseExtractor:
    """
    Robust text extraction utility for LLM responses.
    
    Implements the proven 3-tier extraction pattern:
    1. Try response.text convenience field first
    2. Iterate candidates and concatenate all parts[].text
    3. Fallback to direct content.text if parts missing or empty
    """
    
    @staticmethod
    def extract_text(response: Any) -> str:
        """
        Extract text from an LLM response using the robust 3-tier pattern.
        
        Args:
            response: The raw LLM API response object
            
        Returns:
            str: Extracted text content or empty string if no text found
        """
        if response is None:
            return ""
            
        text_response = ""
        
        try:
            # Tier 1: Try response.text convenience field first
            if hasattr(response, 'text') and response.text:
                text_response = response.text
                logger.debug("Text extracted via response.text (Tier 1)")
                return text_response
            
            # Tier 2: If empty, iterate candidates and concatenate all parts[].text
            if not text_response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                        
                        if text_response:
                            logger.debug("Text extracted via candidates[0].content.parts (Tier 2)")
                            return text_response
                    
                    # Tier 3: Fallback to direct content.text if parts missing or empty
                    if not text_response and hasattr(candidate.content, 'text'):
                        text_response = candidate.content.text or ""
                        if text_response:
                            logger.debug("Text extracted via candidates[0].content.text (Tier 3)")
                            return text_response
            
            # If no text found through any tier
            if not text_response:
                logger.debug("No text content found in LLM response")
                
            return text_response
            
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return ""
    
    @staticmethod
    def extract_text_with_metadata(response: Any) -> Tuple[str, dict]:
        """
        Extract text and provide metadata about the extraction process.
        
        Args:
            response: The raw LLM API response object
            
        Returns:
            Tuple[str, dict]: (extracted_text, metadata)
        """
        if response is None:
            return "", {"extraction_tier": None, "error": "Response is None"}
            
        text_response = ""
        metadata = {"extraction_tier": None, "error": None}
        
        try:
            # Tier 1: Try response.text convenience field first
            if hasattr(response, 'text') and response.text:
                text_response = response.text
                metadata["extraction_tier"] = 1
                metadata["method"] = "response.text"
                return text_response, metadata
            
            # Tier 2: If empty, iterate candidates and concatenate all parts[].text
            if not text_response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        parts_found = 0
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                                parts_found += 1
                        
                        if text_response:
                            metadata["extraction_tier"] = 2
                            metadata["method"] = "candidates[0].content.parts"
                            metadata["parts_processed"] = parts_found
                            return text_response, metadata
                    
                    # Tier 3: Fallback to direct content.text if parts missing or empty
                    if not text_response and hasattr(candidate.content, 'text'):
                        text_response = candidate.content.text or ""
                        if text_response:
                            metadata["extraction_tier"] = 3
                            metadata["method"] = "candidates[0].content.text"
                            return text_response, metadata
            
            # No text found
            metadata["error"] = "No text content found in any tier"
            return text_response, metadata
            
        except Exception as e:
            metadata["error"] = str(e)
            logger.error(f"Error during text extraction: {e}")
            return "", metadata
    
    @staticmethod
    def validate_response_structure(response: Any) -> dict:
        """
        Validate and analyze the structure of an LLM response.
        Useful for debugging and understanding response formats.
        
        Args:
            response: The raw LLM API response object
            
        Returns:
            dict: Analysis of the response structure
        """
        analysis = {
            "has_text_attr": False,
            "has_candidates": False,
            "candidates_count": 0,
            "has_content": False,
            "has_parts": False,
            "parts_count": 0,
            "text_parts_count": 0,
            "has_content_text": False,
            "response_type": type(response).__name__ if response else "None"
        }
        
        if response is None:
            return analysis
            
        try:
            # Check top-level text attribute
            analysis["has_text_attr"] = hasattr(response, 'text') and response.text
            
            # Check candidates structure
            if hasattr(response, 'candidates') and response.candidates:
                analysis["has_candidates"] = True
                analysis["candidates_count"] = len(response.candidates)
                
                if response.candidates:
                    candidate = response.candidates[0]
                    
                    # Check content structure
                    if hasattr(candidate, 'content') and candidate.content:
                        analysis["has_content"] = True
                        
                        # Check parts structure
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            analysis["has_parts"] = True
                            analysis["parts_count"] = len(candidate.content.parts)
                            
                            # Count text parts
                            text_parts = 0
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts += 1
                            analysis["text_parts_count"] = text_parts
                        
                        # Check direct content text
                        analysis["has_content_text"] = hasattr(candidate.content, 'text') and candidate.content.text
                        
        except Exception as e:
            analysis["analysis_error"] = str(e)
            
        return analysis


class LLMResponseExtractorAsync:
    """
    Async version of the LLM Response Extractor.
    Useful for async workflows and mechanisms.
    """
    
    @staticmethod
    async def extract_text(response: Any) -> str:
        """
        Async wrapper for text extraction.
        
        Args:
            response: The raw LLM API response object
            
        Returns:
            str: Extracted text content or empty string if no text found
        """
        return LLMResponseExtractor.extract_text(response)
    
    @staticmethod
    async def extract_text_with_metadata(response: Any) -> Tuple[str, dict]:
        """
        Async wrapper for text extraction with metadata.
        
        Args:
            response: The raw LLM API response object
            
        Returns:
            Tuple[str, dict]: (extracted_text, metadata)
        """
        return LLMResponseExtractor.extract_text_with_metadata(response)


def extract_text_from_response(response: Any) -> str:
    """
    Convenience function for quick text extraction.
    
    Args:
        response: The raw LLM API response object
        
    Returns:
        str: Extracted text content or empty string if no text found
    """
    return LLMResponseExtractor.extract_text(response)


def extract_text_safe(response: Any, fallback_text: str = "") -> str:
    """
    Safe text extraction with custom fallback.
    
    Args:
        response: The raw LLM API response object
        fallback_text: Text to return if extraction fails
        
    Returns:
        str: Extracted text or fallback text
    """
    extracted = LLMResponseExtractor.extract_text(response)
    return extracted if extracted else fallback_text


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    print("LLM Response Extractor Utility")
    print("This module provides robust text extraction for LLM responses")
    print("\nUsage examples:")
    print("  from backend.utils.llm_response_extractor import extract_text_from_response")
    print("  text = extract_text_from_response(llm_response)")
    print("\nFor detailed extraction:")
    print("  from backend.utils.llm_response_extractor import LLMResponseExtractor")
    print("  text, metadata = LLMResponseExtractor.extract_text_with_metadata(llm_response)")
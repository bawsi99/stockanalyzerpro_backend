#!/usr/bin/env python3
"""
Test Script for LLM Response Text Extractor

This script tests the robust text extraction utility with various
mock response formats to ensure it works correctly.
"""

import asyncio
from llm_response_extractor import (
    LLMResponseExtractor,
    LLMResponseExtractorAsync,
    extract_text_from_response,
    extract_text_safe
)


def create_mock_responses():
    """Create various mock response formats for testing."""
    
    # Mock response with direct text attribute (Tier 1)
    class MockResponseTier1:
        def __init__(self):
            self.text = "Response text from tier 1 (direct response.text)"
    
    # Mock response with candidates structure (Tier 2)
    class MockResponseTier2:
        def __init__(self):
            self.text = None  # No direct text
            self.candidates = [MockCandidate()]
    
    class MockCandidate:
        def __init__(self):
            self.content = MockContent()
    
    class MockContent:
        def __init__(self):
            self.parts = [MockPart("Text from tier 2 part 1. "), MockPart("Text from tier 2 part 2.")]
            self.text = None
    
    class MockPart:
        def __init__(self, text):
            self.text = text
    
    # Mock response with content.text fallback (Tier 3)
    class MockResponseTier3:
        def __init__(self):
            self.text = None
            self.candidates = [MockCandidateTier3()]
    
    class MockCandidateTier3:
        def __init__(self):
            self.content = MockContentTier3()
    
    class MockContentTier3:
        def __init__(self):
            self.parts = []  # Empty parts
            self.text = "Response text from tier 3 (content.text fallback)"
    
    # Mock response with no text (should return empty string)
    class MockResponseEmpty:
        def __init__(self):
            self.text = None
            self.candidates = []
    
    return {
        "tier1": MockResponseTier1(),
        "tier2": MockResponseTier2(), 
        "tier3": MockResponseTier3(),
        "empty": MockResponseEmpty(),
        "none": None
    }


def test_basic_extraction():
    """Test basic text extraction functionality."""
    print("=== Testing Basic Text Extraction ===")
    
    responses = create_mock_responses()
    extractor = LLMResponseExtractor()
    
    for name, response in responses.items():
        print(f"\n--- Testing {name.upper()} response ---")
        
        # Test static method
        text = extractor.extract_text(response)
        print(f"Extracted text: '{text}'")
        
        # Test convenience function
        text_conv = extract_text_from_response(response)
        print(f"Convenience function: '{text_conv}'")
        
        # Test safe extraction
        text_safe = extract_text_safe(response, f"Fallback for {name}")
        print(f"Safe extraction: '{text_safe}'")


def test_extraction_with_metadata():
    """Test extraction with metadata and debugging."""
    print("\n\n=== Testing Extraction with Metadata ===")
    
    responses = create_mock_responses()
    extractor = LLMResponseExtractor()
    
    for name, response in responses.items():
        print(f"\n--- Testing {name.upper()} response with metadata ---")
        
        text, metadata = extractor.extract_text_with_metadata(response)
        print(f"Extracted text: '{text}'")
        print(f"Extraction tier: {metadata.get('extraction_tier')}")
        print(f"Extraction method: {metadata.get('method')}")
        print(f"Error: {metadata.get('error')}")
        
        if 'parts_processed' in metadata:
            print(f"Parts processed: {metadata['parts_processed']}")


def test_response_structure_validation():
    """Test response structure validation."""
    print("\n\n=== Testing Response Structure Validation ===")
    
    responses = create_mock_responses()
    extractor = LLMResponseExtractor()
    
    for name, response in responses.items():
        print(f"\n--- Analyzing {name.upper()} response structure ---")
        
        analysis = extractor.validate_response_structure(response)
        print(f"Response type: {analysis['response_type']}")
        print(f"Has text attribute: {analysis['has_text_attr']}")
        print(f"Has candidates: {analysis['has_candidates']}")
        print(f"Candidates count: {analysis['candidates_count']}")
        print(f"Has content: {analysis['has_content']}")
        print(f"Has parts: {analysis['has_parts']}")
        print(f"Parts count: {analysis['parts_count']}")
        print(f"Text parts count: {analysis['text_parts_count']}")
        print(f"Has content text: {analysis['has_content_text']}")


async def test_async_extraction():
    """Test async extraction functionality."""
    print("\n\n=== Testing Async Text Extraction ===")
    
    responses = create_mock_responses()
    
    for name, response in responses.items():
        print(f"\n--- Testing async {name.upper()} response ---")
        
        # Test async method
        text = await LLMResponseExtractorAsync.extract_text(response)
        print(f"Async extracted text: '{text}'")
        
        # Test async with metadata
        text_meta, metadata = await LLMResponseExtractorAsync.extract_text_with_metadata(response)
        print(f"Async with metadata: '{text_meta}'")
        print(f"Metadata tier: {metadata.get('extraction_tier')}")


def main():
    """Run all tests."""
    print("🚀 Testing LLM Response Text Extractor")
    print("=" * 60)
    
    # Run synchronous tests
    test_basic_extraction()
    test_extraction_with_metadata()
    test_response_structure_validation()
    
    # Run async tests
    print("\n" + "=" * 60)
    print("Running async tests...")
    asyncio.run(test_async_extraction())
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
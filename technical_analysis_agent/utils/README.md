# LLM Response Text Extraction Utility

## Overview

This utility provides a robust, standardized text extraction mechanism for LLM responses using the proven 3-tier pattern from the optimized Gemini system. It eliminates "No text content found" errors and ensures consistent text extraction across all LLM integrations.

## Features

- **3-Tier Robust Extraction**: Tries multiple extraction methods automatically
- **Zero Failures**: Never throws exceptions, always returns valid text or empty string
- **Async Support**: Full async/await compatibility
- **Debugging Tools**: Detailed metadata and response structure analysis
- **Easy Integration**: Simple imports and usage patterns

## Installation

The utility is already available in your backend:

```python
from backend.utils.llm_response_extractor import extract_text_from_response
```

## Quick Start

### Basic Usage

```python
from backend.utils.llm_response_extractor import extract_text_from_response

# Your LLM call
response = your_llm_client.generate_content("Analyze the stock market")

# Extract text robustly
analysis_text = extract_text_from_response(response)
```

### With Fallback Text

```python
from backend.utils.llm_response_extractor import extract_text_safe

# Extract with custom fallback
analysis_text = extract_text_safe(
    response, 
    "Analysis temporarily unavailable. Please try again."
)
```

### Async Usage

```python
from backend.utils.llm_response_extractor import LLMResponseExtractorAsync

# Async extraction
analysis_text = await LLMResponseExtractorAsync.extract_text(response)
```

## Advanced Usage

### With Debugging Information

```python
from backend.utils.llm_response_extractor import LLMResponseExtractor

extractor = LLMResponseExtractor()
text, metadata = extractor.extract_text_with_metadata(response)

print(f"Extracted: {text}")
print(f"Used tier: {metadata.get('extraction_tier')}")
print(f"Method: {metadata.get('method')}")
print(f"Error: {metadata.get('error')}")
```

### Response Structure Analysis

```python
from backend.utils.llm_response_extractor import LLMResponseExtractor

extractor = LLMResponseExtractor()
structure = extractor.validate_response_structure(response)

print(f"Response type: {structure['response_type']}")
print(f"Has text: {structure['has_text_attr']}")
print(f"Has candidates: {structure['has_candidates']}")
```

## Integration Patterns

### Pattern 1: Basic New Mechanism

```python
class MyNewMechanism:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def analyze(self, prompt: str) -> str:
        # Your LLM call
        response = self.llm_client.generate_content(prompt)
        
        # Robust extraction
        from backend.utils.llm_response_extractor import extract_text_from_response
        return extract_text_from_response(response)
```

### Pattern 2: With Retry Logic (Like Optimized Gemini)

```python
import asyncio
from backend.utils.llm_response_extractor import LLMResponseExtractor

class MyAdvancedMechanism:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.extractor = LLMResponseExtractor()
    
    async def analyze_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                # Your async LLM call
                response = await self.llm_client.generate_content_async(prompt)
                
                # Extract with metadata for debugging
                text, metadata = self.extractor.extract_text_with_metadata(response)
                
                if text and text.strip():
                    return text
                
                # Log empty response and retry
                print(f"Empty response on attempt {attempt + 1}, retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return "Analysis failed after multiple attempts"
```

### Pattern 3: Primary + Fallback (Sector Agent Style)

```python
from backend.utils.llm_response_extractor import LLMResponseExtractor

class MySectorLikeMechanism:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.extractor = LLMResponseExtractor()
    
    async def analyze(self, prompt: str) -> str:
        # Primary method: Try with code execution
        try:
            response = await self.llm_client.call_with_code_execution(prompt)
            text = self.extractor.extract_text(response)
            
            if text and text.strip():
                return text
        except Exception as e:
            print(f"Primary method failed: {e}")
        
        # Fallback method: Basic call
        try:
            response = await self.llm_client.call_basic(prompt)
            text = self.extractor.extract_text(response)
            return text or "Analysis unavailable"
        except Exception as e:
            print(f"Fallback method failed: {e}")
            return "Analysis system error"
```

## The 3-Tier Pattern

The utility implements the proven 3-tier extraction pattern:

### Tier 1: Direct response.text
```python
if hasattr(response, 'text') and response.text:
    return response.text
```

### Tier 2: Iterate through candidates[0].content.parts
```python
if hasattr(response, 'candidates') and response.candidates:
    candidate = response.candidates[0]
    if hasattr(candidate.content, 'parts'):
        text = ""
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                text += part.text
        return text
```

### Tier 3: Fallback to candidates[0].content.text
```python
if hasattr(candidate.content, 'text'):
    return candidate.content.text or ""
```

## API Reference

### Classes

#### `LLMResponseExtractor`
Static methods for robust text extraction.

- `extract_text(response)` → `str`
- `extract_text_with_metadata(response)` → `Tuple[str, dict]`
- `validate_response_structure(response)` → `dict`

#### `LLMResponseExtractorAsync`
Async wrappers for extraction methods.

- `extract_text(response)` → `str` (async)
- `extract_text_with_metadata(response)` → `Tuple[str, dict]` (async)

### Functions

#### `extract_text_from_response(response)`
Convenience function for quick extraction.

#### `extract_text_safe(response, fallback_text="")`
Safe extraction with custom fallback text.

## Testing

Run the test suite to verify functionality:

```bash
cd backend/utils
python test_extractor.py
```

## Error Handling

The utility is designed to never throw exceptions:

- `None` responses return empty string
- Invalid responses return empty string
- Parsing errors return empty string
- All errors are logged for debugging

## Integration with Existing System

This utility is compatible with:

- ✅ Gemini API responses
- ✅ OpenAI API responses  
- ✅ Any response with `text`, `candidates`, or similar structure
- ✅ Async and sync workflows
- ✅ Existing error handling patterns

## Performance

- **Zero overhead** when text is available in Tier 1
- **Minimal overhead** for Tier 2/3 extraction
- **No network calls** - pure response parsing
- **Memory efficient** - processes responses in-place

## Best Practices

1. **Use the convenience functions** for simple cases
2. **Use the class methods** for debugging and complex scenarios  
3. **Always provide fallback text** for user-facing features
4. **Log extraction metadata** in production for monitoring
5. **Use async methods** in async workflows for consistency

## Migration from Existing Code

### Before (Fragile):
```python
# ❌ Can throw "No text content found" errors
analysis = response.text
```

### After (Robust):
```python
# ✅ Never fails, always returns valid text
from backend.utils.llm_response_extractor import extract_text_from_response
analysis = extract_text_from_response(response)
```

This utility ensures your new mechanisms will be as robust and reliable as the optimized Gemini system!
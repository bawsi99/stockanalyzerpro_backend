#!/usr/bin/env python3
"""
Test script to debug Gemini response text extraction issue
"""

import os
import asyncio
from google import genai
from google.genai import types

async def test_gemini_direct():
    """Test Gemini API directly to debug response structure"""
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY1')
    if not api_key:
        print("❌ No Gemini API key found!")
        return
        
    print(f"🔑 Using API key: ...{api_key[-8:]}")
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Simple test prompt
    prompt = """Analyze this volume data and return a JSON object:
```json
{
  "volume": 1000000,
  "average": 500000,
  "status": "high"
}
```

Return ONLY a JSON object with your analysis."""

    try:
        print("\n📤 Sending request with code execution enabled...")
        
        # Request with code execution
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution)]
        )
        
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=[prompt],
            config=config
        )
        
        print("\n📥 Response received!")
        print(f"Response type: {type(response)}")
        
        # Debug response structure
        print("\n🔍 Response structure:")
        print(f"Has 'text' attribute: {hasattr(response, 'text')}")
        print(f"Has 'candidates' attribute: {hasattr(response, 'candidates')}")
        
        # Try to extract text using response.text
        if hasattr(response, 'text'):
            print(f"\n📝 response.text: {response.text}")
            
        # Check candidates
        if hasattr(response, 'candidates') and response.candidates:
            print(f"\n📋 Number of candidates: {len(response.candidates)}")
            
            for i, candidate in enumerate(response.candidates):
                print(f"\n  Candidate {i}:")
                print(f"    Has 'content': {hasattr(candidate, 'content')}")
                
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    print(f"    Content type: {type(content)}")
                    print(f"    Has 'parts': {hasattr(content, 'parts')}")
                    print(f"    Has 'text': {hasattr(content, 'text')}")
                    
                    if hasattr(content, 'parts') and content.parts:
                        print(f"    Number of parts: {len(content.parts)}")
                        
                        for j, part in enumerate(content.parts):
                            print(f"\n      Part {j}:")
                            print(f"        Type: {type(part)}")
                            
                            # Print all attributes of the part
                            attrs = [attr for attr in dir(part) if not attr.startswith('_')]
                            print(f"        Attributes: {attrs}")
                            
                            if hasattr(part, 'text'):
                                print(f"        Has text: {part.text is not None}")
                                if part.text:
                                    print(f"        Text preview: {part.text[:100]}...")
                                    
                            if hasattr(part, 'executable_code'):
                                print(f"        Has executable_code: {part.executable_code is not None}")
                                
                            if hasattr(part, 'code_execution_result'):
                                print(f"        Has code_execution_result: {part.code_execution_result is not None}")
                                
        # Test text extraction logic
        print("\n🔧 Testing text extraction logic:")
        text_response = ""
        
        # 1. Try response.text convenience field first
        if hasattr(response, 'text') and response.text:
            text_response = response.text
            print(f"✅ Got text from response.text: {len(text_response)} chars")
        
        # 2. If empty, iterate candidates and concatenate all parts[].text
        if not text_response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_response += part.text
                            print(f"✅ Added text from part: {len(part.text)} chars")
                # 3. Fallback to direct content.text if parts missing or empty
                if not text_response and hasattr(candidate.content, 'text'):
                    text_response = candidate.content.text or ""
                    print(f"✅ Got text from content.text: {len(text_response)} chars")
        
        print(f"\n📊 Final extracted text length: {len(text_response)} chars")
        if text_response:
            print(f"📄 Text preview: {text_response[:200]}...")
        else:
            print("❌ No text extracted!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_without_code_execution():
    """Test without code execution to see if that's the issue"""
    
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY1')
    if not api_key:
        print("❌ No Gemini API key found!")
        return
        
    client = genai.Client(api_key=api_key)
    
    prompt = """Return a simple JSON object: {"status": "ok", "value": 42}"""
    
    try:
        print("\n📤 Sending request WITHOUT code execution...")
        
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=[prompt],
            config=None  # No code execution
        )
        
        print(f"\n✅ Response text: {response.text}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def main():
    print("🧪 Testing Gemini Response Text Extraction")
    print("=" * 50)
    
    await test_gemini_direct()
    
    print("\n" + "=" * 50)
    
    await test_without_code_execution()


if __name__ == "__main__":
    asyncio.run(main())
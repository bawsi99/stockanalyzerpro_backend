#!/usr/bin/env python3
"""
Simple Example: How to Send Requests to Gemini using backend/llm

This script shows the most common usage patterns for sending requests
to Gemini LLM using the new backend/llm system.

Usage:
    cd backend/llm/examples
    python simple_usage.py
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from backend/config/.env
from dotenv import load_dotenv
env_path = project_root / "backend" / "config" / ".env"
load_dotenv(env_path)

from backend.llm import get_llm_client


async def example_1_simple_request():
    """Example 1: Simple text request"""
    print("📝 Example 1: Simple Text Request")
    print("-" * 40)
    
    # Create a client using agent configuration
    client = get_llm_client("indicator_agent")
    
    # Your prompt
    prompt = "Explain what RSI indicator means in stock trading in simple terms."
    
    # Send request and get response
    response = await client.generate_text(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")


async def example_2_code_execution():
    """Example 2: Request with code execution"""
    print("🧮 Example 2: Request with Code Execution")
    print("-" * 40)
    
    client = get_llm_client("indicator_agent")
    
    prompt = """
    I have these daily closing prices: [100, 102, 98, 105, 108, 103, 107]
    Calculate the 3-day moving average for the last 3 days using Python code.
    """
    
    # Enable code execution
    response = await client.generate(
        prompt=prompt,
        enable_code_execution=True
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")


async def example_3_different_agents():
    """Example 3: Using different pre-configured agents"""
    print("🎯 Example 3: Different Agent Configurations")
    print("-" * 40)
    
    # Different agents use different models/timeouts
    indicator_client = get_llm_client("indicator_agent")    # Fast model
    risk_client = get_llm_client("risk_agent")             # Pro model
    
    # Same prompt to both agents
    prompt = "What are the key risks in swing trading?"
    
    # Get responses from both
    indicator_response = await indicator_client.generate_text(prompt)
    risk_response = await risk_client.generate_text(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Indicator Agent ({indicator_client.get_provider_info()}): {indicator_response[:100]}...")
    print(f"Risk Agent ({risk_client.get_provider_info()}): {risk_response[:100]}...\n")


async def example_4_custom_configuration():
    """Example 4: Custom provider and model configuration"""
    print("⚙️ Example 4: Custom Configuration")
    print("-" * 40)
    
    # Create client with custom settings
    client = get_llm_client(
        provider="gemini",
        model="gemini-2.5-pro",    # Use the Pro model
        timeout=60,                # 1 minute timeout
        max_retries=2             # Fewer retries
    )
    
    prompt = "Analyze the pros and cons of algorithmic trading strategies."
    
    response = await client.generate_text(prompt)
    
    print(f"Custom client using: {client.get_provider_info()}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response[:200]}...\n")


async def example_5_error_handling():
    """Example 5: Proper error handling"""
    print("🛡️ Example 5: Error Handling")
    print("-" * 40)
    
    client = get_llm_client("indicator_agent")
    
    try:
        # This request has a very short timeout to demonstrate error handling
        response = await client.generate_text(
            prompt="Provide detailed analysis of all major stock indices",
            timeout=1  # Very short timeout - likely to fail
        )
        print(f"Response: {response}")
        
    except asyncio.TimeoutError:
        print("⚠️ Request timed out - this is expected with 1-second timeout")
        
        # Try again with normal timeout
        print("Retrying with normal timeout...")
        response = await client.generate_text(
            prompt="What is MACD indicator?",
            timeout=30
        )
        print(f"Successful response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ Other error occurred: {e}")
    
    print()


async def example_6_real_world_usage():
    """Example 6: Real-world usage in your application"""
    print("🌍 Example 6: Real-World Usage Pattern")
    print("-" * 40)
    
    # This is how you'd typically use it in your actual application
    class StockAnalyzer:
        def __init__(self):
            self.indicator_client = get_llm_client("indicator_agent")
            self.risk_client = get_llm_client("risk_agent")
        
        async def analyze_stock(self, symbol, price, rsi, macd):
            # Create a detailed prompt with your data
            prompt = f"""
            Stock Analysis Request:
            Symbol: {symbol}
            Current Price: ${price}
            RSI: {rsi}
            MACD Signal: {macd}
            
            Provide a brief technical analysis with:
            1. Current trend assessment
            2. Key levels to watch
            3. Short-term outlook
            """
            
            # Get technical analysis
            analysis = await self.indicator_client.generate_text(prompt)
            
            # Get risk assessment
            risk_prompt = f"Assess the risk level for trading {symbol} with RSI at {rsi}"
            risk_assessment = await self.risk_client.generate_text(risk_prompt)
            
            return {
                'symbol': symbol,
                'technical_analysis': analysis,
                'risk_assessment': risk_assessment
            }
    
    # Use the analyzer
    analyzer = StockAnalyzer()
    result = await analyzer.analyze_stock("AAPL", 150, 65, "bullish")
    
    print(f"Analysis for {result['symbol']}:")
    print(f"Technical: {result['technical_analysis'][:150]}...")
    print(f"Risk: {result['risk_assessment'][:100]}...\n")


async def main():
    """Run all examples"""
    print("🚀 backend/llm Usage Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_1_simple_request,
        example_2_code_execution,
        example_3_different_agents,
        example_4_custom_configuration,
        example_5_error_handling,
        example_6_real_world_usage
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"❌ Example failed: {e}\n")
    
    print("=" * 50)
    print("✅ All examples completed!")
    print("\n💡 Key Takeaways:")
    print("1. Use get_llm_client() to create clients")
    print("2. Use await client.generate_text() for simple requests") 
    print("3. Use agent names (e.g., 'indicator_agent') for pre-configured settings")
    print("4. Enable code execution for calculations")
    print("5. Always handle errors appropriately")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test the Volume Anomaly LLM Agent with fixed text extraction
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up to 3.0
sys.path.insert(0, str(project_root))

# Load environment variables from config/.env
from dotenv import load_dotenv
env_path = current_dir / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")

# Import the volume anomaly agent
from backend.agents.volume.volume_anomaly.llm_agent import VolumeAnomalyLLMAgent


async def test_volume_anomaly_with_fix():
    """Test the Volume Anomaly LLM Agent with the fixed text extraction"""
    print("🧪 Testing Volume Anomaly LLM Agent with Text Extraction Fix")
    print("=" * 60)
    
    try:
        # Create the LLM agent
        llm_agent = VolumeAnomalyLLMAgent()
        print(f"✅ Agent created successfully")
        print(f"   Client info: {llm_agent.get_client_info()}")
        
        # Sample analysis data (simulating what the processor would provide)
        sample_analysis_data = {
            "significant_anomalies": [
                {
                    "date": "2024-01-15",
                    "volume_level": 2500000,
                    "z_score": 3.2,
                    "significance": "high",
                    "anomaly_type": "significant_outlier",
                    "likely_cause": "technical_breakout",
                    "price_context": "breakout_up"
                }
            ],
            "current_volume_status": {
                "current_status": "elevated",
                "volume_percentile": 85,
                "z_score": 2.1,
                "vs_mean_ratio": 1.8
            },
            "volume_statistics": {
                "volume_mean": 1200000,
                "volume_std": 400000,
                "volume_cv": 0.33,
                "percentiles": {
                    "percentile_95": 2000000,
                    "percentile_75": 1500000
                }
            },
            "anomaly_patterns": {
                "anomaly_frequency": "medium",
                "anomaly_pattern": "clustered",
                "temporal_clustering": "weekly_pattern",
                "analysis_period_days": 90
            },
            "quality_assessment": {
                "overall_score": 85,
                "detection_quality_score": 32,
                "high_significance_count": 3,
                "data_quality_score": 25
            }
        }
        
        # Create a dummy chart image (just a small PNG)
        import base64
        # 1x1 transparent PNG
        dummy_image = base64.b64decode(
            b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
        )
        
        print("\n📊 Sending analysis request...")
        print(f"   Stock symbol: TEST_STOCK")
        print(f"   Image size: {len(dummy_image)} bytes")
        
        # Call the analysis method
        response = await llm_agent.analyze_volume_anomaly(
            chart_image=dummy_image,
            analysis_data=sample_analysis_data,
            symbol="TEST_STOCK"
        )
        
        print(f"\n✅ Analysis completed!")
        print(f"   Response length: {len(response) if response else 0} characters")
        
        if response:
            # Try to parse as JSON to verify it's valid
            import json
            try:
                parsed = json.loads(response)
                print(f"   ✅ Response is valid JSON")
                print(f"   Keys in response: {list(parsed.keys())}")
            except json.JSONDecodeError:
                print(f"   ⚠️  Response is not valid JSON")
                print(f"   Response preview: {response[:200]}...")
        else:
            print(f"   ❌ Empty response!")
            
        return response is not None and len(response) > 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test"""
    print("🚀 Volume Anomaly LLM Agent Test (with Text Extraction Fix)")
    print("=" * 70)
    
    # Check if API keys are available
    api_keys_found = []
    for i in range(1, 6):
        if os.getenv(f'GEMINI_API_KEY{i}'):
            api_keys_found.append(f'GEMINI_API_KEY{i}')
    if os.getenv('GEMINI_API_KEY'):
        api_keys_found.append('GEMINI_API_KEY')
        
    if api_keys_found:
        print(f"✅ Found {len(api_keys_found)} API key(s): {', '.join(api_keys_found)}")
    else:
        print("⚠️  No API keys found in environment")
        print("   Make sure to set GEMINI_API_KEY or GEMINI_API_KEY1-5")
        # Continue anyway to test the setup
    
    # Run the test
    success = await test_volume_anomaly_with_fix()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Test PASSED! Text extraction is working correctly.")
        print("✅ The empty response issue should now be fixed.")
    else:
        print("❌ Test FAILED! Check the error messages above.")
        
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
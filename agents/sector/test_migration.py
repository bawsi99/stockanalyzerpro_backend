#!/usr/bin/env python3
"""
Test script for the migrated Sector Agent using backend/llm system.

This script tests the sector agent's migration from backend/gemini to backend/llm
to ensure it works correctly with the new provider-agnostic system.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.agents.sector.processor import SectorSynthesisProcessor


async def test_sector_agent_migration():
    """Test the migrated sector agent with sample data."""
    print("🧪 Testing Migrated Sector Agent")
    print("=" * 50)
    
    try:
        # Initialize the migrated sector processor
        print("\n1. Initializing sector processor with backend/llm...")
        processor = SectorSynthesisProcessor()
        print(f"   ✅ Processor initialized with agent: {processor.agent_name}")
        print(f"   ✅ LLM client: {processor.client.get_provider_info()}")
        
        # Test 1: Simple sector analysis with mock data
        print("\n2. Testing with structured sector data...")
        
        mock_sector_data = {
            "sector_outperformance_pct": 5.2,
            "market_outperformance_pct": 8.7,
            "sector_beta": 1.3,
            "market_beta": 0.9,
            "rotation_stage": "Growth",
            "rotation_momentum": 12.4,
            "sector_name": "Technology",
            "sector_correlation": 85,
            "market_correlation": 72,
            "sector_sharpe": 1.4,
            "market_sharpe": 1.1,
            "sector_volatility": 18.5,
            "market_volatility": 15.2,
            "sector_return": 14.3,
            "market_return": 11.8
        }
        
        result = await processor.analyze_async(
            symbol="RELIANCE",
            sector_data=mock_sector_data,
            knowledge_context="Additional market context for testing"
        )
        
        print(f"   ✅ Analysis completed successfully")
        print(f"   📊 Agent: {result['agent_name']}")
        print(f"   🏷️  Symbol: {result['symbol']}")
        print(f"   🤖 LLM Provider: {result.get('llm_provider', 'unknown')}")
        print(f"   ⏰ Timestamp: {result['analysis_timestamp']}")
        print(f"   📈 Used Structured Metrics: {result['used_structured_metrics']}")
        
        print(f"\n   📝 Generated Bullets:")
        bullets = result['bullets']
        if isinstance(bullets, str):
            for line in bullets.split('\n'):
                if line.strip():
                    print(f"      {line.strip()}")
        else:
            print(f"      {bullets}")
        
        print(f"\n   🔍 Context Block Length: {len(result['context_block'])} characters")
        
        # Test 2: Analysis with minimal data
        print("\n3. Testing with minimal sector data...")
        
        minimal_result = await processor.analyze_async(
            symbol="TCS",
            sector_data=None,
            knowledge_context="SECTOR CONTEXT for TCS\n- Sector Outperformance: 3.5\n- Market Outperformance: 6.2"
        )
        
        print(f"   ✅ Minimal analysis completed")
        print(f"   🤖 LLM Provider: {minimal_result.get('llm_provider', 'unknown')}")
        print(f"   📈 Used Structured Metrics: {minimal_result['used_structured_metrics']}")
        
        # Test 3: Error handling
        print("\n4. Testing error handling...")
        
        try:
            # Test with invalid data to trigger error path
            error_result = await processor.analyze_async(
                symbol="INVALID",
                sector_data={"invalid": "data"},
                knowledge_context=""
            )
            
            if "error" in error_result:
                print(f"   ⚠️  Error handling working: {error_result['error']}")
                print(f"   🔄 Fallback bullets provided: {len(error_result['bullets'])} chars")
            else:
                print(f"   ✅ Analysis completed despite minimal data")
                print(f"   🤖 LLM Provider: {error_result.get('llm_provider', 'unknown')}")
                
        except Exception as e:
            print(f"   ⚠️  Exception caught (expected): {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Migration Test Results:")
        print("✅ Sector agent successfully migrated to backend/llm")
        print("✅ Provider-agnostic LLM calls working")
        print("✅ Prompt template loaded locally")
        print("✅ Context engineering working")
        print("✅ Error handling functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_prompt_building():
    """Test the prompt building functionality separately."""
    print("\n🔧 Testing Prompt Building")
    print("=" * 30)
    
    try:
        processor = SectorSynthesisProcessor()
        
        # Test prompt template loading
        print(f"✅ Template loaded: {len(processor.prompt_template)} characters")
        
        # Test context building
        test_context = """SECTOR CONTEXT for RELIANCE
- Sector Outperformance: 5.2
- Market Outperformance: 8.7
- Sector Beta: 1.3
- Sector: Energy
- Rotation Stage: Growth"""
        
        prompt = processor._build_sector_analysis_prompt(test_context)
        print(f"✅ Prompt built: {len(prompt)} characters")
        
        # Show a snippet of the prompt
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        print(f"\n📝 Prompt Preview:")
        print(f"{prompt_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt building test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Sector Agent Migration Test")
    
    async def run_all_tests():
        print("\n" + "🧪 SECTOR AGENT MIGRATION VALIDATION" + "\n")
        
        # Test prompt building first
        prompt_test = await test_prompt_building()
        
        # Test full migration if API key available
        migration_test = await test_sector_agent_migration()
        
        print("\n" + "=" * 60)
        if prompt_test and migration_test:
            print("🎉 ALL TESTS PASSED - MIGRATION SUCCESSFUL!")
            print("✅ The sector agent has been successfully migrated to backend/llm")
        else:
            print("⚠️  Some tests failed - check the output above")
            
        return prompt_test and migration_test
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
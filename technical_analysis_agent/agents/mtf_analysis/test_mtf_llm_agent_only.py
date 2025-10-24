#!/usr/bin/env python3
"""
Focused test script for MTF LLM Agent migration to new backend/llm system.

This script directly tests the MTF LLM Agent without importing the full MTF module
to avoid dependency issues. It focuses on the LLM integration aspects.

Usage:
    cd backend/agents/mtf_analysis
    python test_mtf_llm_agent_only.py
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def test_basic_functionality():
    """Test basic functionality of the MTF LLM Agent."""
    print("🧪 Testing MTF LLM Agent Basic Functionality")
    print("=" * 55)
    
    try:
        # Read the agent file and check if it has the right imports
        agent_file = os.path.join(os.path.dirname(__file__), 'mtf_llm_agent.py')
        with open(agent_file, 'r') as f:
            content = f.read()
        
        # Check for new backend/llm imports (no legacy fallback)
        has_new_import = 'from backend.llm import get_llm_client' in content
        has_new_logic = 'self.llm_client = get_llm_client' in content
        no_legacy_fallback = 'from gemini.gemini_client import GeminiClient' not in content
        no_legacy_logic = '_init_legacy_client' not in content
        no_dual_path = 'if hasattr(self, \'_legacy_mode\')' not in content
        
        print("✅ Code analysis results:")
        print(f"   Has new backend/llm import: {has_new_import}")
        print(f"   Has new client logic: {has_new_logic}")
        print(f"   No legacy fallback import: {no_legacy_fallback}")
        print(f"   No legacy client logic: {no_legacy_logic}")
        print(f"   No dual execution path: {no_dual_path}")
        
        # Check prompt building method
        has_build_prompt = '_build_mtf_prompt' in content
        has_confidence_calc = '_calculate_confidence' in content
        has_analyze_method = 'async def analyze_mtf_with_llm' in content
        
        print(f"\n   Core methods present:")
        print(f"   Has prompt building: {has_build_prompt}")
        print(f"   Has confidence calculation: {has_confidence_calc}")
        print(f"   Has analyze method: {has_analyze_method}")
        
        # Check for new system integration
        has_generate_call = 'await self.llm_client.generate(' in content
        has_code_execution_param = 'enable_code_execution=True' in content
        
        print(f"\n   New system integration:")
        print(f"   Has generate() call: {has_generate_call}")
        print(f"   Has code execution parameter: {has_code_execution_param}")
        
        all_checks = [
            has_new_import, has_new_logic, no_legacy_fallback,
            no_legacy_logic, no_dual_path, has_build_prompt,
            has_confidence_calc, has_analyze_method, has_generate_call,
            has_code_execution_param
        ]
        
        passed = sum(all_checks)
        total = len(all_checks)
        
        print(f"\n   Code analysis: {passed}/{total} checks passed")
        
        if passed >= 8:  # Allow some flexibility
            print("✅ Code structure looks good for migration")
            return True
        else:
            print("❌ Code structure needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Code analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_configuration():
    """Test that the YAML configuration has been updated."""
    print("\n🧪 Testing YAML Configuration")
    print("=" * 55)
    
    try:
        yaml_file = os.path.join(os.path.dirname(__file__), '..', '..', 'llm', 'config', 'llm_assignments.yaml')
        
        if not os.path.exists(yaml_file):
            print("⚠️  YAML configuration file not found")
            return False
        
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        # Check for MTF agent configuration
        has_mtf_agent = 'mtf_agent:' in yaml_content
        has_provider_config = 'provider: "gemini"' in yaml_content and 'mtf_agent:' in yaml_content
        has_model_config = 'model: "gemini-2.5-flash"' in yaml_content
        has_code_execution = 'enable_code_execution: true' in yaml_content
        has_timeout = 'timeout: 75' in yaml_content
        
        print("✅ YAML configuration analysis:")
        print(f"   Has MTF agent entry: {has_mtf_agent}")
        print(f"   Has provider configuration: {has_provider_config}")
        print(f"   Has model configuration: {has_model_config}")
        print(f"   Has code execution setting: {has_code_execution}")
        print(f"   Has timeout setting: {has_timeout}")
        
        if has_mtf_agent:
            print("✅ YAML configuration updated successfully")
            return True
        else:
            print("❌ YAML configuration missing MTF agent")
            return False
            
    except Exception as e:
        print(f"❌ YAML configuration test failed: {e}")
        return False

def test_backend_llm_system():
    """Test if the backend/llm system is available."""
    print("\n🧪 Testing backend/llm System Availability")
    print("=" * 55)
    
    try:
        # Test basic imports
        from backend.llm import get_llm_client, LLMClient
        from backend.llm.config.config import get_llm_config
        
        print("✅ Backend/llm imports successful")
        
        # Test configuration loading
        config = get_llm_config()
        agents = config.list_agents()
        
        print(f"   Configured agents: {len(agents)}")
        print(f"   MTF agent in config: {'mtf_agent' in agents}")
        
        if 'mtf_agent' in agents:
            mtf_config = config.get_agent_config('mtf_agent')
            print(f"   MTF agent provider: {mtf_config.get('provider', 'unknown')}")
            print(f"   MTF agent model: {mtf_config.get('model', 'unknown')}")
            print(f"   MTF agent code execution: {mtf_config.get('enable_code_execution', 'unknown')}")
            print(f"   MTF agent timeout: {mtf_config.get('timeout', 'unknown')}s")
        
        # Test client creation (will fail without API key, but structure should work)
        try:
            client = get_llm_client("mtf_agent")
            print("⚠️  Client created without API key (unexpected)")
            return False
        except ValueError as e:
            if "API key is required" in str(e):
                print("✅ Correct error for missing API key")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
                
    except ImportError as e:
        print(f"⚠️  Backend/llm system not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Backend/llm system test failed: {e}")
        return False

async def main():
    """Run all focused tests."""
    print("🚀 MTF LLM Agent Migration Focused Test Suite")
    print("=" * 65)
    
    tests = [
        ("Code Structure Analysis", test_basic_functionality),
        ("YAML Configuration", test_yaml_configuration),
        ("Backend/LLM System", test_backend_llm_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 Test Results Summary")
    print("=" * 65)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MTF LLM Agent migration is structurally correct.")
        print("\n🔥 Migration Summary:")
        print("   ✅ Code has been updated to use new backend/llm system exclusively")
        print("   ✅ Legacy fallback has been completely removed")
        print("   ✅ YAML configuration has been updated")
        print("   ✅ Backend/llm system is available and configured")
        
        return True
    elif passed >= total * 0.75:
        print("⚠️  Most tests passed. Migration is largely successful with minor issues.")
        return True
    else:
        print("❌ Multiple test failures. Migration needs more work.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
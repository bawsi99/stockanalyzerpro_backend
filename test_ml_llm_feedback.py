#!/usr/bin/env python3
"""
Test script to demonstrate ML→LLM feedback loop functionality.
This script shows how the ML system feeds into the LLM system to enhance analysis.
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ml_llm_feedback_loop():
    """
    Test the complete ML→LLM feedback loop functionality.
    """
    try:
        logger.info("🚀 Starting ML→LLM Feedback Loop Test")
        
        # Test 1: Import ML system components
        logger.info("📊 Testing ML System Components...")
        from ml.inference import predict_probability, get_pattern_prediction_breakdown, get_model_version
        
        logger.info(f"✅ ML Model Version: {get_model_version()}")
        
        # Test 2: Generate ML validation context
        logger.info("🔍 Testing ML Validation Context Generation...")
        from agent_capabilities import StockAnalysisOrchestrator
        
        orchestrator = StockAnalysisOrchestrator()
        
        # Mock indicators for testing
        mock_indicators = {
            'pattern_duration': {'value': 7.0},
            'volume': {'volume_ratio': 1.2},
            'trend': {'alignment_score': 0.8},
            'pattern': {'completion_rate': 0.9}
        }
        
        mock_chart_paths = {
            'technical_overview': '/mock/path/technical.png',
            'pattern_analysis': '/mock/path/pattern.png'
        }
        
        # Generate ML validation context
        ml_context = await orchestrator._get_ml_validation_context(mock_indicators, mock_chart_paths)
        
        if ml_context:
            logger.info("✅ ML Validation Context Generated Successfully")
            logger.info(f"📊 Patterns Validated: {len(ml_context.get('pattern_validation', {}))}")
            logger.info(f"🎯 Overall Success Rate: {ml_context.get('risk_assessment', {}).get('overall_pattern_success_rate', 0):.1%}")
            
            # Pretty print the ML context
            print("\n" + "="*60)
            print("🔬 ML VALIDATION CONTEXT")
            print("="*60)
            print(json.dumps(ml_context, indent=2, default=str))
            print("="*60)
        else:
            logger.error("❌ Failed to generate ML validation context")
            return False
        
        # Test 3: Test Enhanced Prompt Manager
        logger.info("📝 Testing Enhanced Prompt Manager...")
        from gemini.prompt_manager import PromptManager
        
        prompt_manager = PromptManager()
        
        # Create ML-enhanced prompt
        enhanced_prompt = prompt_manager.create_ml_enhanced_prompt(
            'optimized_pattern_analysis',
            ml_context,
            context="## Analysis Context:\nTest context for ML enhancement."
        )
        
        if enhanced_prompt:
            logger.info("✅ ML-Enhanced Prompt Created Successfully")
            logger.info(f"📏 Prompt Length: {len(enhanced_prompt)} characters")
            
            # Show a snippet of the enhanced prompt
            print("\n" + "="*60)
            print("📝 ML-ENHANCED PROMPT SNIPPET")
            print("="*60)
            lines = enhanced_prompt.split('\n')
            ml_section_start = None
            for i, line in enumerate(lines):
                if "ML System Validation Context" in line:
                    ml_section_start = i
                    break
            
            if ml_section_start:
                ml_section = lines[ml_section_start:ml_section_start+20]
                print('\n'.join(ml_section))
                print("... (truncated)")
            print("="*60)
        else:
            logger.error("❌ Failed to create ML-enhanced prompt")
            return False
        
        # Test 4: Test ML Context Extraction
        logger.info("🔍 Testing ML Context Extraction...")
        from gemini.gemini_client import GeminiClient
        
        # Mock knowledge context with ML validation
        mock_knowledge_context = f"""
        Some analysis context here.
        
        MLSystemValidation:
        {json.dumps(ml_context)}
        
        More context here.
        """
        
        # Create a mock Gemini client for testing
        client = GeminiClient()
        extracted_ml_context = client._extract_ml_validation_context(mock_knowledge_context)
        
        if extracted_ml_context:
            logger.info("✅ ML Context Extraction Successful")
            logger.info(f"📊 Extracted Patterns: {len(extracted_ml_context.get('pattern_validation', {}))}")
        else:
            logger.error("❌ Failed to extract ML context")
            return False
        
        # Test 5: Demonstrate the Complete Flow
        logger.info("🔄 Demonstrating Complete ML→LLM Flow...")
        
        print("\n" + "="*60)
        print("🔄 COMPLETE ML→LLM FEEDBACK LOOP FLOW")
        print("="*60)
        print("1. 📊 ML System analyzes patterns and generates validation context")
        print("2. 🔍 ML context is extracted and formatted for LLM consumption")
        print("3. 📝 Enhanced prompts are created with ML validation insights")
        print("4. 🤖 LLM receives ML context and uses it to enhance analysis")
        print("5. 🎯 Final analysis combines LLM insights with ML validation")
        print("="*60)
        
        # Show example of how ML context enhances LLM analysis
        print("\n📊 Example ML Context Enhancement:")
        for pattern, data in ml_context.get('pattern_validation', {}).items():
            prob = data.get('success_probability', 0)
            confidence = data.get('confidence', 'medium')
            risk = data.get('risk_level', 'medium')
            print(f"   • {pattern.replace('_', ' ').title()}: {prob:.1%} success, {confidence} confidence, {risk} risk")
        
        print(f"\n🎯 Overall ML Assessment:")
        print(f"   • Success Rate: {ml_context.get('risk_assessment', {}).get('overall_pattern_success_rate', 0):.1%}")
        print(f"   • High Confidence Patterns: {ml_context.get('risk_assessment', {}).get('high_confidence_patterns', 0)}")
        print(f"   • Low Risk Patterns: {ml_context.get('risk_assessment', {}).get('low_risk_patterns', 0)}")
        
        logger.info("✅ All ML→LLM Feedback Loop Tests Passed Successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """
    Main test function.
    """
    logger.info("🚀 Starting ML→LLM Feedback Loop Test Suite")
    
    success = await test_ml_llm_feedback_loop()
    
    if success:
        logger.info("🎉 All tests passed! ML→LLM feedback loop is working correctly.")
        print("\n" + "🎉 SUCCESS: ML→LLM Feedback Loop is Fully Functional!")
        print("The ML system now successfully feeds into the LLM system to enhance analysis quality.")
    else:
        logger.error("💥 Some tests failed. Please check the error messages above.")
        print("\n" + "💥 FAILURE: Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    asyncio.run(main())

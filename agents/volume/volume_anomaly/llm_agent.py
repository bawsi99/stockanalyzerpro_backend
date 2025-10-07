#!/usr/bin/env python3
"""
Volume Anomaly Detection Agent - LLM Integration Module

This module handles LLM integration for the Volume Anomaly Detection Agent using the new backend/llm system.
It manages all prompt processing internally, eliminating dependencies on backend/gemini components.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

# Import the new LLM system
from llm import get_llm_client


class VolumeAnomalyLLMAgent:
    """
    LLM integration agent for Volume Anomaly Detection
    
    This agent handles all prompt processing internally and uses the new backend/llm system
    directly, eliminating dependencies on backend/gemini components like PromptManager,
    ContextEngineer, and GeminiClient.
    """
    
    def __init__(self):
        """Initialize the LLM agent with backend/llm client"""
        try:
            # Use the volume_anomaly_agent configuration from llm_assignments.yaml
            self.llm_client = get_llm_client("volume_anomaly_agent")
            print("✅ Volume Anomaly LLM Agent initialized with backend/llm")
        except Exception as e:
            print(f"❌ Failed to initialize Volume Anomaly LLM Agent: {e}")
            self.llm_client = None
    
    def build_analysis_prompt(self, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build the complete analysis prompt with all context directly in the agent.
        
        This replaces the backend/gemini PromptManager and template system by building
        the full prompt programmatically within the agent itself.
        
        Args:
            analysis_data: Processed data from VolumeAnomalyProcessor
            symbol: Stock symbol being analyzed
            
        Returns:
            Complete prompt string ready for LLM
        """
        # Base prompt - moved from backend/prompts/volume_anomaly_detection.txt
        base_prompt = """You are a Volume Anomaly Detection Specialist. Your sole purpose is to identify and classify statistical volume outliers and irregular patterns.

## Your Specific Task:
Analyze volume patterns to identify statistical anomalies, classify their significance, and detect irregular trading patterns.

## Key Analysis Points:
- Statistical volume outliers (2σ, 3σ, 4σ deviations)
- Volume pattern irregularities and clustering
- Retail-driven volume spikes vs normal fluctuations
- Temporal anomaly patterns and frequency"""

        # Build analysis context from the processor data
        context = self._build_analysis_context(analysis_data, symbol)
        
        # Output format specification - maintained from original template
        output_format = """## Required Output Format:

Output ONLY a valid JSON object. NO PROSE, NO EXPLANATIONS OUTSIDE THE JSON.

```json
{
  "statistical_anomalies": [
    {
      "date": "YYYY-MM-DD",
      "volume_level": 0,
      "z_score": 0.0,
      "significance": "high/medium/low",
      "anomaly_type": "extreme_outlier/significant_outlier/moderate_outlier",
      "likely_cause": "technical_breakout/news_reaction/market_interest/trading_activity",
      "price_context": "breakout_up/breakout_down/consolidation/normal_move"
    }
  ],
  "anomaly_frequency": "high/medium/low",
  "anomaly_pattern": "clustered/scattered/periodic/irregular",
  "current_volume_status": "elevated/normal/below_average",
  "volume_percentile": 0-100,
  "statistical_summary": {
    "total_outliers": 0,
    "extreme_outliers": 0,
    "significant_outliers": 0
  },
  "confidence_score": 0-100,
  "key_insight": "Brief insight about statistical volume anomalies"
}
```"""

        # Instructions - maintained from original template
        instructions = """## Instructions:
1. Identify statistical volume outliers using z-score analysis
2. Classify anomaly significance based on statistical deviation
3. Focus on retail/market-driven causes (exclude institutional analysis)
4. Assess anomaly patterns and frequency

Focus only on statistical volume outliers and irregular patterns. Ignore institutional activity (handled by separate agent), price trends, or support/resistance analysis."""

        # Combine all sections into the final prompt
        full_prompt = f"""{base_prompt}

## Analysis Context:
{context}

{instructions}

{output_format}"""
        
        return full_prompt
    
    def _build_analysis_context(self, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build the analysis context from processor data.
        
        This replaces the context engineering functionality from backend/gemini
        by directly processing the analysis data into a structured context.
        """
        try:
            # Extract key metrics for context summary
            anomalies = analysis_data.get('significant_anomalies', [])
            current_status = analysis_data.get('current_volume_status', {})
            quality_assessment = analysis_data.get('quality_assessment', {})
            volume_statistics = analysis_data.get('volume_statistics', {})
            anomaly_patterns = analysis_data.get('anomaly_patterns', {})
            
            # Build structured context
            context = f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

VOLUME ANOMALY ANALYSIS DATA:
{json.dumps(analysis_data, indent=2, default=str)}

KEY METRICS SUMMARY:
- Total Anomalies Detected: {len(anomalies)}
- Current Volume Status: {current_status.get('current_status', 'unknown').replace('_', ' ').title()}
- Volume Percentile: {current_status.get('volume_percentile', 0)}th
- Current Z-Score: {current_status.get('z_score', 0):.2f}
- vs Mean Ratio: {current_status.get('vs_mean_ratio', 1.0):.2f}x

VOLUME STATISTICS:
- Mean Volume: {volume_statistics.get('volume_mean', 0):,.0f}
- Standard Deviation: {volume_statistics.get('volume_std', 0):,.0f}
- Coefficient of Variation: {volume_statistics.get('volume_cv', 0):.3f}
- 95th Percentile: {volume_statistics.get('percentiles', {}).get('percentile_95', 0):,.0f}
- 75th Percentile: {volume_statistics.get('percentiles', {}).get('percentile_75', 0):,.0f}

ANOMALY PATTERN ANALYSIS:
- Frequency: {anomaly_patterns.get('anomaly_frequency', 'unknown')}
- Pattern Type: {anomaly_patterns.get('anomaly_pattern', 'unknown')}
- Temporal Clustering: {anomaly_patterns.get('temporal_clustering', 'unknown')}
- Total Analysis Period: {anomaly_patterns.get('analysis_period_days', 0)} days

ANALYSIS QUALITY METRICS:
- Overall Score: {quality_assessment.get('overall_score', 0)}/100
- Detection Quality: {quality_assessment.get('detection_quality_score', 0)}/40
- High Significance Count: {quality_assessment.get('high_significance_count', 0)}
- Data Quality Score: {quality_assessment.get('data_quality_score', 0)}/30

RECENT SIGNIFICANT ANOMALIES:"""
            
            # Add details for recent significant anomalies
            if anomalies:
                for i, anomaly in enumerate(anomalies[:5]):  # Show top 5 anomalies
                    if 'error' not in anomaly:
                        context += f"""
  {i+1}. {anomaly.get('date', 'N/A')}: 
     - Volume: {anomaly.get('volume_level', 0):,}
     - Z-Score: {anomaly.get('z_score', 0):.2f}
     - Significance: {anomaly.get('significance', 'unknown')}
     - Type: {anomaly.get('anomaly_type', 'unknown')}
     - Likely Cause: {anomaly.get('likely_cause', 'unknown')}
     - Price Context: {anomaly.get('price_context', 'unknown')}"""
            else:
                context += "\n  No significant anomalies detected in the analysis period."
            
            context += f"""

Please analyze this comprehensive volume anomaly data to identify statistical outliers and provide insights about irregular volume patterns for {symbol}."""
            
            return context
            
        except Exception as e:
            # Fallback context if processing fails
            return f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

VOLUME ANOMALY ANALYSIS DATA:
{json.dumps(analysis_data, indent=2, default=str)}

Error processing context: {str(e)}

Please analyze this volume anomaly data to identify statistical outliers."""
    
    async def analyze_volume_anomaly(
        self, 
        chart_image: bytes,
        analysis_data: Dict[str, Any], 
        symbol: str
    ) -> Optional[str]:
        """
        Main analysis method using backend/llm system.
        
        This replaces the old GeminiClient.analyze_volume_agent_specific() method
        by calling backend/llm directly with the complete prompt built internally.
        
        Args:
            chart_image: Chart image bytes from VolumeAnomalyCharts
            analysis_data: Processed data from VolumeAnomalyProcessor
            symbol: Stock symbol being analyzed
            
        Returns:
            LLM analysis response as JSON string, or error response
        """
        if not self.llm_client:
            error_msg = "LLM client not initialized"
            print(f"[VOLUME_ANOMALY_LLM] {error_msg}")
            return f'{{"error": "{error_msg}", "agent": "volume_anomaly", "status": "failed"}}'
        
        try:
            # Build the complete prompt internally (no template system needed)
            prompt = self.build_analysis_prompt(analysis_data, symbol)
            
            print(f"[VOLUME_ANOMALY_LLM] Sending analysis request for {symbol}")
            print(f"[VOLUME_ANOMALY_LLM] Prompt length: {len(prompt)} characters")
            print(f"[VOLUME_ANOMALY_LLM] Chart image size: {len(chart_image)} bytes")
            
            # Call backend/llm with image and prompt
            response = await self.llm_client.generate(
                prompt=prompt,
                images=[chart_image],
                enable_code_execution=True  # Enable statistical calculations
            )
            
            print(f"[VOLUME_ANOMALY_LLM] Analysis completed for {symbol}")
            print(f"[VOLUME_ANOMALY_LLM] Response length: {len(response) if response else 0} characters")
            
            # Validate response is not empty
            if not response or not response.strip():
                error_msg = "Empty response from LLM"
                print(f"[VOLUME_ANOMALY_LLM] {error_msg}")
                return f'{{"error": "{error_msg}", "agent": "volume_anomaly", "status": "failed"}}'
            
            return response
            
        except asyncio.TimeoutError:
            error_msg = f"LLM request timed out for {symbol}"
            print(f"[VOLUME_ANOMALY_LLM] {error_msg}")
            return f'{{"error": "{error_msg}", "agent": "volume_anomaly", "status": "timeout"}}'
            
        except Exception as e:
            error_msg = f"Volume anomaly LLM analysis failed for {symbol}: {str(e)}"
            print(f"[VOLUME_ANOMALY_LLM] {error_msg}")
            import traceback
            print(f"[VOLUME_ANOMALY_LLM] Traceback: {traceback.format_exc()}")
            return f'{{"error": "{error_msg}", "agent": "volume_anomaly", "status": "failed"}}'
    
    def get_client_info(self) -> str:
        """Get information about the LLM client being used"""
        if self.llm_client:
            return self.llm_client.get_provider_info()
        return "No LLM client initialized"


# Test function for the new LLM agent
async def test_volume_anomaly_llm_agent():
    """Test the Volume Anomaly LLM Agent"""
    print("🧪 Testing Volume Anomaly LLM Agent")
    print("=" * 50)
    
    try:
        # Create agent
        llm_agent = VolumeAnomalyLLMAgent()
        print(f"✅ Agent created successfully")
        print(f"   Client info: {llm_agent.get_client_info()}")
        
        # Test prompt building
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
        
        prompt = llm_agent.build_analysis_prompt(sample_analysis_data, "TEST_STOCK")
        print(f"✅ Prompt built successfully")
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Contains required sections: {all(section in prompt for section in ['Analysis Context', 'Instructions', 'Required Output Format'])}")
        
        # Test would require actual chart image for full test
        print(f"✅ Volume Anomaly LLM Agent test completed")
        print(f"   Ready for integration with volume agents orchestrator")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_volume_anomaly_llm_agent())
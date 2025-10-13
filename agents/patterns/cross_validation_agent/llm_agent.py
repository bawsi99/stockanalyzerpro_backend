#!/usr/bin/env python3
"""
Cross-Validation LLM Agent

This module handles LLM integration for cross-validation analysis, providing:
- Intelligent validation result interpretation
- Pattern confidence assessment using AI
- Validation method comparison and insights
- Risk assessment based on validation outcomes
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm import get_llm_client

logger = logging.getLogger(__name__)

class CrossValidationLLMAgent:
    """
    LLM Agent specialized in cross-validation analysis interpretation.
    
    Provides AI-powered insights for:
    - Validation result interpretation and analysis
    - Pattern confidence assessment and recommendations
    - Risk evaluation based on validation outcomes
    - Trading decision support with validation context
    """
    
    def __init__(self):
        self.name = "cross_validation_llm"
        self.version = "1.0.0"
        try:
            # Use the cross_validation_agent configuration from llm_assignments.yaml
            self.llm_client = get_llm_client("cross_validation_agent")
            print("✅ Cross-Validation LLM Agent initialized with backend/llm")
        except Exception as e:
            print(f"❌ Failed to initialize Cross-Validation LLM Agent: {e}")
            self.llm_client = None
    
    async def generate_validation_analysis(
        self, 
        validation_data: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]],
        symbol: str = "STOCK",
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive LLM analysis for cross-validation results.
        
        Args:
            validation_data: Cross-validation analysis results
            detected_patterns: Original patterns that were validated
            symbol: Stock symbol
            market_context: Optional market context information
            
        Returns:
            Dictionary containing LLM analysis and insights
        """
        try:
            logger.info(f"[CROSS_VALIDATION_LLM] Generating analysis for {symbol}")
            
            if not validation_data.get('success', False):
                return self._build_error_result("Cross-validation analysis failed - no data to analyze")
            
            # Build comprehensive prompt
            analysis_prompt = self._build_validation_analysis_prompt(
                validation_data, detected_patterns, symbol, market_context
            )
            
            # Get LLM response
            llm_response = await self._get_llm_response(analysis_prompt, symbol)
            
            # Parse and structure the response
            structured_response = self._parse_llm_response(llm_response, validation_data, symbol)
            
            return structured_response
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Analysis generation failed: {e}")
            return self._build_error_result(str(e))
    
    def _build_validation_analysis_prompt(
        self, 
        validation_data: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]],
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive prompt for cross-validation analysis"""
        
        # Extract key information
        validation_summary = validation_data.get('validation_summary', {})
        validation_scores = validation_data.get('validation_scores', {})
        final_confidence = validation_data.get('final_confidence_assessment', {})
        
        patterns_validated = validation_summary.get('patterns_validated', 0)
        methods_used = validation_summary.get('validation_methods_used', 0)
        overall_score = validation_scores.get('overall_score', 0)
        final_conf_score = final_confidence.get('overall_confidence', 0)
        confidence_level = final_confidence.get('confidence_level', 'unknown')
        
        prompt = f"""# Cross-Validation Analysis Report for {symbol}

## ANALYSIS CONTEXT
You are an expert quantitative analyst specializing in pattern validation and risk assessment. Analyze the cross-validation results for detected chart patterns and provide comprehensive insights for trading decisions.

## STOCK INFORMATION
- **Symbol**: {symbol}
- **Patterns Validated**: {patterns_validated}
- **Validation Methods Used**: {methods_used}
- **Overall Validation Score**: {overall_score:.2f}
- **Final Confidence Level**: {confidence_level} ({final_conf_score:.1%})

## DETECTED PATTERNS SUMMARY
Total patterns originally detected: {len(detected_patterns)}

### Original Pattern Detection:
"""
        
        if detected_patterns:
            for i, pattern in enumerate(detected_patterns, 1):
                pattern_name = pattern.get('pattern_name', f'Pattern {i}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                original_reliability = pattern.get('reliability', 'unknown')
                completion = pattern.get('completion_percentage', 0)
                
                prompt += f"""
**{i}. {pattern_name.replace('_', ' ').title()}**
- Type: {pattern_type.title()}
- Original Reliability: {original_reliability.title()}
- Completion: {completion}%
"""
        
        # Add validation method results
        prompt += f"""
## VALIDATION METHODS ANALYSIS

### Method Scores:
"""
        method_scores = validation_scores.get('method_scores', {})
        weights = validation_scores.get('weights_used', {})
        
        for method, score in method_scores.items():
            weight = weights.get(method, 0)
            prompt += f"- **{method.replace('_', ' ').title()}**: {score:.2f} (Weight: {weight:.1%})\n"
        
        # Add detailed validation results
        prompt += f"""

### Detailed Validation Results:

"""
        
        # Statistical Validation
        statistical_validation = validation_data.get('statistical_validation', {})
        if not statistical_validation.get('error'):
            stat_score = statistical_validation.get('overall_statistical_score', 0)
            prompt += f"**Statistical Validation**: {stat_score:.2f}\n"
            validation_results = statistical_validation.get('validation_results', [])
            if validation_results:
                prompt += "- Pattern-specific statistical tests completed\n"
                high_confidence = len([p for p in validation_results if p.get('statistical_confidence') == 'very_high'])
                prompt += f"- {high_confidence} patterns show very high statistical confidence\n"
        
        # Volume Confirmation
        volume_confirmation = validation_data.get('volume_confirmation', {})
        if not volume_confirmation.get('error'):
            vol_score = volume_confirmation.get('overall_volume_score', 0)
            prompt += f"**Volume Confirmation**: {vol_score:.2f}\n"
            confirmation_results = volume_confirmation.get('confirmation_results', [])
            if confirmation_results:
                strong_volume = len([p for p in confirmation_results if p.get('volume_strength') == 'strong'])
                prompt += f"- {strong_volume} patterns show strong volume confirmation\n"
        
        # Time Series Validation
        time_series_validation = validation_data.get('time_series_validation', {})
        if not time_series_validation.get('error'):
            ts_score = time_series_validation.get('overall_time_series_score', 0)
            prompt += f"**Time Series Validation**: {ts_score:.2f}\n"
        
        # Historical Performance
        historical_validation = validation_data.get('historical_validation', {})
        if not historical_validation.get('error'):
            hist_score = historical_validation.get('overall_historical_score', 0)
            prompt += f"**Historical Performance**: {hist_score:.2f}\n"
            performance_results = historical_validation.get('performance_results', [])
            if performance_results:
                excellent_patterns = len([p for p in performance_results if p.get('performance_category') == 'excellent'])
                prompt += f"- {excellent_patterns} patterns have excellent historical performance\n"
        
        # Consistency Analysis
        consistency_analysis = validation_data.get('consistency_analysis', {})
        if not consistency_analysis.get('error'):
            consistency_score = consistency_analysis.get('consistency_score', 0)
            prompt += f"**Pattern Consistency**: {consistency_score:.2f}\n"
            
            conflicts = consistency_analysis.get('pattern_conflicts', [])
            reinforcements = consistency_analysis.get('pattern_reinforcements', [])
            prompt += f"- {len(conflicts)} pattern conflicts detected\n"
            prompt += f"- {len(reinforcements)} pattern reinforcements found\n"
        
        # Alternative Methods
        alternative_validation = validation_data.get('alternative_validation', {})
        if not alternative_validation.get('error'):
            alt_score = alternative_validation.get('overall_alternative_score', 0)
            prompt += f"**Alternative Methods**: {alt_score:.2f}\n"
        
        # Add validation completeness and quality info
        completeness = validation_scores.get('validation_completeness', 0)
        validation_quality = validation_scores.get('validation_quality', 'unknown')
        
        prompt += f"""
## VALIDATION QUALITY ASSESSMENT
- **Validation Completeness**: {completeness:.1%}
- **Validation Quality**: {validation_quality}
- **Methods Successfully Applied**: {len(method_scores)} out of {methods_used}

## CONFIDENCE ASSESSMENT DETAILS
- **Base Validation Score**: {final_confidence.get('base_validation_score', 0):.2f}
- **Method Completeness Factor**: {final_confidence.get('validation_completeness', 0):.2f}
- **Pattern Count Factor**: {final_confidence.get('pattern_count_factor', 1.0):.2f}
- **Final Confidence**: {final_conf_score:.1%}
- **Confidence Category**: {final_confidence.get('confidence_category', 'unknown')}

"""
        
        # Add market context if provided
        if market_context:
            prompt += f"""
## MARKET CONTEXT
- Market Environment: {market_context.get('market_trend', 'unknown')}
- Sector Performance: {market_context.get('sector_performance', 'unknown')}
- Volatility Environment: {market_context.get('volatility_regime', 'unknown')}

"""
        
        # Add pattern-specific validation details if available
        pattern_details = validation_data.get('pattern_validation_details', [])
        if pattern_details:
            prompt += "## PATTERN-SPECIFIC VALIDATION SUMMARY\n"
            for detail in pattern_details:
                pattern_name = detail.get('pattern_name', 'Unknown')
                original_rel = detail.get('original_reliability', 'unknown')
                validation_results = detail.get('validation_results', {})
                
                prompt += f"\n**{pattern_name}** (Original: {original_rel})\n"
                for method, result in validation_results.items():
                    if isinstance(result, dict):
                        score = None
                        for key, value in result.items():
                            if 'score' in key and isinstance(value, (int, float)):
                                score = value
                                break
                        if score is not None:
                            prompt += f"- {method.replace('_', ' ').title()}: {score:.2f}\n"
        
        prompt += f"""
## ANALYSIS REQUIREMENTS

Please provide a comprehensive cross-validation analysis covering:

### 1. VALIDATION RELIABILITY ASSESSMENT
- Overall validation quality and completeness evaluation
- Strengths and weaknesses of the validation process
- Reliability of individual validation methods
- Data quality impact on validation results

### 2. PATTERN CONFIDENCE EVALUATION
- Individual pattern validation assessment
- Confidence level interpretation and implications
- Pattern-specific reliability recommendations
- Validation method agreement analysis

### 3. RISK ASSESSMENT AND WARNINGS
- Validation-based risk evaluation
- Pattern failure probability assessment
- Conflicting validation signals analysis
- Data quality and methodology limitations

### 4. TRADING DECISION FRAMEWORK
- Validation-informed trading recommendations
- Position sizing based on validation confidence
- Entry and exit criteria considering validation results
- Risk management protocols for different confidence levels

### 5. VALIDATION INSIGHTS AND RECOMMENDATIONS
- Key findings from cross-validation analysis
- Most reliable patterns and validation methods
- Areas requiring additional confirmation
- Recommendations for improving validation confidence

## OUTPUT FORMAT
Structure your response as a professional validation analysis report with clear sections and actionable insights. Focus on practical applications while maintaining rigorous analytical standards.

**Important**: Base your analysis strictly on the provided cross-validation data. Highlight both strengths and limitations of the validation process.
"""
        
        return prompt
    
    async def _get_llm_response(self, prompt: str, symbol: str) -> str:
        """Get response from LLM with error handling"""
        try:
            if not self.llm_client:
                return f"Cross-validation analysis for {symbol} could not be completed: LLM client not initialized."
            
            # Make async call to LLM using new backend/llm system
            response, token_usage = await self.llm_client.generate_text(prompt)
            
            if not response or len(response.strip()) < 50:
                return f"Cross-validation analysis for {symbol} could not be completed due to insufficient LLM response."
            
            return response
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] LLM request failed: {e}")
            return f"Cross-validation analysis for {symbol} encountered an error: {str(e)}"
    
    def _parse_llm_response(self, llm_response: str, validation_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse and structure the LLM response"""
        try:
            # Basic response validation
            if "error:" in llm_response.lower() or len(llm_response.strip()) < 100:
                logger.warning(f"[CROSS_VALIDATION_LLM] Poor quality response for {symbol}")
                return self._build_fallback_analysis(validation_data, symbol, llm_response)
            
            # Extract key sections from response (basic parsing)
            sections = self._extract_response_sections(llm_response)
            
            # Build structured result
            result = {
                'success': True,
                'agent_name': self.name,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'cross_validation',
                'confidence_score': validation_data.get('confidence_score', 0),
                
                # Core Analysis Content
                'validation_reliability_assessment': sections.get('validation_reliability', ''),
                'pattern_confidence_evaluation': sections.get('pattern_confidence', ''),
                'risk_assessment': sections.get('risk_assessment', ''),
                'trading_decision_framework': sections.get('trading_framework', ''),
                'validation_insights': sections.get('validation_insights', ''),
                
                # Full Response
                'full_analysis': llm_response,
                'validation_methods_analyzed': len(validation_data.get('validation_scores', {}).get('method_scores', {})),
                'patterns_validated': validation_data.get('validation_summary', {}).get('patterns_validated', 0),
                
                # Quality Metrics
                'response_length': len(llm_response),
                'analysis_quality': self._assess_response_quality(llm_response),
                
                # Validation Context
                'overall_validation_score': validation_data.get('validation_scores', {}).get('overall_score', 0),
                'final_confidence_level': validation_data.get('final_confidence_assessment', {}).get('confidence_level', 'unknown')
            }
            
            logger.info(f"[CROSS_VALIDATION_LLM] Analysis completed for {symbol} ({len(llm_response)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Response parsing failed: {e}")
            return self._build_fallback_analysis(validation_data, symbol, llm_response)
    
    def _extract_response_sections(self, response: str) -> Dict[str, str]:
        """Extract different sections from the LLM response"""
        sections = {}
        
        try:
            # Define section headers to look for
            section_headers = {
                'validation_reliability': ['validation reliability', 'reliability assessment', '1. validation'],
                'pattern_confidence': ['pattern confidence', 'confidence evaluation', '2. pattern'],
                'risk_assessment': ['risk assessment', 'risk evaluation', '3. risk'],
                'trading_framework': ['trading decision', 'decision framework', '4. trading'],
                'validation_insights': ['validation insights', 'key findings', '5. validation']
            }
            
            response_lower = response.lower()
            
            # Extract sections based on headers
            for section_name, headers in section_headers.items():
                section_content = ""
                
                for header in headers:
                    header_pos = response_lower.find(header)
                    if header_pos != -1:
                        # Find the start of content after header
                        content_start = header_pos + len(header)
                        
                        # Find next section or end
                        next_section_pos = len(response)
                        for other_section, other_headers in section_headers.items():
                            if other_section != section_name:
                                for other_header in other_headers:
                                    other_pos = response_lower.find(other_header, content_start + 50)
                                    if other_pos != -1 and other_pos < next_section_pos:
                                        next_section_pos = other_pos
                        
                        # Extract content
                        section_content = response[content_start:next_section_pos].strip()
                        break
                
                sections[section_name] = section_content[:2000]  # Limit length
            
            return sections
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Section extraction failed: {e}")
            return {'full_content': response[:2000]}
    
    def _assess_response_quality(self, response: str) -> str:
        """Assess the quality of the LLM response"""
        try:
            if len(response) < 300:
                return 'poor'
            elif len(response) < 1000:
                return 'fair'
            elif len(response) < 2500:
                return 'good'
            else:
                return 'excellent'
                
        except Exception:
            return 'unknown'
    
    def _build_fallback_analysis(self, validation_data: Dict[str, Any], symbol: str, llm_response: str) -> Dict[str, Any]:
        """Build fallback analysis when LLM response is poor"""
        validation_summary = validation_data.get('validation_summary', {})
        validation_scores = validation_data.get('validation_scores', {})
        final_confidence = validation_data.get('final_confidence_assessment', {})
        
        patterns_validated = validation_summary.get('patterns_validated', 0)
        overall_score = validation_scores.get('overall_score', 0)
        confidence_level = final_confidence.get('confidence_level', 'unknown')
        
        fallback_content = f"""
## Cross-Validation Analysis for {symbol}

### Validation Summary
- {patterns_validated} patterns were validated using multiple methods
- Overall validation score: {overall_score:.2f}
- Final confidence level: {confidence_level}

### Method Results
"""
        
        method_scores = validation_scores.get('method_scores', {})
        for method, score in method_scores.items():
            fallback_content += f"- {method.replace('_', ' ').title()}: {score:.2f}\n"
        
        fallback_content += f"""
### Key Findings
- Validation completeness: {validation_scores.get('validation_completeness', 0):.1%}
- Final confidence: {final_confidence.get('overall_confidence', 0):.1%}
- Recommended approach: {final_confidence.get('recommendation', 'Review validation results carefully')}

*Note: This is a simplified analysis due to limited AI response quality.*
"""
        
        return {
            'success': True,
            'agent_name': self.name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'cross_validation_fallback',
            'confidence_score': validation_data.get('confidence_score', 0),
            'validation_reliability_assessment': fallback_content,
            'pattern_confidence_evaluation': f'Overall confidence level: {confidence_level}',
            'risk_assessment': 'Standard cross-validation risks apply - review individual method results',
            'trading_decision_framework': 'Use validation confidence levels to guide position sizing and risk management',
            'validation_insights': f'{len(method_scores)} validation methods applied with varying results',
            'full_analysis': fallback_content,
            'validation_methods_analyzed': len(method_scores),
            'patterns_validated': patterns_validated,
            'response_length': len(fallback_content),
            'analysis_quality': 'fallback',
            'overall_validation_score': overall_score,
            'final_confidence_level': confidence_level,
            'original_llm_response': llm_response
        }
    
    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """Build error result dictionary"""
        return {
            'success': False,
            'agent_name': self.name,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'cross_validation',
            'confidence_score': 0.0
        }
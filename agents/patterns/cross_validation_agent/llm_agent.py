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
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import sys

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Try importing from backend - use try/except for robustness
try:
    from llm import get_llm_client
except ImportError:
    try:
        from backend.llm import get_llm_client  
    except ImportError:
        # Fallback for testing without LLM
        def get_llm_client(name):
            return None

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
        
        # Load prompt template
        self.template_path = os.path.join(os.path.dirname(__file__), "cross_validation_analysis_template.txt")
        self.prompt_template = None
        self._load_prompt_template()
        
        try:
            # Use the cross_validation_agent configuration from llm_assignments.yaml
            self.llm_client = get_llm_client("cross_validation_agent")
            print("✅ Cross-Validation LLM Agent initialized with backend/llm")
        except Exception as e:
            print(f"❌ Failed to initialize Cross-Validation LLM Agent: {e}")
            self.llm_client = None
    
    def _load_prompt_template(self):
        """Load the prompt template from file"""
        try:
            if os.path.exists(self.template_path):
                with open(self.template_path, 'r', encoding='utf-8') as f:
                    self.prompt_template = f.read()
                logger.info(f"[CROSS_VALIDATION_LLM] Loaded prompt template from {self.template_path}")
            else:
                logger.warning(f"[CROSS_VALIDATION_LLM] Template file not found: {self.template_path}")
                logger.warning(f"[CROSS_VALIDATION_LLM] Using fallback hardcoded template")
                self.prompt_template = None
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Failed to load template: {e}")
            self.prompt_template = None
    
    async def generate_validation_analysis(
        self, 
        validation_data: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]],
        symbol: str = "STOCK",
        market_context: Optional[Dict[str, Any]] = None,
        chart_image_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive LLM analysis for cross-validation results.
        
        Args:
            validation_data: Cross-validation analysis results
            detected_patterns: Original patterns that were validated
            symbol: Stock symbol
            market_context: Optional market context information
            chart_image_path: Optional path to chart image for multimodal analysis
            
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
            
            # Get LLM response (with chart image if available)
            llm_response = await self._get_llm_response(analysis_prompt, symbol, chart_image_bytes)
            
            # Parse and structure the response
            structured_response = self._parse_llm_response(llm_response, validation_data, symbol)
            
            return structured_response
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Analysis generation failed: {e}")
            return self._build_error_result(str(e))
    
    def _build_patterns_section(self, detected_patterns: List[Dict[str, Any]]) -> str:
        """Build the patterns section for template"""
        if not detected_patterns:
            return "No patterns detected."
        
        section = ""
        for i, pattern in enumerate(detected_patterns, 1):
            pattern_name = pattern.get('pattern_name', f'Pattern {i}')
            pattern_type = pattern.get('pattern_type', 'unknown')
            original_reliability = pattern.get('reliability', 'unknown')
            completion = pattern.get('completion_percentage', 0)
            
            # Extract temporal information
            start_date = pattern.get('start_date', 'Unknown')
            end_date = pattern.get('end_date', 'Unknown')
            duration_days = pattern.get('pattern_duration_days', 'Unknown')
            age_days = pattern.get('pattern_age_days', 'Unknown')
            
            # Format temporal display
            if start_date != 'Unknown' and 'T' not in start_date:
                start_display = start_date.split(' ')[0] if ' ' in start_date else start_date
            else:
                start_display = start_date
            
            if end_date != 'Unknown' and 'T' not in end_date:
                end_display = end_date.split(' ')[0] if ' ' in end_date else end_date
            else:
                end_display = end_date
            
            # Duration and age display
            duration_display = f"{duration_days} days" if isinstance(duration_days, (int, float)) else str(duration_days)
            age_display = f"{age_days} days ago" if isinstance(age_days, (int, float)) else str(age_days)
            
            section += f"""
**{i}. {pattern_name.replace('_', ' ').title()}**
- Type: {pattern_type.title()}
- Original Reliability: {original_reliability.title()}
- Completion: {completion}%
- **Formation Period**: {start_display} → {end_display} ({duration_display})
- **Pattern Age**: {age_display}
- **Status**: {pattern.get('completion_status', 'Unknown').title()}
"""
        
        return section
    
    def _build_method_scores_section(self, validation_scores: Dict[str, Any]) -> str:
        """Build method scores section"""
        method_scores = validation_scores.get('method_scores', {})
        weights = validation_scores.get('weights_used', {})
        
        if not method_scores:
            return "No method scores available."
        
        section = ""
        for method, score in method_scores.items():
            weight = weights.get(method, 0)
            section += f"- **{method.replace('_', ' ').title()}**: {score:.2f} (Weight: {weight:.1%})\n"
        
        return section
    
    def _build_validation_results_section(self, validation_data: Dict[str, Any]) -> str:
        """Build validation results section"""
        section = ""
        
        # Statistical Validation
        statistical_validation = validation_data.get('statistical_validation', {})
        if not statistical_validation.get('error'):
            stat_score = statistical_validation.get('overall_statistical_score', 0)
            section += f"**Statistical Validation**: {stat_score:.2f}\n"
            validation_results = statistical_validation.get('validation_results', [])
            if validation_results:
                section += "- Pattern-specific statistical tests completed\n"
                high_confidence = len([p for p in validation_results if p.get('statistical_confidence') == 'very_high'])
                section += f"- {high_confidence} patterns show very high statistical confidence\n"
        
        # Volume Confirmation
        volume_confirmation = validation_data.get('volume_confirmation', {})
        if not volume_confirmation.get('error'):
            vol_score = volume_confirmation.get('overall_volume_score', 0)
            section += f"**Volume Confirmation**: {vol_score:.2f}\n"
            confirmation_results = volume_confirmation.get('confirmation_results', [])
            if confirmation_results:
                strong_volume = len([p for p in confirmation_results if p.get('volume_strength') == 'strong'])
                section += f"- {strong_volume} patterns show strong volume confirmation\n"
        
        # Time Series Validation
        time_series_validation = validation_data.get('time_series_validation', {})
        if not time_series_validation.get('error'):
            ts_score = time_series_validation.get('overall_time_series_score', 0)
            section += f"**Time Series Validation**: {ts_score:.2f}\n"
        
        # Historical Performance
        historical_validation = validation_data.get('historical_validation', {})
        if not historical_validation.get('error'):
            hist_score = historical_validation.get('overall_historical_score', 0)
            section += f"**Historical Performance**: {hist_score:.2f}\n"
            performance_results = historical_validation.get('performance_results', [])
            if performance_results:
                excellent_patterns = len([p for p in performance_results if p.get('performance_category') == 'excellent'])
                section += f"- {excellent_patterns} patterns have excellent historical performance\n"
        
        # Consistency Analysis
        consistency_analysis = validation_data.get('consistency_analysis', {})
        if not consistency_analysis.get('error'):
            consistency_score = consistency_analysis.get('consistency_score', 0)
            section += f"**Pattern Consistency**: {consistency_score:.2f}\n"
            
            conflicts = consistency_analysis.get('pattern_conflicts', [])
            reinforcements = consistency_analysis.get('pattern_reinforcements', [])
            section += f"- {len(conflicts)} pattern conflicts detected\n"
            section += f"- {len(reinforcements)} pattern reinforcements found\n"
        
        # Alternative Methods
        alternative_validation = validation_data.get('alternative_validation', {})
        if not alternative_validation.get('error'):
            alt_score = alternative_validation.get('overall_alternative_score', 0)
            section += f"**Alternative Methods**: {alt_score:.2f}\n"
        
        return section if section else "No validation results available."
    
    def _build_confidence_diagnostics_section(self, final_confidence: Dict[str, Any]) -> str:
        """Build confidence diagnostics section"""
        try:
            factors = final_confidence.get('confidence_factors', {}) or {}
            section = f"""
## CONFIDENCE CALCULATION DIAGNOSTICS
- Preliminary Confidence: {final_confidence.get('preliminary_confidence', 'n/a')}
- Conflict Adjustment: {final_confidence.get('conflict_adjustment', 'n/a')}
- Adjusted (Pre-cap): {final_confidence.get('adjusted_confidence', 'n/a')}
- Max Allowed Confidence (Cap): {final_confidence.get('max_allowed_confidence', 'n/a')}
- Cap Applied: {final_confidence.get('confidence_cap_applied', False)}
- Cap Reason: {final_confidence.get('confidence_cap_reason', 'n/a')}
- Factors:
  - Statistical: {factors.get('statistical_score', 'n/a')}
  - Volume: {factors.get('volume_score', 'n/a')}
  - Weighted base: {factors.get('weighted_base', 'n/a')}
  - Method completeness: {factors.get('method_completeness', 'n/a')}
  - Pattern consistency: {factors.get('pattern_consistency', 'n/a')}
  - Market regime adj: {factors.get('market_regime_adjustment', 'n/a')}
"""
            return section
        except Exception:
            return "\n## CONFIDENCE CALCULATION DIAGNOSTICS\n- Diagnostics not available\n"
    
    def _build_market_structure_section(self, validation_data: Dict[str, Any], market_context: Optional[Dict[str, Any]]) -> str:
        """Build market structure section"""
        # Add market or structure context if provided
        # Prefer param if it's from market_structure_agent; otherwise fallback to validation_data
        ctx_from_results = validation_data.get('market_context') if isinstance(validation_data, dict) else None
        selected_ctx = market_context
        if (not selected_ctx) and ctx_from_results:
            selected_ctx = ctx_from_results
        # If both exist and only results has market_structure_agent source, prefer it
        try:
            if selected_ctx and isinstance(selected_ctx, dict):
                if market_context and ctx_from_results:
                    if (market_context.get('source') != 'market_structure_agent') and (ctx_from_results.get('source') == 'market_structure_agent'):
                        selected_ctx = ctx_from_results
            if selected_ctx and isinstance(selected_ctx, dict):
                sel_source = selected_ctx.get('source')
                logger.info(f"[CROSS_VALIDATION_LLM] Using market context from {'param' if selected_ctx is market_context else 'validation_data'} with source={sel_source}")
        except Exception:
            pass

        if selected_ctx:
            try:
                # Prefer structured context from Market Structure agent
                if isinstance(selected_ctx, dict) and selected_ctx.get('source') == 'market_structure_agent':
                    regime = selected_ctx.get('regime', {}) or {}
                    trend = selected_ctx.get('trend', {}) or {}
                    bos = selected_ctx.get('bos_choch', {}) or {}
                    kl = selected_ctx.get('key_levels', {}) or {}
                    fractal = selected_ctx.get('fractal', {}) or {}

                    # Extract recent break details for enhanced context
                    recent_break = bos.get('recent_structural_break', {})
                    break_details = ""
                    if recent_break:
                        break_date = recent_break.get('date', 'Unknown')
                        break_price = recent_break.get('break_price')
                        prev_level = recent_break.get('previous_level')
                        pct_break = recent_break.get('percentage_break')
                        break_strength = recent_break.get('strength', 'unknown')
                        
                        # Format date for readability
                        if break_date != 'Unknown' and 'T' not in str(break_date):
                            try:
                                from datetime import datetime
                                if '+' in str(break_date):
                                    # Handle timezone
                                    break_date_clean = str(break_date).split('+')[0]
                                else:
                                    break_date_clean = str(break_date).split(' ')[0] if ' ' in str(break_date) else str(break_date)
                            except:
                                break_date_clean = str(break_date)
                        else:
                            break_date_clean = str(break_date)
                            
                        break_details = f"""
- **Recent Structural Break**: {recent_break.get('type', 'unknown').replace('_', ' ').title()}
  - Date: {break_date_clean}
  - Break Price: {break_price} (from previous level: {prev_level})
  - Break Magnitude: {pct_break}% ({break_strength} strength)
  - Days Since Break: {(datetime.now() - datetime.fromisoformat(str(break_date).replace('+05:30', ''))).days if break_date != 'Unknown' and '+' in str(break_date) else 'Unknown'}"""

                    section = f"""
## MARKET STRUCTURE SUMMARY
- **Context Source**: market_structure_agent
- **Market Regime**: {regime.get('regime', 'unknown').title()} (confidence: {regime.get('confidence', 0.0):.0%})
- **Structural Bias**: {selected_ctx.get('structure_bias', 'unknown').title()}
- **Trend Analysis**: {trend.get('direction', 'unknown').title()} | Strength: {trend.get('strength', 'unknown').title()} | Quality: {trend.get('quality', 'unknown').title()}
- **Structure Events**: {bos.get('total_bos_events', 0)} BOS events, {bos.get('total_choch_events', 0)} CHOCH events{break_details}
- **Key Price Levels**:
  - Current Price: {kl.get('current_price', 'NA')}
  - Nearest Support: {kl.get('nearest_support', {}).get('level', 'NA') if kl.get('nearest_support') else 'NA'} ({kl.get('nearest_support', {}).get('distance_pct', 'NA')}% away)
  - Nearest Resistance: {kl.get('nearest_resistance', {}).get('level', 'NA') if kl.get('nearest_resistance') else 'NA'} ({kl.get('nearest_resistance', {}).get('distance_pct', 'NA')}% away)
  - Price Position: {kl.get('price_position_description', 'unknown').replace('_', ' ').title()}
- **Multi-Timeframe Context**: {fractal.get('timeframe_alignment', 'unknown').title()} alignment | Consensus: {fractal.get('trend_consensus', 'unknown').title()}
- **Analysis Timestamp**: {selected_ctx.get('timestamp', 'unknown')}
"""
                    return section
                else:
                    # Generic market context fallback (legacy shape)
                    return f"""
## MARKET CONTEXT
- Market Environment: {selected_ctx.get('market_trend', 'unknown')}
- Sector Performance: {selected_ctx.get('sector_performance', 'unknown')}
- Volatility Environment: {selected_ctx.get('volatility_regime', 'unknown')}
"""
            except Exception as e:
                logger.error(f"[CROSS_VALIDATION_LLM] Failed to build market structure section: {e}")
                return ""
        else:
            # Fallback: include internal regime from validation_data if present
            try:
                mr = validation_data.get('market_regime_analysis', {}) or {}
                if mr:
                    return f"""
## MARKET REGIME (FALLBACK)
- Regime: {mr.get('regime', 'unknown')} (confidence: {mr.get('confidence', 0.0)})
- Volatility: {mr.get('volatility', 'NA')} | Trend Strength: {mr.get('trend_strength', 'NA')} | Price Range: {mr.get('price_range', 'NA')}
"""
                else:
                    return """
## MARKET CONTEXT (NO DATA AVAILABLE)
- No market structure or regime data available for this analysis
- Pattern validation will proceed without market context consideration
"""
            except Exception as e:
                logger.error(f"[CROSS_VALIDATION_LLM] Failed to add fallback market regime: {e}")
                return """
## MARKET CONTEXT (ERROR)
- Market context processing failed completely
"""
    
    def _build_pattern_specific_section(self, validation_data: Dict[str, Any]) -> str:
        """Build pattern-specific validation section"""
        pattern_details = validation_data.get('pattern_validation_details', [])
        if not pattern_details:
            return ""
        
        section = "## PATTERN-SPECIFIC VALIDATION SUMMARY\n"
        for detail in pattern_details:
            pattern_name = detail.get('pattern_name', 'Unknown')
            original_rel = detail.get('original_reliability', 'unknown')
            validation_results = detail.get('validation_results', {})
            
            section += f"\n**{pattern_name}** (Original: {original_rel})\n"
            for method, result in validation_results.items():
                if isinstance(result, dict):
                    score = None
                    for key, value in result.items():
                        if 'score' in key and isinstance(value, (int, float)):
                            score = value
                            break
                    if score is not None:
                        section += f"- {method.replace('_', ' ').title()}: {score:.2f}\n"
        
        return section
    
    def _build_validation_analysis_prompt(
        self, 
        validation_data: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]],
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive prompt for cross-validation analysis using template"""
        
        # Use template if available, otherwise fall back to hardcoded
        if self.prompt_template:
            return self._build_prompt_from_template(validation_data, detected_patterns, symbol, market_context)
        else:
            return self._build_hardcoded_prompt(validation_data, detected_patterns, symbol, market_context)
    
    def _build_prompt_from_template(
        self,
        validation_data: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]],
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt using the loaded template"""
        
        # Extract key information for template variables
        validation_summary = validation_data.get('validation_summary', {})
        validation_scores = validation_data.get('validation_scores', {})
        final_confidence = validation_data.get('final_confidence_assessment', {})
        
        patterns_validated = validation_summary.get('patterns_validated', 0)
        methods_used = validation_summary.get('validation_methods_used', 0)
        overall_score = validation_scores.get('overall_score', 0)
        final_conf_score = final_confidence.get('overall_confidence', 0)
        confidence_level = final_confidence.get('confidence_level', 'unknown')
        
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Build patterns section
        patterns_section = self._build_patterns_section(detected_patterns)
        
        # Build method scores section
        method_scores_section = self._build_method_scores_section(validation_scores)
        
        # Build validation results section
        validation_results_section = self._build_validation_results_section(validation_data)
        
        # Build confidence diagnostics section
        confidence_diagnostics_section = self._build_confidence_diagnostics_section(final_confidence)
        
        # Build market structure section
        market_structure_section = self._build_market_structure_section(validation_data, market_context)
        
        # Build pattern-specific section
        pattern_specific_section = self._build_pattern_specific_section(validation_data)
        
        # Template variable substitution
        template_vars = {
            'symbol': symbol,
            'current_date': current_date,
            'patterns_validated': patterns_validated,
            'methods_used': methods_used,
            'overall_score': overall_score,
            'final_conf_score': final_conf_score,
            'confidence_level': confidence_level,
            'total_patterns': len(detected_patterns),
            'patterns_section': patterns_section,
            'method_scores_section': method_scores_section,
            'validation_results_section': validation_results_section,
            'completeness': validation_scores.get('validation_completeness', 0),
            'validation_quality': validation_scores.get('validation_quality', 'unknown'),
            'methods_applied': len(validation_scores.get('method_scores', {})),
            'methods_total': methods_used,
            'base_validation_score': final_confidence.get('base_validation_score', 0),
            'method_completeness_factor': final_confidence.get('validation_completeness', 0),
            'pattern_count_factor': final_confidence.get('pattern_count_factor', 1.0),
            'final_confidence': final_conf_score,
            'confidence_category': final_confidence.get('confidence_category', 'unknown'),
            'confidence_diagnostics_section': confidence_diagnostics_section,
            'market_structure_section': market_structure_section,
            'pattern_specific_section': pattern_specific_section
        }
        
        # Apply template substitution
        try:
            prompt = self.prompt_template.format(**template_vars)
            return prompt
        except KeyError as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Template variable missing: {e}")
            logger.warning(f"[CROSS_VALIDATION_LLM] Falling back to hardcoded prompt")
            return self._build_hardcoded_prompt(validation_data, detected_patterns, symbol, market_context)
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_LLM] Template formatting failed: {e}")
            return self._build_hardcoded_prompt(validation_data, detected_patterns, symbol, market_context)
    
    def _build_hardcoded_prompt(
        self,
        validation_data: Dict[str, Any],
        detected_patterns: List[Dict[str, Any]],
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Fallback hardcoded prompt (original implementation)"""
        
        # Extract key information
        validation_summary = validation_data.get('validation_summary', {})
        validation_scores = validation_data.get('validation_scores', {})
        final_confidence = validation_data.get('final_confidence_assessment', {})
        
        patterns_validated = validation_summary.get('patterns_validated', 0)
        methods_used = validation_summary.get('validation_methods_used', 0)
        overall_score = validation_scores.get('overall_score', 0)
        final_conf_score = final_confidence.get('overall_confidence', 0)
        confidence_level = final_confidence.get('confidence_level', 'unknown')
        
        # Add current analysis timestamp for context
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Shortened fallback prompt (original was too long)
        prompt = f"""# Cross-Validation Analysis Report for {symbol}

## ANALYSIS CONTEXT
You are an expert quantitative analyst specializing in pattern validation and risk assessment.

**Analysis Date**: {current_date}

## STOCK INFORMATION
- **Symbol**: {symbol}
- **Patterns Validated**: {patterns_validated}
- **Validation Methods Used**: {methods_used}
- **Overall Validation Score**: {overall_score:.2f}
- **Final Confidence Level**: {confidence_level} ({final_conf_score:.1%})

## DETECTED PATTERNS SUMMARY
Total patterns originally detected: {len(detected_patterns)}

## ANALYSIS REQUIREMENTS
Please provide a comprehensive cross-validation analysis covering validation reliability, pattern confidence, risk assessment, and trading decision framework.

## JSON OUTPUT REQUIREMENTS
After the narrative analysis, output exactly one JSON object with key decisions for downstream consumption.

{{
  "symbol": "{symbol}",
  "analysis_date": "{current_date}",
  "overall": {{
    "validation_score": {overall_score:.2f},
    "confidence": {final_conf_score:.2f},
    "confidence_category": "{confidence_level}"
  }}
}}
"""
        
        if detected_patterns:
            for i, pattern in enumerate(detected_patterns, 1):
                pattern_name = pattern.get('pattern_name', f'Pattern {i}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                original_reliability = pattern.get('reliability', 'unknown')
                completion = pattern.get('completion_percentage', 0)
                
                # Extract temporal information
                start_date = pattern.get('start_date', 'Unknown')
                end_date = pattern.get('end_date', 'Unknown')
                duration_days = pattern.get('pattern_duration_days', 'Unknown')
                age_days = pattern.get('pattern_age_days', 'Unknown')
                
                # Format temporal display
                if start_date != 'Unknown' and 'T' not in start_date:
                    start_display = start_date.split(' ')[0] if ' ' in start_date else start_date
                else:
                    start_display = start_date
                
                if end_date != 'Unknown' and 'T' not in end_date:
                    end_display = end_date.split(' ')[0] if ' ' in end_date else end_date
                else:
                    end_display = end_date
                
                # Duration and age display
                duration_display = f"{duration_days} days" if isinstance(duration_days, (int, float)) else str(duration_days)
                age_display = f"{age_days} days ago" if isinstance(age_days, (int, float)) else str(age_days)
                
                prompt += f"""
**{i}. {pattern_name.replace('_', ' ').title()}**
- Type: {pattern_type.title()}
- Original Reliability: {original_reliability.title()}
- Completion: {completion}%
- **Formation Period**: {start_display} → {end_display} ({duration_display})
- **Pattern Age**: {age_display}
- **Status**: {pattern.get('completion_status', 'Unknown').title()}
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
        
        # Add confidence calculation diagnostics (caps and factors)
        try:
            factors = final_confidence.get('confidence_factors', {}) or {}
            prompt += f"""
## CONFIDENCE CALCULATION DIAGNOSTICS
- Preliminary Confidence: {final_confidence.get('preliminary_confidence', 'n/a')}
- Conflict Adjustment: {final_confidence.get('conflict_adjustment', 'n/a')}
- Adjusted (Pre-cap): {final_confidence.get('adjusted_confidence', 'n/a')}
- Max Allowed Confidence (Cap): {final_confidence.get('max_allowed_confidence', 'n/a')}
- Cap Applied: {final_confidence.get('confidence_cap_applied', False)}
- Cap Reason: {final_confidence.get('confidence_cap_reason', 'n/a')}
- Factors:
  - Statistical: {factors.get('statistical_score', 'n/a')}
  - Volume: {factors.get('volume_score', 'n/a')}
  - Weighted base: {factors.get('weighted_base', 'n/a')}
  - Method completeness: {factors.get('method_completeness', 'n/a')}
  - Pattern consistency: {factors.get('pattern_consistency', 'n/a')}
  - Market regime adj: {factors.get('market_regime_adjustment', 'n/a')}
"""
        except Exception:
            pass
        
        # Add market or structure context if provided
        # Prefer param if it's from market_structure_agent; otherwise fallback to validation_data
        ctx_from_results = validation_data.get('market_context') if isinstance(validation_data, dict) else None
        selected_ctx = market_context
        if (not selected_ctx) and ctx_from_results:
            selected_ctx = ctx_from_results
        # If both exist and only results has market_structure_agent source, prefer it
        try:
            if selected_ctx and isinstance(selected_ctx, dict):
                if market_context and ctx_from_results:
                    if (market_context.get('source') != 'market_structure_agent') and (ctx_from_results.get('source') == 'market_structure_agent'):
                        selected_ctx = ctx_from_results
            if selected_ctx and isinstance(selected_ctx, dict):
                sel_source = selected_ctx.get('source')
                logger.info(f"[CROSS_VALIDATION_LLM] Using market context from {'param' if selected_ctx is market_context else 'validation_data'} with source={sel_source}")
        except Exception:
            pass

        if selected_ctx:
            try:
                # Prefer structured context from Market Structure agent
                if isinstance(selected_ctx, dict) and selected_ctx.get('source') == 'market_structure_agent':
                    regime = selected_ctx.get('regime', {}) or {}
                    trend = selected_ctx.get('trend', {}) or {}
                    bos = selected_ctx.get('bos_choch', {}) or {}
                    kl = selected_ctx.get('key_levels', {}) or {}
                    fractal = selected_ctx.get('fractal', {}) or {}

                    # Extract recent break details for enhanced context
                    recent_break = bos.get('recent_structural_break', {})
                    break_details = ""
                    if recent_break:
                        break_date = recent_break.get('date', 'Unknown')
                        break_price = recent_break.get('break_price')
                        prev_level = recent_break.get('previous_level')
                        pct_break = recent_break.get('percentage_break')
                        break_strength = recent_break.get('strength', 'unknown')
                        
                        # Format date for readability
                        if break_date != 'Unknown' and 'T' not in str(break_date):
                            try:
                                from datetime import datetime
                                if '+' in str(break_date):
                                    # Handle timezone
                                    break_date_clean = str(break_date).split('+')[0]
                                else:
                                    break_date_clean = str(break_date).split(' ')[0] if ' ' in str(break_date) else str(break_date)
                            except:
                                break_date_clean = str(break_date)
                        else:
                            break_date_clean = str(break_date)
                            
                        break_details = f"""
- **Recent Structural Break**: {recent_break.get('type', 'unknown').replace('_', ' ').title()}
  - Date: {break_date_clean}
  - Break Price: {break_price} (from previous level: {prev_level})
  - Break Magnitude: {pct_break}% ({break_strength} strength)
  - Days Since Break: {(datetime.now() - datetime.fromisoformat(str(break_date).replace('+05:30', ''))).days if break_date != 'Unknown' and '+' in str(break_date) else 'Unknown'}"""

                    prompt += f"""
## MARKET STRUCTURE SUMMARY
- **Context Source**: market_structure_agent
- **Market Regime**: {regime.get('regime', 'unknown').title()} (confidence: {regime.get('confidence', 0.0):.0%})
- **Structural Bias**: {selected_ctx.get('structure_bias', 'unknown').title()}
- **Trend Analysis**: {trend.get('direction', 'unknown').title()} | Strength: {trend.get('strength', 'unknown').title()} | Quality: {trend.get('quality', 'unknown').title()}
- **Structure Events**: {bos.get('total_bos_events', 0)} BOS events, {bos.get('total_choch_events', 0)} CHOCH events{break_details}
- **Key Price Levels**:
  - Current Price: {kl.get('current_price', 'NA')}
  - Nearest Support: {kl.get('nearest_support', {}).get('level', 'NA') if kl.get('nearest_support') else 'NA'} ({kl.get('nearest_support', {}).get('distance_pct', 'NA')}% away)
  - Nearest Resistance: {kl.get('nearest_resistance', {}).get('level', 'NA') if kl.get('nearest_resistance') else 'NA'} ({kl.get('nearest_resistance', {}).get('distance_pct', 'NA')}% away)
  - Price Position: {kl.get('price_position_description', 'unknown').replace('_', ' ').title()}
- **Multi-Timeframe Context**: {fractal.get('timeframe_alignment', 'unknown').title()} alignment | Consensus: {fractal.get('trend_consensus', 'unknown').title()}
- **Analysis Timestamp**: {selected_ctx.get('timestamp', 'unknown')}
"""
                else:
                    # Generic market context fallback (legacy shape)
                    prompt += f"""
## MARKET CONTEXT
- Market Environment: {selected_ctx.get('market_trend', 'unknown')}
- Sector Performance: {selected_ctx.get('sector_performance', 'unknown')}
- Volatility Environment: {selected_ctx.get('volatility_regime', 'unknown')}
"""
            except Exception as e:
                logger.error(f"[CROSS_VALIDATION_LLM] Failed to build market structure section: {e}")
                # Fallback on error: add minimal context indicator
                try:
                    prompt += f"""
## MARKET CONTEXT (ERROR FALLBACK)
- Market structure analysis encountered an error during prompt generation
- Source: {selected_ctx.get('source', 'unknown') if isinstance(selected_ctx, dict) else 'non-dict'}
- Available keys: {list(selected_ctx.keys()) if isinstance(selected_ctx, dict) else 'none'}
"""
                except Exception:
                    prompt += "\n## MARKET CONTEXT (ERROR FALLBACK)\n- Market context processing failed\n"
        else:
            # Fallback: include internal regime from validation_data if present
            try:
                mr = validation_data.get('market_regime_analysis', {}) or {}
                if mr:
                    prompt += f"""
## MARKET REGIME (FALLBACK)
- Regime: {mr.get('regime', 'unknown')} (confidence: {mr.get('confidence', 0.0)})
- Volatility: {mr.get('volatility', 'NA')} | Trend Strength: {mr.get('trend_strength', 'NA')} | Price Range: {mr.get('price_range', 'NA')}
"""
                else:
                    # No market context at all - add a minimal indicator
                    prompt += """
## MARKET CONTEXT (NO DATA AVAILABLE)
- No market structure or regime data available for this analysis
- Pattern validation will proceed without market context consideration
"""
            except Exception as e:
                logger.error(f"[CROSS_VALIDATION_LLM] Failed to add fallback market regime: {e}")
                prompt += """
## MARKET CONTEXT (ERROR)
- Market context processing failed completely
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
        
        prompt += """
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

### 5. TEMPORAL ANALYSIS AND PATTERN TIMING
- Pattern formation timeline and duration analysis
- Pattern age and recency assessment for trading relevance
- Temporal clustering or pattern overlap analysis
- Formation period quality and market conditions context
- Pattern maturity and potential expiration considerations

### 6. VALIDATION INSIGHTS AND RECOMMENDATIONS
- Key findings from cross-validation analysis
- Most reliable patterns and validation methods
- Areas requiring additional confirmation
- Recommendations for improving validation confidence
- Time-sensitive trading opportunities or warnings

## OUTPUT FORMAT
Structure your response as a professional validation analysis report with clear sections and actionable insights. Focus on practical applications while maintaining rigorous analytical standards.

**Important**: Base your analysis strictly on the provided cross-validation data. Pay special attention to pattern timing and relevance. Highlight both strengths and limitations of the validation process.

## JSON OUTPUT REQUIREMENTS
After the narrative analysis, output exactly one JSON object to summarize key decisions for downstream consumption. Do not include any text after the JSON. Do not wrap the JSON in code fences. If a field is unknown, set it to null.

Schema:
{
  "symbol": string,
  "analysis_date": string,  // ISO date
  "overall": {
    "validation_score": number,       // 0..1
    "confidence": number,             // 0..1 (final)
    "confidence_category": string,    // very_low|low|medium|high|very_high or similar
    "confidence_cap_reason": string|null
  },
  "market_regime": {
    "regime": string,                 // trending|consolidating|volatile|stable|mixed|unknown
    "confidence": number|null
  },
  "top_patterns": [
    {
      "name": string,
      "status": string,               // forming|completed|unknown
      "age_days": number|null,
      "reliability": string|null,     // high|medium|low|unknown
      "highlights": [string],         // key strengths/weaknesses
      "required_confirmations": [string]  // e.g., volume_surge_above_1.5x, close_above_resistance
    }
  ],
  "discarded_patterns": [
    { "name": string, "age_days": number|null, "reason": string }
  ],
  "risk": {
    "level": string,                 // very_low|low|moderate|high|very_high
    "key_risks": [string]
  },
  "decision_guidance": {
    "position_sizing": string,       // e.g., conservative|standard|avoid
    "entry_rules": [string],
    "exit_rules": [string]
  }
}

Rules:
- Output the narrative first, then the JSON as the final content.
- Output exactly one JSON object. No trailing commentary after the JSON.
- Use numbers for numeric fields. Use null for unknowns.
"""
        
        return prompt
    
    async def _get_llm_response(self, prompt: str, symbol: str, chart_image_bytes: Optional[bytes] = None) -> str:
        """Get response from LLM with error handling"""
        try:
            if not self.llm_client:
                return f"Cross-validation analysis for {symbol} could not be completed: LLM client not initialized."
            
            # Prepare chart image if provided - use bytes directly like market structure agent
            chart_images = None
            if chart_image_bytes:
                chart_images = [chart_image_bytes]
                logger.info(f"[CROSS_VALIDATION_LLM] Chart image provided: {len(chart_image_bytes)} bytes")
            else:
                logger.info(f"[CROSS_VALIDATION_LLM] No chart image bytes provided")
            
            # Make async call to LLM using new backend/llm system - FIXED to match working agents
            if chart_images:
                llm_result = await self.llm_client.generate(
                    prompt=prompt,
                    images=chart_images,
                    enable_code_execution=True,
                    timeout=90
                )
            else:
                llm_result = await self.llm_client.generate_text(
                    prompt=prompt,
                    enable_code_execution=True,
                    timeout=90
                )
            
            # Handle different return formats from LLM client
            if isinstance(llm_result, tuple):
                response = llm_result[0]
                token_usage = llm_result[1] if len(llm_result) > 1 else None
            else:
                response = llm_result
                token_usage = None
            
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

            # Try to extract trailing JSON summary
            structured_json = self._extract_json_from_response(llm_response)
            
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
                'final_confidence_level': validation_data.get('final_confidence_assessment', {}).get('confidence_level', 'unknown'),
                
                # Structured JSON (if available)
                'structured_output': structured_json
            }
            
            logger.info(f"[CROSS_VALIDATION_LLM] Analysis completed for {symbol} ({len(llm_response)} chars, structured_json={(structured_json is not None)})")
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
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt to extract a trailing JSON object from the LLM response."""
        try:
            # Heuristic: try to parse the largest trailing JSON block
            text = response.strip()
            # Find last '{' and try parses from there backwards a few times
            last = text.rfind('{')
            attempts = 0
            while last != -1 and attempts < 5:
                candidate = text[last:]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    # Move to previous '{'
                    last = text.rfind('{', 0, last)
                    attempts += 1
            return None
        except Exception:
            return None

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
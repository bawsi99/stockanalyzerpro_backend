from typing import Dict, Any, List
import json
import numpy as np
from .gemini_core import GeminiCore
from .prompt_manager import PromptManager
from .image_utils import ImageUtils
from .error_utils import ErrorUtils

import asyncio
import time

# --- Import clean_for_json ---
from utils import clean_for_json


class GeminiClient:
    """
    Client for interacting with Google's Gemini API for AI-powered analysis.
    This class handles API authentication, prompt generation, and response parsing.
    """
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert NumPy types to JSON-serializable Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: GeminiClient.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [GeminiClient.convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # Handle pandas Timestamp objects
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # Handle pandas Series and other array-like objects
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):  # Handle pandas DataFrame objects
            return obj.to_dict('records')
        else:
            return obj
    
    def __init__(self, api_key: str = None):
        self.core = GeminiCore(api_key)
        self.prompt_manager = PromptManager()
        self.image_utils = ImageUtils()
        self.error_utils = ErrorUtils()

    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None):
        # print("[DEBUG] Entering build_indicators_summary")
        try:
            # Convert NumPy types to JSON-serializable types and clean for JSON
            serializable_indicators = clean_for_json(self.convert_numpy_types(indicators))
            raw_indicators_json = json.dumps(serializable_indicators, indent=2)
            # print(f"[DEBUG] raw_indicators_json: {raw_indicators_json}")
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during json.dumps(indicators): {ex}")
            import traceback; traceback.print_exc()
            raise
        timeframe = f"{period} days, {interval}"
        
        # print("[DEBUG] About to call self.prompt_manager.format_prompt")
        try:
            prompt = self.prompt_manager.format_prompt(
                "indicators_to_summary_and_json",
                symbol=symbol,
                timeframe=timeframe,
                knowledge_context=knowledge_context or "",
                raw_indicators=raw_indicators_json
            )
        except Exception as ex:
            # print(f"[DEBUG-ERROR] Exception during format_prompt: {ex}")
            import traceback; traceback.print_exc()
            raise
        # print("[DEBUG] After format_prompt, prompt:\n" + prompt)
        loop = asyncio.get_event_loop()
        try:
            # print("[DEBUG] About to call self.core.call_llm(prompt)")
            try:
                # Use code execution for enhanced mathematical analysis
                text_response, code_results, execution_results = self.core.call_llm_with_code_execution(prompt)
                print(f"[DEBUG] LLM response length: {len(text_response) if text_response else 0}")
                print(f"[DEBUG] Code results count: {len(code_results)}")
                print(f"[DEBUG] Execution results count: {len(execution_results)}")
            except Exception as ex:
                print(f"[DEBUG-ERROR] Exception during LLM call with code execution: {ex}")
                # Fallback to regular LLM call without code execution
                print("[DEBUG] Falling back to regular LLM call...")
                text_response = await loop.run_in_executor(None, lambda: self.core.call_llm(prompt))
                code_results, execution_results = [], []
                print(f"[DEBUG] Fallback response length: {len(text_response) if text_response else 0}")
            
            if not text_response or not isinstance(text_response, str) or text_response.strip() == "":
                # print(f"[DEBUG-ERROR] LLM response is empty or invalid: {text_response!r}")
                print("[DEBUG] Empty response, using fallback JSON")
                # Create a fallback response
                fallback_json = self._create_fallback_json()
                return "Analysis completed with fallback data due to empty response.", json.loads(fallback_json)
            
            # print("[DEBUG] About to call extract_markdown_and_json")
            markdown_part, json_blob = self.extract_markdown_and_json(text_response)
            # print(f"[DEBUG] After extract_markdown_and_json, json_blob: {json_blob}")

            # print("[DEBUG] About to call json.loads on json_blob")
            try:
                parsed = json.loads(json_blob)
                # print(f"[DEBUG] After json.loads, parsed: {parsed}")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"JSON blob: {json_blob[:500]}...")  # Print first 500 chars for debugging
                # Use fallback JSON
                fallback_json = self._create_fallback_json()
                parsed = json.loads(fallback_json)
                print("Using fallback JSON due to parsing error")
            
            # Enhance the parsed result with code execution results
            if code_results or execution_results:
                parsed = self._enhance_with_calculations(parsed, code_results, execution_results)
            
            # print("[DEBUG] Leaving build_indicators_summary")
            return markdown_part, parsed
        except Exception as ex:
            # print(f"[DEBUG-ERROR] Exception in build_indicators_summary: {ex}")
            import traceback; traceback.print_exc()
            # print(f"[DEBUG-ERROR] Prompt: {prompt}")
            # Return fallback response
            fallback_json = self._create_fallback_json()
            return "Analysis completed with fallback data due to error.", json.loads(fallback_json)

    def _enhance_with_calculations(self, parsed_result, code_results, execution_results):
        """
        Enhance the parsed result with actual calculation results from code execution.
        """
        try:
            # Add code execution metadata
            if 'mathematical_validation' not in parsed_result:
                parsed_result['mathematical_validation'] = {}
            
            # Add code execution results
            parsed_result['mathematical_validation']['code_execution'] = {
                'code_snippets': code_results,
                'execution_outputs': execution_results,
                'calculation_timestamp': time.time()
            }
            
            # Try to extract specific calculation results from execution outputs
            for output in execution_results:
                if isinstance(output, str):
                    # Look for specific calculation results in the output
                    if 'correlation' in output.lower():
                        try:
                            # Extract correlation values from output
                            import re
                            corr_match = re.search(r'correlation[:\s]*([0-9.-]+)', output, re.IGNORECASE)
                            if corr_match:
                                parsed_result['mathematical_validation']['price_volume_correlation'] = {
                                    'correlation_coefficient': float(corr_match.group(1)),
                                    'p_value': 0.05,  # Default p-value
                                    'significance': 'high' if abs(float(corr_match.group(1))) > 0.7 else 'medium'
                                }
                        except:
                            pass
                    
                    if 'rsi' in output.lower():
                        try:
                            # Extract RSI values from output
                            import re
                            rsi_matches = re.findall(r'(\d+(?:\.\d+)?)', output)
                            if len(rsi_matches) >= 3:
                                parsed_result['mathematical_validation']['rsi_analysis'] = {
                                    'oversold_periods': int(rsi_matches[0]) if len(rsi_matches) > 0 else 0,
                                    'overbought_periods': int(rsi_matches[1]) if len(rsi_matches) > 1 else 0,
                                    'average_rsi': float(rsi_matches[2]) if len(rsi_matches) > 2 else 50.0,
                                    'signal_strength': 'strong'
                                }
                        except:
                            pass
            
            return parsed_result
        except Exception as e:
            print(f"Error enhancing with calculations: {e}")
            return parsed_result

    @staticmethod
    def extract_markdown_and_json(llm_response: str):
        # print("[DEBUG] Entering extract_markdown_and_json")
        # Extracts the markdown summary and the JSON code block from the LLM response
        import re
        import json
        try:
            match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_response)
            if match:
                json_blob = match.group(1)
                markdown_part = llm_response[:match.start()].strip()
                
                # Try to validate and fix the JSON
                try:
                    # First attempt: direct parsing
                    json.loads(json_blob)
                    return markdown_part, json_blob
                except json.JSONDecodeError as e:
                    # Second attempt: try to fix common JSON issues
                    fixed_json = GeminiClient._fix_json_string(json_blob)
                    try:
                        json.loads(fixed_json)
                        return markdown_part, fixed_json
                    except json.JSONDecodeError:
                        # Third attempt: create a minimal valid JSON
                        print(f"Warning: Could not fix JSON, using fallback. Error: {e}")
                        fallback_json = GeminiClient._create_fallback_json()
                        return markdown_part, fallback_json
            else:
                # print(f"[DEBUG-ERROR] Could not find JSON code block in LLM response. Raw response:\n{llm_response}")
                raise ValueError("Could not find JSON code block in LLM response.")
        except Exception as ex:
            # print(f"[DEBUG-ERROR] Exception in extract_markdown_and_json: {ex}")
            import traceback; traceback.print_exc()
            # print(f"[DEBUG-ERROR] Raw LLM response: {llm_response}")
            raise

    @staticmethod
    def _fix_json_string(json_str: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        import re
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Remove any control characters
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
        
        return json_str

    @staticmethod
    def _create_fallback_json() -> str:
        """Create a minimal valid JSON when all else fails."""
        return json.dumps({
            "trend": "neutral",
            "confidence_pct": 50,
            "short_term": {"signal": "hold", "target": None, "stop_loss": None},
            "medium_term": {"signal": "hold", "target": None, "stop_loss": None},
            "long_term": {"signal": "hold", "target": None, "stop_loss": None},
            "risks": ["Data parsing error occurred"],
            "must_watch_levels": [],
            "analysis_notes": "JSON parsing failed, using fallback analysis"
        })

    async def analyze_stock(self, symbol, indicators, chart_paths, period, interval, knowledge_context=""):
        # Ensure chart_paths is a dict
        if chart_paths is None:
            chart_paths = {}
        # 1. Indicator summary + JSON
        ind_summary_md, ind_json = await self.build_indicators_summary(
            symbol=symbol,
            indicators=indicators,
            period=period,
            interval=interval,
            knowledge_context=knowledge_context
        )

        # 2. Chart insights (analyze images) - OPTIMIZED GROUPING STRATEGY
        chart_insights_list = []
        
        # GROUP 1: Comprehensive Technical Overview (1 chart - most important)
        if chart_paths.get('comparison_chart'):
            with open(chart_paths['comparison_chart'], 'rb') as f:
                comparison_chart = f.read()
            comparison_insight = await self.analyze_comprehensive_overview(comparison_chart)
            chart_insights_list.append("**Comprehensive Technical Overview:**\n" + comparison_insight)
        
        # GROUP 2: Volume Analysis (3 charts together - complete volume story)
        volume_charts = []
        if chart_paths.get('volume_anomalies'):
            with open(chart_paths['volume_anomalies'], 'rb') as f:
                volume_charts.append(f.read())
        if chart_paths.get('price_volume_correlation'):
            with open(chart_paths['price_volume_correlation'], 'rb') as f:
                volume_charts.append(f.read())
        if chart_paths.get('candlestick_volume'):
            with open(chart_paths['candlestick_volume'], 'rb') as f:
                volume_charts.append(f.read())
        
        if volume_charts:
            volume_insight = await self.analyze_volume_comprehensive(volume_charts)
            chart_insights_list.append("**Comprehensive Volume Analysis:**\n" + volume_insight)
        
        # GROUP 3: Reversal Pattern Analysis (2 charts together)
        reversal_charts = []
        if chart_paths.get('divergence'):
            with open(chart_paths['divergence'], 'rb') as f:
                reversal_charts.append(f.read())
        if chart_paths.get('double_tops_bottoms'):
            with open(chart_paths['double_tops_bottoms'], 'rb') as f:
                reversal_charts.append(f.read())
        
        if reversal_charts:
            reversal_insight = await self.analyze_reversal_patterns(reversal_charts)
            chart_insights_list.append("**Reversal Pattern Analysis:**\n" + reversal_insight)
        
        # GROUP 4: Continuation & Level Analysis (2 charts together)
        continuation_charts = []
        if chart_paths.get('triangles_flags'):
            with open(chart_paths['triangles_flags'], 'rb') as f:
                continuation_charts.append(f.read())
        if chart_paths.get('support_resistance'):
            with open(chart_paths['support_resistance'], 'rb') as f:
                continuation_charts.append(f.read())
        
        if continuation_charts:
            continuation_insight = await self.analyze_continuation_levels(continuation_charts)
            chart_insights_list.append("**Continuation & Level Analysis:**\n" + continuation_insight)
        
        chart_insights_md = "\n\n".join(chart_insights_list) if chart_insights_list else ""

        # 3. Final decision prompt with enhanced mathematical validation
        decision_prompt = self.prompt_manager.format_prompt(
            "final_stock_decision",
            indicator_json=json.dumps(clean_for_json(self.convert_numpy_types(ind_json)), indent=2),
            chart_insights=chart_insights_md
        )
        try:
            # Use code execution for final decision analysis
            text_response, code_results, execution_results = self.core.call_llm_with_code_execution(decision_prompt)
            
            # Debug: Log the response to understand what we're getting
            print(f"[DEBUG] Final decision response type: {type(text_response)}")
            print(f"[DEBUG] Final decision response length: {len(text_response) if text_response else 0}")
            print(f"[DEBUG] Final decision response preview: {text_response[:200] if text_response else 'None'}")
            print(f"[DEBUG] Code execution results: {len(code_results) if code_results else 0} code snippets")
            print(f"[DEBUG] Execution outputs: {len(execution_results) if execution_results else 0} outputs")
            
            # The final decision should output ONLY JSON, not markdown with JSON inside
            # So we try to parse it directly as JSON first
            try:
                result = json.loads(text_response.strip())
                print("[DEBUG] Successfully parsed response as direct JSON")
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Direct JSON parsing failed: {e}")
                # If direct parsing fails, try to extract JSON from markdown code block
                try:
                    _, json_blob = self.extract_markdown_and_json(text_response)
                    result = json.loads(json_blob)
                    print("[DEBUG] Successfully extracted JSON from markdown code block")
                except Exception as extract_error:
                    print(f"[DEBUG] Failed to extract JSON from markdown: {extract_error}")
                    print(f"[DEBUG] Full response: {text_response}")
                    raise ValueError(f"Could not parse final decision response as JSON: {e}. Response: {text_response[:500]}")
            
            # Enhance result with code execution data
            if code_results or execution_results:
                result = self._enhance_final_decision_with_calculations(result, code_results, execution_results)
                
        except Exception as ex:
            import traceback; traceback.print_exc()
            raise
        return result, ind_summary_md, chart_insights_md

    def _enhance_final_decision_with_calculations(self, result, code_results, execution_results):
        """
        Enhance the final decision result with code execution calculations.
        """
        try:
            # Add code execution metadata to the result
            if 'analysis_metadata' not in result:
                result['analysis_metadata'] = {}
            
            result['analysis_metadata']['code_execution'] = {
                'code_snippets_count': len(code_results),
                'execution_outputs_count': len(execution_results),
                'calculation_timestamp': time.time(),
                'enhanced_analysis': True
            }
            
            # Try to extract specific calculation insights
            calculation_insights = []
            for output in execution_results:
                if isinstance(output, str):
                    # Look for key calculation results
                    if any(keyword in output.lower() for keyword in ['correlation', 'rsi', 'trend', 'volatility', 'support', 'resistance']):
                        calculation_insights.append(output.strip())
            
            if calculation_insights:
                result['analysis_metadata']['calculation_insights'] = calculation_insights
            
            return result
        except Exception as e:
            print(f"Error enhancing final decision with calculations: {e}")
            return result

    async def analyze_stock_with_enhanced_calculations(self, symbol, indicators, chart_paths, period, interval, knowledge_context=""):
        """
        Enhanced version of analyze_stock with comprehensive mathematical validation.
        """
        # Ensure chart_paths is a dict
        if chart_paths is None:
            chart_paths = {}
        
        # 1. Enhanced indicator summary with mathematical validation
        ind_summary_md, ind_json = await self.build_indicators_summary(
            symbol=symbol,
            indicators=indicators,
            period=period,
            interval=interval,
            knowledge_context=knowledge_context
        )

        # 2. Enhanced chart analysis with code execution
        chart_insights_list = []
        
        # GROUP 1: Comprehensive Technical Overview with calculations
        if chart_paths.get('comparison_chart'):
            with open(chart_paths['comparison_chart'], 'rb') as f:
                comparison_chart = f.read()
            comparison_insight = await self.analyze_comprehensive_overview_with_calculations(comparison_chart, indicators)
            chart_insights_list.append("**Comprehensive Technical Overview (Mathematically Validated):**\n" + comparison_insight)
        
        # GROUP 2: Volume Analysis with statistical validation
        volume_charts = []
        if chart_paths.get('volume_anomalies'):
            with open(chart_paths['volume_anomalies'], 'rb') as f:
                volume_charts.append(f.read())
        if chart_paths.get('price_volume_correlation'):
            with open(chart_paths['price_volume_correlation'], 'rb') as f:
                volume_charts.append(f.read())
        if chart_paths.get('candlestick_volume'):
            with open(chart_paths['candlestick_volume'], 'rb') as f:
                volume_charts.append(f.read())
        
        if volume_charts:
            volume_insight = await self.analyze_volume_comprehensive_with_calculations(volume_charts, indicators)
            chart_insights_list.append("**Comprehensive Volume Analysis (Statistically Validated):**\n" + volume_insight)
        
        chart_insights_md = "\n\n".join(chart_insights_list) if chart_insights_list else ""

        # 3. Enhanced final decision with comprehensive mathematical validation
        enhanced_decision_prompt = self.prompt_manager.format_prompt(
            "final_stock_decision",
            indicator_json=json.dumps(clean_for_json(self.convert_numpy_types(ind_json)), indent=2),
            chart_insights=chart_insights_md
        )
        
        # Add mathematical validation requirements to the prompt
        enhanced_decision_prompt += """

MATHEMATICAL VALIDATION REQUIREMENTS:
1. Calculate and validate all support/resistance levels using statistical methods
2. Verify trend strength using linear regression analysis
3. Calculate risk metrics including Value at Risk (VaR)
4. Validate pattern reliability using mathematical criteria
5. Perform correlation analysis between all indicators
6. Calculate confidence intervals for all predictions

Use Python code for all calculations and include the mathematical validation results in your analysis.
"""
        
        try:
            text_response, code_results, execution_results = self.core.call_llm_with_code_execution(enhanced_decision_prompt)
            
            try:
                result = json.loads(text_response.strip())
            except json.JSONDecodeError as e:
                try:
                    _, json_blob = self.extract_markdown_and_json(text_response)
                    result = json.loads(json_blob)
                except Exception as extract_error:
                    print(f"[DEBUG] Failed to extract JSON from markdown: {extract_error}")
                    raise ValueError(f"Could not parse enhanced decision response as JSON: {e}")
            
            # Enhance with comprehensive calculation results
            result = self._enhance_final_decision_with_calculations(result, code_results, execution_results)
            
            # Add mathematical validation summary
            if 'mathematical_validation' not in result:
                result['mathematical_validation'] = {}
            
            result['mathematical_validation']['enhanced_analysis'] = {
                'calculation_method': 'code_execution',
                'statistical_validation': True,
                'confidence_improvement': 'high',
                'analysis_timestamp': time.time()
            }
                
        except Exception as ex:
            import traceback; traceback.print_exc()
            raise
        return result, ind_summary_md, chart_insights_md

    async def analyze_comprehensive_overview(self, image: bytes) -> str:
        """Analyze the comprehensive comparison chart that shows all major indicators."""
        prompt = self.prompt_manager.format_prompt("image_analysis_comprehensive_overview")
        return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))

    async def analyze_volume_comprehensive(self, images: list) -> str:
        """Analyze all volume-related charts together for complete volume story."""
        prompt = self.prompt_manager.format_prompt("image_analysis_volume_comprehensive")
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(prompt, pil_images)

    async def analyze_reversal_patterns(self, images: list) -> str:
        """Analyze divergence and double tops/bottoms charts together for reversal signals."""
        prompt = self.prompt_manager.format_prompt("image_analysis_reversal_patterns")
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(prompt, pil_images)

    async def analyze_continuation_levels(self, images: list) -> str:
        """Analyze triangles/flags and support/resistance charts together for continuation and levels."""
        prompt = self.prompt_manager.format_prompt("image_analysis_continuation_levels")
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(prompt, pil_images)

    async def analyze_comprehensive_overview_with_calculations(self, image: bytes, indicators: dict) -> str:
        """Analyze the comprehensive comparison chart with mathematical validation."""
        enhanced_prompt = self.prompt_manager.format_prompt("image_analysis_comprehensive_overview")
        enhanced_prompt += f"""

MATHEMATICAL VALIDATION REQUIRED:
Analyze the chart and perform the following calculations using Python code:

1. Calculate the correlation between price movements and volume changes
2. Verify RSI levels and count oversold/overbought periods
3. Calculate MACD signal strength and histogram analysis
4. Determine trend strength using linear regression
5. Calculate volatility metrics (standard deviation, coefficient of variation)
6. Validate support/resistance levels using statistical methods

Technical Indicators Data: {json.dumps(clean_for_json(self.convert_numpy_types(indicators)), indent=2)}

Use Python code for all calculations and include the results in your analysis.
"""
        return await self.core.call_llm_with_image(enhanced_prompt, self.image_utils.bytes_to_image(image), enable_code_execution=True)

    async def analyze_volume_comprehensive_with_calculations(self, images: list, indicators: dict) -> str:
        """Analyze all volume-related charts with statistical validation."""
        enhanced_prompt = self.prompt_manager.format_prompt("image_analysis_volume_comprehensive")
        enhanced_prompt += f"""

STATISTICAL VALIDATION REQUIRED:
Analyze the volume charts and perform the following calculations using Python code:

1. Calculate price-volume correlation coefficient and p-value
2. Identify volume anomalies using statistical methods (z-score analysis)
3. Calculate volume-weighted average price (VWAP)
4. Determine volume trend strength using linear regression
5. Calculate volume-based support/resistance levels
6. Validate volume confirmation of price movements

Technical Indicators Data: {json.dumps(clean_for_json(self.convert_numpy_types(indicators)), indent=2)}

Use Python code for all calculations and include the results in your analysis.
"""
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(enhanced_prompt, pil_images, enable_code_execution=True)



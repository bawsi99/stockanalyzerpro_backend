from typing import Dict, Any, List
import json
import numpy as np
from .gemini_core import GeminiCore
from .prompt_manager import PromptManager
from .image_utils import ImageUtils
from .error_utils import ErrorUtils
from .debug_logger import debug_logger
from .token_tracker import get_or_create_tracker, AnalysisTokenTracker
from .context_engineer import ContextEngineer, AnalysisType, ContextConfig

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
    
    def __init__(self, api_key: str = None, context_config: ContextConfig = None):
        self.core = GeminiCore(api_key)
        self.prompt_manager = PromptManager()
        self.image_utils = ImageUtils()
        self.error_utils = ErrorUtils()
        self.context_engineer = ContextEngineer(context_config)

    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None, token_tracker=None, mtf_context=None):
        """
        Build comprehensive indicator summary with multi-timeframe and sector analysis support.
        
        Args:
            symbol: Stock symbol
            indicators: Technical indicators data (can be single or multi-timeframe)
            period: Analysis period
            interval: Analysis interval
            knowledge_context: Additional context (includes sector context)
            token_tracker: Token usage tracker
            mtf_context: Multi-timeframe context data
        """
        try:
            # Use context engineering to curate and structure indicators
            curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.INDICATOR_SUMMARY)
            timeframe = f"{period} days, {interval}"
            
            # Enhance context with MTF data if available
            enhanced_context = knowledge_context or ""
            if mtf_context and mtf_context.get('success', False):
                enhanced_context += self._build_mtf_context_for_indicators(mtf_context)
            
            # Extract and enhance sector context if available
            if "SECTOR CONTEXT" in knowledge_context:
                enhanced_context += self._build_sector_context_for_indicators(knowledge_context)
            
            # Structure context using context engineering
            context = self.context_engineer.structure_context(
                curated_indicators, 
                AnalysisType.INDICATOR_SUMMARY, 
                symbol, 
                timeframe, 
                enhanced_context
            )
            
            # Use optimized prompt template
            prompt = self.prompt_manager.format_prompt(
                "optimized_indicators_summary",
                context=context
            )
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during context engineering: {ex}")
            import traceback; traceback.print_exc()
            raise
        
        loop = asyncio.get_event_loop()
        try:
            try:
                # Use code execution for enhanced mathematical analysis
                response, code_results, execution_results = await self.core.call_llm_with_code_execution(prompt, return_full_response=True)
                text_response = response.text if response else ""
                
                # Track token usage if tracker is provided
                if token_tracker:
                    token_tracker.add_token_usage("indicator_summary", response, "gemini-2.5-flash")
                
                print(f"[DEBUG] LLM response length: {len(text_response) if text_response else 0}")
                print(f"[DEBUG] Code results count: {len(code_results)}")
                print(f"[DEBUG] Execution results count: {len(execution_results)}")
            except Exception as ex:
                print(f"[DEBUG-ERROR] Exception during LLM call with code execution: {ex}")
                # Fallback to regular LLM call without code execution
                print("[DEBUG] Falling back to regular LLM call...")
                response = await loop.run_in_executor(None, lambda: self.core.call_llm(prompt, return_full_response=True))
                text_response = response.text if response else ""
                code_results, execution_results = [], []
                
                # Track token usage for fallback call
                if token_tracker and response:
                    token_tracker.add_token_usage("indicator_summary_fallback", response, "gemini-2.5-flash")
                
                print(f"[DEBUG] Fallback response length: {len(text_response) if text_response else 0}")
            
            if not text_response or not isinstance(text_response, str) or text_response.strip() == "":
                print("[DEBUG] Empty response, using fallback JSON")
                # Create a fallback response
                fallback_json = self._create_fallback_json()
                return "Analysis completed with fallback data due to empty response.", json.loads(fallback_json)
            
            markdown_part, json_blob = self.extract_markdown_and_json(text_response)

            try:
                parsed = json.loads(json_blob)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"JSON blob: {json_blob[:500]}...")
                # Use fallback JSON
                fallback_json = self._create_fallback_json()
                parsed = json.loads(fallback_json)
                print("Using fallback JSON due to parsing error")
            
            # Enhance the parsed result with code execution results
            if code_results or execution_results:
                parsed = self._enhance_with_calculations(parsed, code_results, execution_results)
            
            return markdown_part, parsed
        except Exception as ex:
            import traceback; traceback.print_exc()
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
        debug_logger.log_processing_step("Extracting markdown and JSON from LLM response")
        
        # Extracts the markdown summary and the JSON code block from the LLM response
        import re
        import json
        try:
            match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_response)
            if match:
                json_blob = match.group(1)
                markdown_part = llm_response[:match.start()].strip()
                
                debug_logger.log_processing_step("Found JSON code block", f"Length: {len(json_blob)} chars")
                
                # Try to validate and fix the JSON
                try:
                    # First attempt: direct parsing
                    parsed_data = json.loads(json_blob)
                    debug_logger.log_json_parsing(json_blob, True, parsed_data)
                    return markdown_part, json_blob
                except json.JSONDecodeError as e:
                    # Second attempt: try to fix common JSON issues
                    fixed_json = GeminiClient._fix_json_string(json_blob)
                    try:
                        parsed_data = json.loads(fixed_json)
                        debug_logger.log_json_parsing(fixed_json, True, parsed_data)
                        debug_logger.log_processing_step("JSON fixed successfully", f"Original error: {e}")
                        return markdown_part, fixed_json
                    except json.JSONDecodeError as e2:
                        # Third attempt: create a minimal valid JSON
                        debug_logger.log_json_parsing(fixed_json, False, error=e2)
                        debug_logger.log_processing_step("Using fallback JSON", f"Original error: {e}, Fixed error: {e2}")
                        fallback_json = GeminiClient._create_fallback_json()
                        return markdown_part, fallback_json
            else:
                debug_logger.log_error(ValueError("No JSON code block found"), "extract_markdown_and_json", llm_response)
                raise ValueError("Could not find JSON code block in LLM response.")
        except Exception as ex:
            debug_logger.log_error(ex, "extract_markdown_and_json", llm_response)
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
        
        print(f"[ASYNC-OPTIMIZED] Starting optimized analysis for {symbol}...")
        print(f"[ASYNC-OPTIMIZED] Chart paths received: {list(chart_paths.keys()) if chart_paths else 'None'}")
        print(f"[ASYNC-OPTIMIZED] Chart paths content: {chart_paths}")
        
        # START ALL INDEPENDENT LLM CALLS IMMEDIATELY
        # 1. Indicator summary (no dependencies)
        print("[ASYNC-OPTIMIZED] Starting indicator summary analysis...")
        indicator_task = self.build_indicators_summary(
            symbol=symbol,
            indicators=indicators,
            period=period,
            interval=interval,
            knowledge_context=knowledge_context
        )

        # 2. Chart insights (analyze images) - START ALL CHART TASKS IMMEDIATELY
        print("[ASYNC-OPTIMIZED] Starting all chart analysis tasks...")
        chart_analysis_tasks = []
        
        # GROUP 1: Comprehensive Technical Overview (1 chart - most important)
        print(f"[ASYNC-OPTIMIZED] Checking for technical_overview: {chart_paths.get('technical_overview')}")
        if chart_paths.get('technical_overview'):
            try:
                with open(chart_paths['technical_overview'], 'rb') as f:
                    technical_chart = f.read()
                print(f"[ASYNC-OPTIMIZED] Successfully read technical_overview: {len(technical_chart)} bytes")
                task = self.analyze_technical_overview(technical_chart)
                chart_analysis_tasks.append(("technical_overview", task))
                print("[ASYNC-OPTIMIZED] Added technical_overview task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED] Error reading technical_overview: {e}")
        else:
            print("[ASYNC-OPTIMIZED] technical_overview not found in chart_paths")
        
        # GROUP 2: Pattern Analysis (1 chart - comprehensive pattern recognition)
        print(f"[ASYNC-OPTIMIZED] Checking pattern_analysis: {chart_paths.get('pattern_analysis')}")
        if chart_paths.get('pattern_analysis'):
            try:
                with open(chart_paths['pattern_analysis'], 'rb') as f:
                    pattern_chart = f.read()
                print(f"[ASYNC-OPTIMIZED] Successfully read pattern_analysis: {len(pattern_chart)} bytes")
                task = self.analyze_pattern_analysis(pattern_chart, indicators)
                chart_analysis_tasks.append(("pattern_analysis", task))
                print("[ASYNC-OPTIMIZED] Added pattern_analysis task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED] Error reading pattern_analysis: {e}")
        else:
            print("[ASYNC-OPTIMIZED] pattern_analysis not found in chart_paths")
        
        # GROUP 3: Volume Analysis (1 chart - comprehensive volume story)
        print(f"[ASYNC-OPTIMIZED] Checking volume_analysis: {chart_paths.get('volume_analysis')}")
        if chart_paths.get('volume_analysis'):
            try:
                with open(chart_paths['volume_analysis'], 'rb') as f:
                    volume_chart = f.read()
                print(f"[ASYNC-OPTIMIZED] Successfully read volume_analysis: {len(volume_chart)} bytes")
                task = self.analyze_volume_analysis(volume_chart, indicators)
                chart_analysis_tasks.append(("volume_analysis", task))
                print("[ASYNC-OPTIMIZED] Added volume_analysis task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED] Error reading volume_analysis: {e}")
        else:
            print("[ASYNC-OPTIMIZED] volume_analysis not found in chart_paths")
        
        # GROUP 4: Multi-Timeframe Comparison (1 chart - MTF validation)
        print(f"[ASYNC-OPTIMIZED] Checking mtf_comparison: {chart_paths.get('mtf_comparison')}")
        if chart_paths.get('mtf_comparison'):
            try:
                with open(chart_paths['mtf_comparison'], 'rb') as f:
                    mtf_chart = f.read()
                print(f"[ASYNC-OPTIMIZED] Successfully read mtf_comparison: {len(mtf_chart)} bytes")
                task = self.analyze_mtf_comparison(mtf_chart, indicators)
                chart_analysis_tasks.append(("mtf_comparison", task))
                print("[ASYNC-OPTIMIZED] Added mtf_comparison task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED] Error reading mtf_comparison: {e}")
        else:
            print("[ASYNC-OPTIMIZED] mtf_comparison not found in chart_paths")
        
        # EXECUTE ALL INDEPENDENT TASKS IN PARALLEL
        print(f"[ASYNC-OPTIMIZED] Total tasks created: {len(chart_analysis_tasks) + 1}")
        print(f"[ASYNC-OPTIMIZED] Chart analysis tasks: {[name for name, _ in chart_analysis_tasks]}")
        
        # FALLBACK: If no chart tasks were created, create mock tasks for testing
        if len(chart_analysis_tasks) == 0:
            print("[ASYNC-OPTIMIZED] WARNING: No chart tasks created! Creating fallback mock tasks for testing...")
            
            # Create mock chart data for testing
            mock_chart_data = b"mock_chart_data_for_testing"
            
            # Add mock tasks for all chart analysis types
            mock_technical_task = self.analyze_technical_overview(mock_chart_data)
            chart_analysis_tasks.append(("technical_overview_mock", mock_technical_task))
            print("[ASYNC-OPTIMIZED] Added mock technical_overview task")
            
            mock_pattern_task = self.analyze_pattern_analysis(mock_chart_data, indicators)
            chart_analysis_tasks.append(("pattern_analysis_mock", mock_pattern_task))
            print("[ASYNC-OPTIMIZED] Added mock pattern_analysis task")
            
            mock_volume_task = self.analyze_volume_analysis(mock_chart_data, indicators)
            chart_analysis_tasks.append(("volume_analysis_mock", mock_volume_task))
            print("[ASYNC-OPTIMIZED] Added mock volume_analysis task")
            
            mock_mtf_task = self.analyze_mtf_comparison(mock_chart_data, indicators)
            chart_analysis_tasks.append(("mtf_comparison_mock", mock_mtf_task))
            print("[ASYNC-OPTIMIZED] Added mock mtf_comparison task")
            
            print(f"[ASYNC-OPTIMIZED] Created {len(chart_analysis_tasks)} mock chart tasks for testing")
        
        print(f"[ASYNC-OPTIMIZED] Executing {len(chart_analysis_tasks) + 1} independent tasks in parallel...")
        
        # DISABLE RATE LIMITING FOR TRUE PARALLEL EXECUTION
        self.core.disable_rate_limiting()
        
        import asyncio
        parallel_start_time = time.time()
        
        # Log the exact time when parallel execution starts
        print(f"[ASYNC-OPTIMIZED] Parallel execution started at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        # Combine indicator task with chart tasks
        all_tasks = [indicator_task] + [task for _, task in chart_analysis_tasks]
        
        # Log that we're about to gather all tasks
        print(f"[ASYNC-OPTIMIZED] Sending all {len(all_tasks)} tasks to asyncio.gather() at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        parallel_elapsed_time = time.time() - parallel_start_time
        print(f"[ASYNC-OPTIMIZED] Completed all independent tasks in {parallel_elapsed_time:.2f} seconds")
        print(f"[ASYNC-OPTIMIZED] Parallel execution ended at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        # RE-ENABLE RATE LIMITING AFTER PARALLEL EXECUTION
        self.core.enable_rate_limiting()
        
        # Process results and handle exceptions
        ind_summary_md, ind_json = all_results[0] if not isinstance(all_results[0], Exception) else ("", {})
        if isinstance(all_results[0], Exception):
            print(f"[ASYNC-OPTIMIZED] Warning: Indicator summary failed: {all_results[0]}")
        
        # Process chart results
        chart_insights_list = []
        for i, (task_name, _) in enumerate(chart_analysis_tasks, 1):
            result = all_results[i]
            if isinstance(result, Exception):
                print(f"[ASYNC-OPTIMIZED] Warning: {task_name} failed: {result}")
                continue
            
            if task_name == "technical_overview" or task_name == "technical_overview_mock":
                chart_insights_list.append("**Comprehensive Technical Overview:**\n" + result)
            elif task_name == "pattern_analysis" or task_name == "pattern_analysis_mock":
                chart_insights_list.append("**Pattern Analysis:**\n" + result)
            elif task_name == "volume_analysis" or task_name == "volume_analysis_mock":
                chart_insights_list.append("**Volume Analysis:**\n" + result)
            elif task_name == "mtf_comparison" or task_name == "mtf_comparison_mock":
                chart_insights_list.append("**Multi-Timeframe Comparison:**\n" + result)
        
        chart_insights_md = "\n\n".join(chart_insights_list) if chart_insights_list else ""

        # 3. Final decision prompt with enhanced mathematical validation (depends on all previous results)
        print("[ASYNC-OPTIMIZED] Starting final decision analysis...")
        decision_start_time = time.time()
        
        # Create final decision context using context engineering with MTF and sector integration
        final_decision_data = {
            "analysis_focus": "final_decision_synthesis",
            "consensus_analysis": {
                "trend_alignment": self._analyze_trend_alignment(ind_json, chart_insights_md),
                "confidence_score": self._calculate_consensus_confidence(ind_json, chart_insights_md),
                "key_conflicts": self._identify_key_conflicts(ind_json, chart_insights_md)
            },
            "risk_assessment": {
                "primary_risks": self._identify_primary_risks(ind_json, chart_insights_md),
                "risk_level": self._assess_overall_risk(ind_json, chart_insights_md)
            },
            "decision_framework": {
                "entry_strategy": self._determine_entry_strategy(ind_json, chart_insights_md),
                "timeframe": "short_term",
                "position_size": "moderate"
            },
            "multi_timeframe_context": self._extract_mtf_context_from_analysis(ind_json, chart_insights_md),
            "sector_context": self._extract_sector_context_from_analysis(ind_json, chart_insights_md)
        }
        
        # Enhance context with MTF and sector data if available in knowledge_context
        enhanced_knowledge_context = knowledge_context
        if "ENHANCED MULTI-TIMEFRAME ANALYSIS CONTEXT" in knowledge_context:
            enhanced_knowledge_context += self._build_mtf_context_for_final_decision(knowledge_context)
        
        if "SECTOR CONTEXT" in knowledge_context:
            enhanced_knowledge_context += self._build_sector_context_for_final_decision(knowledge_context)
        
        context = self.context_engineer.structure_context(
            final_decision_data, 
            AnalysisType.FINAL_DECISION, 
            symbol, 
            f"{period} days, {interval}", 
            enhanced_knowledge_context
        )
        
        decision_prompt = self.prompt_manager.format_prompt(
            "optimized_final_decision",
            context=context
        )
        try:
            # Use code execution for final decision analysis
            text_response, code_results, execution_results = await self.core.call_llm_with_code_execution(decision_prompt)
            
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
            
            # Enhance result with code execution data and MTF context
            if code_results or execution_results:
                result = self._enhance_final_decision_with_calculations(result, code_results, execution_results)
            
            # Add MTF context to result if available
            if "ENHANCED MULTI-TIMEFRAME ANALYSIS CONTEXT" in knowledge_context:
                result = self._enhance_result_with_mtf_context(result, knowledge_context)
            
            # Add sector context to result if available
            if "SECTOR CONTEXT" in knowledge_context:
                result = self._enhance_result_with_sector_context(result, knowledge_context)
                
        except Exception as ex:
            import traceback; traceback.print_exc()
            raise
        
        decision_elapsed_time = time.time() - decision_start_time
        total_elapsed_time = time.time() - parallel_start_time
        print(f"[ASYNC-OPTIMIZED] Final decision analysis completed in {decision_elapsed_time:.2f} seconds")
        print(f"[ASYNC-OPTIMIZED] Total analysis completed in {total_elapsed_time:.2f} seconds")
        
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
    
    def _analyze_trend_alignment(self, ind_json: dict, chart_insights: str) -> dict:
        """Analyze trend alignment between indicators and chart patterns."""
        try:
            # Extract trend from indicator analysis
            indicator_trend = ind_json.get('market_outlook', {}).get('primary_trend', {}).get('direction', 'neutral')
            indicator_confidence = ind_json.get('market_outlook', {}).get('primary_trend', {}).get('confidence', 50)
            
            # Simple trend alignment analysis
            alignment = {
                "indicator_trend": indicator_trend,
                "indicator_confidence": indicator_confidence,
                "chart_confirmation": "confirming" if "bullish" in chart_insights.lower() and indicator_trend == "bullish" else "mixed",
                "overall_alignment": "strong" if indicator_confidence > 70 else "moderate"
            }
            
            return alignment
        except Exception as e:
            print(f"Error analyzing trend alignment: {e}")
            return {"indicator_trend": "neutral", "confidence": 50, "alignment": "weak"}
    
    def _calculate_consensus_confidence(self, ind_json: dict, chart_insights: str) -> float:
        """Calculate consensus confidence from multiple analyses."""
        try:
            # Get confidence from indicator analysis
            indicator_confidence = ind_json.get('market_outlook', {}).get('primary_trend', {}).get('confidence', 50)
            
            # Simple confidence calculation (can be enhanced)
            base_confidence = indicator_confidence
            
            # Adjust based on chart insights
            if "high confidence" in chart_insights.lower():
                base_confidence += 10
            elif "low confidence" in chart_insights.lower():
                base_confidence -= 10
            
            return min(max(base_confidence, 0), 100)
        except Exception as e:
            print(f"Error calculating consensus confidence: {e}")
            return 50.0
    
    def _identify_key_conflicts(self, ind_json: dict, chart_insights: str) -> list:
        """Identify key conflicts between different analyses."""
        conflicts = []
        
        try:
            # Check for signal conflicts in indicator analysis
            signal_conflicts = ind_json.get('signal_conflicts', {})
            if signal_conflicts.get('has_conflicts', False):
                conflicts.append(signal_conflicts.get('conflict_description', 'Signal conflicts detected'))
            
            # Check for conflicts between indicators and charts
            indicator_trend = ind_json.get('market_outlook', {}).get('primary_trend', {}).get('direction', 'neutral')
            if "bearish" in chart_insights.lower() and indicator_trend == "bullish":
                conflicts.append("Chart patterns suggest bearish while indicators are bullish")
            elif "bullish" in chart_insights.lower() and indicator_trend == "bearish":
                conflicts.append("Chart patterns suggest bullish while indicators are bearish")
            
            return conflicts
        except Exception as e:
            print(f"Error identifying key conflicts: {e}")
            return ["Unable to analyze conflicts"]
    
    def _identify_primary_risks(self, ind_json: dict, chart_insights: str) -> list:
        """Identify primary risks from analysis."""
        risks = []
        
        try:
            # Extract risks from indicator analysis
            risk_management = ind_json.get('risk_management', {})
            key_risks = risk_management.get('key_risks', [])
            
            for risk in key_risks:
                if isinstance(risk, dict):
                    risks.append(risk.get('risk', 'Unknown risk'))
                else:
                    risks.append(str(risk))
            
            # Add chart-specific risks
            if "pattern failure" in chart_insights.lower():
                risks.append("Pattern failure risk")
            if "false signal" in chart_insights.lower():
                risks.append("False signal risk")
            
            return risks[:5]  # Limit to top 5 risks
        except Exception as e:
            print(f"Error identifying primary risks: {e}")
            return ["Market volatility", "Technical analysis limitations"]
    
    def _assess_overall_risk(self, ind_json: dict, chart_insights: str) -> str:
        """Assess overall risk level."""
        try:
            # Get risk indicators
            indicator_confidence = ind_json.get('market_outlook', {}).get('primary_trend', {}).get('confidence', 50)
            signal_conflicts = ind_json.get('signal_conflicts', {}).get('has_conflicts', False)
            
            # Risk assessment logic
            if indicator_confidence < 30 or signal_conflicts:
                return "high"
            elif indicator_confidence < 60:
                return "medium"
            else:
                return "low"
        except Exception as e:
            print(f"Error assessing overall risk: {e}")
            return "medium"
    
    def _determine_entry_strategy(self, ind_json: dict, chart_insights: str) -> str:
        """Determine optimal entry strategy."""
        try:
            # Get trading strategy from indicator analysis
            short_term = ind_json.get('trading_strategy', {}).get('short_term', {})
            entry_strategy = short_term.get('entry_strategy', {}).get('type', 'breakout')
            
            # Adjust based on chart patterns
            if "reversal" in chart_insights.lower():
                entry_strategy = "reversal"
            elif "continuation" in chart_insights.lower():
                entry_strategy = "breakout"
            
            return entry_strategy
        except Exception as e:
            print(f"Error determining entry strategy: {e}")
            return "breakout"

    async def analyze_stock_with_enhanced_calculations(self, symbol, indicators, chart_paths, period, interval, knowledge_context=""):
        """
        Enhanced version of analyze_stock with comprehensive mathematical validation.
        """
        # Ensure chart_paths is a dict
        if chart_paths is None:
            chart_paths = {}
        
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced optimized analysis for {symbol}...")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Chart paths received: {list(chart_paths.keys()) if chart_paths else 'None'}")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Chart paths content: {chart_paths}")
        
        # START ALL INDEPENDENT LLM CALLS IMMEDIATELY
        # 1. Enhanced indicator summary with mathematical validation (no dependencies)
        print("[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced indicator summary analysis...")
        indicator_task = self.build_indicators_summary(
            symbol=symbol,
            indicators=indicators,
            period=period,
            interval=interval,
            knowledge_context=knowledge_context
        )

        # 2. Enhanced chart analysis with code execution - START ALL CHART TASKS IMMEDIATELY
        print("[ASYNC-OPTIMIZED-ENHANCED] Starting all enhanced chart analysis tasks...")
        chart_analysis_tasks = []
        
        # GROUP 1: Comprehensive Technical Overview with calculations
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Checking for comparison_chart: {chart_paths.get('comparison_chart')}")
        if chart_paths.get('comparison_chart'):
            try:
                with open(chart_paths['comparison_chart'], 'rb') as f:
                    comparison_chart = f.read()
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read comparison_chart: {len(comparison_chart)} bytes")
                task = self.analyze_comprehensive_overview_with_calculations(comparison_chart, indicators)
                chart_analysis_tasks.append(("comprehensive_overview_enhanced", task))
                print("[ASYNC-OPTIMIZED-ENHANCED] Added comprehensive_overview_enhanced task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading comparison_chart: {e}")
        else:
            print("[ASYNC-OPTIMIZED-ENHANCED] comparison_chart not found in chart_paths")
        
        # GROUP 2: Volume Analysis with statistical validation
        volume_charts = []
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Checking volume charts:")
        print(f"  - volume_anomalies: {chart_paths.get('volume_anomalies')}")
        print(f"  - price_volume_correlation: {chart_paths.get('price_volume_correlation')}")
        print(f"  - candlestick_volume: {chart_paths.get('candlestick_volume')}")
        
        if chart_paths.get('volume_anomalies'):
            try:
                with open(chart_paths['volume_anomalies'], 'rb') as f:
                    volume_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read volume_anomalies: {len(volume_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading volume_anomalies: {e}")
        
        if chart_paths.get('price_volume_correlation'):
            try:
                with open(chart_paths['price_volume_correlation'], 'rb') as f:
                    volume_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read price_volume_correlation: {len(volume_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading price_volume_correlation: {e}")
        
        if chart_paths.get('candlestick_volume'):
            try:
                with open(chart_paths['candlestick_volume'], 'rb') as f:
                    volume_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read candlestick_volume: {len(volume_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading candlestick_volume: {e}")
        
        if volume_charts:
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Creating volume analysis task with {len(volume_charts)} charts")
            task = self.analyze_volume_comprehensive_with_calculations(volume_charts, indicators)
            chart_analysis_tasks.append(("volume_analysis_enhanced", task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added volume_analysis_enhanced task")
        else:
            print("[ASYNC-OPTIMIZED-ENHANCED] No volume charts available for analysis")
        
        # GROUP 3: Reversal Pattern Analysis with enhanced calculations
        reversal_charts = []
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Checking reversal charts:")
        print(f"  - divergence: {chart_paths.get('divergence')}")
        print(f"  - double_tops_bottoms: {chart_paths.get('double_tops_bottoms')}")
        
        if chart_paths.get('divergence'):
            try:
                with open(chart_paths['divergence'], 'rb') as f:
                    reversal_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read divergence: {len(reversal_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading divergence: {e}")
        
        if chart_paths.get('double_tops_bottoms'):
            try:
                with open(chart_paths['double_tops_bottoms'], 'rb') as f:
                    reversal_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read double_tops_bottoms: {len(reversal_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading double_tops_bottoms: {e}")
        
        if reversal_charts:
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Creating reversal patterns task with {len(reversal_charts)} charts")
            task = self.analyze_reversal_patterns_with_calculations(reversal_charts, indicators)
            chart_analysis_tasks.append(("reversal_patterns_enhanced", task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added reversal_patterns_enhanced task")
        else:
            print("[ASYNC-OPTIMIZED-ENHANCED] No reversal charts available for analysis")
        
        # GROUP 4: Continuation & Level Analysis with enhanced calculations
        continuation_charts = []
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Checking continuation charts:")
        print(f"  - triangles_flags: {chart_paths.get('triangles_flags')}")
        print(f"  - support_resistance: {chart_paths.get('support_resistance')}")
        
        if chart_paths.get('triangles_flags'):
            try:
                with open(chart_paths['triangles_flags'], 'rb') as f:
                    continuation_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read triangles_flags: {len(continuation_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading triangles_flags: {e}")
        
        if chart_paths.get('support_resistance'):
            try:
                with open(chart_paths['support_resistance'], 'rb') as f:
                    continuation_charts.append(f.read())
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read support_resistance: {len(continuation_charts[-1])} bytes")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading support_resistance: {e}")
        
        if continuation_charts:
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Creating continuation levels task with {len(continuation_charts)} charts")
            task = self.analyze_continuation_levels_with_calculations(continuation_charts, indicators)
            chart_analysis_tasks.append(("continuation_levels_enhanced", task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added continuation_levels_enhanced task")
        else:
            print("[ASYNC-OPTIMIZED-ENHANCED] No continuation charts available for analysis")
        
        # EXECUTE ALL INDEPENDENT TASKS IN PARALLEL
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Total tasks created: {len(chart_analysis_tasks) + 1}")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Chart analysis tasks: {[name for name, _ in chart_analysis_tasks]}")
        
        # FALLBACK: If no chart tasks were created, create mock tasks for testing
        if len(chart_analysis_tasks) == 0:
            print("[ASYNC-OPTIMIZED-ENHANCED] WARNING: No chart tasks created! Creating fallback mock tasks for testing...")
            
            # Create mock chart data for testing
            mock_chart_data = b"mock_chart_data_for_testing"
            
            # Add mock tasks for all chart analysis types
            mock_comprehensive_task = self.analyze_comprehensive_overview_with_calculations(mock_chart_data, indicators)
            chart_analysis_tasks.append(("comprehensive_overview_enhanced_mock", mock_comprehensive_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock comprehensive_overview_enhanced task")
            
            mock_volume_task = self.analyze_volume_comprehensive_with_calculations([mock_chart_data, mock_chart_data, mock_chart_data], indicators)
            chart_analysis_tasks.append(("volume_analysis_enhanced_mock", mock_volume_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock volume_analysis_enhanced task")
            
            mock_reversal_task = self.analyze_reversal_patterns_with_calculations([mock_chart_data, mock_chart_data], indicators)
            chart_analysis_tasks.append(("reversal_patterns_enhanced_mock", mock_reversal_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock reversal_patterns_enhanced task")
            
            mock_continuation_task = self.analyze_continuation_levels_with_calculations([mock_chart_data, mock_chart_data], indicators)
            chart_analysis_tasks.append(("continuation_levels_enhanced_mock", mock_continuation_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock continuation_levels_enhanced task")
            
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Created {len(chart_analysis_tasks)} mock chart tasks for testing")
        
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Executing {len(chart_analysis_tasks) + 1} independent enhanced tasks in parallel...")
        
        # DISABLE RATE LIMITING FOR TRUE PARALLEL EXECUTION
        self.core.disable_rate_limiting()
        
        import asyncio
        parallel_start_time = time.time()
        
        # Log the exact time when parallel execution starts
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Parallel execution started at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        # Combine indicator task with chart tasks
        all_tasks = [indicator_task] + [task for _, task in chart_analysis_tasks]
        
        # Log that we're about to gather all tasks
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Sending all {len(all_tasks)} tasks to asyncio.gather() at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        parallel_elapsed_time = time.time() - parallel_start_time
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Completed all independent enhanced tasks in {parallel_elapsed_time:.2f} seconds")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Parallel execution ended at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        # RE-ENABLE RATE LIMITING AFTER PARALLEL EXECUTION
        self.core.enable_rate_limiting()
        
        # Process results and handle exceptions
        ind_summary_md, ind_json = all_results[0] if not isinstance(all_results[0], Exception) else ("", {})
        if isinstance(all_results[0], Exception):
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Warning: Enhanced indicator summary failed: {all_results[0]}")
        
        # Process chart results
        chart_insights_list = []
        for i, (task_name, _) in enumerate(chart_analysis_tasks, 1):
            result = all_results[i]
            if isinstance(result, Exception):
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Warning: {task_name} failed: {result}")
                continue
            
            if task_name == "comprehensive_overview_enhanced" or task_name == "comprehensive_overview_enhanced_mock":
                chart_insights_list.append("**Comprehensive Technical Overview (Mathematically Validated):**\n" + result)
            elif task_name == "volume_analysis_enhanced" or task_name == "volume_analysis_enhanced_mock":
                chart_insights_list.append("**Comprehensive Volume Analysis (Statistically Validated):**\n" + result)
            elif task_name == "reversal_patterns_enhanced" or task_name == "reversal_patterns_enhanced_mock":
                chart_insights_list.append("**Reversal Pattern Analysis (Enhanced):**\n" + result)
            elif task_name == "continuation_levels_enhanced" or task_name == "continuation_levels_enhanced_mock":
                chart_insights_list.append("**Continuation & Level Analysis (Enhanced):**\n" + result)
        
        chart_insights_md = "\n\n".join(chart_insights_list) if chart_insights_list else ""

        # 3. Final decision prompt with enhanced mathematical validation (depends on all previous results)
        print("[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced final decision analysis...")
        decision_start_time = time.time()
        
        decision_prompt = self.prompt_manager.format_prompt(
            "final_stock_decision",
            indicator_json=json.dumps(clean_for_json(self.convert_numpy_types(ind_json)), indent=2),
            chart_insights=chart_insights_md
        )
        try:
            # Use code execution for final decision analysis
            text_response, code_results, execution_results = await self.core.call_llm_with_code_execution(decision_prompt)
            
            # Debug: Log the response to understand what we're getting
            print(f"[DEBUG] Enhanced final decision response type: {type(text_response)}")
            print(f"[DEBUG] Enhanced final decision response length: {len(text_response) if text_response else 0}")
            print(f"[DEBUG] Enhanced final decision response preview: {text_response[:200] if text_response else 'None'}")
            print(f"[DEBUG] Enhanced code execution results: {len(code_results) if code_results else 0} code snippets")
            print(f"[DEBUG] Enhanced execution outputs: {len(execution_results) if execution_results else 0} outputs")
            
            # The final decision should output ONLY JSON, not markdown with JSON inside
            # So we try to parse it directly as JSON first
            try:
                result = json.loads(text_response.strip())
                print("[DEBUG] Successfully parsed enhanced response as direct JSON")
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Enhanced direct JSON parsing failed: {e}")
                # If direct parsing fails, try to extract JSON from markdown code block
                try:
                    _, json_blob = self.extract_markdown_and_json(text_response)
                    result = json.loads(json_blob)
                    print("[DEBUG] Successfully extracted JSON from enhanced markdown code block")
                except Exception as extract_error:
                    print(f"[DEBUG] Failed to extract JSON from enhanced markdown: {extract_error}")
                    print(f"[DEBUG] Enhanced full response: {text_response}")
                    raise ValueError(f"Could not parse enhanced final decision response as JSON: {e}. Response: {text_response[:500]}")
            
            # Enhance result with code execution data
            if code_results or execution_results:
                result = self._enhance_final_decision_with_calculations(result, code_results, execution_results)
                
            # Add MTF context to result if available
            if "ENHANCED MULTI-TIMEFRAME ANALYSIS CONTEXT" in knowledge_context:
                result = self._enhance_result_with_mtf_context(result, knowledge_context)
            
            # Add sector context to result if available
            if "SECTOR CONTEXT" in knowledge_context:
                result = self._enhance_result_with_sector_context(result, knowledge_context)
                
        except Exception as ex:
            import traceback; traceback.print_exc()
            raise
        
        decision_elapsed_time = time.time() - decision_start_time
        total_elapsed_time = time.time() - parallel_start_time
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Enhanced final decision analysis completed in {decision_elapsed_time:.2f} seconds")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Total enhanced analysis completed in {total_elapsed_time:.2f} seconds")
        
        return result, ind_summary_md, chart_insights_md

    async def analyze_comprehensive_overview(self, image: bytes) -> str:
        """Analyze the comprehensive comparison chart that shows all major indicators."""
        prompt = self.prompt_manager.format_prompt("image_analysis_comprehensive_overview")
        # Add the solving line at the very end
        prompt += self.prompt_manager.SOLVING_LINE
        return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))

    async def analyze_volume_comprehensive(self, images: list, indicators: dict = None) -> str:
        """Analyze all volume-related charts together for complete volume story."""
        try:
            # Use context engineering for volume analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.VOLUME_ANALYSIS)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.VOLUME_ANALYSIS, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_volume_analysis", context=context)
            else:
                prompt = self.prompt_manager.format_prompt("image_analysis_volume_comprehensive")
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during volume analysis context engineering: {ex}")
            # Fallback to original method
            prompt = self.prompt_manager.format_prompt("image_analysis_volume_comprehensive")
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)

    async def analyze_reversal_patterns(self, images: list, indicators: dict = None) -> str:
        """Analyze divergence and double tops/bottoms charts together for reversal signals."""
        try:
            # Use context engineering for reversal pattern analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.REVERSAL_PATTERNS)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.REVERSAL_PATTERNS, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_reversal_patterns", context=context)
            else:
                prompt = self.prompt_manager.format_prompt("image_analysis_reversal_patterns")
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during reversal pattern analysis context engineering: {ex}")
            # Fallback to original method
            prompt = self.prompt_manager.format_prompt("image_analysis_reversal_patterns")
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)

    async def analyze_continuation_levels(self, images: list, indicators: dict = None) -> str:
        """Analyze triangles/flags and support/resistance charts together for continuation and levels."""
        try:
            # Use context engineering for continuation level analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.CONTINUATION_LEVELS)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.CONTINUATION_LEVELS, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_continuation_levels", context=context)
            else:
                prompt = self.prompt_manager.format_prompt("image_analysis_continuation_levels")
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during continuation level analysis context engineering: {ex}")
            # Fallback to original method
            prompt = self.prompt_manager.format_prompt("image_analysis_continuation_levels")
            prompt += self.prompt_manager.SOLVING_LINE
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
        
        # Add the solving line at the very end
        enhanced_prompt += self.prompt_manager.SOLVING_LINE
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
        
        # Add the solving line at the very end
        enhanced_prompt += self.prompt_manager.SOLVING_LINE
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(enhanced_prompt, pil_images, enable_code_execution=True)

    async def analyze_reversal_patterns_with_calculations(self, images: list, indicators: dict) -> str:
        """Analyze reversal patterns with enhanced mathematical validation."""
        enhanced_prompt = self.prompt_manager.format_prompt("image_analysis_reversal_patterns")
        enhanced_prompt += f"""

MATHEMATICAL VALIDATION REQUIRED:
Analyze the reversal pattern charts and perform the following calculations using Python code:

1. Calculate pattern reliability score using statistical methods
2. Validate divergence signals using correlation analysis
3. Calculate probability of pattern completion using historical data
4. Determine pattern strength using mathematical criteria
5. Calculate risk-reward ratios for identified patterns
6. Validate pattern confirmation signals

Technical Indicators Data: {json.dumps(clean_for_json(self.convert_numpy_types(indicators)), indent=2)}

Use Python code for all calculations and include the results in your analysis.
"""
        
        # Add the solving line at the very end
        enhanced_prompt += self.prompt_manager.SOLVING_LINE
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(enhanced_prompt, pil_images, enable_code_execution=True)

    async def analyze_continuation_levels_with_calculations(self, images: list, indicators: dict) -> str:
        """Analyze continuation patterns and support/resistance levels with enhanced calculations."""
        enhanced_prompt = self.prompt_manager.format_prompt("image_analysis_continuation_levels")
        enhanced_prompt += f"""

MATHEMATICAL VALIDATION REQUIRED:
Analyze the continuation and level charts and perform the following calculations using Python code:

1. Calculate support/resistance level strength using statistical methods
2. Validate pattern continuation probability using mathematical models
3. Calculate breakout probability and strength
4. Determine level reliability using historical testing
5. Calculate risk metrics for identified levels
6. Validate pattern completion criteria

Technical Indicators Data: {json.dumps(clean_for_json(self.convert_numpy_types(indicators)), indent=2)}

Use Python code for all calculations and include the results in your analysis.
"""
        
        # Add the solving line at the very end
        enhanced_prompt += self.prompt_manager.SOLVING_LINE
        pil_images = [self.image_utils.bytes_to_image(img) for img in images]
        return await self.core.call_llm_with_images(enhanced_prompt, pil_images, enable_code_execution=True)

    async def analyze_technical_overview(self, image: bytes) -> str:
        """Analyze the comprehensive technical overview chart that shows all major indicators and support/resistance."""
        # Provide default context to prevent KeyError
        default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
        prompt = self.prompt_manager.format_prompt("optimized_technical_overview", context=default_context)
        # Add the solving line at the very end
        prompt += self.prompt_manager.SOLVING_LINE
        return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))

    async def analyze_pattern_analysis(self, image: bytes, indicators: dict = None) -> str:
        """Analyze the comprehensive pattern analysis chart showing all reversal and continuation patterns."""
        try:
            # Use context engineering for pattern analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.REVERSAL_PATTERNS)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.REVERSAL_PATTERNS, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_pattern_analysis", context=context)
            else:
                # Provide default context when no indicators are available
                default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
                prompt = self.prompt_manager.format_prompt("optimized_pattern_analysis", context=default_context)
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during pattern analysis context engineering: {ex}")
            print(f"[DEBUG-ERROR] Exception type: {type(ex).__name__}")
            import traceback
            print(f"[DEBUG-ERROR] Traceback: {traceback.format_exc()}")
            # Fallback to original method with default context
            default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
            prompt = self.prompt_manager.format_prompt("optimized_pattern_analysis", context=default_context)
            prompt += self.prompt_manager.SOLVING_LINE
            return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))

    async def analyze_volume_analysis(self, image: bytes, indicators: dict = None) -> str:
        """Analyze the comprehensive volume analysis chart showing all volume patterns and correlations."""
        try:
            # Use context engineering for volume analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.VOLUME_ANALYSIS)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.VOLUME_ANALYSIS, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_volume_analysis", context=context)
            else:
                # Provide default context when no indicators are available
                default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
                prompt = self.prompt_manager.format_prompt("optimized_volume_analysis", context=default_context)
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during volume analysis context engineering: {ex}")
            print(f"[DEBUG-ERROR] Exception type: {type(ex).__name__}")
            import traceback
            print(f"[DEBUG-ERROR] Traceback: {traceback.format_exc()}")
            # Fallback to original method with default context
            default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
            prompt = self.prompt_manager.format_prompt("optimized_volume_analysis", context=default_context)
            prompt += self.prompt_manager.SOLVING_LINE
            return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))

    async def analyze_mtf_comparison(self, image: bytes, indicators: dict = None) -> str:
        """Analyze the multi-timeframe comparison chart for cross-timeframe validation."""
        try:
            # Use context engineering for MTF analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.FINAL_DECISION)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.FINAL_DECISION, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_mtf_comparison", context=context)
            else:
                # Provide default context when no indicators are available
                default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
                prompt = self.prompt_manager.format_prompt("optimized_mtf_comparison", context=default_context)
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during MTF comparison context engineering: {ex}")
            print(f"[DEBUG-ERROR] Exception type: {type(ex).__name__}")
            import traceback
            print(f"[DEBUG-ERROR] Traceback: {traceback.format_exc()}")
            # Fallback to original method with default context
            default_context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
            prompt = self.prompt_manager.format_prompt("optimized_mtf_comparison", context=default_context)
            prompt += self.prompt_manager.SOLVING_LINE
            return await self.core.call_llm_with_image(prompt, self.image_utils.bytes_to_image(image))

    def _build_mtf_context_for_indicators(self, mtf_context):
        """
        Build MTF context specifically for indicator analysis.
        """
        mtf_summary = mtf_context.get('summary', {})
        mtf_validation = mtf_context.get('cross_timeframe_validation', {})
        mtf_timeframes = mtf_context.get('timeframe_analyses', {})
        
        mtf_context_str = f"""

MULTI-TIMEFRAME INDICATOR CONTEXT:
This analysis includes multi-timeframe indicator data across 6 timeframes: 1min, 5min, 15min, 30min, 1hour, and 1day.

OVERALL MTF INDICATOR SUMMARY:
- Consensus Trend: {mtf_summary.get('overall_signal', 'Unknown')}
- Confidence Score: {mtf_summary.get('confidence', 0):.2%}
- Signal Alignment: {mtf_summary.get('signal_alignment', 'Unknown')}
- Risk Level: {mtf_summary.get('risk_level', 'Unknown')}

CROSS-TIMEFRAME INDICATOR VALIDATION:
- Signal Strength: {mtf_validation.get('signal_strength', 0):.2%}
- Supporting Timeframes: {', '.join(mtf_validation.get('supporting_timeframes', []))}
- Conflicting Timeframes: {', '.join(mtf_validation.get('conflicting_timeframes', []))}
- Divergence Detected: {'Yes' if mtf_validation.get('divergence_detected', False) else 'No'}

TIMEFRAME-SPECIFIC INDICATOR ANALYSIS:
"""
        
        # Add individual timeframe indicator analysis
        for timeframe, analysis in mtf_timeframes.items():
            trend = analysis.get('trend', 'Unknown')
            confidence = analysis.get('confidence', 0)
            key_indicators = analysis.get('key_indicators', {})
            
            # Determine importance based on signal quality and confidence
            if confidence > 0.8:
                importance = "🔥 HIGH IMPORTANCE"
            elif confidence > 0.6:
                importance = "⚡ MEDIUM-HIGH IMPORTANCE"
            elif confidence > 0.4:
                importance = "📊 MEDIUM IMPORTANCE"
            else:
                importance = "⚠️ LOW IMPORTANCE"
            
            mtf_context_str += f"""
{timeframe} Timeframe ({importance}):
- Trend: {trend}
- Confidence: {confidence:.2%}
- Key Indicators:
  * RSI: {key_indicators.get('rsi', 'N/A')}
  * MACD Signal: {key_indicators.get('macd_signal', 'Unknown')}
  * Volume Status: {key_indicators.get('volume_status', 'Unknown')}
  * Support Levels: {key_indicators.get('support_levels', [])}
  * Resistance Levels: {key_indicators.get('resistance_levels', [])}
"""
        
        mtf_context_str += f"""
INDICATOR ANALYSIS GUIDELINES:
1. Compare single-timeframe indicators with multi-timeframe consensus
2. Identify indicator divergences across timeframes
3. Weight indicators by timeframe importance and confidence
4. Resolve conflicts between different timeframe signals
5. Focus on high-importance timeframe indicators for primary signals
6. Use lower-importance timeframes for confirmation or early warning
"""
        
        return mtf_context_str

    def _build_sector_context_for_indicators(self, knowledge_context: str) -> str:
        """
        Build sector context specifically for indicator analysis.
        """
        sector_context_str = """
        
SECTOR ANALYSIS INTEGRATION:
This analysis includes comprehensive sector context to enhance technical indicator analysis.

SECTOR INDICATOR GUIDELINES:
1. **Sector Performance Alignment**: Compare stock indicators with sector performance
2. **Sector Rotation Impact**: Assess how sector rotation affects stock indicators
3. **Sector Correlation Analysis**: Consider sector correlations for risk assessment
4. **Sector Momentum Integration**: Incorporate sector momentum in trend analysis
5. **Sector-Based Confidence Adjustment**: Adjust confidence based on sector alignment
6. **Sector Risk Assessment**: Evaluate sector-specific risks and opportunities

SECTOR INDICATOR ANALYSIS FRAMEWORK:
- **Strong Buy Signal**: Stock indicators bullish + Sector outperforming + Sector rotation positive
- **Buy Signal**: Stock indicators bullish + Sector neutral/positive + No major sector conflicts
- **Hold Signal**: Mixed stock indicators + Sector neutral + Wait for clearer signals
- **Sell Signal**: Stock indicators bearish + Sector underperforming + Sector rotation negative
- **Strong Sell Signal**: Stock indicators bearish + Sector underperforming + Sector rotation strongly negative

SECTOR RISK CONSIDERATIONS:
- **Sector Concentration Risk**: High correlation with sector index
- **Sector Rotation Risk**: Sector moving from leading to lagging
- **Sector Volatility Risk**: Sector experiencing high volatility
- **Sector Correlation Risk**: Sector highly correlated with market
- **Sector Momentum Risk**: Sector momentum declining

SECTOR OPPORTUNITY ASSESSMENT:
- **Sector Leadership**: Sector is leading market performance
- **Sector Rotation**: Sector moving from lagging to leading
- **Sector Momentum**: Sector momentum increasing
- **Sector Diversification**: Sector provides portfolio diversification
- **Sector Stability**: Sector showing stable performance
"""
        return sector_context_str

    def _extract_mtf_context_from_analysis(self, ind_json: dict, chart_insights: str) -> dict:
        """
        Extract MTF context from indicator analysis and chart insights.
        """
        mtf_context = {
            "timeframe_alignment": {},
            "signal_consensus": {},
            "conflicts": [],
            "confidence_weighting": {}
        }
        
        # Extract timeframe information from indicator analysis
        if ind_json.get('market_outlook'):
            primary_trend = ind_json['market_outlook'].get('primary_trend', {})
            secondary_trend = ind_json['market_outlook'].get('secondary_trend', {})
            
            mtf_context['signal_consensus'] = {
                'primary_trend': primary_trend.get('direction', 'Unknown'),
                'primary_confidence': primary_trend.get('confidence', 0),
                'secondary_trend': secondary_trend.get('direction', 'Unknown'),
                'secondary_confidence': secondary_trend.get('confidence', 0)
            }
        
        # Extract conflicts from indicator analysis
        if ind_json.get('signal_conflicts', {}).get('has_conflicts', False):
            mtf_context['conflicts'].append({
                'type': 'indicator_conflict',
                'description': ind_json['signal_conflicts'].get('conflict_description', 'Unknown'),
                'resolution': ind_json['signal_conflicts'].get('resolution_guidance', 'Unknown')
            })
        
        return mtf_context

    def _extract_sector_context_from_analysis(self, ind_json: dict, chart_insights: str) -> dict:
        """
        Extract sector context from indicator analysis and chart insights.
        """
        sector_context = {
            "sector_alignment": {},
            "sector_risks": [],
            "sector_opportunities": [],
            "sector_confidence": 0
        }
        
        # Extract sector information from indicator analysis
        if ind_json.get('market_outlook', {}).get('sector_integration'):
            sector_integration = ind_json['market_outlook']['sector_integration']
            sector_context['sector_alignment'] = {
                'performance_alignment': sector_integration.get('sector_performance_alignment', 'neutral'),
                'rotation_impact': sector_integration.get('sector_rotation_impact', 'neutral'),
                'momentum_support': sector_integration.get('sector_momentum_support', 'none'),
                'confidence_boost': sector_integration.get('sector_confidence_boost', 0),
                'risk_adjustment': sector_integration.get('sector_risk_adjustment', 'unchanged')
            }
        
        # Extract sector risks from risk assessment
        if ind_json.get('risk_assessment', {}).get('sector_risk_analysis'):
            sector_risk_analysis = ind_json['risk_assessment']['sector_risk_analysis']
            sector_context['sector_risks'] = [
                sector_risk_analysis.get('sector_performance_risks', ''),
                sector_risk_analysis.get('rotation_risks', ''),
                sector_risk_analysis.get('correlation_risks', '')
            ]
        
        # Extract sector confidence
        if ind_json.get('confidence_metrics', {}).get('sector_confidence'):
            sector_context['sector_confidence'] = ind_json['confidence_metrics']['sector_confidence']
        
        return sector_context

    def _build_mtf_context_for_final_decision(self, knowledge_context: str) -> str:
        """
        Build enhanced MTF context specifically for final decision analysis.
        """
        # Extract MTF information from knowledge context
        mtf_context_str = """
        
FINAL DECISION MTF INTEGRATION:
This final decision must integrate multi-timeframe analysis with single-timeframe indicators and chart patterns.

MTF DECISION GUIDELINES:
1. **Primary Signal**: Use the highest confidence timeframe as primary signal
2. **Confirmation**: Require at least 2 supporting timeframes for high-confidence decisions
3. **Conflict Resolution**: When timeframes conflict, favor higher timeframes for trend direction
4. **Risk Assessment**: Higher timeframe conflicts indicate increased risk
5. **Entry Timing**: Use lower timeframes for precise entry timing
6. **Position Sizing**: Adjust position size based on timeframe alignment

MTF WEIGHTING FRAMEWORK:
- 1day timeframe: 40% weight (trend direction)
- 1hour timeframe: 25% weight (medium-term momentum)
- 30min timeframe: 15% weight (short-term momentum)
- 15min timeframe: 10% weight (entry timing)
- 5min timeframe: 7% weight (entry precision)
- 1min timeframe: 3% weight (micro-timing)

DECISION CRITERIA:
- Strong Buy: 3+ timeframes bullish, no major conflicts, >70% confidence
- Buy: 2+ timeframes bullish, minor conflicts, >60% confidence
- Hold: Mixed signals, significant conflicts, 40-60% confidence
- Sell: 2+ timeframes bearish, minor conflicts, >60% confidence
- Strong Sell: 3+ timeframes bearish, no major conflicts, >70% confidence
"""
        return mtf_context_str

    def _enhance_result_with_mtf_context(self, result: dict, knowledge_context: str) -> dict:
        """
        Enhance the final decision result with MTF context information.
        """
        # Extract MTF summary from knowledge context
        if "OVERALL MTF SUMMARY" in knowledge_context:
            # Parse MTF consensus information
            mtf_consensus = {
                "mtf_trend": "Unknown",
                "mtf_confidence": 0,
                "mtf_alignment": "Unknown",
                "supporting_timeframes": [],
                "conflicting_timeframes": []
            }
            
            # Extract information from knowledge context
            lines = knowledge_context.split('\n')
            for line in lines:
                if "Consensus Trend:" in line:
                    mtf_consensus["mtf_trend"] = line.split("Consensus Trend:")[1].strip()
                elif "Confidence Score:" in line:
                    try:
                        confidence_str = line.split("Confidence Score:")[1].strip()
                        mtf_consensus["mtf_confidence"] = float(confidence_str.replace('%', '')) / 100
                    except:
                        pass
                elif "Signal Alignment:" in line:
                    mtf_consensus["mtf_alignment"] = line.split("Signal Alignment:")[1].strip()
                elif "Supporting Timeframes:" in line:
                    timeframes = line.split("Supporting Timeframes:")[1].strip()
                    mtf_consensus["supporting_timeframes"] = [tf.strip() for tf in timeframes.split(',') if tf.strip()]
                elif "Conflicting Timeframes:" in line:
                    timeframes = line.split("Conflicting Timeframes:")[1].strip()
                    mtf_consensus["conflicting_timeframes"] = [tf.strip() for tf in timeframes.split(',') if tf.strip()]
            
            # Add MTF context to result
            result["mtf_context"] = mtf_consensus
            
            # Adjust confidence based on MTF alignment
            if mtf_consensus["mtf_confidence"] > 0:
                # Weight the final confidence with MTF confidence
                original_confidence = result.get("confidence_pct", 50)
                mtf_weighted_confidence = (original_confidence * 0.7) + (mtf_consensus["mtf_confidence"] * 100 * 0.3)
                result["confidence_pct"] = int(mtf_weighted_confidence)
                
                # Add MTF rationale
                if "rationale" not in result:
                    result["rationale"] = {}
                result["rationale"]["mtf_integration"] = f"MTF consensus: {mtf_consensus['mtf_trend']} with {mtf_consensus['mtf_confidence']:.1%} confidence. Supporting timeframes: {', '.join(mtf_consensus['supporting_timeframes'])}"
        
        return result

    def _build_sector_context_for_final_decision(self, knowledge_context: str) -> str:
        """
        Build enhanced sector context specifically for final decision analysis.
        """
        sector_context_str = """
        
FINAL DECISION SECTOR INTEGRATION:
This final decision must integrate sector analysis with technical indicators and multi-timeframe analysis.

SECTOR DECISION GUIDELINES:
1. **Sector Performance**: Consider sector outperformance/underperformance vs market
2. **Sector Rotation**: Assess sector rotation timing and impact
3. **Sector Correlation**: Evaluate sector correlation with market and other sectors
4. **Sector Momentum**: Consider sector momentum and trend strength
5. **Sector Risk**: Assess sector-specific risks and volatility
6. **Sector Opportunities**: Identify sector-specific opportunities

SECTOR DECISION FRAMEWORK:
- **Strong Buy**: Technical bullish + Sector outperforming + Sector rotation positive + Low sector risk
- **Buy**: Technical bullish + Sector neutral/positive + No major sector conflicts + Moderate sector risk
- **Hold**: Mixed technical signals + Sector neutral + Wait for clearer sector signals + Moderate sector risk
- **Sell**: Technical bearish + Sector underperforming + Sector rotation negative + High sector risk
- **Strong Sell**: Technical bearish + Sector underperforming + Sector rotation strongly negative + High sector risk

SECTOR POSITION SIZING:
- **Large Position**: Strong sector alignment + Low sector risk + Positive sector rotation
- **Medium Position**: Moderate sector alignment + Moderate sector risk + Neutral sector rotation
- **Small Position**: Weak sector alignment + High sector risk + Negative sector rotation
- **No Position**: Conflicting sector signals + Very high sector risk + Strong negative sector rotation

SECTOR TIMING CONSIDERATIONS:
- **Entry Timing**: Consider sector rotation timing for optimal entry
- **Exit Timing**: Monitor sector performance for exit signals
- **Holding Period**: Adjust holding period based on sector momentum
- **Risk Management**: Use sector volatility for position sizing
"""
        return sector_context_str

    def _enhance_result_with_sector_context(self, result: dict, knowledge_context: str) -> dict:
        """
        Enhance the final decision result with sector context information.
        """
        # Extract sector summary from knowledge context
        if "SECTOR CONTEXT" in knowledge_context:
            # Parse sector consensus information
            sector_consensus = {
                "sector_performance": "Unknown",
                "sector_outperformance": 0,
                "sector_beta": 1.0,
                "sector_rotation_impact": "neutral",
                "sector_risks": []
            }
            
            # Extract information from knowledge context
            lines = knowledge_context.split('\n')
            for line in lines:
                if "Market Outperformance:" in line:
                    try:
                        outperformance_str = line.split("Market Outperformance:")[1].strip()
                        sector_consensus["sector_outperformance"] = float(outperformance_str.replace('%', '')) / 100
                    except:
                        pass
                elif "Sector Outperformance:" in line:
                    try:
                        sector_outperformance_str = line.split("Sector Outperformance:")[1].strip()
                        if sector_outperformance_str != 'N/A':
                            sector_consensus["sector_performance"] = float(sector_outperformance_str.replace('%', '')) / 100
                    except:
                        pass
                elif "Sector Beta:" in line:
                    try:
                        beta_str = line.split("Sector Beta:")[1].strip()
                        sector_consensus["sector_beta"] = float(beta_str)
                    except:
                        pass
            
            # Add sector context to result
            result["sector_context"] = sector_consensus
            
            # Adjust confidence based on sector alignment
            if sector_consensus["sector_outperformance"] > 0:
                # Weight the final confidence with sector performance
                original_confidence = result.get("confidence_pct", 50)
                sector_weighted_confidence = (original_confidence * 0.6) + (sector_consensus["sector_outperformance"] * 100 * 0.4)
                result["confidence_pct"] = int(sector_weighted_confidence)
                
                # Add sector rationale
                if "rationale" not in result:
                    result["rationale"] = {}
                result["rationale"]["sector_integration"] = f"Sector outperformance: {sector_consensus['sector_outperformance']:.1%} with beta: {sector_consensus['sector_beta']:.2f}"
        
        return result



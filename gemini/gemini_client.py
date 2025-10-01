from typing import Dict, Any, List, Optional
import json
import numpy as np
from .gemini_core import GeminiCore
from .prompt_manager import PromptManager
from .image_utils import ImageUtils
from .error_utils import ErrorUtils
from .debug_logger import debug_logger
from .token_tracker import get_or_create_tracker, AnalysisTokenTracker
from .context_engineer import ContextEngineer, AnalysisType, ContextConfig
from .schema import coerce_ai_analysis, AIAnalysisSchema

import asyncio
import time

# --- Import clean_for_json ---
from core.utils import clean_for_json


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
    
    def __init__(self, api_key: str = None, context_config: ContextConfig = None, agent_name: str = None):
        self.core = GeminiCore(api_key, agent_name=agent_name)
        self.prompt_manager = PromptManager()
        self.image_utils = ImageUtils()
        self.error_utils = ErrorUtils()
        self.context_engineer = ContextEngineer(context_config)
        self.agent_name = agent_name

    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None, token_tracker=None, mtf_context=None, curated_indicators: Optional[dict] = None):
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
            # NEW: Require curated indicators provided by the new indicator agents path
            if curated_indicators is None:
                raise ValueError("curated_indicators must be provided (from new IndicatorAgentsOrchestrator). Old curator is removed.")
            timeframe = f"{period} days, {interval}"
            
            # Enhance context with MTF data if available
            enhanced_context = knowledge_context or ""
            if mtf_context and mtf_context.get('success', False):
                enhanced_context += self._build_mtf_context_for_indicators(mtf_context)
            
            # Extract and enhance sector context if available
            if knowledge_context and "SECTOR CONTEXT" in knowledge_context:
                enhanced_context += self._build_sector_context_for_indicators(knowledge_context)
            
            # Extract ML guidance if present
            try:
                ml_block = self._extract_labeled_json_block(knowledge_context or "", label="MLSystemValidation:")
                ml_guidance_text = self._build_ml_guidance_text(ml_block) if ml_block else ""
            except Exception:
                ml_guidance_text = ""

            # Structure context using context engineering
            context = self.context_engineer.structure_context(
                curated_indicators, 
                AnalysisType.INDICATOR_SUMMARY, 
                symbol, 
                timeframe, 
                (enhanced_context + ("\n\n" + ml_guidance_text if ml_guidance_text else ""))
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
        print("indicator summary agent request sent")
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
            
            # Schema validation/coercion
            try:
                validated: AIAnalysisSchema = coerce_ai_analysis(parsed)
                parsed = json.loads(validated.model_dump_json())
            except Exception as ve:
                print(f"[SCHEMA] AIAnalysis schema validation failed: {ve}")
                # Keep parsed but ensure minimal fallback keys exist
                parsed.setdefault("trend", parsed.get("trend", "neutral"))
                parsed.setdefault("confidence_pct", parsed.get("confidence_pct", 50))

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

    def _build_comprehensive_context(self, enhanced_ind_json: dict, chart_insights: str, knowledge_context: str) -> str:
        """
        Build comprehensive context for the optimized final decision template.
        This combines all analysis data into a structured format.
        """
        try:
            context_sections = []
            
            # 1. Technical Indicators Analysis
            context_sections.append("## Technical Indicators Analysis")
            context_sections.append(json.dumps(clean_for_json(self.convert_numpy_types(enhanced_ind_json)), indent=2))
            
            # 2. Chart Pattern Insights
            if chart_insights and chart_insights.strip():
                context_sections.append("\n## Chart Pattern Insights")
                context_sections.append(chart_insights)
            
            # 3. Multi-Timeframe Context
            mtf_context = self._extract_labeled_json_block(knowledge_context or "", label="MultiTimeframeContext:")
            if mtf_context:
                context_sections.append("\n## Multi-Timeframe Analysis Context")
                context_sections.append(json.dumps(clean_for_json(mtf_context), indent=2))
            
            # 4. Sector Context
            if "SECTOR CONTEXT" in (knowledge_context or ""):
                context_sections.append("\n## Sector Analysis Context")
                sector_lines = [line for line in (knowledge_context or "").split('\n') if 'SECTOR' in line or 'Market Outperformance:' in line or 'Sector Outperformance:' in line or 'Sector Beta:' in line]
                context_sections.append('\n'.join(sector_lines))
            
            # 5. Advanced Analysis Context
            adv_context = self._extract_labeled_json_block(knowledge_context or "", label="AdvancedAnalysisDigest:")
            if adv_context:
                context_sections.append("\n## Advanced Analysis Context")
                context_sections.append(json.dumps(clean_for_json(adv_context), indent=2))
            
            # 6. ML System Context
            ml_context = self._extract_labeled_json_block(knowledge_context or "", label="MLSystemValidation:")
            if ml_context:
                context_sections.append("\n## ML System Context")
                context_sections.append(json.dumps(clean_for_json(ml_context), indent=2))
            
            # 7. Existing Trading Strategy (for consistency)
            existing_strategy = enhanced_ind_json.get('existing_trading_strategy', {})
            if existing_strategy:
                context_sections.append("\n## EXISTING TRADING STRATEGY (Use as Foundation)")
                context_sections.append("The following targets and stop losses were calculated in the previous analysis phase:")
                context_sections.append(json.dumps(clean_for_json(existing_strategy), indent=2))
                context_sections.append("IMPORTANT: Use these values as your foundation and only modify them if you have strong technical reasons based on the comprehensive analysis above.")
            
            comprehensive_context = '\n'.join(context_sections)
            
            print(f"[DEBUG] Built comprehensive context with {len(context_sections)} sections, total length: {len(comprehensive_context)}")
            return comprehensive_context
            
        except Exception as e:
            print(f"[DEBUG] Error building comprehensive context: {e}")
            # Fallback to basic context
            return f"## Technical Analysis\n{json.dumps(enhanced_ind_json, indent=2)}\n\n## Chart Insights\n{chart_insights}"
    
    def _extract_existing_trading_strategy(self, ind_json: dict) -> dict:
        """
        Extract existing trading strategy data from indicator JSON for consistency in final decision.
        """
        try:
            if not isinstance(ind_json, dict):
                return {}
            
            trading_strategy = ind_json.get('trading_strategy', {})
            if not trading_strategy:
                return {}
            
            # Extract key trading data for each timeframe
            existing_strategy = {}
            
            # Short term
            short_term = trading_strategy.get('short_term', {})
            if short_term:
                existing_strategy['short_term'] = {
                    'entry_range': short_term.get('entry_strategy', {}).get('entry_range', []),
                    'stop_loss': short_term.get('exit_strategy', {}).get('stop_loss'),
                    'targets': [t.get('price') if isinstance(t, dict) else t for t in short_term.get('exit_strategy', {}).get('targets', [])],
                    'bias': short_term.get('bias'),
                    'confidence': short_term.get('confidence')
                }
            
            # Medium term
            medium_term = trading_strategy.get('medium_term', {})
            if medium_term:
                existing_strategy['medium_term'] = {
                    'entry_range': medium_term.get('entry_strategy', {}).get('entry_range', []),
                    'stop_loss': medium_term.get('exit_strategy', {}).get('stop_loss'),
                    'targets': [t.get('price') if isinstance(t, dict) else t for t in medium_term.get('exit_strategy', {}).get('targets', [])],
                    'bias': medium_term.get('bias'),
                    'confidence': medium_term.get('confidence')
                }
            
            # Long term
            long_term = trading_strategy.get('long_term', {})
            if long_term:
                existing_strategy['long_term'] = {
                    'fair_value_range': long_term.get('fair_value_range', []),
                    'investment_rating': long_term.get('investment_rating'),
                    'accumulation_zone': long_term.get('key_levels', {}).get('accumulation_zone', []),
                    'bias': long_term.get('bias'),
                    'confidence': long_term.get('confidence')
                }
            
            print(f"[DEBUG] Extracted trading strategy for timeframes: {list(existing_strategy.keys())}")
            return existing_strategy
            
        except Exception as e:
            print(f"[DEBUG] Error extracting existing trading strategy: {e}")
            return {}
    
    def _build_ml_guidance_text(self, ml_block: dict) -> str:
        """
        Convert compact ML block into short guidance text for prompts with clear weighting rules.
        """
        try:
            if not isinstance(ml_block, dict) or not ml_block:
                return ""
            lines = ["## ML System Guidance (Use as Evidence):"]
            price = ml_block.get("price") or {}
            vol = ml_block.get("volatility") or {}
            reg = ml_block.get("market_regime") or {}
            cons = ml_block.get("consensus") or {}
            patt = ml_block.get("pattern_ml") or {}

            if price:
                lines.append(f"- ML price: direction={price.get('direction')}, magnitude={price.get('magnitude')}, confidence={price.get('confidence')}")
            if vol:
                lines.append(f"- Volatility: current={vol.get('current')}, predicted={vol.get('predicted')}, regime={vol.get('regime')}")
            if reg:
                lines.append(f"- Market regime: {reg.get('regime')} (confidence={reg.get('confidence')})")
            if patt:
                lines.append(f"- Pattern ML: success_probability={patt.get('success_probability')}, confidence={patt.get('confidence')}, signal={patt.get('signal')}")
            if cons:
                lines.append(f"- ML consensus: {cons.get('overall_signal')} (confidence={cons.get('confidence')}, risk={cons.get('risk_level')})")

            # Guardrails
            lines.append("- Prefer higher-timeframe consensus when ML conflicts with MTF/indicators.")
            lines.append("- If volatility regime is 'high', reflect elevated risk and down-weight aggressive entries.")
            lines.append("- When ML confidence is low (<60%), defer to technical analysis consensus.")
            return "\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    def extract_markdown_and_json(llm_response: str):
        debug_logger.log_processing_step("Extracting markdown and JSON from LLM response")
        
        # Extracts the markdown summary and the JSON code block from the LLM response
        import re
        import json
        try:
            match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_response or "")
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
                # Try to find any JSON object in the output as a last attempt
                any_match = re.search(r"(\{[\s\S]+\})", llm_response or "")
                if any_match:
                    blob = any_match.group(1)
                    try:
                        json.loads(blob)
                        return (llm_response, blob)
                    except Exception:
                        pass
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



    # ===== Auxiliary synthesis helpers to align with rules.txt (chunk, label, single-purpose) =====
    def _extract_labeled_json_block(self, context: str, label: str) -> dict | None:
        try:
            if not context or label not in context:
                return None
            after = context.split(label, 1)[1].strip()
            # Try to read a JSON object starting there
            import re
            m = re.search(r"\{[\s\S]+\}", after)
            if not m:
                return None
            return json.loads(m.group(0))
        except Exception:
            return None

    async def synthesize_mtf_summary(self, mtf_json: dict) -> str:
        """Single-purpose: Synthesize MTF JSON into structured decision with timeframe weighting."""
        try:
            prompt = self.prompt_manager.format_prompt(
                "optimized_mtf_comparison",
                context=f"""
[Source: MultiTimeframeContext]
Multi-Timeframe Analysis Data:
{json.dumps(mtf_json, indent=2)[:8000]}
"""
            ) + self.prompt_manager.SOLVING_LINE
            text = await self.core.call_llm(prompt)
            return text or ""
        except Exception:
            return ""

    async def synthesize_risk_summary(self, adv_json: dict) -> str:
        """Single-purpose: Summarize risk/stress/scenario digest in 5 bullets."""
        try:
            prompt = self.prompt_manager.format_prompt(
                "risk_synthesis_template",
                context=f"""
[Source: AdvancedAnalysisDigest]
Analyze the following risk/stress/scenario analysis data and provide focused risk synthesis.

Risk Analysis Data:
{json.dumps(adv_json, indent=2)[:4000]}
"""
            ) + self.prompt_manager.SOLVING_LINE
            text = await self.core.call_llm(prompt)
            return text or ""
        except Exception:
            return ""

    async def synthesize_sector_summary(self, knowledge_context: str) -> str:
        """Single-purpose: Summarize sector context lines into 4 bullets (concise, timeframed).
        Uses a robust LLM call path with tolerant text extraction and retries to avoid intermittent empty responses.
        """
        try:
            import re
            ctx = knowledge_context or ""
            # Extract key metrics from knowledge_context
            def extract(pattern: str) -> str | None:
                m = re.search(pattern, ctx, re.IGNORECASE)
                return m.group(1).strip() if m else None

            sector_out = extract(r"-\s*Sector\s*Outperformance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            market_out = extract(r"-\s*Market\s*Outperformance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            sector_beta = extract(r"-\s*Sector\s*Beta:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            market_beta = extract(r"-\s*Market\s*Beta:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            rotation_stage = extract(r"-\s*Rotation\s*Stage:\s*([A-Za-z]+)")
            rotation_mom = extract(r"-\s*Rotation\s*Momentum:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            sector_name = extract(r"-\s*Sector:\s*(.+)")
            
            # Extract additional metrics
            sector_corr = extract(r"-\s*Sector\s*Correlation:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_corr = extract(r"-\s*Market\s*Correlation:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            sector_sharpe = extract(r"-\s*Sector\s*Sharpe:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_sharpe = extract(r"-\s*Market\s*Sharpe:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            sector_vol = extract(r"-\s*Sector\s*Volatility:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_vol = extract(r"-\s*Market\s*Volatility:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            sector_ret = extract(r"-\s*Sector\s*Return:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_ret = extract(r"-\s*Market\s*Return:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")

            # Build concise, explicit context with timeframes
            metrics_lines = []
            if sector_out is not None:
                metrics_lines.append(f"- Sector Outperformance (12m): {sector_out}%")
            if market_out is not None:
                metrics_lines.append(f"- Market Outperformance (12m): {market_out}%")
            if sector_beta is not None:
                metrics_lines.append(f"- Sector Beta (12m): {sector_beta}")
            if rotation_stage is not None:
                metrics_lines.append(f"- Rotation Stage (3m): {rotation_stage}")
            if rotation_mom is not None:
                metrics_lines.append(f"- Rotation Momentum (3m): {rotation_mom}%")

            additional_lines = []
            if sector_name:
                additional_lines.append(f"- Sector: {sector_name}")
            if market_beta is not None:
                additional_lines.append(f"- Market Beta (12m): {market_beta}")
            
            # Enhanced metrics in additional context
            if sector_corr is not None:
                additional_lines.append(f"- Sector Correlation: {sector_corr}%")
            if market_corr is not None:
                additional_lines.append(f"- Market Correlation: {market_corr}%")
            if sector_sharpe is not None:
                additional_lines.append(f"- Sector Sharpe: {sector_sharpe}")
            if market_sharpe is not None:
                additional_lines.append(f"- Market Sharpe: {market_sharpe}")
            if sector_vol is not None:
                additional_lines.append(f"- Sector Volatility: {sector_vol}%")
            if market_vol is not None:
                additional_lines.append(f"- Market Volatility: {market_vol}%")
            if sector_ret is not None:
                additional_lines.append(f"- Sector Return: {sector_ret}%")
            if market_ret is not None:
                additional_lines.append(f"- Market Return: {market_ret}%")

            concise_context = f"""
[Source: SectorContext]
Timeframes: Relative performance and beta = 12m; Rotation = 3m

Sector Metrics:
{chr(10).join(metrics_lines) if metrics_lines else 'N/A'}

Additional Context (if available):
{chr(10).join(additional_lines) if additional_lines else 'None'}
"""

            prompt = self.prompt_manager.format_prompt(
                "sector_synthesis_template",
                context=concise_context
            ) + self.prompt_manager.SOLVING_LINE

            # Prefer the robust path with retries and tolerant extraction
            text, _code, _exec = await self.core.call_llm_with_code_execution(
                prompt, return_full_response=False
            )
            if text and isinstance(text, str) and text.strip():
                return text

            # Fallback once to the basic call
            fallback = await self.core.call_llm(prompt)
            return fallback or ""
        except Exception:
            return ""

    async def verify_and_format_final_json(self, result: dict) -> dict:
        """Single-purpose: Verify/normalize final JSON to schema; return corrected JSON."""
        try:
            prompt = self.prompt_manager.format_prompt(
                "optimized_indicators_summary",
                context=f"""
Task: Validate and normalize the following JSON to ensure fields exist: trend, confidence_pct, signals (optional), rationale (optional).
If keys missing, add conservative defaults. Return only JSON.

JSON:
{json.dumps(result)}
"""
            ) + self.prompt_manager.SOLVING_LINE
            text = await self.core.call_llm(prompt)
            try:
                return json.loads(text)
            except Exception:
                _, blob = self.extract_markdown_and_json(text or "")
                return json.loads(blob)
        except Exception:
            return result

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
    
    def _create_basic_key_indicators_from_raw(self, indicators: dict) -> dict:
        """Create basic key indicators structure from raw indicators for fallback."""
        try:
            key_indicators = {}
            
            # Create basic trend indicators from raw data
            if isinstance(indicators, dict):
                # Moving averages
                mov = indicators.get('moving_averages', {})
                if isinstance(mov, dict):
                    trend_indicators = {
                        "direction": "neutral",
                        "strength": "weak", 
                        "confidence": 0.3  # Low confidence for fallback
                    }
                    
                    # Add numeric values if available
                    for key in ['sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50']:
                        if key in mov:
                            try:
                                trend_indicators[key] = round(float(mov[key]), 2)
                            except:
                                pass
                    
                    # Add percentage values if available
                    for key in ['price_to_sma_200', 'sma_20_to_sma_50']:
                        if key in mov:
                            try:
                                trend_indicators[key] = round(float(mov[key]), 2)
                            except:
                                pass
                    
                    # Add boolean values if available
                    for key in ['golden_cross', 'death_cross']:
                        if key in mov:
                            try:
                                trend_indicators[key] = bool(mov[key])
                            except:
                                trend_indicators[key] = False
                    
                    key_indicators["trend_indicators"] = trend_indicators
                
                # Momentum indicators
                momentum_indicators = {
                    "direction": "neutral",
                    "strength": "weak",
                    "confidence": 0.3,
                    "rsi_status": "neutral"
                }
                
                # RSI
                rsi = indicators.get('rsi')
                if isinstance(rsi, dict) and 'rsi_14' in rsi:
                    try:
                        rsi_val = float(rsi['rsi_14'])
                        momentum_indicators["rsi_current"] = round(rsi_val, 2)
                        
                        if rsi_val > 70:
                            momentum_indicators["rsi_status"] = "overbought"
                        elif rsi_val < 30:
                            momentum_indicators["rsi_status"] = "oversold"
                        else:
                            momentum_indicators["rsi_status"] = "neutral"
                    except:
                        pass
                
                # MACD  
                macd = indicators.get('macd')
                if isinstance(macd, dict) and 'histogram' in macd:
                    try:
                        hist_val = float(macd['histogram'])
                        momentum_indicators["macd"] = {
                            "histogram": round(hist_val, 2),
                            "trend": "bullish" if hist_val > 0 else "bearish"
                        }
                        
                        if hist_val > 0:
                            momentum_indicators["direction"] = "bullish"
                        elif hist_val < 0:
                            momentum_indicators["direction"] = "bearish"
                    except:
                        pass
                
                key_indicators["momentum_indicators"] = momentum_indicators
                
                # Volume indicators
                vol = indicators.get('volume', {})
                if isinstance(vol, dict) and 'volume_ratio' in vol:
                    try:
                        vol_ratio = float(vol['volume_ratio'])
                        key_indicators["volume_indicators"] = {
                            "volume_ratio": round(vol_ratio, 2),
                            "volume_trend": "high" if vol_ratio > 1.5 else "low" if vol_ratio < 0.8 else "neutral"
                        }
                    except:
                        pass
            
            return key_indicators
            
        except Exception as e:
            print(f"Error creating basic key indicators from raw: {e}")
            return {}

    async def analyze_stock_with_enhanced_calculations(self, symbol, indicators, chart_paths, period, interval, knowledge_context="", exchange: str = "NSE"):
        """
        Enhanced version of analyze_stock with comprehensive mathematical validation.
        Now includes ML system feedback to enhance LLM analysis.
        """
        # Store current knowledge context for ML enhancement
        self._current_knowledge_context = knowledge_context
        
        # Ensure chart_paths is a dict
        if chart_paths is None:
            chart_paths = {}
        
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced optimized analysis for {symbol}...")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Chart paths received: {list(chart_paths.keys()) if chart_paths else 'None'}")
        # Print chart metadata without binary data
        chart_metadata = {}
        for chart_name, chart_info in chart_paths.items():
            if isinstance(chart_info, dict):
                chart_metadata[chart_name] = {
                    'type': chart_info.get('type'),
                    'format': chart_info.get('format'),
                    'size_bytes': chart_info.get('size_bytes'),
                    'chart_type': chart_info.get('chart_type'),
                    'symbol': chart_info.get('symbol'),
                    'interval': chart_info.get('interval')
                }
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Chart metadata: {chart_metadata}")
        
        # START ALL INDEPENDENT LLM CALLS IMMEDIATELY
        # 1. Enhanced indicator summary with mathematical validation (no dependencies)
        print("[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced indicator summary analysis...")
        
        # Create fallback curated indicators since this path doesn't have access to the integration manager
        fallback_curated_indicators = {
            "analysis_focus": "technical_indicators_summary",
            "key_indicators": self._create_basic_key_indicators_from_raw(indicators),
            "critical_levels": {},
            "conflict_analysis_needed": False,
            "detected_conflicts": {"has_conflicts": False, "conflict_count": 0, "conflict_list": []},
            "fallback_used": True,
            "source": "analyze_stock_with_enhanced_calculations_fallback"
        }
        
        indicator_task = self.build_indicators_summary(
            symbol=symbol,
            indicators=indicators,
            period=period,
            interval=interval,
            knowledge_context=knowledge_context,
            curated_indicators=fallback_curated_indicators
        )

        # 2. Enhanced chart analysis with code execution - START ALL CHART TASKS IMMEDIATELY
        print("[ASYNC-OPTIMIZED-ENHANCED] Starting all enhanced chart analysis tasks...")
        chart_analysis_tasks = []
        
        # GROUP 1: Technical Overview (comprehensive technical analysis)
        # print(f"[ASYNC-OPTIMIZED-ENHANCED] Checking for technical_overview: {chart_paths.get('technical_overview')}")
        if chart_paths.get('technical_overview') and chart_paths['technical_overview'].get('type') == 'image_bytes':
            try:
                # Load image bytes directly from chart data
                technical_chart = chart_paths['technical_overview']['data']
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read technical_overview: {len(technical_chart)} bytes")
                task = self.analyze_technical_overview(technical_chart)
                chart_analysis_tasks.append(("technical_overview_enhanced", task))
                print("[ASYNC-OPTIMIZED-ENHANCED] Added technical_overview_enhanced task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading technical_overview: {e}")
        else:
            print("[ASYNC-OPTIMIZED-ENHANCED] technical_overview not found or not in image_bytes format")
        
        # GROUP 2: Pattern Analysis (all pattern recognition)
        # print(f"[ASYNC-OPTIMIZED-ENHANCED] Checking for pattern_analysis: {chart_paths.get('pattern_analysis')}")
        if chart_paths.get('pattern_analysis') and chart_paths['pattern_analysis'].get('type') == 'image_bytes':
            try:
                # Load image bytes directly from chart data
                pattern_chart = chart_paths['pattern_analysis']['data']
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Successfully read pattern_analysis: {len(pattern_chart)} bytes")
                task = self.analyze_pattern_analysis(pattern_chart, indicators)
                chart_analysis_tasks.append(("pattern_analysis_enhanced", task))
                print("[ASYNC-OPTIMIZED-ENHANCED] Added pattern_analysis_enhanced task")
            except Exception as e:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Error reading pattern_analysis: {e}")
        else:
            print("[ASYNC-OPTIMIZED-ENHANCED] pattern_analysis not found or not in image_bytes format")
        
        # GROUP 3: Volume Analysis
        # Replaced by distributed volume agents system handled in the orchestrator.
        print("[ASYNC-OPTIMIZED-ENHANCED] Skipping legacy volume_analysis_enhanced task (using distributed volume agents instead)")
        
        # GROUP 4: Multi-Timeframe Comparison (MTF validation)
        # REMOVED: Old mtf_comparison chart analysis - this chart was misleading as it showed
        # multi-period moving averages on a single timeframe rather than true multi-timeframe data.
        # The new MTF visualization (backend/agents/mtf_analysis/visualization.py) will be used
        # with its own dedicated LLM synthesis function in the future.
        print("[ASYNC-OPTIMIZED-ENHANCED] Skipping legacy mtf_comparison chart analysis (chart removed from orchestrator)")
        
        # 3. AUXILIARY SYNTHESIS TASKS (rules: chunk and label sources; keep tasks single-purpose)
        aux_tasks = []
        parsed_mtf = self._extract_labeled_json_block(knowledge_context, label="MultiTimeframeContext:")
        parsed_adv = self._extract_labeled_json_block(knowledge_context, label="AdvancedAnalysisDigest:")
        has_sector = ("SECTOR CONTEXT" in (knowledge_context or ""))

        if parsed_mtf:
            aux_tasks.append(("mtf_synthesis", self.synthesize_mtf_summary(parsed_mtf)))
        if parsed_adv:
            aux_tasks.append(("risk_synthesis", self.synthesize_risk_summary(parsed_adv)))
        if has_sector:
            aux_tasks.append(("sector_synthesis", self.synthesize_sector_summary(knowledge_context)))
        
        # Add pattern detection as a parallel task
        try:
            from .parallel_pattern_detection import parallel_pattern_detection
            # Fetch stock data for pattern detection via central data provider (non-blocking)
            stock_data_for_patterns = None
            try:
                from services.central_data_provider import CentralDataProvider
                central_data_provider = CentralDataProvider()
                stock_data_for_patterns = await central_data_provider.get_stock_data_async(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    period=period,
                )
            except Exception as e_fetch:
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Warning: could not fetch stock data for pattern detection: {e_fetch}")

            if stock_data_for_patterns is not None and not getattr(stock_data_for_patterns, 'empty', True):
                pattern_task = parallel_pattern_detection.detect_patterns_async(stock_data_for_patterns)
                aux_tasks.append(("pattern_detection", pattern_task))
                print("[ASYNC-OPTIMIZED-ENHANCED] Added pattern_detection task")
            else:
                print("[ASYNC-OPTIMIZED-ENHANCED] Skipping pattern_detection task due to missing data")
        except Exception as e:
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Error adding pattern detection task: {e}")

        # REMOVED: Duplicate volume agents call - this is now handled in orchestrator.py
        # to avoid running volume agents multiple times for the same analysis
        print("[VOLUME_AGENT_DEBUG] Volume agents analysis handled by orchestrator - skipping duplicate call")

        # EXECUTE ALL INDEPENDENT TASKS IN PARALLEL
        total_independent = 1 + len(chart_analysis_tasks) + len(aux_tasks)
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Total tasks created: {total_independent}")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Chart analysis tasks: {[name for name, _ in chart_analysis_tasks]}")
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Aux tasks: {[name for name, _ in aux_tasks]}")
        
        # FALLBACK: If no chart tasks were created, create mock tasks for testing
        if len(chart_analysis_tasks) == 0:
            print("[ASYNC-OPTIMIZED-ENHANCED] WARNING: No chart tasks created! Creating fallback mock tasks for testing...")
            
            # Create mock chart data for testing
            mock_chart_data = b"mock_chart_data_for_testing"
            
            # Add mock tasks for all chart analysis types
            mock_technical_task = self.analyze_technical_overview(mock_chart_data)
            chart_analysis_tasks.append(("technical_overview_enhanced_mock", mock_technical_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock technical_overview_enhanced task")
            
            mock_pattern_task = self.analyze_pattern_analysis(mock_chart_data, indicators)
            chart_analysis_tasks.append(("pattern_analysis_enhanced_mock", mock_pattern_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock pattern_analysis_enhanced task")
            
            mock_volume_task = self.analyze_volume_analysis(mock_chart_data, indicators)
            chart_analysis_tasks.append(("volume_analysis_enhanced_mock", mock_volume_task))
            print("[ASYNC-OPTIMIZED-ENHANCED] Added mock volume_analysis_enhanced task")
            
            # REMOVED: Mock mtf_comparison task - old chart was misleading
            # mock_mtf_task = self.analyze_mtf_comparison(mock_chart_data, indicators)
            # chart_analysis_tasks.append(("mtf_comparison_enhanced_mock", mock_mtf_task))
            # print("[ASYNC-OPTIMIZED-ENHANCED] Added mock mtf_comparison_enhanced task")
            
            print(f"[ASYNC-OPTIMIZED-ENHANCED] Created {len(chart_analysis_tasks)} mock chart tasks for testing")
        
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Executing {len(chart_analysis_tasks) + 1} independent enhanced tasks in parallel...")
        
        # DISABLE RATE LIMITING FOR TRUE PARALLEL EXECUTION
        self.core.disable_rate_limiting()
        
        import asyncio
        parallel_start_time = time.time()
        
        # Log the exact time when parallel execution starts
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Parallel execution started at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        # Combine indicator task with chart tasks and auxiliary synthesis tasks
        all_tasks = [indicator_task] + [task for _, task in chart_analysis_tasks] + [task for _, task in aux_tasks]
        
        # Create a mapping of tasks to their names for result processing
        task_to_name = {}
        task_to_name[indicator_task] = "indicator_summary"
        for name, task in chart_analysis_tasks:
            task_to_name[task] = name
        for name, task in aux_tasks:
            task_to_name[task] = name
        
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
            
            if task_name == "technical_overview_enhanced" or task_name == "technical_overview_enhanced_mock":
                chart_insights_list.append("**Technical Overview (Comprehensive Analysis):**\n" + result)
            elif task_name == "pattern_analysis_enhanced" or task_name == "pattern_analysis_enhanced_mock":
                chart_insights_list.append("**Pattern Analysis (All Pattern Recognition):**\n" + result)
            elif task_name == "volume_analysis_enhanced" or task_name == "volume_analysis_enhanced_mock":
                chart_insights_list.append("**Volume Analysis (Complete Volume Story):**\n" + result)
            # REMOVED: mtf_comparison_enhanced result processing - old chart was misleading
            # elif task_name == "mtf_comparison_enhanced" or task_name == "mtf_comparison_enhanced_mock":
            #     chart_insights_list.append("**Multi-Timeframe Comparison (MTF Validation):**\n" + result)
        
        # Process auxiliary synthesis results
        aux_insights = []
        aux_offset = 1 + len(chart_analysis_tasks)
        for j, (aux_name, _) in enumerate(aux_tasks, aux_offset):
            result = all_results[j]
            if isinstance(result, Exception):
                print(f"[ASYNC-OPTIMIZED-ENHANCED] Warning: {aux_name} failed: {result}")
                continue
            title_map = {
                "mtf_synthesis": "Multi-Timeframe Synthesis",
                "risk_synthesis": "Risk and Scenario Synthesis", 
                "sector_synthesis": "Sector Context Synthesis",
            }
            aux_insights.append(f"**{title_map.get(aux_name, aux_name)}:**\n" + str(result))

        chart_insights_md = "\n\n".join(chart_insights_list + aux_insights) if (chart_insights_list or aux_insights) else ""

        # Volume agents context is now handled in orchestrator.py - no need to inject here
        final_knowledge_context = knowledge_context
        print("[VOLUME_AGENT_DEBUG] Volume agents context handled by orchestrator - using original knowledge context")

        # 3. Final decision prompt with enhanced mathematical validation (depends on all previous results)
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced final decision analysis...")
        decision_start_time = time.time()
        
        # Extract existing trading strategy data for consistency
        existing_trading_strategy = self._extract_existing_trading_strategy(ind_json)
        
        # Add existing trading strategy context to indicator JSON for final decision
        enhanced_ind_json = ind_json.copy() if isinstance(ind_json, dict) else {}
        if existing_trading_strategy:
            enhanced_ind_json['existing_trading_strategy'] = existing_trading_strategy
            print(f"[DEBUG] Added existing trading strategy to final decision context: {list(existing_trading_strategy.keys())}")
        
        decision_prompt = self.prompt_manager.format_prompt(
            "optimized_final_decision",
            context=self._build_comprehensive_context(enhanced_ind_json, chart_insights_md, final_knowledge_context)
        )
        # Append ML guidance to final decision prompt if present in knowledge context
        try:
            ml_block = self._extract_labeled_json_block(knowledge_context or "", label="MLSystemValidation:")
            ml_guidance_text = self._build_ml_guidance_text(ml_block) if ml_block else ""
            if ml_guidance_text:
                decision_prompt += "\n\n" + ml_guidance_text
        except Exception:
            pass
        try:
            # Use code execution for final decision analysis
            text_response, code_results, execution_results = await self.core.call_llm_with_code_execution(decision_prompt)
            
            # Debug: Log the response to understand what we're getting
            print(f"[DEBUG] Enhanced final decision response type: {type(text_response)}")
            print(f"[DEBUG] Enhanced final decision response length: {len(text_response) if text_response else 0}")
            print(f"[DEBUG] Enhanced final decision response preview: {text_response[:200] if text_response else 'None'}")
            print(f"[DEBUG] Enhanced code execution results: {len(code_results) if code_results else 0} code snippets")
            print(f"[DEBUG] Enhanced execution outputs: {len(execution_results) if execution_results else 0} outputs")
            
            # The final decision should output ONLY JSON. Guard for empty or malformed responses.
            if not text_response or not str(text_response).strip():
                print("[DEBUG] Empty final decision response; using fallback JSON.")
                result = json.loads(self._create_fallback_json())
                result.setdefault('analysis_metadata', {})
                result['analysis_metadata']['fallback_reason'] = 'empty_final_decision_response'
            else:
                # Try direct JSON first
                try:
                    result = json.loads(text_response.strip())
                    print("[DEBUG] Successfully parsed enhanced response as direct JSON")
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] Enhanced direct JSON parsing failed: {e}")
                    # Try to extract JSON from markdown code block
                    try:
                        _, json_blob = self.extract_markdown_and_json(text_response)
                        result = json.loads(json_blob)
                        print("[DEBUG] Successfully extracted JSON from enhanced markdown code block")
                    except Exception as extract_error:
                        print(f"[DEBUG] Failed to extract JSON from enhanced markdown: {extract_error}")
                        print(f"[DEBUG] Enhanced full response: {text_response}")
                        # Final fallback
                        result = json.loads(self._create_fallback_json())
                        result.setdefault('analysis_metadata', {})
                        result['analysis_metadata']['fallback_reason'] = 'unparsable_final_decision_response'
            
            # Enhance result with code execution data
            if code_results or execution_results:
                result = self._enhance_final_decision_with_calculations(result, code_results, execution_results)

            # Schema verification/normalization pass removed to avoid extra LLM call post final decision
                
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

    async def run_final_decision(self, ind_json: dict, chart_insights_md: str, knowledge_context: str) -> dict:
        """Run only the final decision LLM call given precomputed indicator JSON and chart insights.
        Adds ML guidance if present in knowledge_context and applies parsing fallbacks.
        """
        final_knowledge_context = knowledge_context or ""
        print("[ASYNC-OPTIMIZED-ENHANCED] Starting enhanced final decision analysis (standalone)...")
        start_ts = time.time()

        existing_trading_strategy = self._extract_existing_trading_strategy(ind_json)
        enhanced_ind_json = ind_json.copy() if isinstance(ind_json, dict) else {}
        if existing_trading_strategy:
            enhanced_ind_json['existing_trading_strategy'] = existing_trading_strategy

        decision_prompt = self.prompt_manager.format_prompt(
            "optimized_final_decision",
            context=self._build_comprehensive_context(enhanced_ind_json, chart_insights_md or "", final_knowledge_context)
        )
        try:
            ml_block = self._extract_labeled_json_block(knowledge_context or "", label="MLSystemValidation:")
            ml_guidance_text = self._build_ml_guidance_text(ml_block) if ml_block else ""
            if ml_guidance_text:
                decision_prompt += "\n\n" + ml_guidance_text
        except Exception:
            pass

        # Simple retry for transient 503s
        attempts = 0
        while True:
            attempts += 1
            try:
                text_response, code_results, execution_results = await self.core.call_llm_with_code_execution(decision_prompt)
                if not text_response or not str(text_response).strip():
                    result = json.loads(self._create_fallback_json())
                    result.setdefault('analysis_metadata', {})
                    result['analysis_metadata']['fallback_reason'] = 'empty_final_decision_response'
                else:
                    try:
                        result = json.loads(text_response.strip())
                    except json.JSONDecodeError:
                        try:
                            _, json_blob = self.extract_markdown_and_json(text_response)
                            result = json.loads(json_blob)
                        except Exception:
                            result = json.loads(self._create_fallback_json())
                            result.setdefault('analysis_metadata', {})
                            result['analysis_metadata']['fallback_reason'] = 'unparsable_final_decision_response'
                if code_results or execution_results:
                    result = self._enhance_final_decision_with_calculations(result, code_results, execution_results)
                if "ENHANCED MULTI-TIMEFRAME ANALYSIS CONTEXT" in (knowledge_context or ""):
                    result = self._enhance_result_with_mtf_context(result, knowledge_context)
                if "SECTOR CONTEXT" in (knowledge_context or ""):
                    result = self._enhance_result_with_sector_context(result, knowledge_context)
                break
            except Exception as ex:
                # Best-effort transient retry only once
                if attempts < 2 and "503" in str(ex):
                    await asyncio.sleep(1.0)
                    continue
                raise

        elapsed = time.time() - start_ts
        print(f"[ASYNC-OPTIMIZED-ENHANCED] Standalone final decision completed in {elapsed:.2f}s")
        return result


    # DEPRECATED: This method has been replaced by the distributed volume agents system
    # The new system provides more specialized and accurate analysis through 5 coordinated agents
    # async def analyze_volume_comprehensive(self, images: list, indicators: dict = None) -> str:
    #     """DEPRECATED: Use volume agents system instead"""
    #     pass

    async def analyze_reversal_patterns(self, images: list, indicators: dict = None) -> str:
        """Analyze divergence and double tops/bottoms charts together for reversal signals."""
        try:
            # Use context engineering for reversal pattern analysis
            if indicators:
                curated_indicators = self.context_engineer.curate_indicators(indicators, AnalysisType.REVERSAL_PATTERNS)
                context = self.context_engineer.structure_context(curated_indicators, AnalysisType.REVERSAL_PATTERNS, "", "", "")
                prompt = self.prompt_manager.format_prompt("optimized_reversal_patterns", context=context)
            else:
                # Use optimized template with minimal context
                context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and reversal indicators."
                prompt = self.prompt_manager.format_prompt("optimized_reversal_patterns", context=context)
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during reversal pattern analysis context engineering: {ex}")
            # Fallback with minimal context
            context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and reversal indicators."
            prompt = self.prompt_manager.format_prompt("optimized_reversal_patterns", context=context)
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
                # Use optimized template with minimal context
                context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and continuation indicators."
                prompt = self.prompt_manager.format_prompt("optimized_continuation_levels", context=context)
            
            # Add the solving line at the very end
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)
        except Exception as ex:
            print(f"[DEBUG-ERROR] Exception during continuation level analysis context engineering: {ex}")
            # Fallback with minimal context
            context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and continuation indicators."
            prompt = self.prompt_manager.format_prompt("optimized_continuation_levels", context=context)
            prompt += self.prompt_manager.SOLVING_LINE
            pil_images = [self.image_utils.bytes_to_image(img) for img in images]
            return await self.core.call_llm_with_images(prompt, pil_images)


    # DEPRECATED: This method has been replaced by the distributed volume agents system
    # The volume agents provide specialized statistical validation through individual processors
    # async def analyze_volume_comprehensive_with_calculations(self, images: list, indicators: dict) -> str:
    #     """DEPRECATED: Use volume agents system instead"""
    #     pass

    async def analyze_reversal_patterns_with_calculations(self, images: list, indicators: dict) -> str:
        """Analyze reversal patterns with enhanced mathematical validation."""
        # Use optimized template with minimal context
        context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and reversal indicators."
        enhanced_prompt = self.prompt_manager.format_prompt("optimized_reversal_patterns", context=context)
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
        # Use optimized template with minimal context
        context = "## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and continuation indicators."
        enhanced_prompt = self.prompt_manager.format_prompt("optimized_continuation_levels", context=context)
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
        pil_image = await self.image_utils.bytes_to_image_async(image)
        return await self.core.call_llm_with_image(prompt, pil_image)

    async def analyze_pattern_analysis(self, image: bytes, indicators: dict = None) -> str:
        """
        Analyze the comprehensive pattern analysis chart showing all reversal and continuation patterns.
        Now enhanced with ML validation context for better pattern analysis.
        """
        try:
            # Get ML validation context if available
            ml_context = self._extract_ml_validation_context(getattr(self, '_current_knowledge_context', ''))
            
            # Create enhanced prompt with ML context
            if ml_context and self.prompt_manager:
                try:
                    enhanced_prompt = self.prompt_manager.create_ml_enhanced_prompt(
                        'optimized_pattern_analysis',
                        ml_context,
                        context=f"## Analysis Context:\nAnalyze the pattern analysis chart with ML validation insights. Use the ML system's pattern success probabilities and risk assessments to enhance your analysis."
                    )
                except Exception as e:
                    print(f"Failed to create ML-enhanced prompt, using default: {e}")
                    enhanced_prompt = self.prompt_manager.format_prompt(
                        'optimized_pattern_analysis',
                        context="## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
                    )
            else:
                # Fallback to default prompt
                enhanced_prompt = self.prompt_manager.format_prompt(
                    'optimized_pattern_analysis',
                    context="## Analysis Context:\nNo additional context provided. Analyze the chart based on visual patterns and technical indicators."
                )
            
            pil_image = await self.image_utils.bytes_to_image_async(image)
            return await self.core.call_llm_with_image(enhanced_prompt, pil_image)
            
        except Exception as e:
            print(f"Error in analyze_pattern_analysis: {e}")
            return f"Pattern analysis failed: {str(e)}"

    async def analyze_volume_analysis(self, image: bytes, indicators: dict = None) -> str:
        """
        Analyze the comprehensive volume analysis chart.
        UPDATED: Now provides fallback to old method while volume agents system is the primary approach.
        """
        try:
            # FALLBACK: Use the old single volume analysis method for chart analysis
            # The distributed volume agents system is used in the main orchestrator workflow
            # This method remains for backward compatibility with chart analysis
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
            pil_image = await self.image_utils.bytes_to_image_async(image)
            return await self.core.call_llm_with_image(prompt, pil_image)
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
            pil_image = await self.image_utils.bytes_to_image_async(image)
            return await self.core.call_llm_with_image(prompt, pil_image)
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

    async def analyze_volume_agent_specific(self, chart_image: bytes, prompt_text: str, agent_name: str) -> str:
        """
        Analyze volume agent chart using agent-specific prompt and chart image.
        This method is called by individual volume agents for LLM analysis.
        
        Args:
            chart_image: Chart image bytes generated by the volume agent
            prompt_text: Agent-specific prompt text with analysis context
            agent_name: Name of the volume agent (for template mapping)
        
        Returns:
            LLM analysis response as string
        """
        try:
            # Map agent names to their corresponding prompt templates
            agent_template_map = {
                'volume_anomaly': 'volume_anomaly_detection',
                'institutional_activity': 'institutional_activity_analysis', 
                'volume_confirmation': 'volume_confirmation_analysis',
                'support_resistance': 'volume_support_resistance',
                'volume_momentum': 'volume_trend_momentum'
            }
            
            # Get the appropriate template for this agent
            template_name = agent_template_map.get(agent_name)
            if not template_name:
                # Fallback: use the prompt text directly if no template mapping found
                print(f"[VOLUME_AGENT_DEBUG] No template mapping found for {agent_name}, using provided prompt text")
                final_prompt = prompt_text
            else:
                # Use the agent-specific template with the provided context
                try:
                    final_prompt = self.prompt_manager.format_prompt(template_name, context=prompt_text)
                except Exception as template_error:
                    print(f"[VOLUME_AGENT_DEBUG] Template formatting failed for {agent_name}: {template_error}, using provided prompt text")
                    final_prompt = prompt_text
            
            # Add the solving line at the end
            final_prompt += self.prompt_manager.SOLVING_LINE
            
            # Convert bytes to PIL Image and call LLM
            pil_image = await self.image_utils.bytes_to_image_async(chart_image)
            response = await self.core.call_llm_with_image(final_prompt, pil_image, enable_code_execution=True)
            
            print(f"[VOLUME_AGENT_DEBUG] {agent_name} LLM analysis completed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Volume agent {agent_name} LLM analysis failed: {str(e)}"
            print(f"[VOLUME_AGENT_DEBUG] {error_msg}")
            import traceback
            print(f"[VOLUME_AGENT_DEBUG] Traceback: {traceback.format_exc()}")
            
            # Return a structured error response that won't break the volume agent
            return f"{{\"error\": \"{error_msg}\", \"agent\": \"{agent_name}\", \"status\": \"failed\"}}"

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

    def _extract_ml_validation_context(self, knowledge_context: str) -> dict:
        """
        Extract ML system validation context from the knowledge context.
        This allows the LLM to utilize ML validation insights for better analysis.
        """
        try:
            if not knowledge_context:
                return {}
            
            # Look for MLSystemValidation block
            ml_block = self._extract_labeled_json_block(knowledge_context, label="MLSystemValidation:")
            if ml_block:
                return ml_block
            return {}
            
        except Exception as e:
            print(f"Error extracting ML validation context: {e}")
            return {}

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



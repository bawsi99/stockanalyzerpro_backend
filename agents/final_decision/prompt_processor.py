#!/usr/bin/env python3
"""
Final Decision Agent - Prompt Processing Module

This module contains all the prompt processing logic that was previously
handled by the Gemini backend. It includes:
- Template loading and formatting
- Context engineering and structuring
- Response parsing and validation
- Fallback handling
"""

import json
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Import clean_for_json utility
try:
    from core.utils import clean_for_json
except ImportError:
    # Fallback implementation if core.utils is not available
    def clean_for_json(obj):
        """Convert object to JSON-safe format"""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif str(type(obj).__name__).startswith('numpy'):
            return float(obj) if hasattr(obj, 'item') else str(obj)
        else:
            return obj

class FinalDecisionPromptProcessor:
    """Handles all prompt processing for the final decision agent"""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            # Default to same directory as this agent
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = current_dir
        
        self.prompts_dir = os.path.abspath(prompts_dir)
        self.solving_line = "\n\nLet me solve this by .."
        
    def load_template(self, template_name: str) -> str:
        """Load prompt template from file"""
        template_path = os.path.join(self.prompts_dir, f"{template_name}.txt")
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format prompt template with provided context"""
        template = self.load_template(template_name)
        if not template:
            raise FileNotFoundError(f"Prompt template '{template_name}.txt' not found in {self.prompts_dir}")
        
        try:
            # Handle context containing JSON data safely
            if '{context}' in template and 'context' in kwargs:
                context = kwargs['context']
                # Replace {context} manually to avoid formatting issues with JSON
                safe_context = self._escape_context_braces(context)
                formatted = template.replace('{context}', safe_context)
                # Format the rest of the template normally
                remaining_kwargs = {k: v for k, v in kwargs.items() if k != 'context'}
                if remaining_kwargs:
                    formatted = formatted.format(**remaining_kwargs)
                return formatted
            else:
                return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            # Fallback formatting
            if 'context' in kwargs:
                return template.replace('{context}', str(kwargs['context']))
            return template
    
    def _escape_context_braces(self, context: str) -> str:
        """Escape curly braces in context to prevent template formatting conflicts"""
        if not context or not ('{' in context and '}' in context):
            return context
        # Escape braces in JSON data
        return context.replace('{', '{{').replace('}', '}}')
    
    def inject_context_blocks(
        self,
        knowledge_context: str,
        mtf_context: Optional[Dict[str, Any]],
        sector_bullets: Optional[str],
        risk_bullets: Optional[str],
        advanced_digest: Optional[Dict[str, Any]] = None,
        volume_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Inject labeled JSON blocks and synthesis sections into knowledge context"""
        parts = [knowledge_context or ""]

        # Add MTF context as labeled JSON block
        if mtf_context and isinstance(mtf_context, dict):
            try:
                parts.append("MultiTimeframeContext:\n" + json.dumps(mtf_context))
            except Exception:
                pass

        # Add Advanced analysis digest as a labeled JSON block
        if advanced_digest and isinstance(advanced_digest, dict) and len(advanced_digest) > 0:
            try:
                parts.append("AdvancedAnalysisDigest:\n" + json.dumps(advanced_digest))
            except Exception:
                pass

        # Add Volume analysis - prefer combined LLM analysis text over raw JSON structure
        if volume_analysis and isinstance(volume_analysis, dict):
            try:
                # Check if we have combined LLM analysis (preferred)
                combined_llm_analysis = volume_analysis.get('combined_llm_analysis', '')
                if combined_llm_analysis and isinstance(combined_llm_analysis, str) and len(combined_llm_analysis.strip()) > 0:
                    # Use the human-readable LLM analysis summary
                    parts.append("VOLUME ANALYSIS CONTEXT\n" + combined_llm_analysis.strip())
                else:
                    # Fallback to structured JSON if no LLM analysis available
                    parts.append("VolumeAnalysisContext:\n" + json.dumps(volume_analysis))
            except Exception:
                pass

        # Add Sector and Risk synthesis sections for human-readable context
        if sector_bullets and sector_bullets.strip():
            parts.append("SECTOR CONTEXT\n" + sector_bullets.strip())
        if risk_bullets and risk_bullets.strip():
            parts.append("RISK CONTEXT\n" + risk_bullets.strip())

        return "\n\n".join([p for p in parts if p])
    
    def build_comprehensive_context(self, enhanced_ind_json: Any, chart_insights: str, knowledge_context: str) -> str:
        """Build comprehensive context for the final decision template"""
        try:
            context_sections = []
            
            # 1. Technical Indicators Analysis
            context_sections.append("## Technical Indicators Analysis")
            if isinstance(enhanced_ind_json, str):
                # Embed raw JSON blob as-is (no re-serialization)
                context_sections.append(enhanced_ind_json)
            else:
                context_sections.append(json.dumps(clean_for_json(self._convert_numpy_types(enhanced_ind_json)), indent=2))
            
            # 2. Chart Pattern Insights
            if chart_insights and chart_insights.strip():
                context_sections.append("\n## Chart Pattern Insights")
                context_sections.append(chart_insights)
            
            # 3. Multi-Timeframe Context
            mtf_context = self.extract_labeled_json_block(knowledge_context or "", label="MultiTimeframeContext:")
            if mtf_context:
                context_sections.append("\n## Multi-Timeframe Analysis Context")
                context_sections.append(json.dumps(clean_for_json(mtf_context), indent=2))
            
            # 4. Sector Context
            if "SECTOR CONTEXT" in (knowledge_context or ""):
                context_sections.append("\n## Sector Analysis Context")
                # Extract everything after "SECTOR CONTEXT:" 
                sector_start = knowledge_context.find("SECTOR CONTEXT:")
                if sector_start != -1:
                    sector_content = knowledge_context[sector_start + len("SECTOR CONTEXT:"):].strip()
                    context_sections.append(sector_content)
            
            # 5. Advanced Analysis Context
            adv_context = self.extract_labeled_json_block(knowledge_context or "", label="AdvancedAnalysisDigest:")
            if adv_context:
                context_sections.append("\n## Advanced Analysis Context")
                context_sections.append(json.dumps(clean_for_json(adv_context), indent=2))
            
            # 6. ML System Context
            ml_context = self.extract_labeled_json_block(knowledge_context or "", label="MLSystemValidation:")
            if ml_context:
                context_sections.append("\n## ML System Context")
                context_sections.append(json.dumps(clean_for_json(ml_context), indent=2))
            
            # 7. Existing Trading Strategy (for consistency) — only when indicators payload is a dict
            if isinstance(enhanced_ind_json, dict):
                existing_strategy = enhanced_ind_json.get('existing_trading_strategy', {})
                if existing_strategy:
                    context_sections.append("\n## EXISTING TRADING STRATEGY (Use as Foundation)")
                    context_sections.append("The following targets and stop losses were calculated in the previous analysis phase:")
                    context_sections.append(json.dumps(clean_for_json(existing_strategy), indent=2))
                    context_sections.append("IMPORTANT: Use these values as your foundation and only modify them if you have strong technical reasons based on the comprehensive analysis above.")
            
            comprehensive_context = '\n'.join(context_sections)
            return comprehensive_context
            
        except Exception as e:
            print(f"[DEBUG] Error building comprehensive context: {e}")
            # Fallback to basic context
            return f"## Technical Analysis\n{json.dumps(enhanced_ind_json, indent=2)}\n\n## Chart Insights\n{chart_insights}"
    
    def extract_existing_trading_strategy(self, ind_json: dict) -> dict:
        """Extract existing trading strategy data from indicator JSON for consistency"""
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
            
            return existing_strategy
            
        except Exception as e:
            print(f"[DEBUG] Error extracting existing trading strategy: {e}")
            return {}
    
    def extract_labeled_json_block(self, context: str, label: str) -> Optional[dict]:
        """Extract a labeled JSON block from context"""
        try:
            if not context or label not in context:
                return None
            after = context.split(label, 1)[1].strip()
            # Try to read a JSON object starting there
            m = re.search(r"\{[\s\S]+\}", after)
            if not m:
                return None
            return json.loads(m.group(0))
        except Exception:
            return None
    
    def build_ml_guidance_text(self, ml_block: dict) -> str:
        """Convert compact ML block into short guidance text for prompts"""
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
    
    def extract_markdown_and_json(self, llm_response: str) -> Tuple[str, str]:
        """Extract markdown summary and JSON from LLM response"""
        try:
            match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_response or "")
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
                    fixed_json = self._fix_json_string(json_blob)
                    try:
                        json.loads(fixed_json)
                        return markdown_part, fixed_json
                    except json.JSONDecodeError:
                        # Third attempt: create a minimal valid JSON
                        fallback_json = self._create_fallback_json()
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
                raise ValueError("Could not find JSON code block in LLM response.")
        except Exception:
            raise
    
    def _fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Remove any control characters
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
        
        return json_str
    
    def _create_fallback_json(self) -> str:
        """Create a minimal valid JSON when all else fails"""
        return json.dumps({
            "trend": "neutral",
            "confidence_pct": 50,
            "short_term": {"signal": "hold", "target": None, "stop_loss": None},
            "medium_term": {"signal": "hold", "target": None, "stop_loss": None},
            "long_term": {"signal": "hold", "target": None, "stop_loss": None},
            "risks": ["Data parsing error occurred"],
            "must_watch_levels": [],
            "analysis_notes": "JSON parsing failed, using fallback analysis",
            "timestamp": datetime.now().isoformat()
        })
    
    def enhance_final_decision_with_calculations(self, result: dict, code_results: List[str], execution_results: List[str]) -> dict:
        """Enhance the final decision result with code execution calculations"""
        try:
            # Add code execution metadata to the result
            if 'analysis_metadata' not in result:
                result['analysis_metadata'] = {}
            
            result['analysis_metadata']['code_execution'] = {
                'code_snippets_count': len(code_results),
                'execution_outputs_count': len(execution_results),
                'calculation_timestamp': datetime.now().timestamp(),
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
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to JSON-serializable Python types"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # Handle pandas Timestamp objects
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # Handle pandas Series and other array-like objects
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):  # Handle pandas DataFrame objects
            return obj.to_dict('records')
        else:
            return obj
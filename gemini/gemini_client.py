from typing import Dict, Any, List
import json
from .gemini_core import GeminiCore
from .prompt_manager import PromptManager
from .image_utils import ImageUtils
from .error_utils import ErrorUtils

import asyncio


class GeminiClient:
    """
    Client for interacting with Google's Gemini API for AI-powered analysis.
    This class handles API authentication, prompt generation, and response parsing.
    """
    def __init__(self, api_key: str = None):
        self.core = GeminiCore(api_key)
        self.prompt_manager = PromptManager()
        self.image_utils = ImageUtils()
        self.error_utils = ErrorUtils()

    async def build_indicators_summary(self, symbol, indicators, period, interval, knowledge_context=None):
        # print("[DEBUG] Entering build_indicators_summary")
        try:
            raw_indicators_json = json.dumps(indicators, indent=2)
            # print(f"[DEBUG] raw_indicators_json: {raw_indicators_json}")
        except Exception as ex:
            # print(f"[DEBUG-ERROR] Exception during json.dumps(indicators): {ex}")
            import traceback; traceback.print_exc()
            # print(f"[DEBUG-ERROR] indicators: {repr(indicators)}")
            raise
        timeframe = f"{period} days, {interval}"
        
        # Extract additional context if available
        additional_context = indicators.get('additional_context', {})
        additional_context_json = json.dumps(additional_context, indent=2) if additional_context else "{}"
        
        # print("[DEBUG] About to call self.prompt_manager.format_prompt")
        try:
            prompt = self.prompt_manager.format_prompt(
                "indicators_to_summary_and_json",
                symbol=symbol,
                timeframe=timeframe,
                knowledge_context=knowledge_context or "",
                raw_indicators=raw_indicators_json,
                additional_context=additional_context_json
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
                llm_response = await loop.run_in_executor(None, lambda: self.core.call_llm(prompt))
                # print(f"[DEBUG] LLM response (raw): {llm_response!r}")
            except Exception as ex:
                # print(f"[DEBUG-ERROR] Exception during LLM call: {ex}")
                import traceback; traceback.print_exc()
                raise
            if not llm_response or not isinstance(llm_response, str) or llm_response.strip() == "":
                # print(f"[DEBUG-ERROR] LLM response is empty or invalid: {llm_response!r}")
                raise ValueError("LLM response is empty or invalid")
            # print("[DEBUG] About to call extract_markdown_and_json")
            markdown_part, json_blob = self.extract_markdown_and_json(llm_response)
            # print(f"[DEBUG] After extract_markdown_and_json, json_blob: {json_blob}")

            # print("[DEBUG] About to call json.loads on json_blob")
            parsed = json.loads(json_blob)
            # print(f"[DEBUG] After json.loads, parsed: {parsed}")
            # print("[DEBUG] Leaving build_indicators_summary")
            return markdown_part, parsed
        except Exception as ex:
            # print(f"[DEBUG-ERROR] Exception in build_indicators_summary: {ex}")
            import traceback; traceback.print_exc()
            # print(f"[DEBUG-ERROR] Prompt: {prompt}")
            raise

    @staticmethod
    def extract_markdown_and_json(llm_response: str):
        # print("[DEBUG] Entering extract_markdown_and_json")
        # Extracts the markdown summary and the JSON code block from the LLM response
        import re
        try:
            match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_response)
            if match:
                json_blob = match.group(1)
                markdown_part = llm_response[:match.start()].strip()
                return markdown_part, json_blob
            else:
                # print(f"[DEBUG-ERROR] Could not find JSON code block in LLM response. Raw response:\n{llm_response}")
                raise ValueError("Could not find JSON code block in LLM response.")
        except Exception as ex:
            # print(f"[DEBUG-ERROR] Exception in extract_markdown_and_json: {ex}")
            import traceback; traceback.print_exc()
            # print(f"[DEBUG-ERROR] Raw LLM response: {llm_response}")
            raise

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

        # 3. Final decision prompt (now includes chart_insights and additional context)
        # Check if additional context is available in indicators
        additional_context = indicators.get('additional_context', {})
        
        decision_prompt = self.prompt_manager.format_prompt(
            "final_stock_decision",
            indicator_json=json.dumps(ind_json, indent=2),
            chart_insights=chart_insights_md,
            additional_context=json.dumps(additional_context, indent=2) if additional_context else "{}"
        )
        try:
            decision_response = self.core.call_llm(decision_prompt)
            
            # Debug: Log the response to understand what we're getting
            print(f"[DEBUG] Final decision response type: {type(decision_response)}")
            print(f"[DEBUG] Final decision response length: {len(decision_response) if decision_response else 0}")
            print(f"[DEBUG] Final decision response preview: {decision_response[:200] if decision_response else 'None'}")
            
            # The final decision should output ONLY JSON, not markdown with JSON inside
            # So we try to parse it directly as JSON first
            try:
                result = json.loads(decision_response.strip())
                print("[DEBUG] Successfully parsed response as direct JSON")
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Direct JSON parsing failed: {e}")
                # If direct parsing fails, try to extract JSON from markdown code block
                try:
                    _, json_blob = self.extract_markdown_and_json(decision_response)
                    result = json.loads(json_blob)
                    print("[DEBUG] Successfully extracted JSON from markdown code block")
                except Exception as extract_error:
                    print(f"[DEBUG] Failed to extract JSON from markdown: {extract_error}")
                    print(f"[DEBUG] Full response: {decision_response}")
                    raise ValueError(f"Could not parse final decision response as JSON: {e}. Response: {decision_response[:500]}")
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



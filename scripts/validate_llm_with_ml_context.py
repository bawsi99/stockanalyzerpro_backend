#!/usr/bin/env python3
"""
Validate that LLM prompts correctly incorporate ML guidance from MLSystemValidation.
Runs GeminiClient.build_indicators_summary with and without ML guidance and shows diffs.
Run:
  GEMINI_API_KEY=... PYTHONPATH=backend python backend/scripts/validate_llm_with_ml_context.py
"""

import os
import sys
import json
import asyncio
from datetime import datetime


def add_backend_to_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root not in sys.path:
        sys.path.insert(0, root)


async def run_validation():
    from gemini.gemini_client import GeminiClient
    client = GeminiClient()

    # Minimal indicators for test
    indicators = {
        'moving_averages': {'sma_20': 100.0, 'sma_50': 98.5, 'sma_200': 90.0},
        'rsi': {'rsi_14': 55.0, 'trend': 'neutral', 'status': 'neutral'},
        'macd': {'macd_line': 0.2, 'signal_line': 0.1, 'histogram': 0.1},
        'volume': {'volume_ratio': 1.1}
    }

    symbol = 'TEST'
    period = 30
    interval = 'day'

    base_context = ""
    ml_context = {
        "price": {"direction": "up", "magnitude": 0.8, "confidence": 0.72},
        "volatility": {"current": 0.02, "predicted": 0.025, "regime": "normal"},
        "market_regime": {"regime": "bullish", "confidence": 0.65},
        "consensus": {"overall_signal": "buy", "confidence": 0.68, "risk_level": "medium"}
    }

    # 1) Without ML block
    md_no_ml, json_no_ml = await client.build_indicators_summary(
        symbol, indicators, period, interval, knowledge_context=base_context
    )

    # 2) With ML block injected
    with_ml_context = base_context + "\n\nMLSystemValidation:\n" + json.dumps(ml_context)
    md_with_ml, json_with_ml = await client.build_indicators_summary(
        symbol, indicators, period, interval, knowledge_context=with_ml_context
    )

    print("\n=== Indicator Summary (no ML): ===\n")
    print(md_no_ml[:800])
    print("\nJSON keys:", list(json_no_ml.keys())[:20])

    print("\n=== Indicator Summary (with ML): ===\n")
    print(md_with_ml[:800])
    print("\nJSON keys:", list(json_with_ml.keys())[:20])

    print("\n✅ validate_llm_with_ml_context: Completed LLM calls.")


def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️ GEMINI_API_KEY not set; this validation will fail to call the LLM.")
    add_backend_to_path()
    asyncio.run(run_validation())


if __name__ == "__main__":
    main()



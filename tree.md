# Codebase Call Tree & Architecture (Updated)

---

## Entry Points

- **main.py**: CLI entry point for stock analysis.
- **api.py**: FastAPI HTTP endpoint for programmatic access.

---

## Core Orchestrator: `StockAnalysisOrchestrator` (agent_capabilities.py)

- **__init__**: Sets up ZerodhaDataClient, GeminiClient, TechnicalIndicators, PatternVisualizer, IndicatorComparisonAnalyzer, DataCollector, and state cache.
- **authenticate**: Authenticates with Zerodha API.
- **_get_or_create_state**: Manages per-stock/exchange state (AnalysisState dataclass).
- **retrieve_stock_data**: Fetches historical data (with market-hours logic).
- **calculate_indicators**: Uses DataCollector to compute all technical indicators.
- **create_visualizations**: Generates and saves all pattern/indicator charts using PatternVisualizer and ChartVisualizer.
- **compare_indicators**: Aggregates indicator signals into a consensus using IndicatorComparisonAnalyzer.
- **serialize_indicators**: Converts pandas/numpy objects to JSON-serializable format.
- **orchestrate_llm_analysis (async)**: Coordinates LLM analysis via GeminiClient (returns result, indicator summary markdown, chart insights markdown).
- **analyze_with_ai (async)**: Wrapper for orchestrate_llm_analysis.
- **analyze_stock (async)**: Main orchestrator method. Calls all above, returns full analysis and data.
- **AnalysisState (dataclass)**: Caches indicators, consensus, results, last_updated.

---

## Technical Indicators (technical_indicators.py)

- **TechnicalIndicators**: Static methods for all major indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, ADX, Ichimoku, Fibonacci, Pivot Points, Support/Resistance).
- **IndicatorComparisonAnalyzer**: Aggregates indicator signals into a consensus (bullish/bearish/neutral, with strength and percentages).
- **DataCollector**: Gathers all indicator data for a given DataFrame.

---

## Pattern Recognition & Visualization (patterns/)

- **PatternRecognition**: Detects peaks/lows, divergences, double tops/bottoms, triangles, flags, support/resistance, and volume anomalies.
- **PatternVisualizer**: Plots all detected patterns and technical features.
- **ChartVisualizer**: Specialized for multi-panel and comparison charts.

---

## Data Client (zerodha_client.py)

- **ZerodhaDataClient**: Handles authentication, token management, and data retrieval from Zerodha's KiteConnect API.
- **CacheManager**: Caches historical data to avoid redundant API calls, with market-hours awareness.

---

## LLM Integration (gemini/)

- **GeminiClient**: Handles prompt construction, LLM calls, and response parsing. Optimized chart analysis with 4 logical groups.
- **GeminiCore**: Manages API key, rate limiting, and actual LLM calls (text and multi-modal with images).
- **PromptManager**: Loads and formats prompt templates from the prompts/ directory.
- **ImageUtils/ErrorUtils**: Helpers for image conversion and error handling.

---

## RAG (Retrieval-Augmented Generation) (rag_milvus.py)

- **SBERTEmbedding**: Embeds documents/queries for vector search.
- **RAGRetriever**: Loads, splits, indexes, and retrieves knowledge documents using Milvus and LangChain.

---

## Output Structure

- **Per-stock output directories** (e.g., output/RELIANCE/) contain:
  - PNG charts for each pattern/indicator.
  - results.json with only the fields required by the frontend (see README.md for the exact schema). All extra fields (e.g., debug, recommendation, chart_insights) are omitted for efficiency and frontend compatibility.

---

## Prompt Templates (prompts/)

- Templates for each LLM prompt type (indicator summary, image analysis, final decision, etc.), loaded dynamically by PromptManager.
- Optimized for 4 chart analysis groups: comprehensive overview, volume analysis, reversal patterns, continuation & levels.

---

# Call Tree (Indented)

main.py
  └─ main()
      └─ StockAnalysisOrchestrator
          ├─ authenticate()
          └─ analyze_stock() [async]
              ├─ retrieve_stock_data()
              ├─ calculate_indicators()
              ├─ create_visualizations()
              ├─ compare_indicators()
              └─ analyze_with_ai() [async]
                  └─ orchestrate_llm_analysis() [async]
                      └─ GeminiClient.analyze_stock() [async]
                          ├─ build_indicators_summary() [async]
                          │   └─ PromptManager.format_prompt()
                          │   └─ GeminiCore.call_llm()
                          ├─ analyze_comprehensive_overview() [async]
                          ├─ analyze_volume_comprehensive() [async]
                          ├─ analyze_reversal_patterns() [async]
                          ├─ analyze_continuation_levels() [async]
                          └─ PromptManager.format_prompt() (final decision)
                          └─ GeminiCore.call_llm()
              └─ serialize_indicators()

api.py
  └─ analyze() [POST /analyze]
      └─ StockAnalysisOrchestrator
          ├─ authenticate()
          └─ analyze_stock() [async]
      └─ make_json_serializable()

agent_capabilities.py
  └─ StockAnalysisOrchestrator
      ├─ __init__()
      ├─ authenticate()
      ├─ _get_or_create_state()
      ├─ retrieve_stock_data()
      ├─ calculate_indicators()
      ├─ create_visualizations()
      ├─ compare_indicators()
      ├─ serialize_indicators()
      ├─ orchestrate_llm_analysis() [async]
      ├─ analyze_with_ai() [async]
      └─ analyze_stock() [async]
  └─ AnalysisState (dataclass)

technical_indicators.py
  ├─ TechnicalIndicators
  ├─ IndicatorComparisonAnalyzer
  └─ DataCollector

patterns/recognition.py
  └─ PatternRecognition

patterns/visualization.py
  ├─ PatternVisualizer
  └─ ChartVisualizer

zerodha_client.py
  ├─ ZerodhaDataClient
  └─ CacheManager

gemini/gemini_client.py
  └─ GeminiClient
      ├─ build_indicators_summary() [async]
      ├─ analyze_stock() [async]
      ├─ analyze_comprehensive_overview() [async]
      ├─ analyze_volume_comprehensive() [async]
      ├─ analyze_reversal_patterns() [async]
      └─ analyze_continuation_levels() [async]

gemini/gemini_core.py
  └─ GeminiCore

gemini/prompt_manager.py
  └─ PromptManager

rag_milvus.py
  ├─ SBERTEmbedding
  └─ RAGRetriever

output/
  └─ [STOCK]/
      ├─ *_comparison_chart.png
      ├─ *_divergence.png
      ├─ *_double_tops_bottoms.png
      ├─ *_support_resistance.png
      ├─ *_triangles_flags.png
      ├─ *_volume_anomalies.png
      ├─ *_price_volume_correlation.png
      ├─ *_candlestick_volume.png
      └─ results.json

prompts/
  ├─ image_analysis_comprehensive_overview.txt
  ├─ image_analysis_volume_comprehensive.txt
  ├─ image_analysis_reversal_patterns.txt
  ├─ image_analysis_continuation_levels.txt
  ├─ indicators_to_summary_and_json.txt
  ├─ final_stock_decision.txt
  └─ meta_prompt.txt

---

# Architectural Highlights

- **Async LLM Pipeline:** All AI/LLM analysis is async, allowing for efficient multi-modal (text + image) analysis.
- **Optimized Chart Analysis:** 8 charts grouped into 4 logical analysis groups for better efficiency and insights.
- **Extensible Patterns:** Pattern recognition and visualization are modular, supporting easy addition of new patterns.
- **Stateful Orchestration:** Caching and state management allow for efficient repeated analysis and incremental updates.
- **Separation of Concerns:** Data retrieval, indicator calculation, pattern recognition, visualization, and LLM analysis are cleanly separated.
- **RAG Ready:** RAG logic is present but not yet deeply integrated into the main orchestrator.
## Usage

### Basic Usage

Run the agent with a specific stock symbol:

```bash
python main.py --stock RELIANCE
```

### Advanced Usage

```bash
python main.py --stock RELIANCE --exchange NSE --period 365 --interval 60minute        
```

### Command Line Arguments

- `--stock`: Stock symbol to analyze (required)
- `--exchange`: Stock exchange (default: NSE)
- `--period`: Analysis period in days (default: 365)
- `--output`: Output directory (default: ./output)
- `--interval`: period of each unit data

---

## Architecture & Structure (2024 Update)

- **Main Orchestrator:**
  - The core workflow is managed by `StockAnalysisOrchestrator` (in `agent_capabilities.py`).
  - Coordinates data retrieval, indicator calculation, pattern recognition, visualization, async LLM/AI analysis, and report generation.
  - Uses a stateful cache (`AnalysisState`) for efficient repeated analysis and incremental updates.

- **Technical Indicators & Visualization:**
  - All technical indicator calculations are in `technical_indicators.py` (`TechnicalIndicators`).
  - Indicator consensus and aggregation logic is in `IndicatorComparisonAnalyzer`.
  - All charting and visualization logic is in `patterns/visualization.py` (`PatternVisualizer`, `ChartVisualizer`).

- **Pattern Recognition:**
  - All pattern detection logic (peaks/lows, divergences, double tops/bottoms, triangles, flags, support/resistance, volume anomalies) is in `patterns/recognition.py` (`PatternRecognition`).
  - Visualization of these patterns is handled by `PatternVisualizer` and `ChartVisualizer`.

- **LLM/AI Analysis (Async, Multi-modal):**
  - All LLM-powered analysis is handled by `GeminiClient` (in `gemini/gemini_client.py`), orchestrated asynchronously.
  - The main LLM analysis method, `GeminiClient.analyze_stock`, returns a tuple: `(result, ind_summary_md, chart_insights_md)`.
    - `result`: Parsed LLM JSON (final recommendation, targets, risk, etc.)
    - `ind_summary_md`: Markdown summary of technical indicator analysis (for reporting/UI/context)
    - `chart_insights_md`: Markdown insights from image-based chart analysis
  - Multi-modal (text + image) analysis is supported via async calls to Gemini LLM.
  - Prompt templates are managed in `prompts/` and loaded by `PromptManager`.

- **Data Client:**
  - `ZerodhaDataClient` (in `zerodha_client.py`) handles authentication and historical data retrieval from Zerodha's KiteConnect API.
  - `CacheManager` provides market-aware caching of historical data.

- **RAG (Retrieval-Augmented Generation) Ready:**
  - `rag_milvus.py` provides SBERT-based embedding and Milvus-powered retrieval for knowledge-augmented LLM analysis (not yet deeply integrated).

- **Output Structure:**
  - Each stock analyzed gets its own output directory (e.g., `output/RELIANCE/`), containing:
    - PNG charts for each pattern/indicator
    - `results.json` with only the fields required by the frontend (see API Response Fields below)

---

## Extensibility

- **Adding New Indicators or Visualizations:**
  - Add calculation logic in `technical_indicators.py` (or a new module if appropriate).
  - Register new indicators in the indicator registry.
  - Use plotting helpers in `patterns/visualization.py` for consistent charting.
  - Use centralized constants for thresholds to ensure consistency.

- **Adding New Patterns:**
  - Add detection logic to `PatternRecognition`.
  - Add visualization logic to `PatternVisualizer`.

- **Prompt Engineering:**
  - Add or modify prompt templates in the `prompts/` directory.
  - Use `PromptManager` to load and format prompts for LLM calls.

---

## Logging

- Logging is used throughout for error and warning cases, especially in chart generation, data conversion, and LLM calls.

---


## API Usage

- The FastAPI server (see `api.py`) exposes a `/analyze` endpoint for programmatic access.
- Returns JSON with only the fields required by the frontend, including base64-encoded chart images and serializable data.

### API Response Fields

The `/analyze` endpoint returns only the following fields (and their nested subfields):

- `success`: boolean
- `stock_symbol`: string
- `exchange`: string
- `analysis_period`: string
- `interval`: string
- `timestamp`: string
- `results`: object (see below)
- `data`: array of OHLCV records

#### `results` object includes:
- `consensus`, `indicators`, `charts`, `ai_analysis`, `indicator_summary_md`, `summary`, `overlays`
- Optional: `support_levels`, `resistance_levels`, `triangle_patterns`, `flag_patterns`, `volume_anomalies_detailed`

#### `data` array:
Each entry contains: `date`, `open`, `high`, `low`, `close`, `volume`

All other fields are omitted for efficiency.

#### Example API Response

```json
{
  "success": true,
  "stock_symbol": "RELIANCE",
  "exchange": "NSE",
  "analysis_period": "365 days",
  "interval": "day",
  "timestamp": "2025-07-11T12:34:56.789Z",
  "results": {
    "consensus": { /* ... */ },
    "indicators": { /* ... */ },
    "charts": { /* ... */ },
    "ai_analysis": { /* ... */ },
    "indicator_summary_md": "...",
    "summary": { /* ... */ },
    "overlays": { /* ... */ },
    "support_levels": [/* ... */],
    "resistance_levels": [/* ... */]
  },
  "data": [
    {"date": "2025-07-10", "open": 100, "high": 110, "low": 95, "close": 108, "volume": 123456},
    // ...
  ]
}
```

---

## Notes on Extensibility & Reporting

- The LLM/AI analysis step provides both a structured result and markdown summaries for easy integration into reports, dashboards, or further LLM prompts.
- The codebase is highly modular and easy to extend for new indicators, patterns, or AI workflows.

---

For a detailed call tree and relationships, see `tree.md` in the repository.



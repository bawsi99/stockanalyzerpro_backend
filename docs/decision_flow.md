# Stock Analysis Service — Decision-Making Flow Documentation

## Scope
- Starting point: `backend/services/analysis_service.py`
- Traced modules:
  - `backend/core/orchestrator.py`
  - `backend/agents/mtf_analysis/` (new agent-based MTF system)
    - `core/processor.py` (CoreMTFProcessor)
    - `orchestrator.py` (MTFOrchestrator)
    - `mtf_agents.py` (MTFAgentsOrchestrator)
    - `integration_manager.py` (MTFAgentIntegrationManager)
    - Specialized agents: `intraday/`, `swing/`, `position/`
  - `backend/agents/sector/` (new sector agent refactor)
    - `benchmarking.py` (SectorBenchmarkingProvider)
    - `processor.py` (SectorSynthesisProcessor)
    - `cache_manager.py` (SectorCacheManager + `cache_config.json`)
    - `classifier.py`, `enhanced_classifier.py`
  - `backend/agents/risk_analysis/`
    - `quantitative_risk/processor.py` (QuantitativeRiskProcessor)
    - `risk_llm_agent.py` (LLM risk synthesis)
  - `backend/ml/analysis/mtf_analysis.py` (legacy EnhancedMultiTimeframeAnalyzer - still available)
  - `backend/analysis/advanced_analysis.py`
  - `backend/ml/indicators/technical_indicators.py`
  - `backend/api/responses.py`
- Includes volume and risk agents endpoints and database service integration

---

## High-level Responsibilities
- Orchestrates stock analysis: data retrieval, indicator computation, optional charts, sector context, pattern recognition, risk analysis, ML predictions, and AI-based decision.
- Exposes REST endpoints for:
  - Enhanced analysis, async analysis, enhanced MTF
  - Indicators, patterns, charts
  - Sector benchmarking, sector lists, sector stocks, sector performance, sector comparison
  - Volume agents (group and single)
  - Operational utilities: health, CORS, storage, Redis cache
- Persists analysis results to a separate Database Service via HTTP.

---

## Configuration and Dependencies

### Environment variables
- `DATABASE_SERVICE_URL` (default: `http://localhost:8003`)
- `ANALYSIS_SERVICE_URL` (loopback, default: `http://localhost:8002`)
- `CORS_ORIGINS` (comma-separated list)
- Zerodha: `ZERODHA_API_KEY`, `ZERODHA_ACCESS_TOKEN`
- Scheduler: `ENABLE_SCHEDULED_CALIBRATION`, `CALIB_SYMBOLS`, `CALIB_FIXTURES_DIR`

### External services and components
- Zerodha clients:
  - `ZerodhaDataClient` (historical and tokens)
  - Optional WebSocket rolling window snapshot for intraday data
- `services.enhanced_data_service` for optimal/cached data
- `GeminiClient` for LLM (indicator summary, enhanced chart analysis, final decision)
- Database Service (HTTP) for persisting results and resolving users
- Redis cache manager (data caching; images not stored in Redis)
- Chart manager (chart storage paths; actual generation is in-memory now)
- Sector classification and benchmarking providers

### In-process cache
- `VOLUME_PREFETCH_CACHE` keyed by `correlation_id`, to reuse prefetched `stock_data` and `indicators` for volume agents and risk analysis (sector agent may also reuse `stock_data`).

### File-based sector cache (agents/sector/cache_manager.py)
- Managed by `SectorCacheManager` with manifest and `cache_config.json`
- Caches only sector-agnostic data that is safe to reuse across stocks in the same sector:
  - `sector_rotation` (stage, momentum)
  - `sector_correlation` (summary/metrics)
- Does NOT cache stock-specific data:
  - Stock-vs-sector benchmarking outputs
  - LLM sector synthesis bullets (these include stock-specific metrics)
- Cache invalidation: time-based (refresh interval), scheduled refresh, or price-change threshold; manual invalidation supported.

---

## Service Lifecycle

### Startup
- Validate/log Zerodha credential presence
- Initialize chart manager, Redis cache manager, storage directories
- Initialize sector classifiers
- Log signals weighting profiles
- Optionally start a weekly calibration scheduler when `ENABLE_SCHEDULED_CALIBRATION=1`

### Shutdown
- Cancels background tasks and performs best-effort cleanup

---

## Data Models (Requests)

- AnalysisRequest
  - `stock` (required), `exchange="NSE"`, `period=365`, `interval="day"`, `output?`, `sector?`, `user_id?`, `email?`
- EnhancedAnalysisRequest
  - Same as AnalysisRequest plus `enable_code_execution` (default `True`)
- SectorAnalysisRequest
  - `sector`, `period`
- SectorComparisonRequest
  - `sectors[]`, `period`
- IndicatorsRequest
  - `symbol`, `exchange="NSE"`, `interval="1d"`, `indicators` (CSV)
- VolumeAgentRequest
  - `symbol`, `exchange="NSE"`, `interval="day"`, `period=365`, `correlation_id?`, `return_prompt?`
- RiskAnalysisRequest
  - `symbol`, `exchange="NSE"`, `interval="day"`, `period=365`, `correlation_id?`, `return_prompt?`, `timeframes=["short","medium","long"]`

---

## Core Endpoints and Decision Flow

### 1) POST /analyze
- Backward-compatible shim: transforms to EnhancedAnalysisRequest and forwards to `/analyze/enhanced`.

---

### 2) POST /analyze/enhanced
Enhanced decision-making path. Runs deterministic computations and parallel analyses and then performs a final LLM decision.

Step-by-step:
1) Resolve User ID
- If `user_id` provided → use it.
- Else if `email` provided → POST `{email}` to `{DATABASE_SERVICE_URL}/users/resolve-id` with retry/backoff.
- Else generate a new UUID.

2) Initialize orchestrator
- `orchestrator = StockAnalysisOrchestrator()`

3) Retrieve stock data
- `orchestrator.retrieve_stock_data(symbol, exchange, interval, period)`:
  - Tries EnhancedDataService first (maps internal intervals to EDS granularity, and adds attrs like `data_freshness`, `market_status`)
  - Optionally uses WebSocket rolling window snapshot if market open
  - Falls back to Zerodha historical API
- Errors:
  - `ValueError` → 400
  - Other exceptions → 500

4) Calculate indicators
- `TechnicalIndicators.calculate_all_indicators_optimized(df, symbol)`:
  - If < 20 bars → minimal indicators fallback with data quality flags and default signals
  - Else optimized full analysis (current values only; ratios; crosses; MACD/RSI/Bollinger; volume metrics)

5) Launch independent tasks in parallel
- Volume agents (loopback REST): POST `/agents/volume/analyze-all`
  - Reuses `stock_data`/`indicators` via `VOLUME_PREFETCH_CACHE` (using `correlation_id`)
- Risk analysis (loopback REST): POST `/agents/risk/analyze-all`
  - Reuses `stock_data`/`indicators` via `VOLUME_PREFETCH_CACHE` (using `correlation_id`)
  - Computes quantitative metrics (VaR/ES/Sharpe), stress tests (historical/Monte Carlo/sector/market), scenarios, and LLM risk bullets for short/medium/long horizons
- Multi-timeframe analysis (MTF) - **New Agent-Based Architecture**:
  - `mtf_agent_integration_manager.get_comprehensive_mtf_analysis(symbol, exchange)`
  - Uses `MTFAgentsOrchestrator` which coordinates:
    - **Core MTF Processor**: `CoreMTFProcessor.analyze_comprehensive_mtf()` - fundamental MTF analysis across all timeframes
    - **Specialized MTF Agents** (run in parallel with timeouts):
      - `IntradayMTFProcessor` (weight: 0.25, timeout: 120s) - scalping and short-term trading
      - `SwingMTFProcessor` (weight: 0.35, timeout: 150s) - medium-term swing trading
      - `PositionMTFProcessor` (weight: 0.40, timeout: 180s) - long-term position trading
  - Computes per-timeframe signals (6 timeframes: 1min, 5min, 15min, 30min, 1hour, 1day)
  - Performs cross-timeframe consensus validation with dynamic signal quality weighting
  - Detects conflicts, divergences (bullish/bearish), severity scoring
  - Aggregates results into `AggregatedMTFAnalysis` with:
    - Core analysis, individual agent results, unified analysis
    - Consensus signals, trading recommendations, overall confidence
    - Agent success/failure tracking, processing times
  - **Optimized timeframe periods** (reduced for better signal-to-noise):
    - 1min: 1 day, 5min: 3 days, 15min: 7 days, 30min: 120 days, 1hour: 180 days, 1day: 365 days
- Advanced analysis:
  - `advanced_analysis_provider.generate_advanced_analysis(stock_data, symbol, indicators)`
  - Produces scenario probabilities, risk/stress summaries, compact digest
- Sector context (optional, if `request.sector`):
  - `SectorBenchmarkingProvider.get_optimized_comprehensive_sector_analysis`
  - Returns benchmarking, rotation, correlation, optimization metrics
- Indicator summary LLM:
  - Curated indicators via agents manager (best effort)
  - `orchestrator.gemini_client.build_indicators_summary(...)` for markdown + structured JSON

6) Join results and normalize
- Any task failure → normalized to empty context; analysis continues.

7) Prepare final decision inputs
- Collect the context components: sector_bullets (from sector synthesis), an MTF subset (LLM insights or summary), risk bullets (if available), and the advanced analysis digest.
- The centralized FinalDecisionProcessor assembles the labeled knowledge context internally.

8) Final decision agent (centralized)
- `agents.final_decision.processor.FinalDecisionProcessor.analyze_async(symbol, ind_json, mtf_context_subset, sector_bullets, advanced_digest, risk_bullets=None, chart_insights="", base_knowledge_context="")`
- Internally builds labeled context (MultiTimeframeContext, AdvancedAnalysisDigest, SECTOR CONTEXT) and invokes Gemini via GeminiClient code-exec path
- Returns `ai_analysis` via `fd_result['result']` (trend, confidence_pct, guidance, etc.)

9) Optional ML predictions
- `UnifiedMLManager` best-effort train/get prediction, then clear caches.

10) Build frontend response
- `FrontendResponseBuilder.build_frontend_response(...)`
  - Normalizes OHLCV
  - Computes price delta/%, pivots inferred from interval anchor (D/W/M/Q)
  - Adds volume bands where available
- Integrates ai_analysis, indicator_summary, chart_insights, indicators, sector_context, mtf_context, risk_context, advanced_analysis, ml_predictions

11) Persist to Database Service
- `make_json_serializable(frontend_response)`
- POST `{DATABASE_SERVICE_URL}/analyses/store` with `analysis`, `user_id`, `symbol`, `exchange`, `period`, `interval`
- Logs `analysis_id` on success (non-fatal on failure)

12) Return
- `200 OK` with serialized response
- On exceptions: traces + structured 400/500 responses

Inputs/Outputs:
- Input: EnhancedAnalysisRequest
- Intermediate:
  - DataFrame with datetime index + attrs (`data_freshness`, `market_status`)
  - Indicators dict (optimized or minimal)
  - Parallel contexts: volume agents, MTF, advanced digest, sector context, indicator summary JSON
- Output: Frontend response JSON structure with all contexts and UI-friendly fields

---

### 3) POST /analyze/async
- Authenticates Zerodha; calls `orchestrator.analyze_stock_with_async_index_data`.
- Serializes and returns: `{ success, message, data, analysis_type="async_index_data", timestamp }`.

---

### 4) POST /analyze/enhanced-mtf
- Performs direct comprehensive multi-timeframe analysis:
  - `mtf_agent_integration_manager.get_comprehensive_mtf_analysis(symbol, exchange)`
  - Uses the new agent-based MTF architecture (CoreMTFProcessor + specialized agents)
  - Returns aggregated results from core processor and all specialized agents (intraday, swing, position)
  - Includes consensus signals, trading recommendations, and cross-agent validation
- Persists result to Database Service (non-fatal if storing fails).
- Adds request metadata and returns results.
- **Note**: Legacy `EnhancedMultiTimeframeAnalyzer` still exists in `ml/analysis/mtf_analysis.py` but is no longer used by this endpoint.

---

## Supplementary Endpoints

### Health and utilities
- GET `/health`
  - Checks Zerodha credentials, Gemini key presence, sector classifier availability, Database Service connectivity, event loop status.

### Sector analytics
- POST `/agents/sector/analyze-all` (New sector agent path)
  - End-to-end sector analysis for a given stock and its sector using the refactored agents under `backend/agents/sector`
  - Flow (summary):
    1) Prefetch reuse (optional): If `correlation_id` provided and present in `VOLUME_PREFETCH_CACHE`, reuse `stock_data`
    2) Sector detection: Use `SectorClassifier` (or user-provided `sector`)
    3) File cache lookup: `SectorCacheManager.get_cached_analysis(sector)` to fetch sector-agnostic data (rotation/correlation)
    4) Compute fresh stock-specific benchmarking: `SectorBenchmarkingProvider.get_optimized_comprehensive_sector_analysis(...)`
       - Always computed fresh per stock (never cached)
    5) Generate fresh sector synthesis bullets via `SectorSynthesisProcessor`
       - Always regenerated per stock (LLM synthesis contains stock-specific metrics)
       - Synthesis prompt includes enhanced metrics: correlation, Sharpe, volatility, returns for sector and market
    6) Update file cache with sector-agnostic data only (rotation, correlation). Do NOT cache synthesis or benchmarking
    7) Return comprehensive response: benchmarking, rotation, correlation, optimization metrics, synthesis bullets

- POST `/sector/benchmark`
- POST `/sector/benchmark/async`
  - Retrieves df via orchestrator and computes comprehensive benchmarking. Returns serialized results.

- GET `/sector/list`
  - Lists all available sectors.

- GET `/sector/{sector_name}/stocks`
  - Returns sector display name, primary index, and member stocks.

- GET `/sector/{sector_name}/performance?period=365`
  - Retrieves sector index data; computes cumulative return and annualized volatility; returns summary.

- POST `/sector/compare`
  - For each sector: fetches its index data, computes return/volatility; tolerates per-sector failures.

#### Sector agent components (overview)
- `agents/sector/benchmarking.py`: SectorBenchmarkingProvider (optimized comprehensive benchmarking)
- `agents/sector/processor.py`: SectorSynthesisProcessor (LLM bullets via `GeminiClient.synthesize_sector_summary`)
- `agents/sector/cache_manager.py`: SectorCacheManager (file-based cache for sector-agnostic data)
- `agents/sector/cache_config.json`: Cache settings (enabled, refresh interval, price thresholds)
- `agents/sector/classifier.py`, `enhanced_classifier.py`: Sector mapping and enhanced filters

#### Caching policy (critical)
- Cache only sector-agnostic context (rotation, correlation)
- Never cache stock-specific benchmarking or sector synthesis (LLM) since they include stock-specific data

### Indicators, patterns, charts
- GET `/stock/{symbol}/indicators?interval=1day&exchange=NSE&indicators=rsi,macd,sma,ema,bollinger`
  - Maps frontend interval to backend interval; retrieves df; computes requested indicators; returns arrays aligned to timestamps.

- GET `/patterns/{symbol}?interval=1day&exchange=NSE&pattern_types=all`
  - Retrieves df; runs pattern detectors (candlestick, double top/bottom, H&S, triangles...); returns timestamps and recent patterns.

- GET `/charts/{symbol}?interval=1day&…&chart_types=all`
  - Retrieves df; computes indicators; generates in-memory charts via `orchestrator.create_visualizations`; converts to base64; returns metadata and images.

### Volume agents
- POST `/agents/volume/analyze-all`
  - Reuses prefetched data via `VOLUME_PREFETCH_CACHE` using `correlation_id`.
  - `VolumeAgentIntegrationManager.get_comprehensive_volume_analysis` aggregates multi-agent outputs.
  - Returns structured aggregated analysis.

- Single-agent endpoints:
  - `/agents/volume/anomaly`
  - `/agents/volume/institutional`
  - `/agents/volume/confirmation`
  - `/agents/volume/support-resistance`
  - `/agents/volume/momentum`
  - Each retrieves df, computes indicators, runs a single agent via `VolumeAgentsOrchestrator._execute_agent`, returns structured result; can include prompt when `return_prompt=True`.

### Risk analysis
- POST `/agents/risk/analyze-all`
  - Reuses prefetched data via `VOLUME_PREFETCH_CACHE` using `correlation_id`.
  - Performs quantitative metrics (VaR, Expected Shortfall, Sharpe), multi-scenario stress tests, and produces LLM-synthesized risk bullets across timeframes.
  - Returns a structured risk summary with scores and decision-ready bullets; integrated into the final decision agent via `risk_bullets`.

### Storage and cache
- Charts storage:
  - GET `/charts/storage/stats`, POST `/charts/cleanup`, DELETE `/charts/{symbol}/{interval}`, DELETE `/charts/all`
  - Images are generated in-memory; Redis image storage is removed; file storage stats and cleanup remain.

- Redis cache management:
  - GET `/redis/cache/stats`
  - POST `/redis/cache/clear?data_type=...`
  - DELETE `/redis/cache/stock/{symbol}`
  - GET `/redis/cache/stock/{symbol}?exchange=&interval=&period=`
  - Redis is used for data caching (not images).

### Root and CORS
- GET `/` returns basic service info and canonical endpoints.
- Global `OPTIONS /{path:path}` returns 200 for CORS preflights.

---

## Quick Endpoint Reference
- POST `/analyze` → shim to `/analyze/enhanced`
- POST `/analyze/enhanced` → orchestrates parallel tasks:
  - Volume agents → POST `/agents/volume/analyze-all` → VolumeAgentIntegrationManager.get_comprehensive_volume_analysis
  - Risk analysis → POST `/agents/risk/analyze-all` → QuantitativeRiskProcessor + risk_llm_agent
  - MTF analysis → mtf_agent_integration_manager.get_comprehensive_mtf_analysis → MTFAgentsOrchestrator
  - Advanced analysis → advanced_analysis_provider.generate_advanced_analysis
  - Sector analysis → POST `/agents/sector/analyze-all` → SectorBenchmarkingProvider + SectorSynthesisProcessor (+ SectorCacheManager)
  - Indicator summary → GeminiClient.build_indicators_summary
  - Final decision → FinalDecisionProcessor.analyze_async
- POST `/analyze/enhanced-mtf` → mtf_agent_integration_manager.get_comprehensive_mtf_analysis (returns aggregated MTF results)
- Volume single-agent endpoints → VolumeAgentsOrchestrator._execute_agent
- Sector metadata endpoints → SectorClassifier

## Orchestrator Highlights (`core/orchestrator.py`)
- `retrieve_stock_data`:
  - EDS → WebSocket snapshot (if market hours) → Zerodha historical; normalizes index and sets attrs.
- `calculate_indicators`:
  - Uses `DataCollector.collect_all_data` and `TechnicalIndicators` where appropriate.
- `create_visualizations`:
  - Produces in-memory images for multiple chart types using `ChartVisualizer`; integrates volume agents result if present.
- `enhanced_analyze_stock`:
  - Legacy end-to-end method retained; the service layer now orchestrates parallel tasks. Still prepares contexts (sector, advanced, MTF) and calls `enhanced_analyze_with_ai`.
- `enhanced_analyze_with_ai`:
  - Builds supplemental deterministic contexts:
    - Signals summary (`data.signals.scoring`)
    - Multi-timeframe context
    - Advanced digest
    - Volume agents’ compact context
    - Optional compact ML validation block
  - Calls `gemini_client.analyze_stock_with_enhanced_calculations`.

---

## TechnicalIndicators Highlights (`ml/indicators/technical_indicators.py`)
- `calculate_all_indicators_optimized(df, stock_symbol, prefetch=None)`:
  - If insufficient data (< 20), returns minimal indicators with data-quality flags and safe defaults.
  - Else returns optimized full analysis with:
    - Current SMA(20/50/200 with fallbacks), EMA(20/50)
    - Ratios (price_to_sma_200, sma20_to_sma50)
    - Golden/death cross detection
    - MACD/RSI/Bollinger
    - Volume metrics
    - Trend summaries and data-quality metadata
- Individual indicator calculators for RSI/MACD/Bollinger/etc. used across the service.

---

## Multi-Timeframe Analysis System

### New Agent-Based Architecture (`agents/mtf_analysis/`)
The MTF system now follows an agent-based orchestration pattern similar to volume agents:

#### Core Components:
1. **MTFAgentIntegrationManager** (`integration_manager.py`):
   - Main entry point: `get_comprehensive_mtf_analysis(symbol, exchange, include_agents=None)`
   - Orchestrates the entire MTF analysis pipeline
   - Returns tuple: (success: bool, results: dict)

2. **MTFAgentsOrchestrator** (`mtf_agents.py`):
   - Central orchestrator managing all MTF agents
   - Executes core analysis and specialized agents in parallel
   - Aggregates results with weighted consensus building
   - **Agent Configuration**:
     - Intraday: weight=0.25, timeout=120s (scalping/short-term)
     - Swing: weight=0.35, timeout=150s (medium-term)
     - Position: weight=0.40, timeout=180s (long-term)
   - Generates `AggregatedMTFAnalysis` with:
     - Individual agent results and core analysis
     - Unified analysis with cross-agent validation
     - Consensus signals and trading recommendations
     - Overall confidence score (weighted combination)

3. **CoreMTFProcessor** (`core/processor.py`):
   - Fundamental multi-timeframe technical analysis engine
   - `analyze_comprehensive_mtf(symbol, exchange)` - main entry point
   - Fetches data for 6 timeframes (1min, 5min, 15min, 30min, 1hour, 1day)
   - **Optimized timeframe periods** (reduced from legacy for better signal-to-noise):
     - 1min: 1 day (was 30), 5min: 3 days (was 60), 15min: 7 days (was 90)
     - 30min: 120 days, 1hour: 180 days, 1day: 365 days
   - Computes indicators per timeframe with appropriate indicator sets
   - Generates signals with weighted trend analysis
   - Performs cross-timeframe validation with dynamic signal quality weighting
   - Detects divergences (bullish/bearish) between higher and lower timeframes
   - Calculates conflict severity based on weighted conflict scores
   - Returns `MTFAnalysisResult` dataclass

4. **Specialized Agent Processors**:
   - `IntradayMTFProcessor` (`intraday/processor.py`) - short-term trading signals
   - `SwingMTFProcessor` (`swing/processor.py`) - medium-term swing analysis
   - `PositionMTFProcessor` (`position/processor.py`) - long-term position insights
   - Each implements `analyze_async(mtf_data, indicators, context)` method

5. **MTFOrchestrator** (`orchestrator.py`):
   - Alternative orchestrator for direct agent coordination
   - Fetches MTF data and runs agent processors concurrently
   - Provides cross-timeframe validation using core processor results

#### Key Features:
- **Dynamic Signal Quality Weighting**: Weights adjusted based on volume, trend consistency, MACD strength, support/resistance proximity
- **Comprehensive Logging**: `MTFAgentsLogger` tracks operation IDs, agent execution, processing times
- **Parallel Execution**: All agents run concurrently with individual timeouts and error handling
- **Consensus Building**: Weighted voting across core and agents to determine final signal
- **Agent Agreement Bonus**: Confidence boost when all agents agree on direction

### Legacy System (`ml/analysis/mtf_analysis.py`)
- **EnhancedMultiTimeframeAnalyzer** still exists but is no longer used by service endpoints
- Original implementation with fixed timeframe periods (30/60/90/120/180/365 days)
- Replaced by the new agent-based architecture for better modularity and parallel processing
- Kept for backward compatibility but not actively maintained

### Usage in Service Layer:
- Results are passed into LLM context as `mtf_context` in `/analyze/enhanced`
- Returned directly with full details by `/analyze/enhanced-mtf`
- Integrated into parallel task execution with volume agents, advanced analysis, and sector context

---

## Frontend Response Builder (`backend/api/responses.py`)
- Normalizes OHLCV and interval display.
- Computes price change and pivots based on inferred anchor (D/W/M/Q).
- Adds volume support/resistance bands (if available).
- Integrates LLM outputs and deterministic contexts into a UI-friendly schema with safe guards for missing data.

---

## Database Service Integration
- Uses robust HTTP with retry/backoff:
  - `POST /users/resolve-id` (email → user_id)
  - `POST /analyses/store` (persist analysis result)
- Storage failures are logged and non-fatal to the main analysis response.

---

## Error Handling and Timeouts
- Parallel tasks are wrapped with logging and per-task timeouts (≈90–200s).
- Failures are normalized to empty contexts to keep the decision flow robust.
- Data retrieval and indicator errors are short-circuited with specific HTTP statuses.
- Extensive timing and error logs around LLM calls and HTTP I/O.

---

## Call Graph Summary (from `/analyze/enhanced`)
- Request → Resolve user → Orchestrator
- Retrieve data → Compute indicators
- Parallel (with logging + timeouts):
  - Volume agents (loopback REST, reuse prefetched data, timeout: 200s)
  - Risk analysis (timeout: 200s)
  - **MTF analysis** (timeout: 120s):
    - `mtf_agent_integration_manager.get_comprehensive_mtf_analysis()`
    - → `MTFAgentsOrchestrator.analyze_comprehensive_mtf()`
    - → Executes `CoreMTFProcessor` (essential, must succeed)
    - → Executes specialized agents in parallel (intraday, swing, position)
    - → Aggregates results with consensus building
    - → Returns `AggregatedMTFAnalysis`
  - Advanced analysis digest (timeout: 90s)
  - Sector context (if requested, timeout: 120s)
  - Indicator summary LLM (timeout: 120s)
- Join → Prepare final-decision inputs → FinalDecisionProcessor.analyze_async (centralized)
- Optional ML predictions
- Build frontend response → Persist to Database Service → Return JSON

---

## Notes on Charts
- Enhanced decision path avoids chart generation (chart_paths = {}).
- Dedicated `/charts` endpoint generates in-memory PNGs and returns base64 with metadata.
- Conversion and cleanup functions reduce memory pressure (scoped functions, GC hints).

---

## CORS and OPTIONS
- CORS origins derived from `CORS_ORIGINS` env variable.
- Global `OPTIONS` handler ensures preflight success for all routes.

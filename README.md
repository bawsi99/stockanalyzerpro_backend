# Stock Analysis System (Backend)

AI-powered stock analysis backend providing technical analysis, pattern recognition, sector benchmarking, and real-time streaming.

## Monorepo structure (v3)

This backend is one sub-repo in the v3 monorepo:
- frontend/ — Web application (UI) consuming backend services
- backend/ — FastAPI services, agents, ML, data integration (this repo)
- stockpulse-pro-theme/ — Theme/assets for the marketing/docs site

Each sub-repo has its own README and .env; configure CORS and service URLs so frontend ↔ backend work locally and in prod.

## 🚀 Highlights

- AI-only analysis via Gemini with multi-provider, pluggable LLM client (Gemini, OpenAI, Claude)
- 25+ indicators, advanced pattern recognition (market structure + cross-validation)
- Agent suite: Volume, Risk, Patterns, MTF, Sector, Final Decision, Indicator Summary
- Real-time data and WebSocket streaming from Data Service
- Token analytics, multi-key rotation, model-based cost tracking
- Optimized prefetch cache for parallel agents

---

## 🏗️ Backend architecture

### Services (updated)
- Data Service (`services/data_service.py`)
  - Historical data, market status, token mapping
  - WebSocket streaming at `/ws/stream` and auth endpoints
- Analysis Service (`services/analysis_service.py`)
  - AI analysis orchestration, indicators, patterns, sector, MTF, advanced digest
- Database Service (`services/database_service.py`)
  - Persistence via Supabase; analysis storage and retrieval
- Enhanced Data Service (`services/enhanced_data_service.py`)
  - Optimization layer used by Data Service

Notes:
- The old consolidated_service and separate websocket_service were removed; WebSocket lives in Data Service.

### Core components
- Orchestrator (`core/orchestrator.py`), Indicators (`ml/indicators/technical_indicators.py`)
- Patterns (`patterns/recognition.py`, `patterns/visualization.py`)
- LLM client (`llm/client.py`), token tracker (`llm/token_counter.py`), key manager (`llm/key_manager.py`)
- Sector (`agents/sector/*`), Volume (`agents/volume/*`), Risk (`agents/risk_analysis/*`)
- MTF agents (`agents/mtf_analysis/*`), Final Decision (`agents/final_decision/processor.py`)

For a detailed directory view, see `tree.md`.

### Analysis flow (high level)
1) Fetch OHLCV (optimized historical → live fallback when needed)
2) Compute indicators and prepare agent context (prefetch cache)
3) Run agents in parallel (volume, risk, patterns, MTF, sector, advanced, indicator summary)
4) FinalDecisionProcessor synthesizes results via LLM
5) Response assembly + persistence to Database Service

---

## 🔌 Ports and process layout

Defaults (overridable via env):
- DATA_PORT=8001 → Data + WebSocket
- ANALYSIS_PORT=8002 → Analysis
- DATABASE_PORT=8003 → Database

---

## 📡 WebSocket streaming (Data Service)

Endpoint: `GET /ws/stream`
- Actions: `history`, `register_alert`, `remove_alert`
- Supports JWT or API key auth when `REQUIRE_AUTH=true`

Example history request:
```json
{
  "action": "history",
  "token": "<INSTRUMENT_TOKEN>",
  "timeframe": "1m",
  "count": 50
}
```

---

## 📊 Key REST endpoints

Analysis Service:
- `POST /analyze` (shim), `POST /analyze/enhanced`, `POST /analyze/enhanced-mtf`
- Volume: `POST /agents/volume/analyze-all` and per-agent endpoints
- Risk: `POST /agents/risk/analyze-all`
- Patterns: `POST /agents/patterns/analyze-all`
- Sector: `POST /agents/sector/analyze-all`, plus `GET /sector/list` and helpers

Data Service:
- `POST /data/fetch`, `GET /data/market-status`, `GET /data/token-mapping`
- WebSocket: `GET /ws/stream`, auth: `POST /auth/token`, `GET /auth/verify`

Database Service:
- `POST /analyses/store`, `GET /analyses/user/{user_id}`, summaries/filters

---

## 🧠 Universal LLM system

- Provider-agnostic client with YAML assignments: `llm/config/llm_assignments.yaml`
- Token usage analytics per model and agent
- Multi-key rotation: round-robin, agent-specific, single-key

Example assignments:
```yaml
default:
  provider: gemini
  model: gemini-2.5-flash
  api_key_strategy: round_robin
agents:
  final_decision_agent:
    model: gemini-2.5-pro
  mtf_llm_agent:
    model: gemini-2.5-pro
```

---

## 🔒 Security, CORS, and auth

- Set `CORS_ORIGINS` to a comma-separated list of allowed origins (must include your frontend URL)
- WebSocket/Data Service auth:
  - `REQUIRE_AUTH=true|false`
  - `API_KEYS` (comma-separated) for header `X-API-Key`
  - `JWT_SECRET` to enable JWT-based WebSocket auth (`/auth/token`, `/auth/verify`)

---

## 🛠️ Setup

Prereqs: Python 3.10+, Zerodha credentials, Gemini key(s), Supabase project, optional Redis.

Install:
```bash
pip install -r config/requirements.txt
```

Run services (3 terminals):
```bash
ANALYSIS_PORT=8002 python services/analysis_service.py
DATA_PORT=8001     python services/data_service.py
DATABASE_PORT=8003 python services/database_service.py
```

Environment (.env):
```bash
# Zerodha
ZERODHA_API_KEY=...
ZERODHA_ACCESS_TOKEN=...

# LLM
GEMINI_API_KEY1=...
GEMINI_API_KEY2=...
GOOGLE_GEMINI_API_KEY=...

# Optional providers
# OPENAI_API_KEY1=...
# CLAUDE_API_KEY1=...

# Database (Supabase)
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...

# Redis
REDIS_URL=redis://localhost:6379

# Auth / CORS
JWT_SECRET=...
REQUIRE_AUTH=false
API_KEYS=
CORS_ORIGINS=http://localhost:3000

# Service URLs (if services talk over HTTP)
DATA_SERVICE_URL=http://localhost:8001
ANALYSIS_SERVICE_URL=http://localhost:8002
DATABASE_SERVICE_URL=http://localhost:8003
```

Test mode (always-open market) for local UI testing:
```bash
MARKET_HOURS_TEST_MODE=true
FORCE_MARKET_OPEN=true
```

---

## ⚙️ Important notes

- WebSocket is integrated into Data Service; there is no separate websocket_service
- Detailed structure is in `tree.md`; sector categories are symlinked via `sector_category -> data/sector_category`
- Keep `CORS_ORIGINS` in sync with your frontend host(s)
- Supabase service key is required by Database Service; store securely

---

## 📚 Docs and guides

- Decision flow: `docs/decision_flow.md`
- LLM Usage: `llm/USAGE_GUIDE.md`
- Token tracking: `TOKEN_TRACKING_GUIDE.md`
- Env config: `ENV_CONFIG_GUIDE.md`

## 🔄 Recent updates (v3)
- Sub-repo split (frontend, backend, theme) and port standardization
- WebSocket moved under Data Service; consolidated service deprecated
- Expanded pattern system (market structure + cross-validation)
- Unified LLM system with model assignments and key rotation
- Prefetch cache optimization across agents


# Backend (FastAPI services)

This directory is a Git submodule that aggregates backend services for StockAnalyzer-Pro.

## Submodules
- Technical Analysis Agent: `technical_analysis_agent/` — https://github.com/bawsi99/stockanalyzerpro_backend_technical_analysis_agent
  - FastAPI microservices: Analysis (8002), Data + WebSocket (8001), Database (8003)
  - Active development for backend analysis happens in this repo

## Getting started

Initialize/update submodules:
```bash
git submodule update --init --recursive
```

Install dependencies (Python 3.10+):
```bash
cd technical_analysis_agent
pip install -r config/requirements.txt
```

Run services (three terminals):
```bash
# Analysis Service
ANALYSIS_PORT=8002 python services/analysis_service.py

# Data Service (includes WebSocket /ws/stream)
DATA_PORT=8001 python services/data_service.py

# Database Service (Supabase persistence)
DATABASE_PORT=8003 python services/database_service.py
```

Environment (.env in `technical_analysis_agent/config/.env`):
- Zerodha API: `ZERODHA_API_KEY`, `ZERODHA_ACCESS_TOKEN` (+ optional distributed keys 1–5)
- LLM (Gemini): `GEMINI_API_KEY1`, `GEMINI_API_KEY2` (optional), `GOOGLE_GEMINI_API_KEY`
- Database: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`
- CORS: `CORS_ORIGINS`

## Architecture (high level)
- Technical Analysis Agent executes five specialized domains in parallel and synthesizes results:
  - Volume, Risk, Patterns, Multi-Timeframe (MTF), Sector → Final Decision Agent for unified guidance; Indicator Summary supports context
- Data Service provides optimized historical fetch + real-time WebSocket streaming
- Database Service persists analyses with retry logic

For full details, see `technical_analysis_agent/README.md` and `technical_analysis_agent/docs/decision_flow.md`.
#!/usr/bin/env python3
"""
Mockup: Generate production-like charts as PNGs for 5 stocks using real data

Pipeline (mirrors production steps without modifying existing agent modules):
1) Load environment and imports
2) Initialize data sources (Orchestrator preferred, fallback Zerodha client)
3) For each symbol:
   - Fetch OHLCV (365d, daily)
   - Compute Market Structure context (regime, S/R, recent break)
   - Render a 1600x900 PNG chart (price + volume, overlay key levels and regime)
   - Save PNG (fallback to HTML if PNG export fails)

Usage:
  python -m agents.patterns.cross_validation_agent.mock_chart_images \
    --symbols RELIANCE,TCS,INFY,HDFCBANK,ITC --period 365 --outdir charts_mock

Notes:
- Requires plotly and kaleido for PNG export. If kaleido is not available, the script will save HTML instead.
- This mockup focuses on price/volume + market-structure overlays. Pattern geometry overlays can be added later.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock_chart_images")

# Ensure backend is importable
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Load .env
try:
    import dotenv
    env_path = os.path.join(BACKEND_DIR, 'config', '.env')
    if os.path.exists(env_path):
        dotenv.load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded .env from {env_path}")
except Exception as e:
    logger.warning(f"Could not load dotenv: {e}")

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data providers
try:
    from core.orchestrator import StockAnalysisOrchestrator
except Exception as e:
    logger.warning(f"Orchestrator import failed: {e}")
    StockAnalysisOrchestrator = None

try:
    from zerodha.client import ZerodhaDataClient
except Exception as e:
    logger.warning(f"ZerodhaDataClient import failed: {e}")
    ZerodhaDataClient = None

# Market structure processor
try:
    from agents.patterns.market_structure_agent.processor import MarketStructureProcessor
except Exception as e:
    logger.error(f"MarketStructureProcessor import failed: {e}")
    MarketStructureProcessor = None

# Color palette (Okabe–Ito + semantic)
COLORS = {
    'bull': '#009E73',
    'bear': '#D55E00',
    'sma20': '#0072B2',
    'sma50': '#CC79A7',
    'support': '#009E73',
    'resistance': '#D55E00',
    'current_price': '#6c757d',
    'regime_trending': '#0072B2',
    'regime_consolidating': '#7f7f7f',
    'regime_volatile': '#E69F00',
    'regime_stable': '#009E73',
}


def _init_data_sources():
    orch = None
    zc = None
    if StockAnalysisOrchestrator:
        try:
            orch = StockAnalysisOrchestrator()
            logger.info("Orchestrator initialized")
        except Exception as e:
            logger.warning(f"Orchestrator init failed: {e}")
            orch = None
    if ZerodhaDataClient:
        try:
            zc = ZerodhaDataClient()
            logger.info("ZerodhaDataClient initialized")
        except Exception as e:
            logger.warning(f"Zerodha client init failed: {e}")
            zc = None
    return orch, zc


def _fetch_data(symbol: str, period_days: int, orch, zc) -> Optional[pd.DataFrame]:
    logger.info(f"Fetching data for {symbol} ({period_days}d)")
    # Preferred: orchestrator
    if orch:
        try:
            df = orch.retrieve_stock_data_sync(symbol=symbol, exchange="NSE", interval="day", period=period_days)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize_df(df)
        except Exception as e:
            logger.warning(f"Orchestrator fetch failed for {symbol}: {e}")
    # Fallback: Zerodha client
    if zc:
        try:
            # Prefer async if available; else sync
            if hasattr(zc, 'get_historical_data'):
                df = zc.get_historical_data(symbol, "NSE", "day", None, None, period_days)
            else:
                logger.warning("Zerodha client missing get_historical_data; cannot fetch")
                df = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize_df(df)
        except Exception as e:
            logger.warning(f"Zerodha fetch failed for {symbol}: {e}")
    logger.error(f"All data sources failed for {symbol}")
    return None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns and index
    cols = {c.lower(): c for c in df.columns}
    for needed in ['open', 'high', 'low', 'close', 'volume']:
        if needed not in [c.lower() for c in df.columns]:
            raise ValueError(f"Missing required column: {needed}")
    if 'date' not in [c.lower() for c in df.columns]:
        # Try index
        if df.index.name and df.index.name.lower() == 'date':
            df = df.reset_index()
        else:
            df = df.copy()
            df['date'] = df.index
    # Normalize column names to lower
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    return df


def _compute_ms_context(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if not MarketStructureProcessor:
        return None
    try:
        ms = MarketStructureProcessor()
        result = ms.process_market_structure_data(df)
        if result and result.get('success'):
            # Build compact context similarly to agent._build_market_context_from_structure
            ta = result.get('trend_analysis', {}) or {}
            bos = result.get('bos_choch_analysis', {}) or {}
            kl = result.get('key_levels', {}) or {}
            cs = result.get('current_state', {}) or {}
            fr = result.get('fractal_analysis', {}) or {}
            current_price = kl.get('current_price')
            def dist_pct(level):
                try:
                    if current_price is None or level is None or float(current_price) == 0:
                        return None
                    return round(100.0 * (float(level) - float(current_price)) / float(current_price), 2)
                except Exception:
                    return None
            ns = kl.get('nearest_support') or {}
            nr = kl.get('nearest_resistance') or {}
            regime = ta.get('market_regime', {}) or {}
            return {
                'source': 'market_structure_agent',
                'timestamp': datetime.now().isoformat(),
                'regime': {
                    'regime': regime.get('regime', 'unknown'),
                    'confidence': regime.get('confidence', 0.0)
                },
                'structure_bias': bos.get('structural_bias', 'unknown'),
                'trend': {
                    'direction': ta.get('trend_direction', 'unknown'),
                    'strength': ta.get('trend_strength', 'unknown'),
                    'quality': ta.get('trend_quality', 'unknown')
                },
                'bos_choch': {
                    'total_bos_events': bos.get('total_bos_events', 0),
                    'total_choch_events': bos.get('total_choch_events', 0),
                    'recent_structural_break': bos.get('recent_structural_break')
                },
                'key_levels': {
                    'current_price': current_price,
                    'nearest_support': (
                        {'level': ns.get('level'), 'distance_pct': dist_pct(ns.get('level'))} if ns else None
                    ),
                    'nearest_resistance': (
                        {'level': nr.get('level'), 'distance_pct': dist_pct(nr.get('level'))} if nr else None
                    ),
                    'price_position_description': cs.get('price_position_description', 'unknown')
                },
                'fractal': {
                    'timeframe_alignment': fr.get('timeframe_alignment', 'unknown'),
                    'trend_consensus': fr.get('trend_consensus', 'unknown')
                }
            }
        return None
    except Exception as e:
        logger.warning(f"Market structure compute failed: {e}")
        return None


def _regime_badge(regime: str) -> Dict[str, Any]:
    regime_key = (regime or 'unknown').lower()
    color_map = {
        'trending': COLORS['regime_trending'],
        'consolidating': COLORS['regime_consolidating'],
        'volatile': COLORS['regime_volatile'],
        'stable': COLORS['regime_stable'],
    }
    return {
        'bg': color_map.get(regime_key, '#7f7f7f'),
        'txt': (regime or 'unknown').title()
    }


def _render_chart(symbol: str, df: pd.DataFrame, ms_ctx: Optional[Dict[str, Any]], outdir: Path) -> Path:
    # Compute SMAs
    work = df.copy()
    work['sma20'] = work['close'].rolling(20).mean()
    work['sma50'] = work['close'].rolling(50).mean()

    # Subplots: price (70%), volume (30%)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.03)

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=work.index,
            open=work['open'], high=work['high'], low=work['low'], close=work['close'],
            increasing_line_color=COLORS['bull'], increasing_fillcolor=COLORS['bull'],
            decreasing_line_color=COLORS['bear'], decreasing_fillcolor=COLORS['bear'],
            name='Price'
        ), row=1, col=1
    )

    # SMAs
    fig.add_trace(go.Scatter(x=work.index, y=work['sma20'], name='SMA20', line=dict(color=COLORS['sma20'], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=work.index, y=work['sma50'], name='SMA50', line=dict(color=COLORS['sma50'], width=1)), row=1, col=1)

    # Current price line
    last_close = float(work['close'].iloc[-1])
    fig.add_hline(y=last_close, line_dash='dot', line_color=COLORS['current_price'], annotation_text=f"{last_close:.2f}", annotation_position='top left', row=1, col=1)

    # Key levels & market structure overlays
    if ms_ctx and isinstance(ms_ctx, dict):
        kl = ms_ctx.get('key_levels', {}) or {}
        ns = kl.get('nearest_support') or {}
        nr = kl.get('nearest_resistance') or {}
        if ns and ns.get('level') is not None:
            fig.add_hline(y=float(ns['level']), line_dash='dash', line_color=COLORS['support'],
                          annotation_text=f"S: {ns['level']} ({ns.get('distance_pct')}%)", annotation_position='bottom left', row=1, col=1)
        if nr and nr.get('level') is not None:
            fig.add_hline(y=float(nr['level']), line_dash='dash', line_color=COLORS['resistance'],
                          annotation_text=f"R: {nr['level']} ({nr.get('distance_pct')}%)", annotation_position='top right', row=1, col=1)
        # Recent break marker
        rb = (ms_ctx.get('bos_choch') or {}).get('recent_structural_break')
        if rb and rb.get('date'):
            try:
                # Plot vertical line at break date within visible range
                bdate = pd.to_datetime(rb['date'])
                fig.add_vline(x=bdate, line=dict(color=COLORS['bear'] if 'bear' in str(rb.get('type','')).lower() else COLORS['bull'], width=1),
                              annotation_text=f"{rb.get('type','break')} ({rb.get('percentage_break','?')}%)", annotation_position='top', row=1, col=1)
            except Exception:
                pass
        # Regime badge annotation
        regime = (ms_ctx.get('regime') or {}).get('regime', 'unknown')
        conf = (ms_ctx.get('regime') or {}).get('confidence', 0)
        badge = _regime_badge(regime)
        fig.add_annotation(
            xref='paper', yref='paper', x=0.99, y=0.98, xanchor='right', yanchor='top',
            text=f"{badge['txt']} ({conf:.0%})",
            showarrow=False,
            font=dict(color='white'),
            bgcolor=badge['bg'],
            bordercolor='black', borderwidth=0.5, opacity=0.9
        )

    # Volume
    fig.add_trace(
        go.Bar(x=work.index, y=work['volume'], name='Volume', marker_color='rgba(100,100,100,0.7)'),
        row=2, col=1
    )
    # Volume baseline (20-bar average)
    work['vma20'] = work['volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=work.index, y=work['vma20'], name='VMA20', line=dict(color='#666', width=1, dash='dot')), row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"{symbol} — Price/Volume with Market Structure",
        height=900, width=1600,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Export
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{symbol}_price_volume_ms.png"
    try:
        # Try PNG export (requires kaleido)
        fig.write_image(str(png_path), scale=2)
        logger.info(f"Saved PNG: {png_path}")
        return png_path
    except Exception as e:
        logger.warning(f"PNG export failed for {symbol}: {e}. Saving HTML fallback.")
        html_path = outdir / f"{symbol}_price_volume_ms.html"
        fig.write_html(str(html_path))
        return html_path


def main(symbols: List[str], period_days: int, outdir: str):
    orch, zc = _init_data_sources()
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for sym in symbols:
        df = _fetch_data(sym, period_days, orch, zc)
        if df is None or len(df) < 50:
            logger.error(f"Skipping {sym}: insufficient data")
            continue
        ms_ctx = _compute_ms_context(df)
        img_path = _render_chart(sym, df, ms_ctx, out_path)
        results.append({'symbol': sym, 'output': str(img_path), 'market_context': ms_ctx})

    # Save manifest for review
    manifest = out_path / f"manifest_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    with open(manifest, 'w', encoding='utf-8') as f:
        json.dump({'generated': results, 'symbols': symbols, 'period_days': period_days}, f, indent=2, default=str)
    logger.info(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mockup chart image generation for LLM input")
    parser.add_argument('--symbols', type=str, default='RELIANCE,TCS,INFY,HDFCBANK,ITC')
    parser.add_argument('--period', type=int, default=365)
    parser.add_argument('--outdir', type=str, default=str(Path(__file__).parent / 'charts_mock'))
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    main(syms, args.period, args.outdir)

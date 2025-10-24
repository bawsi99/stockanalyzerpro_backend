#!/usr/bin/env python3
"""
Market Structure Agent - Multi-Stock End-to-End Test

Runs the full market-structure pipeline (processor -> charts -> LLM) on multiple
symbols using REAL MARKET DATA where available (with synthetic fallback). Stores
all inputs/outputs: prompts, images, and LLM responses for later inspection.
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# Load environment variables (from backend/config/.env if present)
try:
    import dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', '.env')
    dotenv.load_dotenv(dotenv_path=env_path)
    print(f"✅ Environment variables loaded from: {env_path}")
except Exception:
    print("⚠️ python-dotenv not available or failed to load; relying on process env")

# Ensure backend is importable
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
BACKEND_DIR = os.path.abspath(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Imports from the project
from agents.patterns.market_structure_agent.agent import MarketStructureAgent

# Real data clients (optional; fall back to synthetic if unavailable)
try:
    from core.orchestrator import StockAnalysisOrchestrator
except Exception as e:
    print(f"⚠️ Could not import StockAnalysisOrchestrator: {e}")
    StockAnalysisOrchestrator = None

try:
    from zerodha.client import ZerodhaDataClient
except Exception as e:
    print(f"⚠️ Could not import ZerodhaDataClient: {e}")
    ZerodhaDataClient = None

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("market_structure_multi_stock_test")


class MarketStructureMultiStockTester:
    """Comprehensive E2E tester for the Market Structure Agent."""

    def __init__(self, output_dir: str = None) -> None:
        self.agent = MarketStructureAgent()

        # Real data helpers
        self.orchestrator = None
        self.zerodha_client = None
        self.use_real_data = False

        if StockAnalysisOrchestrator is not None:
            try:
                self.orchestrator = StockAnalysisOrchestrator()
            except Exception as e:
                logger.warning(f"Orchestrator init failed: {e}")
        if ZerodhaDataClient is not None:
            try:
                self.zerodha_client = ZerodhaDataClient()
            except Exception as e:
                logger.warning(f"Zerodha client init failed: {e}")

        self.use_real_data = bool(self.orchestrator or self.zerodha_client)

        # Output directory
        base = Path(__file__).parent
        # Use timestamped output dir if not specified
        if output_dir is None:
            output_dir = f"market_structure_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = (base / output_dir) if not os.path.isabs(output_dir) else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Config
        self.config: Dict[str, Any] = {
            "save_individual_results": True,
            "save_prompts_responses": True,
            "save_charts": True,
            "max_concurrent": 3,
        }

        # In-memory aggregation
        self.results: List[Dict[str, Any]] = []

    async def run(self, symbols: List[str], periods: List[int] = [365], max_concurrent: int = 3, save_results: bool = True) -> Dict[str, Any]:
        start = datetime.now()
        cases = self._build_cases(symbols, periods)
        logger.info(f"[MARKET_STRUCTURE_TEST] Running {len(cases)} cases | Real data: {self.use_real_data}")

        sem = asyncio.Semaphore(max_concurrent)

        async def runner(tc: Dict[str, Any]):
            async with sem:
                return await self._run_single(tc)

        raw = await asyncio.gather(*[runner(tc) for tc in cases], return_exceptions=True)
        for i, r in enumerate(raw):
            if isinstance(r, Exception):
                logger.error(f"Test case {cases[i]['test_id']} failed: {r}")
                self.results.append(self._failed_result(cases[i], str(r)))
            else:
                self.results.append(r)

        report = self._summarize_results(start)
        if save_results:
            await self._save_report(report)
        return report

    def _build_cases(self, symbols: List[str], periods: List[int]) -> List[Dict[str, Any]]:
        cases = []
        for symbol in symbols:
            for period in periods:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                test_case = {
                    "symbol": symbol,
                    "period_days": period,
                    "test_id": f"{symbol}_{period}d_market_structure_{ts}",
                    "start_date": datetime.now() - timedelta(days=period + 10),  # Add buffer for data
                    "end_date": datetime.now()
                }
                cases.append(test_case)
        return cases

    async def _run_single(self, tc: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        symbol = tc["symbol"]
        period_days = tc["period_days"]
        test_id = tc["test_id"]
        logger.info(f"[MARKET_STRUCTURE_TEST] ▶️ {test_id}")

        # Prepare output directory for this test
        tdir = self.output_dir / test_id
        tdir.mkdir(exist_ok=True)
        prompts_dir = tdir / "prompts_responses"
        prompts_dir.mkdir(exist_ok=True)

        # Fetch data (real -> fallback synthetic)
        df, source = await self._get_stock_data(symbol, period_days)
        if df is None or len(df) < 30:
            return self._failed_result(tc, "Insufficient data after fetch/fallback")

        # Run full market structure pipeline
        try:
            analysis = await self.agent.analyze_complete(stock_data=df, symbol=symbol)
        except Exception as e:
            logger.error(f"Agent execution failed for {symbol}: {e}")
            return self._failed_result(tc, f"Agent execution failed: {e}")

        # Save chart (if produced)
        chart_path = None
        chart_bytes = analysis.get("chart_image")
        if chart_bytes and self.config["save_charts"]:
            chart_path = str(tdir / f"{test_id}_chart.png")
            try:
                with open(chart_path, "wb") as f:
                    f.write(chart_bytes)
            except Exception as e:
                logger.warning(f"Failed to save chart for {test_id}: {e}")
                chart_path = None

        # Rebuild and save prompt + response
        prompt_path = None
        response_path = None
        try:
            tech = analysis.get("technical_analysis", {})
            llm_agent = self.agent.llm_agent
            prompt = llm_agent.build_analysis_prompt(tech, symbol)
            prompt_path = str(prompts_dir / f"{test_id}_prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                # Attach chart metadata header
                header = [
                    "# MARKET STRUCTURE PROMPT METADATA",
                    f"# Symbol: {symbol}",
                    f"# Chart Path: {chart_path or 'N/A'}",
                    f"# Generated: {datetime.now().isoformat()}",
                    "",
                ]
                f.write("\n".join(header) + "\n")
                f.write(prompt)

            # Save LLM response (string) if available
            llm_resp = analysis.get("llm_analysis")
            if llm_resp:
                response_path = str(prompts_dir / f"{test_id}_response.txt")
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write(llm_resp)
                # Also attempt to parse the JSON payload and save separately
                try:
                    parsed = self._extract_json_from_response(llm_resp)
                    if parsed is not None:
                        with open(prompts_dir / f"{test_id}_response_parsed.json", "w", encoding="utf-8") as jf:
                            json.dump(parsed, jf, indent=2, default=str)
                except Exception as _:
                    pass
        except Exception as e:
            logger.warning(f"Failed to save prompt/response for {test_id}: {e}")

        duration = time.time() - t0
        result = {
            "test_id": test_id,
            "symbol": symbol,
            "period_days": period_days,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "data_points": len(df),
            "data_source": source,
            "success": analysis.get("success", False),
            "processing_time": analysis.get("processing_time", duration),
            "confidence_score": analysis.get("confidence_score", 0),
            "has_llm_analysis": analysis.get("has_llm_analysis", False),
            "chart_saved": bool(chart_path),
            "chart_path": chart_path,
            "prompt_path": prompt_path,
            "response_path": response_path,
            # Persist a lighter copy (full result can be large due to bytes)
            "summary": self._summarize_single(analysis),
        }

        # Save per-test artifacts
        if self.config["save_individual_results"]:
            await self._save_individual(result, analysis, tdir)

        # Log
        logger.info(
            f"[MARKET_STRUCTURE_TEST] ✅ {test_id} | success={result['success']} "
            f"chart={'yes' if result['chart_saved'] else 'no'} llm={result['has_llm_analysis']} "
            f"time={result['duration']:.2f}s"
        )
        return result

    async def _get_stock_data(self, symbol: str, period_days: int) -> (Optional[pd.DataFrame], str):
        # Prefer orchestrator
        if self.orchestrator is not None:
            try:
                df = await self.orchestrator.retrieve_stock_data(symbol=symbol, exchange="NSE", interval="day", period=period_days)
                if df is not None and not df.empty:
                    # Ensure required columns
                    if all(c in df.columns for c in ["open", "high", "low", "close"]) and ("volume" in df.columns):
                        return df, "real_market_data"
            except Exception as e:
                logger.warning(f"Orchestrator fetch failed for {symbol}: {e}")

        # Fallback to Zerodha client
        if self.zerodha_client is not None:
            try:
                if hasattr(self.zerodha_client, 'get_historical_data_async'):
                    df = await self.zerodha_client.get_historical_data_async(symbol=symbol, exchange="NSE", interval="day", period=period_days)
                else:
                    loop = asyncio.get_event_loop()
                    df = await loop.run_in_executor(None, self.zerodha_client.get_historical_data, symbol, "NSE", "day", None, None, period_days)
                if df is not None and not df.empty:
                    # Normalize index/columns
                    if 'date' not in df.columns:
                        if df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index()
                            if 'date' not in df.columns:
                                df['date'] = df.iloc[:, 0]
                        else:
                            df['date'] = df.index
                            df = df.reset_index(drop=True)
                    df = df.sort_values('date').reset_index(drop=True).set_index('date')
                    req_cols = ["open", "high", "low", "close", "volume"]
                    if all(c in df.columns for c in req_cols):
                        return df, "real_market_data"
            except Exception as e:
                logger.warning(f"Zerodha fetch failed for {symbol}: {e}")

        # Synthetic fallback
        df = self._generate_synthetic(symbol, period_days)
        return df, "synthetic_fallback" if df is not None else (None, "unavailable")

    def _generate_synthetic(self, symbol: str, period_days: int) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now().date()
            start = end - timedelta(days=period_days)
            dates = pd.date_range(start=start, end=end, freq='D')

            rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
            base = rng.uniform(80, 600)
            drift = rng.normal(0.0006, 0.0003, len(dates)).cumsum()
            noise = rng.normal(0, 0.02, len(dates))
            cyc = 0.04 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
            price = base * np.exp(drift + noise + cyc)

            rows = []
            prev = price[0]
            for d, c in zip(dates, price):
                vol = rng.uniform(1e5, 1e6)
                o = c * (1 + rng.normal(0, 0.003))
                h = max(o, c) * (1 + rng.uniform(0.0, 0.02))
                l = min(o, c) * (1 - rng.uniform(0.0, 0.02))
                rows.append({"date": d, "open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2), "volume": int(vol)})
                prev = c
            df = pd.DataFrame(rows).set_index("date")
            return df
        except Exception as e:
            logger.error(f"Synthetic generation failed for {symbol}: {e}")
            return None

    async def _save_individual(self, result: Dict[str, Any], full: Dict[str, Any], tdir: Path) -> None:
        try:
            # Light result
            with open(tdir / "result_summary.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Full (without raw bytes) – remove chart bytes to keep file small
            full_copy = dict(full)
            if isinstance(full_copy.get("chart_image"), (bytes, bytearray)):
                full_copy["chart_image"] = f"<bytes:{len(full_copy['chart_image'])}B>"
            with open(tdir / "full_result.json", "w") as f:
                json.dump(full_copy, f, indent=2, default=str)

            # Chart info
            chart_path = result.get("chart_path")
            if chart_path and os.path.exists(chart_path):
                info = {
                    "chart_path": chart_path,
                    "size_bytes": os.path.getsize(chart_path),
                    "saved": True,
                }
                with open(tdir / "chart_info.json", "w") as f:
                    json.dump(info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save artifacts for {result.get('test_id')}: {e}")

    def _summarize_single(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tech = analysis.get("technical_analysis", {})
            sq = (tech.get("structure_quality") or {})
            ta = (tech.get("trend_analysis") or {})
            bos = (tech.get("bos_choch_analysis") or {})
            swings = (tech.get("swing_points") or {})
            return {
                "structure_quality": {
                    "rating": sq.get("quality_rating"),
                    "score": sq.get("quality_score"),
                },
                "trend": {
                    "direction": ta.get("trend_direction"),
                    "strength": ta.get("trend_strength"),
                    "quality": ta.get("trend_quality"),
                },
                "structural_breaks": {
                    "bias": bos.get("structural_bias"),
                    "total_bos": bos.get("total_bos_events"),
                    "total_choch": bos.get("total_choch_events"),
                },
                "swing_points": {
                    "total": swings.get("total_swings"),
                    "density": swings.get("swing_density"),
                },
                "has_llm": analysis.get("has_llm_analysis", False),
                "confidence": analysis.get("confidence_score", 0),
            }
        except Exception:
            return {"summary": "unavailable"}

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from a response possibly wrapped in ```json fences."""
        try:
            s = text.strip()
            if s.startswith("```"):
                start = s.find('{')
                end = s.rfind('}')
                if start != -1 and end != -1 and end > start:
                    return json.loads(s[start:end+1])
            # Fallback: try direct parse
            return json.loads(s)
        except Exception:
            return None

    def _failed_result(self, tc: Dict[str, Any], error: str) -> Dict[str, Any]:
        return {
            "test_id": tc.get("test_id"),
            "symbol": tc.get("symbol"),
            "period_days": tc.get("period_days"),
            "timestamp": datetime.now().isoformat(),
            "duration": 0.0,
            "data_points": 0,
            "data_source": "unavailable",
            "success": False,
            "error": error,
        }

    def _summarize_results(self, start_time: datetime) -> Dict[str, Any]:
        total = len(self.results)
        success = sum(1 for r in self.results if r.get("success"))
        llm_ok = sum(1 for r in self.results if r.get("has_llm_analysis"))
        charts = sum(1 for r in self.results if r.get("chart_saved"))
        dur = (datetime.now() - start_time).total_seconds()
        avg_conf = np.mean([r.get("confidence_score", 0) for r in self.results if r.get("success")]) if success else 0.0

        return {
            "metadata": {
                "tester": "market_structure_multi_stock_tester",
                "version": "1.0.0",
                "started": start_time.isoformat(),
                "ended": datetime.now().isoformat(),
                "duration": dur,
            },
            "counts": {
                "total_tests": total,
                "successful": success,
                "chart_saved": charts,
                "llm_analysis_success": llm_ok,
            },
            "metrics": {
                "avg_confidence_score": float(avg_conf) if not np.isnan(avg_conf) else 0.0,
            },
            "individual_results": self.results,
        }

    async def _save_report(self, report: Dict[str, Any]) -> None:
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(self.output_dir / f"market_structure_report_{ts}.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            # Simple text summary
            with open(self.output_dir / "summary.txt", "w") as f:
                f.write(
                    "\n".join(
                        [
                            f"Market Structure Test Summary",
                            f"Total: {report['counts']['total_tests']}",
                            f"Successful: {report['counts']['successful']}",
                            f"Charts Saved: {report['counts']['chart_saved']}",
                            f"LLM Success: {report['counts']['llm_analysis_success']}",
                            f"Avg Confidence: {report['metrics']['avg_confidence_score']:.2f}",
                        ]
                    )
                )
            logger.info(f"Results saved to {self.output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")


async def main() -> Dict[str, Any]:
    tester = MarketStructureMultiStockTester(output_dir="market_structure_test_results")

    print("Data Source Configuration:")
    print(f"  Real Data Available: {tester.use_real_data}")
    print(f"  Orchestrator: {'Available' if tester.orchestrator else 'Not Available'}")
    print(f"  Zerodha Client: {'Available' if tester.zerodha_client else 'Not Available'}")

    #symbols = ["INFY", "RELIANCE", "TCS", "HDFCBANK"]
    symbols = ["WIPRO"]
    periods = [365]
    max_concurrent = tester.config.get("max_concurrent", 3)

    print("Starting Market Structure Agent Multi-Stock Test...")
    report = await tester.run(symbols, periods, max_concurrent=max_concurrent, save_results=True)

    print("\n" + "=" * 60)
    print("MARKET STRUCTURE TEST RESULTS SUMMARY")
    print("=" * 60)
    counts = report.get("counts", {})
    metrics = report.get("metrics", {})
    print(f"Total: {counts.get('total_tests', 0)} | Successful: {counts.get('successful', 0)} | Charts: {counts.get('chart_saved', 0)} | LLM: {counts.get('llm_analysis_success', 0)}")
    print(f"Avg Confidence: {metrics.get('avg_confidence_score', 0):.2f}")
    print(f"Artifacts saved to: {Path(tester.output_dir).resolve()}")

    return report


if __name__ == "__main__":
    asyncio.run(main())

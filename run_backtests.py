#!/usr/bin/env python3
import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from .backtesting import Backtester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtests")


def _validate_args(args) -> None:
    if not args.symbols or not isinstance(args.symbols, list):
        raise ValueError("--symbols must be a list of symbols")
    for s in args.symbols:
        if not re.match(r"^[A-Z0-9_\.:-]+$", s):
            raise ValueError(f"Invalid symbol format: {s}")
    if args.days <= 0:
        raise ValueError("--days must be positive")
    if args.lookahead <= 0:
        raise ValueError("--lookahead must be positive")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests for the pattern analysis system")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE"], help="Symbols to backtest")
    parser.add_argument("--days", type=int, default=int(os.getenv("BACKTEST_DAYS", 180)), help="Historical days to test")
    parser.add_argument("--lookahead", type=int, default=int(os.getenv("BACKTEST_LOOKAHEAD_DAYS", 3)), help="Days to look ahead for outcome validation")
    parser.add_argument("--exchange", type=str, default=os.getenv("BACKTEST_EXCHANGE", "NSE"), help="Exchange (default: NSE)")
    parser.add_argument("--interval", type=str, default=os.getenv("BACKTEST_INTERVAL", "day"), help="Interval ('day' recommended)")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("BACKTEST_SUCCESS_THRESHOLD_PCT", 1.0)), help="Success threshold in % move (default 1.0)")
    parser.add_argument("--workers", type=int, default=int(os.getenv("BACKTEST_WORKERS", 4)), help="Parallel workers for multi-symbol backtests")
    args = parser.parse_args()

    _validate_args(args)

    end = datetime.now().date()
    start = end - timedelta(days=int(args.days))

    bt = Backtester(exchange=args.exchange)

    def run_one(sym: str):
        logger.info(f"Running backtest for {sym} from {start} to {end} (lookahead={args.lookahead}d)")
        return sym, bt.run(
            symbol=sym,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            interval=args.interval,
            lookahead_days=int(args.lookahead),
            success_threshold_pct=float(args.threshold),
        )

    # Parallel per-symbol runs
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one, s): s for s in args.symbols}
        for fut in as_completed(futures):
            sym, summary = fut.result()
            if not summary:
                logger.info(f"No patterns detected for {sym} in period")
                continue
            logger.info(f"Results for {sym}:")
            for ptype, stats in summary.items():
                logger.info(
                    f"  {ptype:28s} detected={stats['detected']:4d} confirmed={stats['confirmed']:4d} success={stats['success']:4d} rate={stats['success_rate']:5.1f}%"
                )


if __name__ == "__main__":
    main()



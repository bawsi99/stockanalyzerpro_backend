import json
import os
import argparse
import sys
from pathlib import Path
import asyncio

# Add the backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from analysis.orchestrator import StockAnalysisOrchestrator


def main():
    """
    Main function to run the Stock Analysis Agent.
    Parses command line arguments and runs the analysis workflow.
    """
    parser = argparse.ArgumentParser(description='Stock Analysis Agent for Indian Stock Market (NSE)')
    parser.add_argument('--stock', type=str, required=True, help='Stock symbol to analyze (e.g., RELIANCE)')
    parser.add_argument('--exchange', type=str, default='NSE', choices=['NSE'], help='Stock exchange (default: NSE)')
    parser.add_argument('--period', type=int, default=365, help='Analysis period in days (default: 365)')
    parser.add_argument(
        '--interval',
        type=str,
        default='day',
        choices=[
            'minute', '3minute', '5minute', '10minute', '15minute', '30minute', '60minute', 'day',
            'week', 'month'
        ],
        help='Data interval (default: day). Supported: minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day, week, month.'
    )
    args = parser.parse_args()
    print(f"Starting analysis for {args.stock} ({args.exchange})")
    print(f"Analysis period: {args.period} days")
    print(f"Charts will be stored in Redis")
    try:
        orchestrator = StockAnalysisOrchestrator()
        orchestrator.authenticate()
        try:
            results, success_message, error_message = asyncio.run(orchestrator.enhanced_analyze_stock(
                symbol=args.stock,
                exchange=args.exchange,
                period=args.period,
                interval=args.interval
            ))
            if error_message:
                print(f"Error during analysis: {error_message}")
                return
            def debug_print_types(obj, prefix="results"):
                """Recursively print types of all keys in a nested object for debugging."""
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            if k == 'trend' or 'trend' in k:
                                pass
                            debug_print_types(v, prefix=f"{prefix}['{k}']")
                        except Exception as ex:
                            pass
                elif isinstance(obj, list):
                    for idx, item in enumerate(obj):
                        try:
                            debug_print_types(item, prefix=f"{prefix}[{idx}]")
                        except Exception as ex:
                            pass
            debug_print_types(results)
            print("Analysis completed successfully.")
        except Exception as e:
            print(f"Exception during analysis: {e}")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")

if __name__ == '__main__':
    main()

import json
import os
import sys
import argparse
from agent_capabilities import StockAnalysisOrchestrator
import asyncio
from utils import clean_for_json


def main():
    """
    Main function to run the Stock Analysis Agent.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Analysis Agent for Indian Stock Market (NSE)')
    parser.add_argument('--stock', type=str, required=True, help='Stock symbol to analyze (e.g., RELIANCE)')
    parser.add_argument('--exchange', type=str, default='NSE', help='Stock exchange (default: NSE)')
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
    # Create output directory
    output = f'./output/{args.stock}'
    os.makedirs(output, exist_ok=True)
    
    print(f"Starting analysis for {args.stock} ({args.exchange})")
    print(f"Analysis period: {args.period} days")
    print(f"Output directory: {output}")
    
    # Set API keys if provided (REMOVED, now always fetched from os.getenv in client)

    try:
        #print("Using full orchestrator with A2A, RAG, and MCP capabilities")
        # Create orchestrator
        orchestrator = StockAnalysisOrchestrator()
        
        # Authenticate
        orchestrator.authenticate()
        
        # Analyze stock  
        try:
            results, success_message, error_message = asyncio.run(orchestrator.analyze_stock(
                symbol=args.stock,
                exchange=args.exchange,
                period=args.period,
                interval=args.interval,
                output_dir=output
            ))
            
            if error_message:
                print(f"Error during analysis: {error_message}")
                return
            # print(f"[DEBUG-POST] analyze_stock returned: results type={type(results)}, data type={type(data)}")

            # DEBUG: Print all keys and types in results before serialization, with exception handling
            def debug_print_types(obj, prefix="results"):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            # print(f"[DEBUG] {prefix}['{k}']: type={type(v)}")
                            if k == 'trend' or 'trend' in k:
                                # print(f"[DEBUG] {prefix}['{k}']: value={repr(v)}")
                                pass
                            debug_print_types(v, prefix=f"{prefix}['{k}']")
                        except Exception as ex:
                            # print(f"[DEBUG-ERROR] Exception at {prefix}['{k}']: {ex}")
                            pass
                elif isinstance(obj, list):
                    for idx, item in enumerate(obj):
                        try:
                            debug_print_types(item, prefix=f"{prefix}[{idx}]")
                        except Exception as ex:
                            # print(f"[DEBUG-ERROR] Exception at {prefix}[{idx}]: {ex}")
                            pass
            debug_print_types(results)
            with open(f"{output}/results.json", "w") as f:
                json.dump(clean_for_json(results), f)

            # Print the markdown summary if present
            if 'indicator_summary_md' in results:
                print("\nTechnical Indicator Summary (Markdown):\n")
                print(results['indicator_summary_md'])

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return
            
        
        # Print recommendation summary
        if isinstance(results, dict) and "recommendation" in results:
            print("\nInvestment Recommendation Summary:")
            # Print first paragraph of recommendation
            print(results['recommendation'].split('\n\n')[0])
        
        print(f"\nAnalysis completed successfully. Results saved to {output}")
        
        # List generated files
        print("\nGenerated files:")
        for filename in os.listdir(output):
            if args.stock in filename:
                file_path = os.path.join(output, filename)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {filename} ({file_size:.1f} KB)")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":

    start_time = os.times()[4]  # Get start time in seconds
    print(f"Starting Stock Analysis Agent at {start_time} seconds")
    main()
    end_time = os.times()[4]  # Get end time in seconds
    elapsed_time = end_time - start_time
    print(f"Stock Analysis Agent completed in {elapsed_time:.2f} seconds")

#!/usr/bin/env python3
"""
Debug test to isolate the data loading issue
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zerodha_client import ZerodhaDataClient

def test_zerodha_client():
    """Test the Zerodha client directly."""
    print("Testing Zerodha client...")
    
    try:
        client = ZerodhaDataClient()
        
        # Test the method signature
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Calling get_historical_data with:")
        print(f"  symbol: RELIANCE")
        print(f"  exchange: NSE")
        print(f"  interval: day")
        print(f"  from_date: {start_date}")
        print(f"  to_date: {end_date}")
        
        data = client.get_historical_data(
            symbol="RELIANCE",
            exchange="NSE",
            interval="day",
            from_date=start_date,
            to_date=end_date
        )
        
        if data is not None and not data.empty:
            print(f"✅ Success! Loaded {len(data)} records")
            print(f"Columns: {list(data.columns)}")
            print(f"First few rows:")
            print(data.head())
        else:
            print("❌ No data returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zerodha_client()

#!/usr/bin/env python3
"""
Simple Test - Verify that analysis storage works without column reference errors
"""

import os
import sys
import uuid
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables")
except ImportError:
    print("⚠️ python-dotenv not installed")

from supabase_client import get_supabase_client

def test_simple_analysis_storage():
    """Test simple analysis storage without triggers."""
    print("🧪 Testing Simple Analysis Storage")
    print("=" * 50)
    
    try:
        supabase = get_supabase_client()
        
        # Get existing user
        user_result = supabase.table("profiles").select("id").limit(1).execute()
        if not user_result.data:
            print("❌ No existing users found")
            return False
        
        user_id = user_result.data[0]["id"]
        print(f"✅ Using existing user: {user_id}")
        
        # Create simple analysis data
        analysis_data = {
            "user_id": user_id,
            "stock_symbol": "RELIANCE",
            "analysis_data_json": {
                "ai_analysis": {
                    "trend": "Bullish",
                    "confidence_pct": 75.0,
                    "short_term": {
                        "horizon_days": 10,
                        "entry_range": [100.0, 105.0],
                        "stop_loss": 98.0,
                        "targets": [110.0, 115.0]
                    }
                },
                "indicators": {
                    "rsi": {"value": 65.5, "signal": "neutral", "strength": "moderate"},
                    "macd": {"value": 2.5, "signal": "bullish", "strength": "strong"}
                },
                "summary": {
                    "overall_signal": "buy",
                    "risk_level": "medium",
                    "analysis_quality": "high",
                    "confidence_score": 75
                },
                "metadata": {
                    "current_price": 105.0,
                    "price_change_pct": 2.5,
                    "sector": "Technology",
                    "analysis_timestamp": datetime.now().isoformat()
                }
            },
            "exchange": "NSE",
            "period_days": 30,
            "interval": "day",
            "overall_signal": "buy",
            "confidence_score": 75.0,
            "risk_level": "medium",
            "current_price": 105.0,
            "price_change_percentage": 2.5,
            "sector": "Technology",
            "analysis_type": "standard",
            "analysis_quality": "high",
            "mathematical_validation": False,
            "metadata": {"test": True}
        }
        
        print("Attempting to store analysis...")
        result = supabase.table("stock_analyses").insert(analysis_data).execute()
        
        if result.data:
            analysis_id = result.data[0]["id"]
            print(f"✅ Analysis stored successfully! ID: {analysis_id}")
            
            # Verify the data was stored correctly
            stored_data = supabase.table("stock_analyses").select("*").eq("id", analysis_id).execute()
            if stored_data.data:
                stored_analysis = stored_data.data[0]
                print(f"✅ Retrieved stored analysis:")
                print(f"   - Stock: {stored_analysis['stock_symbol']}")
                print(f"   - Signal: {stored_analysis['overall_signal']}")
                print(f"   - Confidence: {stored_analysis['confidence_score']}")
                print(f"   - Analysis data present: {'analysis_data_json' in stored_analysis}")
                
                # Clean up
                supabase.table("stock_analyses").delete().eq("id", analysis_id).execute()
                print("✅ Test data cleaned up")
                
                return True
            else:
                print("❌ Failed to retrieve stored analysis")
                return False
        else:
            print("❌ Failed to store analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 MAIN ISSUE VERIFICATION")
    print("=" * 50)
    
    if test_simple_analysis_storage():
        print("\n🎉 SUCCESS! The main column reference issue is FIXED!")
        print("✅ Analysis storage now works without column reference ambiguity errors.")
        print("✅ The analysis service should work properly now.")
        return True
    else:
        print("\n❌ FAILED! The issue is still present.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
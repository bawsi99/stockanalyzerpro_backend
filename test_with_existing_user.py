#!/usr/bin/env python3
"""
Test script to store JSON data using an existing user from the profiles table.
This avoids RLS policy issues by using an existing user.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_supabase_client_anon():
    """Get Supabase client using anon key for testing."""
    from supabase import create_client
    
    # Use the anon key from the frontend configuration
    SUPABASE_URL = "https://fjpxcnmogmspguftkvik.supabase.co"
    SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqcHhjbm1vZ21zcGd1ZnRrdmlrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkxNTI1MDMsImV4cCI6MjA2NDcyODUwM30.KmuSs44GrfxKQdUI1iZzAneTU9mLFEa5XiHVFnGfvww"
    
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read and parse JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"✅ Successfully read JSON file: {file_path}")
        print(f"📊 Data keys: {list(data.keys())}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def get_existing_user(supabase) -> Optional[str]:
    """Get an existing user from the profiles table."""
    try:
        # Query existing profiles
        result = supabase.table("profiles").select("id, email, full_name").limit(5).execute()
        
        if result.data and len(result.data) > 0:
            user = result.data[0]
            print(f"✅ Found existing user: {user['email']} (ID: {user['id']})")
            return user['id']
        else:
            print("❌ No existing users found in profiles table")
            return None
            
    except Exception as e:
        print(f"❌ Error getting existing user: {e}")
        return None

def store_analysis_directly(supabase, analysis_data: Dict[str, Any], user_id: str, symbol: str) -> bool:
    """Store analysis data directly in the stock_analyses table."""
    try:
        # Prepare analysis data for storage
        analysis_record = {
            "user_id": user_id,
            "stock_symbol": symbol,
            "analysis_data_json": analysis_data,
            "exchange": "NSE",
            "period_days": 365,
            "interval": "day",
            "overall_signal": analysis_data.get("ai_analysis", {}).get("trend"),
            "confidence_score": analysis_data.get("ai_analysis", {}).get("confidence_pct"),
            "risk_level": "medium",
            "current_price": 1429.0,
            "price_change_percentage": 0.0,
            "sector": "Oil & Gas",
            "analysis_type": "standard",
            "analysis_quality": "standard",
            "mathematical_validation": True,
            "chart_paths": None,
            "metadata": {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "data_source": "test_script"
            },
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "llm_calls_count": 0,
            "token_usage_breakdown": {}
        }
        
        # Insert the analysis record
        result = supabase.table("stock_analyses").insert(analysis_record).execute()
        
        if result.data:
            analysis_id = result.data[0]["id"]
            print(f"✅ Successfully stored analysis with ID: {analysis_id}")
            return True
        else:
            print("❌ Failed to store analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error storing analysis: {e}")
        return False

def main():
    """Main function to test JSON storage with existing user."""
    print("🚀 Starting JSON storage test with existing user...")
    
    # File path
    json_file_path = "output/RELIANCE/results.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ File not found: {json_file_path}")
        print("Please make sure the file exists and the path is correct.")
        return
    
    # Get Supabase client with anon key
    try:
        supabase = get_supabase_client_anon()
        print("✅ Connected to Supabase with anon key")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return
    
    # Read JSON file
    analysis_data = read_json_file(json_file_path)
    if not analysis_data:
        print("❌ Failed to read JSON file")
        return
    
    # Get existing user
    user_id = get_existing_user(supabase)
    if not user_id:
        print("❌ Failed to get existing user")
        print("Please make sure there's at least one user in the profiles table.")
        return
    
    # Store analysis
    symbol = "RELIANCE"
    success = store_analysis_directly(supabase, analysis_data, user_id, symbol)
    
    if success:
        print(f"🎉 Successfully stored analysis for {symbol}")
        print(f"👤 User ID: {user_id}")
        print("✅ Test completed successfully!")
    else:
        print("❌ Failed to store analysis")
        print("❌ Test failed!")

if __name__ == "__main__":
    main() 
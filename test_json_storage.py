#!/usr/bin/env python3
"""
Test script to read JSON file and store it in the database.
This script works around the foreign key constraint issue by using a simpler approach.
"""

import json
import uuid
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supabase_client import get_supabase_client

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as dictionary
    """
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

def create_test_user(supabase) -> str:
    """
    Create a test user in the profiles table.
    
    Args:
        supabase: Supabase client
        
    Returns:
        User ID string
    """
    try:
        # Generate a new UUID for the user
        user_id = str(uuid.uuid4())
        
        # Create profile data
        profile_data = {
            "email": f"test_user_{user_id[:8]}@test.local",
            "full_name": "Test User",
            "subscription_tier": "free",
            "preferences": {},
            "analysis_count": 0,
            "favorite_stocks": [],
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z"
        }
        
        # Try to insert profile (without specifying id to let database generate it)
        result = supabase.table("profiles").insert(profile_data).execute()
        
        if result.data:
            # Get the generated ID from the response
            generated_id = result.data[0]["id"]
            print(f"✅ Created test user with ID: {generated_id}")
            return generated_id
        else:
            print("❌ Failed to create test user")
            return None
            
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        return None

def store_analysis_directly(supabase, analysis_data: Dict[str, Any], user_id: str, symbol: str) -> bool:
    """
    Store analysis data directly in the stock_analyses table.
    
    Args:
        supabase: Supabase client
        analysis_data: Analysis data dictionary
        user_id: User ID
        symbol: Stock symbol
        
    Returns:
        bool: True if successful, False otherwise
    """
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
            "risk_level": "medium",  # Default value
            "current_price": 1429.0,  # From the JSON data
            "price_change_percentage": 0.0,  # Default value
            "sector": "Oil & Gas",  # RELIANCE sector
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
    """Main function to test JSON storage."""
    print("🚀 Starting JSON storage test...")
    
    # File path
    json_file_path = "output/RELIANCE/results.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ File not found: {json_file_path}")
        print("Please make sure the file exists and the path is correct.")
        return
    
    # Read JSON file
    analysis_data = read_json_file(json_file_path)
    if not analysis_data:
        print("❌ Failed to read JSON file")
        return
    
    # Get Supabase client
    try:
        supabase = get_supabase_client()
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return
    
    # Create test user
    user_id = create_test_user(supabase)
    if not user_id:
        print("❌ Failed to create test user")
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
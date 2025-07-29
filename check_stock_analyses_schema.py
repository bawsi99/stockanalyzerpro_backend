"""
Check Stock Analyses Table Schema

This script checks the actual schema of the stock_analyses table to understand
what columns exist and what the constraints are.
"""

import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

from supabase_client import get_supabase_client

def check_stock_analyses_schema():
    """Check the stock_analyses table schema."""
    print("🔍 Checking Stock Analyses Table Schema")
    print("=" * 50)
    
    try:
        supabase = get_supabase_client()
        
        # Try to get a sample record to understand the structure
        result = supabase.table("stock_analyses").select("*").limit(1).execute()
        
        if result.data:
            print("✅ Found existing stock analyses")
            sample_analysis = result.data[0]
            print(f"Sample analysis keys: {list(sample_analysis.keys())}")
            print(f"Sample analysis data:")
            for key, value in sample_analysis.items():
                print(f"  {key}: {type(value).__name__} = {value}")
        else:
            print("⚠️ No existing stock analyses found")
            
        # Try to get table information by attempting to insert with minimal data
        print("\n🔧 Testing minimal insert to understand required fields...")
        
        # Get existing user
        user_result = supabase.table("profiles").select("id").limit(1).execute()
        if user_result.data:
            user_id = user_result.data[0]["id"]
            
            # Try minimal insert
            minimal_data = {
                "user_id": user_id,
                "stock_symbol": "TEST"
            }
            
            try:
                test_result = supabase.table("stock_analyses").insert(minimal_data).execute()
                if test_result.data:
                    print("✅ Minimal insert succeeded")
                    test_id = test_result.data[0]["id"]
                    # Clean up
                    supabase.table("stock_analyses").delete().eq("id", test_id).execute()
                    print("✅ Cleaned up test record")
                else:
                    print("❌ Minimal insert failed")
            except Exception as insert_error:
                print(f"❌ Minimal insert error: {insert_error}")
        else:
            print("❌ No users found for testing")
            
    except Exception as e:
        print(f"❌ Error checking stock_analyses schema: {e}")

def main():
    """Main function."""
    check_stock_analyses_schema()

if __name__ == "__main__":
    main() 
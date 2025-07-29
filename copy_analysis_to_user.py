#!/usr/bin/env python3
"""
Script to copy an existing analysis to a new user for testing.
"""

import uuid
from datetime import datetime
from supabase_client import get_supabase_client

def copy_analysis_to_user(source_user_id: str, target_user_id: str):
    """Copy an analysis from one user to another."""
    try:
        supabase = get_supabase_client()
        
        # Get an analysis from the source user
        result = supabase.table("stock_analyses_simple").select("*").eq("user_id", source_user_id).limit(1).execute()
        
        if not result.data:
            print(f"❌ No analyses found for user: {source_user_id}")
            return None
        
        # Get the first analysis
        original_analysis = result.data[0]
        
        # Create a new analysis record for the target user
        new_analysis_data = {
            "user_id": target_user_id,
            "stock_symbol": original_analysis["stock_symbol"],
            "analysis_data": original_analysis["analysis_data"],
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z"
        }
        
        # Insert the new analysis
        insert_result = supabase.table("stock_analyses_simple").insert(new_analysis_data).execute()
        
        if insert_result.data:
            new_analysis_id = insert_result.data[0]["id"]
            print(f"✅ Successfully copied analysis to user {target_user_id}")
            print(f"📊 Analysis ID: {new_analysis_id}")
            print(f"📈 Stock: {original_analysis['stock_symbol']}")
            return new_analysis_id
        else:
            print("❌ Failed to copy analysis")
            return None
            
    except Exception as e:
        print(f"❌ Error copying analysis: {e}")
        return None

if __name__ == "__main__":
    # Source user ID (one of the existing users with analyses)
    source_user_id = "6036aee5-624c-4275-8482-f77d32723c32"
    
    # Target user ID (the new test user we just created)
    target_user_id = "ad6f0476-07d4-40d1-b9e2-4c31175edd37"
    
    print(f"🔧 Copying analysis from user {source_user_id} to user {target_user_id}...")
    analysis_id = copy_analysis_to_user(source_user_id, target_user_id)
    
    if analysis_id:
        print(f"\n🎉 Analysis copied successfully!")
        print(f"💡 The test user {target_user_id} now has access to the analysis")
    else:
        print("❌ Failed to copy analysis") 
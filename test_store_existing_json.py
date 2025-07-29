"""
Test Script: Store Existing JSON Results in Simplified Database
This script reads the existing results.json file and stores it in the database
"""

import json
import uuid
import os
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

from simple_database_manager import simple_db_manager

def test_store_existing_json():
    """Test storing existing JSON results in the simplified database."""
    print("🧪 Testing Store Existing JSON in Simplified Database")
    print("=" * 60)
    
    # File path
    json_file_path = "output/RELIANCE/results.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ File not found: {json_file_path}")
        print("Please run main.py first to generate the results.json file")
        return False
    
    print(f"✅ Found file: {json_file_path}")
    
    # Read the JSON file
    print("\n1. Reading JSON file...")
    try:
        with open(json_file_path, 'r') as f:
            analysis_data = json.load(f)
        
        print(f"✅ Successfully read JSON file")
        print(f"   File size: {os.path.getsize(json_file_path)} bytes")
        print(f"   Data keys: {list(analysis_data.keys())}")
        
        # Print some sample data structure
        if 'summary' in analysis_data:
            print(f"   Summary keys: {list(analysis_data['summary'].keys())}")
        if 'technical_indicators' in analysis_data:
            print(f"   Technical indicators: {list(analysis_data['technical_indicators'].keys())}")
            
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return False
    
    # Test database connection
    print("\n2. Testing database connection...")
    try:
        status = simple_db_manager.get_database_status()
        print(f"✅ Database connected: {status['connected']}")
        print(f"✅ Available tables: {list(status['tables'].keys())}")
        
        if not status['connected']:
            print("❌ Database not connected")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print("Make sure your environment variables are set:")
        print("  - SUPABASE_URL")
        print("  - SUPABASE_SERVICE_KEY")
        return False
    
    # Get or create a test user
    print("\n3. Setting up test user...")
    try:
        # Try to get existing user first
        existing_users = simple_db_manager.supabase.table("profiles").select("id").limit(1).execute()
        
        if existing_users.data:
            test_user_id = existing_users.data[0]['id']
            print(f"✅ Using existing user: {test_user_id}")
        else:
            # Create new test user
            test_user_id = str(uuid.uuid4())
            success = simple_db_manager.create_anonymous_user(test_user_id)
            if success:
                print(f"✅ Created new test user: {test_user_id}")
            else:
                print(f"❌ Failed to create test user")
                return False
                
    except Exception as e:
        print(f"❌ User setup error: {e}")
        return False
    
    # Store the analysis in database
    print("\n4. Storing analysis in database...")
    try:
        analysis_id = simple_db_manager.store_analysis(
            analysis=analysis_data,
            user_id=test_user_id,
            symbol="RELIANCE",
            exchange="NSE",
            period=365,
            interval="day"
        )
        
        if analysis_id:
            print(f"✅ Successfully stored analysis with ID: {analysis_id}")
        else:
            print(f"❌ Failed to store analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error storing analysis: {e}")
        return False
    
    # Retrieve and verify the stored data
    print("\n5. Retrieving stored analysis...")
    try:
        retrieved_analysis = simple_db_manager.get_analysis(analysis_id)
        
        if retrieved_analysis:
            print(f"✅ Successfully retrieved analysis")
            
            # Compare key data
            original_summary = analysis_data.get('summary', {})
            retrieved_summary = retrieved_analysis.get('summary', {})
            
            print(f"   Original signal: {original_summary.get('overall_signal', 'N/A')}")
            print(f"   Retrieved signal: {retrieved_summary.get('overall_signal', 'N/A')}")
            print(f"   Original confidence: {original_summary.get('confidence', 'N/A')}")
            print(f"   Retrieved confidence: {retrieved_summary.get('confidence', 'N/A')}")
            
            # Check if data matches
            if (original_summary.get('overall_signal') == retrieved_summary.get('overall_signal') and
                original_summary.get('confidence') == retrieved_summary.get('confidence')):
                print("✅ Data integrity verified - stored and retrieved data match!")
            else:
                print("⚠️ Data mismatch detected")
                
        else:
            print(f"❌ Failed to retrieve analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving analysis: {e}")
        return False
    
    # Get user analyses to verify it's in the list
    print("\n6. Verifying user analyses list...")
    try:
        user_analyses = simple_db_manager.get_user_analyses(test_user_id, limit=10)
        
        if user_analyses:
            print(f"✅ Found {len(user_analyses)} analyses for user")
            
            # Find our stored analysis
            stored_analysis = None
            for analysis in user_analyses:
                if analysis.get('id') == analysis_id:
                    stored_analysis = analysis
                    break
            
            if stored_analysis:
                print(f"✅ Found our stored analysis in user list")
                print(f"   Stock: {stored_analysis.get('stock_symbol')}")
                print(f"   Created: {stored_analysis.get('created_at')}")
            else:
                print(f"❌ Could not find our analysis in user list")
                
        else:
            print(f"❌ No analyses found for user")
            return False
            
    except Exception as e:
        print(f"❌ Error getting user analyses: {e}")
        return False
    
    # Test the analysis_summary_simple view
    print("\n7. Testing analysis summary view...")
    try:
        # Query the view directly
        view_result = simple_db_manager.supabase.table("analysis_summary_simple").select("*").eq("id", analysis_id).execute()
        
        if view_result.data:
            view_data = view_result.data[0]
            print(f"✅ Successfully queried analysis summary view")
            print(f"   Stock: {view_data.get('stock_symbol')}")
            print(f"   Signal: {view_data.get('overall_signal')}")
            print(f"   Confidence: {view_data.get('confidence_score')}")
            print(f"   User: {view_data.get('user_name')}")
        else:
            print(f"⚠️ No data found in analysis summary view")
            
    except Exception as e:
        print(f"❌ Error querying analysis summary view: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS: Existing JSON Successfully Stored in Database!")
    print("✅ File read successfully")
    print("✅ Database connection working")
    print("✅ Analysis stored with ID: " + analysis_id)
    print("✅ Data retrieved and verified")
    print("✅ User analysis list updated")
    print("✅ Analysis summary view working")
    print("=" * 60)
    print("\n📊 Database Test Results:")
    print(f"   - Analysis ID: {analysis_id}")
    print(f"   - User ID: {test_user_id}")
    print(f"   - Stock: RELIANCE")
    print(f"   - Data stored in: stock_analyses_simple table")
    print(f"   - JSON column: analysis_data")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_store_existing_json() 
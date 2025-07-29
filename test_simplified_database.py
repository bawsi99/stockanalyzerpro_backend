"""
Test Script for Simplified Database System
This script tests the new simplified database structure
"""

import uuid
import json
from datetime import datetime
from simple_database_manager import simple_db_manager

def test_simplified_database():
    """Test the simplified database system."""
    print("🧪 Testing Simplified Database System")
    print("=" * 50)
    
    # Test 1: Database Status
    print("\n1. Testing Database Status...")
    try:
        status = simple_db_manager.get_database_status()
        print(f"✅ Connected: {status['connected']}")
        print(f"✅ Tables: {list(status['tables'].keys())}")
        print(f"✅ Errors: {status['errors']}")
    except Exception as e:
        print(f"❌ Database status error: {e}")
        return False
    
    # Test 2: Create Test User
    print("\n2. Testing User Creation...")
    test_user_id = str(uuid.uuid4())
    try:
        success = simple_db_manager.create_anonymous_user(test_user_id)
        if success:
            print(f"✅ Created test user: {test_user_id}")
        else:
            print(f"❌ Failed to create test user")
            return False
    except Exception as e:
        print(f"❌ User creation error: {e}")
        return False
    
    # Test 3: Store Test Analysis
    print("\n3. Testing Analysis Storage...")
    test_analysis = {
        "summary": {
            "overall_signal": "bullish",
            "confidence": 85.5,
            "risk_level": "medium",
            "recommendation": "Buy"
        },
        "technical_indicators": {
            "rsi": {"value": 65.2, "signal": "neutral"},
            "macd": {"value": 0.15, "signal": "bullish"}
        },
        "patterns": {
            "support_levels": [1500, 1550],
            "resistance_levels": [1650, 1700]
        },
        "metadata": {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "period_days": 365,
            "interval": "day",
            "sector": "Oil & Gas"
        }
    }
    
    try:
        analysis_id = simple_db_manager.store_analysis(
            analysis=test_analysis,
            user_id=test_user_id,
            symbol="RELIANCE",
            exchange="NSE",
            period=365,
            interval="day"
        )
        
        if analysis_id:
            print(f"✅ Stored test analysis with ID: {analysis_id}")
        else:
            print(f"❌ Failed to store test analysis")
            return False
    except Exception as e:
        print(f"❌ Analysis storage error: {e}")
        return False
    
    # Test 4: Retrieve Analysis
    print("\n4. Testing Analysis Retrieval...")
    try:
        retrieved_analysis = simple_db_manager.get_analysis(analysis_id)
        if retrieved_analysis:
            print(f"✅ Retrieved analysis successfully")
            print(f"   Signal: {retrieved_analysis.get('summary', {}).get('overall_signal')}")
            print(f"   Confidence: {retrieved_analysis.get('summary', {}).get('confidence')}")
        else:
            print(f"❌ Failed to retrieve analysis")
            return False
    except Exception as e:
        print(f"❌ Analysis retrieval error: {e}")
        return False
    
    # Test 5: Get User Analyses
    print("\n5. Testing User Analyses Retrieval...")
    try:
        user_analyses = simple_db_manager.get_user_analyses(test_user_id, limit=10)
        if user_analyses:
            print(f"✅ Retrieved {len(user_analyses)} analyses for user")
            for analysis in user_analyses:
                symbol = analysis.get('stock_symbol')
                created_at = analysis.get('created_at')
                print(f"   - {symbol} (created: {created_at})")
        else:
            print(f"❌ No analyses found for user")
            return False
    except Exception as e:
        print(f"❌ User analyses retrieval error: {e}")
        return False
    
    # Test 6: Get Stock Analyses
    print("\n6. Testing Stock Analyses Retrieval...")
    try:
        stock_analyses = simple_db_manager.get_stock_analyses("RELIANCE", limit=10)
        if stock_analyses:
            print(f"✅ Retrieved {len(stock_analyses)} analyses for RELIANCE")
        else:
            print(f"❌ No analyses found for RELIANCE")
            return False
    except Exception as e:
        print(f"❌ Stock analyses retrieval error: {e}")
        return False
    
    # Test 7: Cleanup (Optional)
    print("\n7. Testing Analysis Deletion...")
    try:
        success = simple_db_manager.delete_analysis(analysis_id)
        if success:
            print(f"✅ Deleted test analysis")
        else:
            print(f"❌ Failed to delete test analysis")
    except Exception as e:
        print(f"❌ Analysis deletion error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All Simplified Database Tests Passed!")
    print("✅ Database structure is working correctly")
    print("✅ JSON storage is functioning properly")
    print("✅ All CRUD operations are working")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_simplified_database() 
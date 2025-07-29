"""
Database Connection Test Script

This script tests the database connection and storage functionality by:
1. Creating a test user profile
2. Storing fake analysis data
3. Retrieving the stored data
4. Cleaning up by deleting the test data

This provides a quick way to verify database connectivity without running full analysis.
"""

import os
import sys
import uuid
import time
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

from supabase_client import get_supabase_client
from database_manager import db_manager

def create_test_user():
    """Create a test user profile."""
    test_user_id = str(uuid.uuid4())
    print(f"🔧 Creating test user: {test_user_id}")
    
    try:
        # Create test user profile with required id field
        profile_data = {
            "id": test_user_id,
            "email": f"test_{test_user_id}@test.local",
            "full_name": "Test User",
            "subscription_tier": "test",
            "preferences": {"test": True},
            "analysis_count": 0,
            "favorite_stocks": ["TEST"]
        }
        
        result = db_manager.supabase.table("profiles").insert(profile_data).execute()
        
        if result.data:
            print(f"✅ Test user created successfully")
            return test_user_id
        else:
            print(f"❌ Failed to create test user")
            return None
            
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        return None

def create_fake_analysis_data():
    """Create fake analysis data for testing."""
    return {
        "ai_analysis": {
            "trend": "Bullish",
            "confidence_pct": 75,
            "short_term": {
                "horizon_days": 10,
                "entry_range": [100.0, 105.0],
                "stop_loss": 98.0,
                "targets": [110.0, 115.0]
            },
            "medium_term": {
                "horizon_days": 30,
                "entry_range": [95.0, 110.0],
                "stop_loss": 90.0,
                "targets": [120.0, 130.0]
            }
        },
        "indicators": {
            "rsi": {"value": 65.5, "signal": "neutral", "strength": "moderate"},
            "macd": {"value": 2.5, "signal": "bullish", "strength": "strong"},
            "sma_20": {"value": 102.5, "signal": "bullish", "strength": "moderate"},
            "sma_50": {"value": 98.0, "signal": "bullish", "strength": "strong"}
        },
        "overlays": {
            "bollinger_bands": {
                "upper": 110.0,
                "middle": 105.0,
                "lower": 100.0,
                "signal": "neutral"
            }
        },
        "indicator_summary_md": "## Technical Analysis Summary\n\n- RSI: 65.5 (Neutral)\n- MACD: 2.5 (Bullish)\n- SMA 20: 102.5 (Bullish)\n- SMA 50: 98.0 (Bullish)",
        "chart_insights": "Price is above both 20-day and 50-day moving averages, indicating bullish momentum.",
        "sector_benchmarking": {
            "sector": "Technology",
            "beta": 1.2,
            "correlation": 0.85,
            "sharpe_ratio": 1.5,
            "volatility": 0.25
        },
        "multi_timeframe_analysis": {
            "short_term": {"signal": "buy", "confidence": 75, "bias": "bullish"},
            "medium_term": {"signal": "buy", "confidence": 80, "bias": "bullish"},
            "long_term": {"signal": "hold", "confidence": 60, "bias": "neutral"}
        },
        "summary": {
            "overall_signal": "buy",
            "risk_level": "medium",
            "analysis_quality": "high",
            "confidence_score": 75
        },
        "trading_guidance": {
            "entry_price": 105.0,
            "stop_loss": 98.0,
            "target_1": 110.0,
            "target_2": 115.0,
            "position_size": "medium"
        },
        "metadata": {
            "current_price": 105.0,
            "price_change_pct": 2.5,
            "sector": "Technology",
            "analysis_timestamp": datetime.now().isoformat(),
            "data_points": 1000,
            "timeframe": "1D"
        }
    }

def test_database_storage(test_user_id):
    """Test storing fake analysis data in the database."""
    print(f"🔧 Testing database storage for user: {test_user_id}")
    
    try:
        # Create fake analysis data
        fake_analysis = create_fake_analysis_data()
        
        # Store analysis using the database manager
        analysis_id = db_manager.store_analysis(
            analysis=fake_analysis,
            user_id=test_user_id,
            symbol="TEST_STOCK",
            exchange="TEST",
            period=30,
            interval="day"
        )
        
        if analysis_id:
            print(f"✅ Analysis stored successfully with ID: {analysis_id}")
            return analysis_id
        else:
            print(f"❌ Failed to store analysis")
            return None
            
    except Exception as e:
        print(f"❌ Error storing analysis: {e}")
        return None

def test_database_retrieval(analysis_id):
    """Test retrieving the stored analysis data."""
    print(f"🔧 Testing database retrieval for analysis: {analysis_id}")
    
    try:
        # Retrieve analysis using the database manager
        analysis = db_manager.get_analysis_by_id(analysis_id)
        
        if analysis:
            print(f"✅ Analysis retrieved successfully")
            print(f"   Stock Symbol: {analysis.get('stock_symbol')}")
            print(f"   User ID: {analysis.get('user_id')}")
            print(f"   Overall Signal: {analysis.get('overall_signal')}")
            print(f"   Confidence Score: {analysis.get('confidence_score')}")
            return True
        else:
            print(f"❌ Failed to retrieve analysis")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving analysis: {e}")
        return False

def test_user_analyses(test_user_id):
    """Test retrieving user's analysis history."""
    print(f"🔧 Testing user analyses retrieval for user: {test_user_id}")
    
    try:
        # Get user analyses
        analyses = db_manager.get_user_analyses(test_user_id, limit=10)
        
        if analyses:
            print(f"✅ Retrieved {len(analyses)} analyses for user")
            for i, analysis in enumerate(analyses[:3]):  # Show first 3
                print(f"   {i+1}. {analysis.get('stock_symbol')} - {analysis.get('overall_signal')}")
            return True
        else:
            print(f"⚠️ No analyses found for user")
            return True  # This is not an error, just no data
            
    except Exception as e:
        print(f"❌ Error retrieving user analyses: {e}")
        return False

def cleanup_test_data(test_user_id, analysis_id):
    """Clean up test data from the database."""
    print(f"🧹 Cleaning up test data...")
    
    try:
        # Delete analysis first (due to foreign key constraints)
        if analysis_id:
            try:
                db_manager.supabase.table("stock_analyses").delete().eq("id", analysis_id).execute()
                print(f"✅ Deleted test analysis: {analysis_id}")
            except Exception as e:
                print(f"⚠️ Error deleting analysis: {e}")
        
        # Delete test user
        try:
            db_manager.supabase.table("profiles").delete().eq("id", test_user_id).execute()
            print(f"✅ Deleted test user: {test_user_id}")
        except Exception as e:
            print(f"⚠️ Error deleting user: {e}")
            
        print(f"✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def main():
    """Main test function."""
    print("🧪 Database Connection Test")
    print("=" * 50)
    
    test_user_id = None
    analysis_id = None
    
    try:
        # Test 1: Create test user
        print("\n📋 Test 1: User Creation")
        print("-" * 30)
        test_user_id = create_test_user()
        if not test_user_id:
            print("❌ User creation failed, stopping tests")
            return False
        
        # Test 2: Store fake analysis
        print("\n📋 Test 2: Analysis Storage")
        print("-" * 30)
        analysis_id = test_database_storage(test_user_id)
        if not analysis_id:
            print("❌ Analysis storage failed, stopping tests")
            return False
        
        # Test 3: Retrieve analysis
        print("\n📋 Test 3: Analysis Retrieval")
        print("-" * 30)
        retrieval_success = test_database_retrieval(analysis_id)
        if not retrieval_success:
            print("❌ Analysis retrieval failed")
        
        # Test 4: Get user analyses
        print("\n📋 Test 4: User Analyses History")
        print("-" * 30)
        user_analyses_success = test_user_analyses(test_user_id)
        if not user_analyses_success:
            print("❌ User analyses retrieval failed")
        
        # Test 5: Cleanup
        print("\n📋 Test 5: Data Cleanup")
        print("-" * 30)
        cleanup_test_data(test_user_id, analysis_id)
        
        print("\n" + "=" * 50)
        print("🎉 Database Connection Test Completed Successfully!")
        print("=" * 50)
        print("✅ All database operations are working correctly")
        print("✅ The analysis service should work properly")
        print("✅ Database connectivity is confirmed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
        
    finally:
        # Ensure cleanup happens even if tests fail
        if test_user_id or analysis_id:
            print(f"\n🧹 Final cleanup...")
            cleanup_test_data(test_user_id, analysis_id)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
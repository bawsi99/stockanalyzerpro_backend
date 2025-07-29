#!/usr/bin/env python3
"""
Print Analysis Request Details

This script demonstrates and prints the details of analysis requests
to show how the email-based user ID mapping works.
"""

import os
import sys
import json
import requests
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from simple_database_manager import simple_db_manager

def print_request_details():
    """Print detailed analysis request information."""
    print("🔍 ANALYSIS REQUEST DETAILS")
    print("=" * 60)
    
    # Get existing user for testing
    try:
        result = simple_db_manager.supabase.table("profiles").select("id, email").limit(1).execute()
        if not result.data:
            print("❌ No users found in profiles table")
            return
            
        user = result.data[0]
        user_id = user.get('id')
        user_email = user.get('email')
        
        print(f"👤 User Information:")
        print(f"   - User ID: {user_id}")
        print(f"   - Email: {user_email}")
        print()
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return
    
    # Example 1: Analysis request with email
    print("📤 EXAMPLE 1: Analysis Request with Email")
    print("-" * 40)
    
    request_with_email = {
        "stock": "RELIANCE",
        "exchange": "NSE",
        "period": 30,
        "interval": "day",
        "email": user_email
    }
    
    print("Request Payload:")
    print(json.dumps(request_with_email, indent=2))
    print()
    
    print("Expected Backend Processing:")
    print(f"1. Extract email: {user_email}")
    print(f"2. Look up user ID in profiles table")
    print(f"3. Map email '{user_email}' → User ID '{user_id}'")
    print(f"4. Store analysis with user_id: {user_id}")
    print()
    
    # Example 2: Analysis request without email
    print("📤 EXAMPLE 2: Analysis Request without Email")
    print("-" * 40)
    
    request_without_email = {
        "stock": "TCS",
        "exchange": "NSE",
        "period": 30,
        "interval": "day"
        # No email provided
    }
    
    print("Request Payload:")
    print(json.dumps(request_without_email, indent=2))
    print()
    
    print("Expected Backend Processing:")
    print("1. No email provided in request")
    print("2. User ID resolution fails")
    print("3. Analysis completes but is NOT stored")
    print("4. Warning message logged")
    print()
    
    # Example 3: Analysis request with invalid email
    print("📤 EXAMPLE 3: Analysis Request with Invalid Email")
    print("-" * 40)
    
    request_invalid_email = {
        "stock": "INFY",
        "exchange": "NSE",
        "period": 30,
        "interval": "day",
        "email": "nonexistent@example.com"
    }
    
    print("Request Payload:")
    print(json.dumps(request_invalid_email, indent=2))
    print()
    
    print("Expected Backend Processing:")
    print("1. Extract email: nonexistent@example.com")
    print("2. Look up user ID in profiles table")
    print("3. User not found for email")
    print("4. User ID resolution fails")
    print("5. Analysis completes but is NOT stored")
    print("6. Error message logged")
    print()
    
    # Show actual database query
    print("🔍 DATABASE QUERY EXAMPLE")
    print("-" * 40)
    
    print("When backend receives email, it runs this query:")
    print(f"SELECT id FROM profiles WHERE email = '{user_email}';")
    print()
    print("Expected result:")
    print(f"id: {user_id}")
    print()
    
    # Show storage details
    print("💾 STORAGE DETAILS")
    print("-" * 40)
    
    print("Analysis will be stored in stock_analyses_simple table:")
    storage_example = {
        "user_id": user_id,
        "stock_symbol": "RELIANCE",
        "exchange": "NSE",
        "period": 30,
        "interval": "day",
        "analysis_data": {
            "consensus": {"overall_signal": "bullish"},
            "indicators": {"rsi": 65.5},
            "ai_analysis": {"trend": "bullish"}
        },
        "created_at": datetime.now().isoformat()
    }
    
    print("Storage Record:")
    print(json.dumps(storage_example, indent=2))
    print()
    
    # Show frontend code example
    print("🎨 FRONTEND CODE EXAMPLE")
    print("-" * 40)
    
    frontend_code = '''
// In NewStockAnalysis.tsx
const { user } = useAuth(); // Gets user with email

const payload = {
  stock: formData.stock.toUpperCase(),
  exchange: formData.exchange,
  period: parseInt(formData.period),
  interval: formData.interval,
  sector: formData.sector === "none" ? null : formData.sector || null,
  email: user?.email // ✅ User email included automatically
};

// Send to backend
const data = await apiService.analyzeStock(payload);
'''
    
    print(frontend_code)
    print()
    
    # Show backend code example
    print("⚙️ BACKEND CODE EXAMPLE")
    print("-" * 40)
    
    backend_code = '''
# In analysis_service.py
@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    # ... analysis logic ...
    
    # Resolve user ID from email
    try:
        resolved_user_id = resolve_user_id(
            user_id=request.user_id,
            email=request.email  # ✅ Email from frontend
        )
        
        # Store with correct user ID
        analysis_id = simple_db_manager.store_analysis(
            analysis=results,
            user_id=resolved_user_id,  # ✅ Mapped user ID
            symbol=request.stock,
            ...
        )
        
    except ValueError as e:
        # Analysis works but not stored if email invalid
        print(f"⚠️ Analysis completed but not stored")
'''
    
    print(backend_code)
    print()
    
    print("🎯 SUMMARY")
    print("=" * 60)
    print("✅ Frontend sends user email in analysis requests")
    print("✅ Backend maps email to user ID from profiles table")
    print("✅ Analysis stored with correct user ID")
    print("✅ No anonymous user generation")
    print("✅ User analysis history properly linked")
    print()
    print("🚀 Ready to test with frontend!")

if __name__ == "__main__":
    print_request_details() 
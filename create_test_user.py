#!/usr/bin/env python3
"""
Script to create a test user profile for frontend testing.
"""

import uuid
from datetime import datetime
from supabase_client import get_supabase_client

def create_test_user():
    """Create a test user profile for frontend testing."""
    try:
        supabase = get_supabase_client()
        
        # Generate a test user ID
        user_id = str(uuid.uuid4())
        
        # Create profile data
        profile_data = {
            "id": user_id,
            "email": "test@example.com",
            "full_name": "Test User",
            "subscription_tier": "free",
            "preferences": {},
            "analysis_count": 0,
            "favorite_stocks": [],
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z"
        }
        
        # Insert profile
        result = supabase.table("profiles").insert(profile_data).execute()
        
        if result.data:
            created_id = result.data[0]["id"]
            print(f"✅ Created test user with ID: {created_id}")
            print(f"📧 Email: test@example.com")
            print(f"🔑 Use this ID in your frontend localStorage: {created_id}")
            return created_id
        else:
            print("❌ Failed to create test user")
            return None
            
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Creating test user for frontend testing...")
    user_id = create_test_user()
    if user_id:
        print(f"\n🎉 Test user created successfully!")
        print(f"💡 To use this user in the frontend:")
        print(f"   1. Open browser developer tools")
        print(f"   2. Go to Application > Local Storage")
        print(f"   3. Set 'user_id' to: {user_id}")
        print(f"   4. Set 'user_email' to: test@example.com")
        print(f"   5. Refresh the page")
    else:
        print("❌ Failed to create test user") 
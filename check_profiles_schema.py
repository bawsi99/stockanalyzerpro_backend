"""
Check Profiles Table Schema

This script checks the actual schema of the profiles table to understand
what fields are required and what the constraints are.
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

def check_profiles_schema():
    """Check the profiles table schema."""
    print("🔍 Checking Profiles Table Schema")
    print("=" * 50)
    
    try:
        supabase = get_supabase_client()
        
        # Try to get a sample profile to understand the structure
        result = supabase.table("profiles").select("*").limit(1).execute()
        
        if result.data:
            print("✅ Found existing profiles")
            sample_profile = result.data[0]
            print(f"Sample profile keys: {list(sample_profile.keys())}")
            print(f"Sample profile data:")
            for key, value in sample_profile.items():
                print(f"  {key}: {type(value).__name__} = {value}")
        else:
            print("⚠️ No existing profiles found")
            
        # Try to get table information
        try:
            # This might not work with Supabase, but worth trying
            schema_result = supabase.rpc('get_table_info', {'table_name': 'profiles'}).execute()
            print(f"Schema info: {schema_result}")
        except Exception as schema_error:
            print(f"Could not get schema info: {schema_error}")
            
    except Exception as e:
        print(f"❌ Error checking profiles schema: {e}")

def try_simple_profile_creation():
    """Try creating a simple profile with minimal fields."""
    print("\n🔧 Trying Simple Profile Creation")
    print("-" * 30)
    
    try:
        supabase = get_supabase_client()
        
        # Try with minimal fields
        simple_profile = {
            "email": "test@example.com",
            "full_name": "Test User"
        }
        
        print(f"Trying to create profile with: {simple_profile}")
        result = supabase.table("profiles").insert(simple_profile).execute()
        
        if result.data:
            print(f"✅ Simple profile created successfully")
            print(f"Created profile: {result.data[0]}")
            
            # Clean up
            profile_id = result.data[0]["id"]
            supabase.table("profiles").delete().eq("id", profile_id).execute()
            print(f"✅ Cleaned up test profile")
            
            return True
        else:
            print(f"❌ Failed to create simple profile")
            return False
            
    except Exception as e:
        print(f"❌ Error creating simple profile: {e}")
        return False

def main():
    """Main function."""
    check_profiles_schema()
    try_simple_profile_creation()

if __name__ == "__main__":
    main() 
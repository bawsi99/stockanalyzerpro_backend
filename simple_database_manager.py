"""
Simple Database Manager - Handles all database operations with a simplified schema
Just one table: stock_analyses_simple with a JSON column for everything
"""

import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import math
import numpy as np
import pandas as pd
from supabase_client import get_supabase_client

class SimpleDatabaseManager:
    """Manages database operations with a simplified schema."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
    
    def _sanitize(self, obj: Any) -> Any:
        """Recursively convert values to JSON-safe types: replace NaN/Inf with None and
        convert numpy/pandas types to native Python types."""
        try:
            # Primitives
            if obj is None or isinstance(obj, (str, int, bool)):
                return obj
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            # Numpy scalars
            if isinstance(obj, np.floating):
                val = float(obj)
                return val if math.isfinite(val) else None
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            # Collections
            if isinstance(obj, dict):
                return {k: self._sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [self._sanitize(v) for v in obj]
            # Pandas
            if isinstance(obj, pd.Series):
                return [self._sanitize(v) for v in obj.tolist()]
            if isinstance(obj, pd.DataFrame):
                return [self._sanitize(r) for r in obj.to_dict(orient='records')]
            # Fallback: leave as-is
            return obj
        except Exception:
            return None
    
    def create_anonymous_user(self, user_id: str) -> bool:
        """
        Create an anonymous user profile in the profiles table.
        
        Args:
            user_id: UUID string for the user
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate UUID format
            uuid.UUID(user_id)
            
            # Check if profiles table exists
            try:
                test_result = self.supabase.table("profiles").select("id").limit(1).execute()
                print(f"✅ Profiles table is accessible")
            except Exception as table_error:
                print(f"❌ Profiles table error: {table_error}")
                return False
            
            # Create anonymous user profile
            profile_data = {
                "id": user_id,
                "email": f"anonymous_{user_id}@system.local",
                "full_name": "Anonymous User",
                "subscription_tier": "free",
                "preferences": {},
                "analysis_count": 0,
                "favorite_stocks": [],
                "created_at": datetime.now().isoformat() + "Z",
                "updated_at": datetime.now().isoformat() + "Z"
            }
            
            # Try to insert profile
            try:
                result = self.supabase.table("profiles").insert(profile_data).execute()
                
                if result.data:
                    print(f"✅ Created anonymous user profile: {user_id}")
                    return True
                else:
                    print(f"❌ Failed to create anonymous user profile: {user_id}")
                    return False
                    
            except Exception as insert_error:
                # If foreign key constraint fails, try without the id field
                if "foreign key constraint" in str(insert_error).lower():
                    print(f"⚠️ Foreign key constraint detected, trying alternative approach")
                    
                    # Try creating profile without specifying id (let database generate it)
                    alt_profile_data = {
                        "email": f"anonymous_{user_id}@system.local",
                        "full_name": "Anonymous User",
                        "subscription_tier": "free",
                        "preferences": {},
                        "analysis_count": 0,
                        "favorite_stocks": [],
                        "created_at": datetime.now().isoformat() + "Z",
                        "updated_at": datetime.now().isoformat() + "Z"
                    }
                    
                    alt_result = self.supabase.table("profiles").insert(alt_profile_data).execute()
                    
                    if alt_result.data:
                        print(f"✅ Created anonymous user profile with generated ID")
                        return True
                    else:
                        print(f"❌ Failed to create profile with alternative approach")
                        return False
                else:
                    print(f"❌ Error creating profile: {insert_error}")
                    return False
                
        except Exception as e:
            print(f"❌ Error creating anonymous user: {e}")
            return False
    
    def ensure_user_exists(self, user_id: str) -> bool:
        """
        Ensure a user exists in the profiles table, create if not.
        
        Args:
            user_id: UUID string for the user
            
        Returns:
            bool: True if user exists or was created successfully
        """
        try:
            # Check if user exists
            result = self.supabase.table("profiles").select("id").eq("id", user_id).execute()
            
            if result.data:
                return True  # User exists
            else:
                # Create anonymous user
                return self.create_anonymous_user(user_id)
                
        except Exception as e:
            print(f"❌ Error checking/creating user: {e}")
            return False
    
    def store_analysis(self, analysis: dict, user_id: str, symbol: str, 
                      exchange: str = "NSE", period: int = 365, interval: str = "day") -> Optional[str]:
        """
        Store analysis results in the simplified database.
        
        Args:
            analysis: Analysis results dictionary (everything goes in JSON column)
            user_id: User ID (should be a valid UUID string)
            symbol: Stock symbol
            exchange: Stock exchange
            period: Analysis period in days
            interval: Data interval
            
        Returns:
            str: Analysis ID if successful, None otherwise
        """
        try:
            # Validate inputs
            if not analysis or not isinstance(analysis, dict):
                raise ValueError("Invalid analysis data. Expected non-empty dictionary.")
            
            if not user_id:
                raise ValueError("User ID is required.")
            
            # Validate that user_id is a proper UUID
            try:
                uuid.UUID(user_id)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid user_id format: {user_id}. Expected valid UUID.")
            
            if not symbol:
                raise ValueError("Stock symbol is required.")
            
            # Ensure user exists
            if not self.ensure_user_exists(user_id):
                raise ValueError(f"Failed to ensure user exists: {user_id}")
            
            # Sanitize analysis payload to be JSON-compliant
            sanitized_analysis = self._sanitize(analysis)

            # Prepare the complete analysis data
            complete_analysis_data = {
                # Original analysis data
                **(sanitized_analysis or {}),
                
                # Add metadata
                "metadata": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "period_days": period,
                    "interval": interval,
                    "user_id": user_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "simplified_storage",
                    "version": "2.0"
                }
            }
            
            # Store in the simplified table
            analysis_record = self._sanitize({
                "user_id": user_id,
                "stock_symbol": symbol,
                "analysis_data": complete_analysis_data,
                "created_at": datetime.now().isoformat() + "Z",
                "updated_at": datetime.now().isoformat() + "Z"
            })
            
            result = self.supabase.table("stock_analyses_simple").insert(analysis_record).execute()
            
            if result.data and len(result.data) > 0:
                analysis_id = result.data[0]["id"]
                print(f"✅ Successfully stored analysis for {symbol} with ID: {analysis_id}")
                
                # Update user's analysis count
                self._update_user_analysis_count(user_id)
                
                return analysis_id
            else:
                print(f"❌ Failed to store analysis for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error storing analysis: {e}")
            return None
    
    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        """
        Retrieve analysis by ID.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            dict: Analysis data or None if not found
        """
        try:
            result = self.supabase.table("stock_analyses_simple").select("*").eq("id", analysis_id).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]["analysis_data"]
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error retrieving analysis: {e}")
            return None
    
    def get_user_analyses(self, user_id: str, limit: int = 50) -> List[dict]:
        """
        Get all analyses for a user.
        
        Args:
            user_id: User ID (UUID)
            limit: Maximum number of analyses to return
            
        Returns:
            List[dict]: List of analysis records
        """
        try:
            # Validate that user_id is not empty
            if not user_id or not user_id.strip():
                print(f"❌ Empty user_id provided")
                return []
            
            user_id = user_id.strip()
            
            # Validate that user_id is a proper UUID
            try:
                uuid.UUID(user_id)
            except (ValueError, TypeError):
                print(f"❌ Invalid UUID format: {user_id}")
                return []
            
            result = self.supabase.table("stock_analyses_simple").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            
            if result.data:
                return result.data
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving user analyses: {e}")
            return []
    
    def get_stock_analyses(self, symbol: str, limit: int = 50) -> List[dict]:
        """
        Get all analyses for a specific stock.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of analyses to return
            
        Returns:
            List[dict]: List of analysis records
        """
        try:
            result = self.supabase.table("stock_analyses_simple").select("*").eq("stock_symbol", symbol).order("created_at", desc=True).limit(limit).execute()
            
            if result.data:
                return result.data
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving stock analyses: {e}")
            return []
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis by ID.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.supabase.table("stock_analyses_simple").delete().eq("id", analysis_id).execute()
            
            if result.data:
                print(f"✅ Successfully deleted analysis: {analysis_id}")
                return True
            else:
                print(f"❌ Failed to delete analysis: {analysis_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting analysis: {e}")
            return False
    
    def _update_user_analysis_count(self, user_id: str):
        """Update the user's analysis count in profiles table."""
        try:
            # Get current count
            result = self.supabase.table("profiles").select("analysis_count").eq("id", user_id).execute()
            
            if result.data and len(result.data) > 0:
                current_count = result.data[0].get("analysis_count", 0) or 0
                new_count = current_count + 1
                
                # Update count
                self.supabase.table("profiles").update({"analysis_count": new_count}).eq("id", user_id).execute()
                
        except Exception as e:
            print(f"⚠️ Warning: Could not update user analysis count: {e}")
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[dict]:
        """
        Get a specific analysis by ID.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Optional[dict]: Analysis record or None if not found
        """
        try:
            result = self.supabase.table("stock_analyses_simple").select("*").eq("id", analysis_id).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error retrieving analysis by ID: {e}")
            return None
    
    def get_analyses_by_signal(self, signal: str, user_id: Optional[str] = None, limit: int = 20) -> List[dict]:
        """
        Get analyses filtered by signal type.
        
        Args:
            signal: Signal type to filter by
            user_id: Optional user ID (UUID) to filter by
            limit: Maximum number of records to return
            
        Returns:
            List[dict]: List of analysis records
        """
        try:
            query = self.supabase.table("stock_analyses_simple").select("*")
            
            # Add signal filter
            query = query.eq("signal", signal)
            
            # Add user filter if provided
            if user_id:
                # Validate UUID format
                try:
                    uuid.UUID(user_id)
                    query = query.eq("user_id", user_id)
                except (ValueError, TypeError):
                    print(f"❌ Invalid UUID format for user_id: {user_id}")
                    return []
            
            result = query.order("created_at", desc=True).limit(limit).execute()
            
            if result.data:
                return result.data
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving analyses by signal: {e}")
            return []
    
    def get_analyses_by_sector(self, sector: str, user_id: Optional[str] = None, limit: int = 20) -> List[dict]:
        """
        Get analyses filtered by sector.
        
        Args:
            sector: Sector to filter by
            user_id: Optional user ID (UUID) to filter by
            limit: Maximum number of records to return
            
        Returns:
            List[dict]: List of analysis records
        """
        try:
            query = self.supabase.table("stock_analyses_simple").select("*")
            
            # Add sector filter
            query = query.eq("sector", sector)
            
            # Add user filter if provided
            if user_id:
                # Validate UUID format
                try:
                    uuid.UUID(user_id)
                    query = query.eq("user_id", user_id)
                except (ValueError, TypeError):
                    print(f"❌ Invalid UUID format for user_id: {user_id}")
                    return []
            
            result = query.order("created_at", desc=True).limit(limit).execute()
            
            if result.data:
                return result.data
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving analyses by sector: {e}")
            return []
    
    def get_high_confidence_analyses(self, min_confidence: float = 80.0, user_id: Optional[str] = None, limit: int = 20) -> List[dict]:
        """
        Get analyses with confidence above threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            user_id: Optional user ID (UUID) to filter by
            limit: Maximum number of records to return
            
        Returns:
            List[dict]: List of analysis records
        """
        try:
            query = self.supabase.table("stock_analyses_simple").select("*")
            
            # Add confidence filter (gte = greater than or equal)
            query = query.gte("confidence", min_confidence)
            
            # Add user filter if provided
            if user_id:
                # Validate UUID format
                try:
                    uuid.UUID(user_id)
                    query = query.eq("user_id", user_id)
                except (ValueError, TypeError):
                    print(f"❌ Invalid UUID format for user_id: {user_id}")
                    return []
            
            result = query.order("created_at", desc=True).limit(limit).execute()
            
            if result.data:
                return result.data
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving high confidence analyses: {e}")
            return []
    
    def get_user_id_by_email(self, email: str) -> Optional[str]:
        """
        Get user ID (UUID) from email address.
        
        Args:
            email: User's email address
            
        Returns:
            Optional[str]: User ID (UUID) or None if not found
        """
        try:
            result = self.supabase.table("profiles").select("id").eq("email", email).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0].get("id")
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error retrieving user ID by email: {e}")
            return None
    
    def get_database_status(self) -> dict:
        """
        Get database connection and table status.
        
        Returns:
            dict: Database status information
        """
        status = {
            "connected": False,
            "tables": {},
            "errors": []
        }
        
        try:
            # Test connection
            test_result = self.supabase.table("profiles").select("id").limit(1).execute()
            status["connected"] = True
            
            # Check tables
            tables_to_check = ["profiles", "stock_analyses_simple"]
            
            for table in tables_to_check:
                try:
                    result = self.supabase.table(table).select("id").limit(1).execute()
                    status["tables"][table] = {
                        "exists": True,
                        "accessible": True,
                        "record_count": len(result.data) if result.data else 0
                    }
                except Exception as table_error:
                    status["tables"][table] = {
                        "exists": False,
                        "accessible": False,
                        "error": str(table_error)
                    }
                    status["errors"].append(f"Table {table}: {str(table_error)}")
                    
        except Exception as e:
            status["connected"] = False
            status["errors"].append(f"Connection error: {str(e)}")
        
        return status

# Global simple database manager instance
simple_db_manager = SimpleDatabaseManager() 
"""
Database Manager - Handles all database operations and ensures proper data storage
"""

import uuid
import time
from typing import Dict, List, Optional, Any
from supabase_client import get_supabase_client

class DatabaseManager:
    """Manages all database operations for the stock analysis system."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
    
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
            
            # Create anonymous user profile
            profile_data = {
                "id": user_id,
                "email": None,
                "full_name": "Anonymous User",
                "subscription_tier": "free",
                "preferences": {},
                "analysis_count": 0,
                "favorite_stocks": []
            }
            
            result = self.supabase.table("profiles").insert(profile_data).execute()
            
            if result.data:
                print(f"✅ Created anonymous user profile: {user_id}")
                return True
            else:
                print(f"❌ Failed to create anonymous user profile: {user_id}")
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
        Store analysis results in the database with proper validation and user management.
        
        Args:
            analysis: Analysis results dictionary
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
            
            if not symbol or not isinstance(symbol, str):
                raise ValueError(f"Invalid symbol: {symbol}. Expected non-empty string.")
            
            # Validate UUID format
            try:
                uuid.UUID(user_id)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid user_id format: {user_id}. Expected valid UUID.")
            
            # Ensure user exists
            if not self.ensure_user_exists(user_id):
                raise ValueError(f"Failed to ensure user exists: {user_id}")
            
            # Prepare analysis data for stock_analyses_simple table
            # This table only accepts: id, user_id, stock_symbol, analysis_data, created_at, updated_at
            analysis_data = {
                "user_id": user_id,
                "stock_symbol": symbol,
                "analysis_data": analysis  # Store complete analysis in JSONB
            }
            
            # Insert analysis
            result = self.supabase.table("stock_analyses_simple").insert(analysis_data).execute()
            
            if result.data:
                analysis_id = result.data[0]["id"]
                print(f"✅ Stored analysis successfully: {analysis_id}")
                
                # Note: Related data storage is disabled for simple table structure
                # All analysis data is stored in the main analysis_data JSON field
                
                return analysis_id
            else:
                print("❌ Failed to store analysis")
                return None
                
        except Exception as e:
            print(f"❌ Error storing analysis: {e}")
            return None
    
    def _store_related_data(self, analysis_id: str, analysis: dict):
        """
        Store related analysis data in separate tables.
        
        Args:
            analysis_id: ID of the main analysis record
            analysis: Analysis data dictionary
        """
        try:
            # Store technical indicators
            if "indicators" in analysis:
                self._store_technical_indicators(analysis_id, analysis["indicators"])
            
            # Store pattern recognition
            if "patterns" in analysis:
                self._store_pattern_recognition(analysis_id, analysis["patterns"])
            
            # Store trading levels
            if "trading_levels" in analysis:
                self._store_trading_levels(analysis_id, analysis["trading_levels"])
            
            # Store volume analysis
            if "volume_analysis" in analysis:
                self._store_volume_analysis(analysis_id, analysis["volume_analysis"])
            
            # Store risk management
            if "risk_management" in analysis:
                self._store_risk_management(analysis_id, analysis["risk_management"])
            
            # Store sector benchmarking
            if "sector_benchmarking" in analysis:
                self._store_sector_benchmarking(analysis_id, analysis["sector_benchmarking"])
            
            # Store multi-timeframe analysis
            if "multi_timeframe_analysis" in analysis:
                self._store_multi_timeframe_analysis(analysis_id, analysis["multi_timeframe_analysis"])
                
        except Exception as e:
            print(f"⚠️ Warning: Error storing related data: {e}")
    
    def _store_technical_indicators(self, analysis_id: str, indicators: dict):
        """Store technical indicators data."""
        try:
            for indicator_name, indicator_data in indicators.items():
                if isinstance(indicator_data, dict) and "value" in indicator_data:
                    indicator_record = {
                        "analysis_id": analysis_id,
                        "indicator_type": "technical",
                        "indicator_name": indicator_name,
                        "value": indicator_data.get("value"),
                        "signal": indicator_data.get("signal"),
                        "strength": indicator_data.get("strength"),
                        "metadata": indicator_data
                    }
                    self.supabase.table("technical_indicators").insert(indicator_record).execute()
        except Exception as e:
            print(f"Error storing technical indicators: {e}")
    
    def _store_pattern_recognition(self, analysis_id: str, patterns: dict):
        """Store pattern recognition data."""
        try:
            for pattern_type, pattern_list in patterns.items():
                if isinstance(pattern_list, list):
                    for pattern in pattern_list:
                        if isinstance(pattern, dict):
                            pattern_record = {
                                "analysis_id": analysis_id,
                                "pattern_type": pattern_type,
                                "pattern_name": pattern.get("name", pattern_type),
                                "confidence": pattern.get("confidence"),
                                "direction": pattern.get("direction"),
                                "start_date": pattern.get("start_date"),
                                "end_date": pattern.get("end_date"),
                                "start_price": pattern.get("start_price"),
                                "end_price": pattern.get("end_price"),
                                "target_price": pattern.get("target_price"),
                                "stop_loss": pattern.get("stop_loss"),
                                "metadata": pattern
                            }
                            self.supabase.table("pattern_recognition").insert(pattern_record).execute()
        except Exception as e:
            print(f"Error storing pattern recognition: {e}")
    
    def _store_trading_levels(self, analysis_id: str, levels: dict):
        """Store trading levels data."""
        try:
            for level_type, level_list in levels.items():
                if isinstance(level_list, list):
                    for level in level_list:
                        if isinstance(level, dict):
                            level_record = {
                                "analysis_id": analysis_id,
                                "level_type": level_type,
                                "price_level": level.get("price"),
                                "strength": level.get("strength"),
                                "volume_confirmation": level.get("volume_confirmation"),
                                "description": level.get("description"),
                                "metadata": level
                            }
                            self.supabase.table("trading_levels").insert(level_record).execute()
        except Exception as e:
            print(f"Error storing trading levels: {e}")
    
    def _store_volume_analysis(self, analysis_id: str, volume_data: dict):
        """Store volume analysis data."""
        try:
            for volume_type, volume_info in volume_data.items():
                if isinstance(volume_info, dict):
                    volume_record = {
                        "analysis_id": analysis_id,
                        "volume_type": volume_type,
                        "date": volume_info.get("date"),
                        "volume": volume_info.get("volume"),
                        "price": volume_info.get("price"),
                        "significance": volume_info.get("significance"),
                        "description": volume_info.get("description"),
                        "metadata": volume_info
                    }
                    self.supabase.table("volume_analysis").insert(volume_record).execute()
        except Exception as e:
            print(f"Error storing volume analysis: {e}")
    
    def _store_risk_management(self, analysis_id: str, risk_data: dict):
        """Store risk management data."""
        try:
            for risk_type, risk_info in risk_data.items():
                if isinstance(risk_info, dict):
                    risk_record = {
                        "analysis_id": analysis_id,
                        "risk_type": risk_type,
                        "risk_level": risk_info.get("level", "medium"),
                        "risk_score": risk_info.get("score"),
                        "description": risk_info.get("description"),
                        "mitigation_strategy": risk_info.get("mitigation"),
                        "stop_loss_level": risk_info.get("stop_loss"),
                        "take_profit_level": risk_info.get("take_profit"),
                        "position_size_recommendation": risk_info.get("position_size"),
                        "metadata": risk_info
                    }
                    self.supabase.table("risk_management").insert(risk_record).execute()
        except Exception as e:
            print(f"Error storing risk management: {e}")
    
    def _store_sector_benchmarking(self, analysis_id: str, sector_data: dict):
        """Store sector benchmarking data."""
        try:
            if isinstance(sector_data, dict):
                sector_record = {
                    "analysis_id": analysis_id,
                    "sector": sector_data.get("sector"),
                    "sector_index": sector_data.get("sector_index"),
                    "beta": sector_data.get("beta"),
                    "correlation": sector_data.get("correlation"),
                    "sharpe_ratio": sector_data.get("sharpe_ratio"),
                    "volatility": sector_data.get("volatility"),
                    "max_drawdown": sector_data.get("max_drawdown"),
                    "cumulative_return": sector_data.get("cumulative_return"),
                    "annualized_return": sector_data.get("annualized_return"),
                    "sector_beta": sector_data.get("sector_beta"),
                    "sector_correlation": sector_data.get("sector_correlation"),
                    "sector_sharpe_ratio": sector_data.get("sector_sharpe_ratio"),
                    "sector_volatility": sector_data.get("sector_volatility"),
                    "sector_max_drawdown": sector_data.get("sector_max_drawdown"),
                    "sector_cumulative_return": sector_data.get("sector_cumulative_return"),
                    "sector_annualized_return": sector_data.get("sector_annualized_return"),
                    "excess_return": sector_data.get("excess_return"),
                    "sector_excess_return": sector_data.get("sector_excess_return"),
                    "metadata": sector_data
                }
                self.supabase.table("sector_benchmarking").insert(sector_record).execute()
        except Exception as e:
            print(f"Error storing sector benchmarking: {e}")
    
    def _store_multi_timeframe_analysis(self, analysis_id: str, mtf_data: dict):
        """Store multi-timeframe analysis data."""
        try:
            for timeframe, tf_data in mtf_data.items():
                if isinstance(tf_data, dict):
                    mtf_record = {
                        "analysis_id": analysis_id,
                        "timeframe": timeframe,
                        "signal": tf_data.get("signal"),
                        "confidence": tf_data.get("confidence"),
                        "bias": tf_data.get("bias"),
                        "entry_range_min": tf_data.get("entry_range", {}).get("min"),
                        "entry_range_max": tf_data.get("entry_range", {}).get("max"),
                        "target_1": tf_data.get("targets", [None])[0] if tf_data.get("targets") else None,
                        "target_2": tf_data.get("targets", [None, None])[1] if len(tf_data.get("targets", [])) > 1 else None,
                        "stop_loss": tf_data.get("stop_loss"),
                        "metadata": tf_data
                    }
                    self.supabase.table("multi_timeframe_analysis").insert(mtf_record).execute()
        except Exception as e:
            print(f"Error storing multi-timeframe analysis: {e}")
    
    def get_user_analyses(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get analysis history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis records
        """
        try:
            result = self.supabase.table("stock_analyses_simple")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting user analyses: {e}")
            return []
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """
        Get a specific analysis by ID.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Analysis record or None
        """
        try:
            result = self.supabase.table("stock_analyses_simple")\
                .select("*")\
                .eq("id", analysis_id)\
                .execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Error getting analysis by ID: {e}")
            return None
    
    def update_user_analysis_count(self, user_id: str):
        """Update the analysis count for a user."""
        try:
            # Get current count
            result = self.supabase.table("profiles")\
                .select("analysis_count")\
                .eq("id", user_id)\
                .execute()
            
            if result.data:
                current_count = result.data[0].get("analysis_count", 0)
                new_count = current_count + 1
                
                # Update count
                self.supabase.table("profiles")\
                    .update({"analysis_count": new_count, "last_analysis_date": time.time()})\
                    .eq("id", user_id)\
                    .execute()
                    
        except Exception as e:
            print(f"Error updating user analysis count: {e}")

    def get_analyses_by_signal(self, signal: str, user_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        Get analyses filtered by signal type.
        
        Args:
            signal: Signal type to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis records
        """
        try:
            query = self.supabase.table("stock_analyses_simple")\
                .select("*")\
                .eq("overall_signal", signal)\
                .order("created_at", desc=True)\
                .limit(limit)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting analyses by signal: {e}")
            return []

    def get_analyses_by_sector(self, sector: str, user_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        Get analyses filtered by sector.
        
        Args:
            sector: Sector to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis records
        """
        try:
            query = self.supabase.table("stock_analyses_simple")\
                .select("*")\
                .eq("sector", sector)\
                .order("created_at", desc=True)\
                .limit(limit)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting analyses by sector: {e}")
            return []

    def get_user_id_by_email(self, email: str) -> Optional[str]:
        """
        Get user ID by email address.
        
        Args:
            email: User email address
            
        Returns:
            User ID or None if not found
        """
        try:
            result = self.supabase.table("profiles")\
                .select("id")\
                .eq("email", email)\
                .execute()
            
            return result.data[0]["id"] if result.data else None
            
        except Exception as e:
            print(f"Error getting user ID by email: {e}")
            return None

# Global database manager instance
db_manager = DatabaseManager() 
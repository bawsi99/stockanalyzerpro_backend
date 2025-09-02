import os
import uuid
import time
import json
import asyncio
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

import dotenv
dotenv.load_dotenv()

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from supabase import create_client, Client # Import Supabase client

app = FastAPI(title="Database Service", version="1.0.0")

# Load CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
print(CORS_ORIGINS)
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
# print(f"🔧 Database Service CORS_ORIGINS (raw from env): {os.getenv("CORS_ORIGINS", "")}")
print(f"🔧 Database Service CORS_ORIGINS (processed for middleware): {CORS_ORIGINS}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DatabaseManager:
    """Manages all database operations for the stock analysis system."""
    
    def __init__(self):
        # Database URL from environment variable
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

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
                return analysis_id
            else:
                print("❌ Failed to store analysis")
                return None
                
        except Exception as e:
            print(f"❌ Error storing analysis: {e}")
            return None
    
    def get_user_analyses(self, user_id: str, offset: int = 0, limit: int = 10) -> List[Dict]:
        """
        Get analysis history for a user with pagination.

        Args:
            user_id: User ID
            offset: Number of records to skip
            limit: Maximum number of analyses to return per request

        Returns:
            List of analysis records
        """
        try:
            result = self.supabase.table("stock_analyses_simple")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .range(offset, offset + limit - 1)\
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
                .order("created_at", desc=True)\
                .limit(limit)
            
            # Filter by signal if 'analysis_data' contains 'trading_guidance' with 'signal'
            if signal:
                query = query.filter("analysis_data->trading_guidance->signal", "eq", signal)
            
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
                .order("created_at", desc=True)\
                .limit(limit)
            
            # Filter by sector if 'analysis_data' contains 'sector_context' with 'sector'
            if sector:
                query = query.filter("analysis_data->sector_context->sector", "eq", sector)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting analyses by sector: {e}")
            return []

    def get_high_confidence_analyses(self, min_confidence: float = 80.0, user_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        Get analyses with confidence above a threshold.
        
        Args:
            min_confidence: Minimum confidence level (0-100)
            user_id: Optional user ID to filter by
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis records
        """
        try:
            query = self.supabase.table("stock_analyses_simple")\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(limit)
            
            # Filter by confidence in analysis_data->ai_analysis->confidence
            # Supabase doesn't directly support JSONB numeric comparisons like '>='
            # We'll fetch and filter in-memory for now, or improve later with RLS
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            
            filtered_analyses = []
            if result.data:
                for analysis in result.data:
                    ai_analysis = analysis.get("analysis_data", {}).get("ai_analysis", {})
                    confidence = ai_analysis.get("overall_confidence", 0) # Assuming confidence is under ai_analysis and named overall_confidence
                    if confidence >= min_confidence:
                        filtered_analyses.append(analysis)
            
            return filtered_analyses
            
        except Exception as e:
            print(f"Error getting high confidence analyses: {e}")
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

# Helper function for JSON serialization
def make_json_serializable(obj):
    """Recursively convert objects to JSON serializable format."""
    if isinstance(obj, (str, int, type(None))):
        return obj
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, float):
        if obj == float('inf') or obj == float('-inf') or obj != obj: # Check for NaN and Inf
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        try:
            # For types like numpy integers, floats, booleans, and pandas Timestamps
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return str(obj) # Fallback to string for unknown types
        except:
            return str(obj)

def resolve_user_id(user_id: Optional[str] = None, email: Optional[str] = None) -> str:
    """
    Resolve user ID from provided user_id or email.
    Email mapping is the primary method for user identification.
    
    Args:
        user_id: Optional user ID (UUID)
        email: Optional user email for ID mapping
        
    Returns:
        str: Valid user ID (UUID)
        
    Raises:
        ValueError: If no valid user ID can be resolved
    """
    try:
        # If user_id is provided and valid, use it
        if user_id:
            try:
                uuid.UUID(user_id)
                # Ensure user exists
                db_manager.ensure_user_exists(user_id)
                print(f"✅ Using provided user ID: {user_id}")
                return user_id
            except (ValueError, TypeError):
                print(f"⚠️ Invalid user_id format: {user_id}")
        
        # If email is provided, try to get user ID from email
        if email:
            resolved_user_id = db_manager.get_user_id_by_email(email)
            if resolved_user_id:
                print(f"✅ Resolved user ID from email: {email} -> {resolved_user_id}")
                return resolved_user_id
            else:
                print(f"❌ User not found for email: {email}")
                raise ValueError(f"User not found for email: {email}")
        
        # No user_id or email provided
        raise ValueError("No user_id or email provided for analysis request")
        
    except Exception as e:
        print(f"❌ Error resolving user ID: {e}")
        raise ValueError(f"Failed to resolve user ID: {e}")

# Pydantic Models for requests
class AnalysisStoreRequest(BaseModel):
    analysis: Dict[str, Any]
    user_id: str
    symbol: str
    exchange: str = "NSE"
    period: int = 365
    interval: str = "day"

class UserAnalysisRequest(BaseModel):
    user_id: str
    offset: int = 0
    limit: int = 10

class AnalysisByIdRequest(BaseModel):
    analysis_id: str

class FilteredAnalysesRequest(BaseModel):
    filter_value: str = Field(..., description="Value to filter by (signal, sector, or min_confidence)")
    user_id: Optional[str] = Field(None, description="Optional user ID to filter by")
    limit: int = 20

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for the Database Service."""
    return {
        "status": "healthy",
        "service": "Database Service",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyses/store")
async def store_analysis_endpoint(request: AnalysisStoreRequest):
    """Store analysis results in the database."""
    try:
        analysis_id = db_manager.store_analysis(
            analysis=make_json_serializable(request.analysis),
            user_id=request.user_id,
            symbol=request.symbol,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval
        )
        if analysis_id:
            return JSONResponse(content={"success": True, "analysis_id": analysis_id})
        raise HTTPException(status_code=500, detail="Failed to store analysis.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing analysis: {str(e)}")

@app.get("/analyses/user/{user_id}")
async def get_user_analyses_endpoint(user_id: str, offset: int = 0, limit: int = 10):
    """Get analysis history for a user with pagination."""
    try:
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty.")
        
        actual_user_id = resolve_user_id(user_id=user_id) # Using the resolve_user_id helper
        
        analyses = db_manager.get_user_analyses(actual_user_id, offset, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analyses: {str(e)}")

@app.get("/analyses/{analysis_id}")
async def get_analysis_by_id_endpoint(analysis_id: str):
    """Get a specific analysis by ID."""
    try:
        if not analysis_id or not analysis_id.strip():
            raise HTTPException(status_code=400, detail="analysis_id cannot be empty.")
        
        analysis = db_manager.get_analysis_by_id(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
        
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

@app.get("/analyses/signal/{signal}")
async def get_analyses_by_signal_endpoint(signal: str, user_id: Optional[str] = None, limit: int = 20):
    """Get analyses filtered by signal type."""
    try:
        actual_user_id = None
        if user_id:
            actual_user_id = resolve_user_id(user_id=user_id)
            
        analyses = db_manager.get_analyses_by_signal(signal, actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses by signal: {str(e)}")

@app.get("/analyses/sector/{sector}")
async def get_analyses_by_sector_endpoint(sector: str, user_id: Optional[str] = None, limit: int = 20):
    """Get analyses filtered by sector."""
    try:
        actual_user_id = None
        if user_id:
            actual_user_id = resolve_user_id(user_id=user_id)
            
        analyses = db_manager.get_analyses_by_sector(sector, actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses by sector: {str(e)}")

@app.get("/analyses/confidence/{min_confidence}")
async def get_high_confidence_analyses_endpoint(min_confidence: float = 80.0, user_id: Optional[str] = None, limit: int = 20):
    """Get analyses with confidence above threshold."""
    try:
        actual_user_id = None
        if user_id:
            actual_user_id = resolve_user_id(user_id=user_id)
            
        analyses = db_manager.get_high_confidence_analyses(min_confidence, actual_user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch high confidence analyses: {str(e)}")

@app.get("/analyses/summary/user/{user_id}")
async def get_user_analysis_summary_endpoint(user_id: str):
    """Get analysis summary for a user."""
    try:
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty.")
        
        actual_user_id = resolve_user_id(user_id=user_id)
        
        analyses = db_manager.get_user_analyses(actual_user_id, 50) # Fetch up to 50 for summary
        
        summary = {
            "total_analyses": len(analyses),
            "unique_stocks": len(set(analysis.get("stock_symbol", "") for analysis in analyses)),
            "recent_analyses": analyses[:5] if analyses else [],
            "sectors_analyzed": list(set(analysis.get("analysis_data", {}).get("sector_context", {}).get("sector", "") for analysis in analyses if analysis.get("analysis_data", {}).get("sector_context", {}).get("sector")))
        }
        
        return {
            "success": True,
            "summary": summary
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analysis summary: {str(e)}")

# Root route
@app.get("/")
async def root():
    """Root endpoint for the Database Service."""
    return {
        "service": "Database Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "store_analysis": "/analyses/store",
            "user_analyses": "/analyses/user/{user_id}"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("DATABASE_PORT", 8003))
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    
    print(f"🚀 Starting Database Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

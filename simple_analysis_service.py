"""
Simple Analysis Service - Minimal version for user analysis endpoints
"""

import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from simple_database_manager import simple_db_manager

app = FastAPI(title="Simple Analysis Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "simple_analysis_service"}

@app.get("/analyses/user/{user_id}")
async def get_user_analyses(user_id: str, limit: int = 50):
    """Get analysis history for a user."""
    try:
        # Validate that user_id is a proper UUID
        try:
            uuid.UUID(user_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid user_id format: {user_id}. Expected valid UUID.")
        
        analyses = simple_db_manager.get_user_analyses(user_id, limit)
        return {
            "success": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analyses: {str(e)}")

@app.get("/analyses/summary/user/{user_id}")
async def get_user_analysis_summary(user_id: str, limit: int = 50):
    """Get analysis summary for a user."""
    try:
        # Validate that user_id is a proper UUID
        try:
            uuid.UUID(user_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid user_id format: {user_id}. Expected valid UUID.")
        
        analyses = simple_db_manager.get_user_analyses(user_id, limit)
        
        # Create summary
        summary = {
            "total_analyses": len(analyses),
            "unique_stocks": len(set(analysis.get("stock_symbol", "") for analysis in analyses)),
            "recent_analyses": analyses[:5] if analyses else [],
            "sectors_analyzed": list(set(analysis.get("sector", "") for analysis in analyses if analysis.get("sector")))
        }
        
        return {
            "success": True,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analysis summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
from supabase_client import get_supabase_client

def extract_analysis_fields(analysis: dict, user_id: str, symbol: str) -> dict:
    # Extract all required fields for stock_analyses table
    return {
        "user_id": user_id,
        "stock_symbol": symbol,
        "analysis_data": analysis,  # full JSON
        "overall_signal": analysis.get("summary", {}).get("overall_signal"),
        "confidence_score": analysis.get("ai_analysis", {}).get("confidence_pct"),
        "risk_level": analysis.get("summary", {}).get("risk_level"),
        "current_price": analysis.get("metadata", {}).get("current_price"),
        "price_change_percentage": analysis.get("metadata", {}).get("price_change_pct"),
        "sector": analysis.get("metadata", {}).get("sector"),
        "analysis_type": analysis.get("analysis_type", "standard"),
        "analysis_quality": analysis.get("summary", {}).get("analysis_quality", "standard"),
        "mathematical_validation": analysis.get("mathematical_validation", False),
        "chart_paths": analysis.get("chart_paths"),
        "metadata": analysis.get("metadata", {}),
    }

def store_analysis_in_supabase(analysis: dict, user_id: str, symbol: str):
    supabase = get_supabase_client()
    data = extract_analysis_fields(analysis, user_id, symbol)
    result = supabase.table("stock_analyses").insert(data).execute()
    return result 
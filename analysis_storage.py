from database_manager import db_manager

def store_analysis_in_supabase(analysis: dict, user_id: str, symbol: str, 
                              exchange: str = "NSE", period: int = 365, interval: str = "day"):
    """
    Store analysis results in Supabase with proper validation and user management.
    
    Args:
        analysis: Analysis results dictionary
        user_id: User ID (should be a valid UUID string)
        symbol: Stock symbol
        exchange: Stock exchange (default: "NSE")
        period: Analysis period in days (default: 365)
        interval: Data interval (default: "day")
    
    Returns:
        str: Analysis ID if successful, None otherwise
    
    Raises:
        ValueError: If user_id is not a valid UUID or other validation fails
        Exception: If Supabase operation fails
    """
    try:
        # Use the database manager to handle all storage operations
        analysis_id = db_manager.store_analysis(
            analysis=analysis,
            user_id=user_id,
            symbol=symbol,
            exchange=exchange,
            period=period,
            interval=interval
        )
        
        if analysis_id:
            # Update user analysis count
            db_manager.update_user_analysis_count(user_id)
            return analysis_id
        else:
            raise Exception("Failed to store analysis")
            
    except Exception as e:
        print(f"Error storing analysis in Supabase: {e}")
        raise 
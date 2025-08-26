"""
Database Setup Script

This script helps set up the required database tables for the stock analysis system.
Run this script to create missing tables and fix schema issues.
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

def create_profiles_table(supabase):
    """Create the profiles table if it doesn't exist."""
    try:
        # Check if profiles table exists
        result = supabase.table("profiles").select("id").limit(1).execute()
        print("✅ Profiles table already exists")
        return True
    except Exception as e:
        print(f"❌ Profiles table error: {e}")
        print("⚠️ You need to create the profiles table manually in Supabase")
        print("   Required columns: id, email, full_name, subscription_tier, preferences, analysis_count, favorite_stocks")
        return False

def create_stock_analyses_table(supabase):
    """Create the stock_analyses_simple table if it doesn't exist."""
    try:
        # Check if stock_analyses_simple table exists
        result = supabase.table("stock_analyses_simple").select("id").limit(1).execute()
        print("✅ Stock analyses simple table already exists")
        return True
    except Exception as e:
        print(f"❌ Stock analyses simple table error: {e}")
        print("⚠️ You need to create the stock_analyses_simple table manually in Supabase")
        print("   Required columns: id, user_id, stock_symbol, analysis_data, created_at")
        return False

def create_technical_indicators_table(supabase):
    """Create the technical_indicators table if it doesn't exist."""
    try:
        result = supabase.table("technical_indicators").select("id").limit(1).execute()
        print("✅ Technical indicators table already exists")
        return True
    except Exception as e:
        print(f"❌ Technical indicators table error: {e}")
        print("⚠️ You need to create the technical_indicators table manually in Supabase")
        return False

def create_pattern_recognition_table(supabase):
    """Create the pattern_recognition table if it doesn't exist."""
    try:
        result = supabase.table("pattern_recognition").select("id").limit(1).execute()
        print("✅ Pattern recognition table already exists")
        return True
    except Exception as e:
        print(f"❌ Pattern recognition table error: {e}")
        print("⚠️ You need to create the pattern_recognition table manually in Supabase")
        return False

def create_trading_levels_table(supabase):
    """Create the trading_levels table if it doesn't exist."""
    try:
        result = supabase.table("trading_levels").select("id").limit(1).execute()
        print("✅ Trading levels table already exists")
        return True
    except Exception as e:
        print(f"❌ Trading levels table error: {e}")
        print("⚠️ You need to create the trading_levels table manually in Supabase")
        return False

def create_volume_analysis_table(supabase):
    """Create the volume_analysis table if it doesn't exist."""
    try:
        result = supabase.table("volume_analysis").select("id").limit(1).execute()
        print("✅ Volume analysis table already exists")
        return True
    except Exception as e:
        print(f"❌ Volume analysis table error: {e}")
        print("⚠️ You need to create the volume_analysis table manually in Supabase")
        return False

def create_risk_management_table(supabase):
    """Create the risk_management table if it doesn't exist."""
    try:
        result = supabase.table("risk_management").select("id").limit(1).execute()
        print("✅ Risk management table already exists")
        return True
    except Exception as e:
        print(f"❌ Risk management table error: {e}")
        print("⚠️ You need to create the risk_management table manually in Supabase")
        return False

def create_sector_benchmarking_table(supabase):
    """Create the sector_benchmarking table if it doesn't exist."""
    try:
        result = supabase.table("sector_benchmarking").select("id").limit(1).execute()
        print("✅ Sector benchmarking table already exists")
        return True
    except Exception as e:
        print(f"❌ Sector benchmarking table error: {e}")
        print("⚠️ You need to create the sector_benchmarking table manually in Supabase")
        return False

def create_multi_timeframe_analysis_table(supabase):
    """Create the multi_timeframe_analysis table if it doesn't exist."""
    try:
        result = supabase.table("multi_timeframe_analysis").select("id").limit(1).execute()
        print("✅ Multi timeframe analysis table already exists")
        return True
    except Exception as e:
        print(f"❌ Multi timeframe analysis table error: {e}")
        print("⚠️ You need to create the multi_timeframe_analysis table manually in Supabase")
        return False

def main():
    """Main setup function."""
    print("🔧 Database Setup Script")
    print("=" * 50)
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        print("✅ Connected to Supabase")
        
        # Check and create tables
        tables_status = {}
        
        tables_status["profiles"] = create_profiles_table(supabase)
        tables_status["stock_analyses_simple"] = create_stock_analyses_table(supabase)
        tables_status["technical_indicators"] = create_technical_indicators_table(supabase)
        tables_status["pattern_recognition"] = create_pattern_recognition_table(supabase)
        tables_status["trading_levels"] = create_trading_levels_table(supabase)
        tables_status["volume_analysis"] = create_volume_analysis_table(supabase)
        tables_status["risk_management"] = create_risk_management_table(supabase)
        tables_status["sector_benchmarking"] = create_sector_benchmarking_table(supabase)
        tables_status["multi_timeframe_analysis"] = create_multi_timeframe_analysis_table(supabase)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Database Setup Summary")
        print("=" * 50)
        
        existing_tables = sum(1 for status in tables_status.values() if status)
        total_tables = len(tables_status)
        
        print(f"✅ Existing tables: {existing_tables}/{total_tables}")
        
        for table_name, status in tables_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {table_name}")
        
        if existing_tables == total_tables:
            print("\n🎉 All tables are ready! The analysis service should work properly.")
        else:
            print(f"\n⚠️ {total_tables - existing_tables} tables are missing.")
            print("Please create the missing tables in your Supabase dashboard.")
            print("\nRequired SQL for missing tables:")
            print_sql_schema()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

def print_sql_schema():
    """Print SQL schema for missing tables."""
    print("""
-- Profiles table
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY,
    email TEXT,
    full_name TEXT,
    subscription_tier TEXT DEFAULT 'free',
    preferences JSONB DEFAULT '{}',
    analysis_count INTEGER DEFAULT 0,
    favorite_stocks TEXT[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Stock analyses table
CREATE TABLE IF NOT EXISTS stock_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES profiles(id),
    stock_symbol TEXT NOT NULL,
    analysis_data_json JSONB NOT NULL,
    exchange TEXT DEFAULT 'NSE',
    period_days INTEGER DEFAULT 365,
    interval TEXT DEFAULT 'day',
    overall_signal TEXT,
    confidence_score FLOAT,
    risk_level TEXT,
    current_price FLOAT,
    price_change_percentage FLOAT,
    sector TEXT,
    analysis_type TEXT DEFAULT 'standard',
    analysis_quality TEXT DEFAULT 'standard',
    mathematical_validation BOOLEAN DEFAULT FALSE,
    chart_paths JSONB,
    metadata JSONB,
    -- Token tracking fields
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    llm_calls_count INTEGER DEFAULT 0,
    token_usage_breakdown JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS technical_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    indicator_type TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    value FLOAT,
    signal TEXT,
    strength TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Pattern recognition table
CREATE TABLE IF NOT EXISTS pattern_recognition (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    pattern_type TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    confidence FLOAT,
    direction TEXT,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    start_price FLOAT,
    end_price FLOAT,
    target_price FLOAT,
    stop_loss FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trading levels table
CREATE TABLE IF NOT EXISTS trading_levels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    level_type TEXT NOT NULL,
    price_level FLOAT NOT NULL,
    strength TEXT,
    volume_confirmation BOOLEAN,
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Volume analysis table
CREATE TABLE IF NOT EXISTS volume_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    volume_type TEXT NOT NULL,
    date TIMESTAMP,
    volume BIGINT,
    price FLOAT,
    significance TEXT,
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Risk management table
CREATE TABLE IF NOT EXISTS risk_management (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    risk_type TEXT NOT NULL,
    risk_level TEXT DEFAULT 'medium',
    risk_score FLOAT,
    description TEXT,
    mitigation_strategy TEXT,
    stop_loss_level FLOAT,
    take_profit_level FLOAT,
    position_size_recommendation TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sector benchmarking table
CREATE TABLE IF NOT EXISTS sector_benchmarking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    sector TEXT,
    sector_index TEXT,
    beta FLOAT,
    correlation FLOAT,
    sharpe_ratio FLOAT,
    volatility FLOAT,
    max_drawdown FLOAT,
    cumulative_return FLOAT,
    annualized_return FLOAT,
    sector_beta FLOAT,
    sector_correlation FLOAT,
    sector_sharpe_ratio FLOAT,
    sector_volatility FLOAT,
    sector_max_drawdown FLOAT,
    sector_cumulative_return FLOAT,
    sector_annualized_return FLOAT,
    excess_return FLOAT,
    sector_excess_return FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Multi timeframe analysis table
CREATE TABLE IF NOT EXISTS multi_timeframe_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES stock_analyses(id) ON DELETE CASCADE,
    timeframe TEXT NOT NULL,
    signal TEXT,
    confidence FLOAT,
    bias TEXT,
    entry_range_min FLOAT,
    entry_range_max FLOAT,
    target_1 FLOAT,
    target_2 FLOAT,
    stop_loss FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
""")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
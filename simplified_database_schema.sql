-- Simplified Database Schema
-- This script removes all complex tables and creates one simple table for storing analysis data

-- Drop all the complex tables that are causing issues
DROP TABLE IF EXISTS multi_timeframe_analysis CASCADE;
DROP TABLE IF EXISTS volume_analysis CASCADE;
DROP TABLE IF EXISTS risk_management CASCADE;
DROP TABLE IF EXISTS trading_levels CASCADE;
DROP TABLE IF EXISTS pattern_recognition CASCADE;
DROP TABLE IF EXISTS technical_indicators CASCADE;
DROP TABLE IF EXISTS sector_benchmarking CASCADE;
DROP TABLE IF EXISTS stock_analyses CASCADE;

-- Drop any views that depend on these tables
DROP VIEW IF EXISTS analysis_summary_view CASCADE;
DROP VIEW IF EXISTS sector_performance_view CASCADE;
DROP VIEW IF EXISTS user_analysis_history_view CASCADE;

-- Create one simple table for storing all analysis data
CREATE TABLE IF NOT EXISTS stock_analyses_simple (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    stock_symbol TEXT NOT NULL,
    analysis_data JSONB NOT NULL, -- This stores everything in one JSON column
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_analyses_simple_user_id ON stock_analyses_simple(user_id);
CREATE INDEX IF NOT EXISTS idx_stock_analyses_simple_stock_symbol ON stock_analyses_simple(stock_symbol);
CREATE INDEX IF NOT EXISTS idx_stock_analyses_simple_created_at ON stock_analyses_simple(created_at);

-- Create a simple view for common queries
CREATE OR REPLACE VIEW analysis_summary_simple AS
SELECT 
    sa.id,
    sa.stock_symbol,
    sa.user_id,
    p.email as user_email,
    p.full_name as user_name,
    sa.analysis_data->>'overall_signal' as overall_signal,
    (sa.analysis_data->>'confidence_score')::float as confidence_score,
    sa.analysis_data->>'risk_level' as risk_level,
    (sa.analysis_data->>'current_price')::float as current_price,
    sa.analysis_data->>'sector' as sector,
    sa.created_at
FROM stock_analyses_simple sa
LEFT JOIN profiles p ON sa.user_id = p.id;

-- Add comments for documentation
COMMENT ON TABLE stock_analyses_simple IS 'Simple table storing all stock analysis data in JSON format';
COMMENT ON COLUMN stock_analyses_simple.analysis_data IS 'JSON column containing all analysis results, indicators, patterns, charts, etc.';

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON stock_analyses_simple TO authenticated;
-- GRANT SELECT ON analysis_summary_simple TO authenticated; 
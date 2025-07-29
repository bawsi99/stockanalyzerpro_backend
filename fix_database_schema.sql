-- =====================================================
-- DATABASE SCHEMA FIX FOR TRADERPRO
-- Fix column name mismatch: analysis_data -> analysis_data_json
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- 1. FIX STOCK_ANALYSES TABLE COLUMN NAME
-- =====================================================

-- Check if the column exists and rename it
DO $$
BEGIN
    -- Check if analysis_data column exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'stock_analyses' 
        AND column_name = 'analysis_data'
    ) THEN
        -- Rename the column to analysis_data_json
        ALTER TABLE stock_analyses RENAME COLUMN analysis_data TO analysis_data_json;
        RAISE NOTICE 'Column analysis_data renamed to analysis_data_json';
    ELSE
        RAISE NOTICE 'Column analysis_data does not exist, checking for analysis_data_json';
    END IF;
    
    -- Check if analysis_data_json column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'stock_analyses' 
        AND column_name = 'analysis_data_json'
    ) THEN
        -- Add the column if it doesn't exist
        ALTER TABLE stock_analyses ADD COLUMN analysis_data_json JSONB;
        RAISE NOTICE 'Column analysis_data_json added';
    ELSE
        RAISE NOTICE 'Column analysis_data_json already exists';
    END IF;
END $$;

-- =====================================================
-- 2. UPDATE DATABASE FUNCTIONS TO USE NEW COLUMN NAME
-- =====================================================

-- Update the extract_technical_indicators function
CREATE OR REPLACE FUNCTION extract_technical_indicators(analysis_id UUID)
RETURNS VOID AS $$
DECLARE
    analysis_data JSONB;
    indicators JSONB;
    indicator_type TEXT;
    indicator_name TEXT;
    indicator_value JSONB;
BEGIN
    -- Get analysis data using the new column name
    SELECT analysis_data_json INTO analysis_data 
    FROM stock_analyses 
    WHERE id = analysis_id;
    
    IF analysis_data IS NULL THEN
        RETURN;
    END IF;
    
    -- Extract indicators from the JSON structure
    indicators = analysis_data->'indicators';
    
    IF indicators IS NULL THEN
        RETURN;
    END IF;
    
    -- Clear existing indicators for this analysis
    DELETE FROM technical_indicators WHERE analysis_id = extract_technical_indicators.analysis_id;
    
    -- Extract moving averages
    IF indicators ? 'moving_averages' THEN
        FOR indicator_name, indicator_value IN SELECT * FROM jsonb_each(indicators->'moving_averages')
        LOOP
            INSERT INTO technical_indicators (analysis_id, indicator_type, indicator_name, value, signal, strength)
            VALUES (
                analysis_id,
                'moving_average',
                indicator_name,
                (indicator_value->>'value')::DECIMAL,
                indicator_value->>'signal',
                (indicator_value->>'strength')::DECIMAL
            );
        END LOOP;
    END IF;
    
    -- Extract RSI
    IF indicators ? 'rsi' THEN
        INSERT INTO technical_indicators (analysis_id, indicator_type, indicator_name, value, signal, strength)
        VALUES (
            analysis_id,
            'momentum',
            'RSI',
            (indicators->'rsi'->>'value')::DECIMAL,
            indicators->'rsi'->>'signal',
            (indicators->'rsi'->>'strength')::DECIMAL
        );
    END IF;
    
    -- Extract MACD
    IF indicators ? 'macd' THEN
        INSERT INTO technical_indicators (analysis_id, indicator_type, indicator_name, value, signal, strength)
        VALUES (
            analysis_id,
            'momentum',
            'MACD',
            (indicators->'macd'->>'value')::DECIMAL,
            indicators->'macd'->>'signal',
            (indicators->'macd'->>'strength')::DECIMAL
        );
    END IF;
    
    -- Extract Bollinger Bands
    IF indicators ? 'bollinger_bands' THEN
        INSERT INTO technical_indicators (analysis_id, indicator_type, indicator_name, value, signal, strength)
        VALUES (
            analysis_id,
            'volatility',
            'Bollinger_Bands',
            (indicators->'bollinger_bands'->>'value')::DECIMAL,
            indicators->'bollinger_bands'->>'signal',
            (indicators->'bollinger_bands'->>'strength')::DECIMAL
        );
    END IF;
    
END;
$$ LANGUAGE plpgsql;

-- Update the extract_sector_benchmarking function
CREATE OR REPLACE FUNCTION extract_sector_benchmarking(analysis_id UUID)
RETURNS VOID AS $$
DECLARE
    analysis_data JSONB;
    sector_data JSONB;
BEGIN
    -- Get analysis data using the new column name
    SELECT analysis_data_json INTO analysis_data 
    FROM stock_analyses 
    WHERE id = analysis_id;
    
    IF analysis_data IS NULL THEN
        RETURN;
    END IF;
    
    -- Extract sector benchmarking
    sector_data = analysis_data->'sector_benchmarking';
    
    IF sector_data IS NULL THEN
        RETURN;
    END IF;
    
    -- Clear existing sector data for this analysis
    DELETE FROM sector_benchmarking WHERE analysis_id = extract_sector_benchmarking.analysis_id;
    
    -- Insert sector benchmarking data
    INSERT INTO sector_benchmarking (
        analysis_id,
        sector,
        sector_index,
        beta,
        correlation,
        sharpe_ratio,
        volatility,
        max_drawdown,
        cumulative_return,
        annualized_return,
        sector_beta,
        sector_correlation,
        sector_sharpe_ratio,
        sector_volatility,
        sector_max_drawdown,
        sector_cumulative_return,
        sector_annualized_return,
        excess_return,
        sector_excess_return
    ) VALUES (
        analysis_id,
        sector_data->'sector_info'->>'sector',
        sector_data->'sector_info'->>'sector_index',
        (sector_data->'market_benchmarking'->>'beta')::DECIMAL,
        (sector_data->'market_benchmarking'->>'correlation')::DECIMAL,
        (sector_data->'market_benchmarking'->>'sharpe_ratio')::DECIMAL,
        (sector_data->'market_benchmarking'->>'volatility')::DECIMAL,
        (sector_data->'market_benchmarking'->>'max_drawdown')::DECIMAL,
        (sector_data->'market_benchmarking'->>'cumulative_return')::DECIMAL,
        (sector_data->'market_benchmarking'->>'annualized_return')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_beta')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_correlation')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_sharpe_ratio')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_volatility')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_max_drawdown')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_cumulative_return')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_annualized_return')::DECIMAL,
        (sector_data->'market_benchmarking'->>'excess_return')::DECIMAL,
        (sector_data->'sector_benchmarking'->>'sector_excess_return')::DECIMAL
    );
    
END;
$$ LANGUAGE plpgsql;

-- Update the extract_pattern_recognition function
CREATE OR REPLACE FUNCTION extract_pattern_recognition(analysis_id UUID)
RETURNS VOID AS $$
DECLARE
    analysis_data JSONB;
    overlays JSONB;
    patterns JSONB;
    pattern JSONB;
BEGIN
    -- Get analysis data using the new column name
    SELECT analysis_data_json INTO analysis_data 
    FROM stock_analyses 
    WHERE id = analysis_id;
    
    IF analysis_data IS NULL THEN
        RETURN;
    END IF;
    
    -- Extract overlays (contains patterns)
    overlays = analysis_data->'overlays';
    
    IF overlays IS NULL THEN
        RETURN;
    END IF;
    
    -- Clear existing patterns for this analysis
    DELETE FROM pattern_recognition WHERE analysis_id = extract_pattern_recognition.analysis_id;
    
    -- Extract triangle patterns
    IF overlays ? 'triangle_patterns' THEN
        FOR pattern IN SELECT * FROM jsonb_array_elements(overlays->'triangle_patterns')
        LOOP
            INSERT INTO pattern_recognition (
                analysis_id, pattern_type, pattern_name, confidence, direction,
                start_date, end_date, start_price, end_price, target_price, stop_loss
            ) VALUES (
                analysis_id,
                'triangle',
                pattern->>'type',
                (pattern->>'confidence')::DECIMAL,
                pattern->>'direction',
                (pattern->>'start_date')::DATE,
                (pattern->>'end_date')::DATE,
                (pattern->>'start_price')::DECIMAL,
                (pattern->>'end_price')::DECIMAL,
                (pattern->>'target_price')::DECIMAL,
                (pattern->>'stop_loss')::DECIMAL
            );
        END LOOP;
    END IF;
    
    -- Extract flag patterns
    IF overlays ? 'flag_patterns' THEN
        FOR pattern IN SELECT * FROM jsonb_array_elements(overlays->'flag_patterns')
        LOOP
            INSERT INTO pattern_recognition (
                analysis_id, pattern_type, pattern_name, confidence, direction,
                start_date, end_date, start_price, end_price, target_price, stop_loss
            ) VALUES (
                analysis_id,
                'flag',
                pattern->>'type',
                (pattern->>'confidence')::DECIMAL,
                pattern->>'direction',
                (pattern->>'start_date')::DATE,
                (pattern->>'end_date')::DATE,
                (pattern->>'start_price')::DECIMAL,
                (pattern->>'end_price')::DECIMAL,
                (pattern->>'target_price')::DECIMAL,
                (pattern->>'stop_loss')::DECIMAL
            );
        END LOOP;
    END IF;
    
END;
$$ LANGUAGE plpgsql;

-- Update the extract_trading_levels function
CREATE OR REPLACE FUNCTION extract_trading_levels(analysis_id UUID)
RETURNS VOID AS $$
DECLARE
    analysis_data JSONB;
    ai_analysis JSONB;
    trading_guidance JSONB;
    levels JSONB;
    level JSONB;
BEGIN
    -- Get analysis data using the new column name
    SELECT analysis_data_json INTO analysis_data 
    FROM stock_analyses 
    WHERE id = analysis_id;
    
    IF analysis_data IS NULL THEN
        RETURN;
    END IF;
    
    -- Extract AI analysis
    ai_analysis = analysis_data->'ai_analysis';
    
    IF ai_analysis IS NULL THEN
        RETURN;
    END IF;
    
    -- Clear existing trading levels for this analysis
    DELETE FROM trading_levels WHERE analysis_id = extract_trading_levels.analysis_id;
    
    -- Extract key levels from AI analysis
    IF ai_analysis ? 'must_watch_levels' THEN
        FOR level IN SELECT * FROM jsonb_array_elements(ai_analysis->'must_watch_levels')
        LOOP
            INSERT INTO trading_levels (
                analysis_id, level_type, price_level, strength, description
            ) VALUES (
                analysis_id,
                'key_level',
                (level->>'price')::DECIMAL,
                (level->>'strength')::DECIMAL,
                level->>'description'
            );
        END LOOP;
    END IF;
    
    -- Extract trading guidance levels
    trading_guidance = ai_analysis->'trading_strategy';
    
    IF trading_guidance IS NOT NULL THEN
        -- Short term levels
        IF trading_guidance ? 'short_term' THEN
            INSERT INTO trading_levels (analysis_id, level_type, price_level, strength, description)
            VALUES (
                analysis_id,
                'entry',
                (trading_guidance->'short_term'->>'entry_range'->0)::DECIMAL,
                0.8,
                'Short term entry level'
            );
            
            INSERT INTO trading_levels (analysis_id, level_type, price_level, strength, description)
            VALUES (
                analysis_id,
                'target',
                (trading_guidance->'short_term'->>'targets'->0)::DECIMAL,
                0.9,
                'Short term target 1'
            );
            
            INSERT INTO trading_levels (analysis_id, level_type, price_level, strength, description)
            VALUES (
                analysis_id,
                'stop_loss',
                (trading_guidance->'short_term'->>'stop_loss')::DECIMAL,
                0.9,
                'Short term stop loss'
            );
        END IF;
    END IF;
    
END;
$$ LANGUAGE plpgsql;

-- Update the extract_volume_analysis function
CREATE OR REPLACE FUNCTION extract_volume_analysis(analysis_id UUID)
RETURNS VOID AS $$
DECLARE
    analysis_data JSONB;
    overlays JSONB;
    volume_anomalies JSONB;
    anomaly JSONB;
BEGIN
    -- Get analysis data using the new column name
    SELECT analysis_data_json INTO analysis_data 
    FROM stock_analyses 
    WHERE id = analysis_id;
    
    IF analysis_data IS NULL THEN
        RETURN;
    END IF;
    
    -- Extract overlays
    overlays = analysis_data->'overlays';
    
    IF overlays IS NULL THEN
        RETURN;
    END IF;
    
    -- Clear existing volume analysis for this analysis
    DELETE FROM volume_analysis WHERE analysis_id = extract_volume_analysis.analysis_id;
    
    -- Extract volume anomalies
    IF overlays ? 'volume_anomalies' THEN
        FOR anomaly IN SELECT * FROM jsonb_array_elements(overlays->'volume_anomalies')
        LOOP
            INSERT INTO volume_analysis (
                analysis_id, volume_type, date, volume, price, significance, description
            ) VALUES (
                analysis_id,
                'anomaly',
                (anomaly->>'date')::DATE,
                (anomaly->>'volume')::DECIMAL,
                (anomaly->>'price')::DECIMAL,
                0.8,
                'Volume anomaly detected'
            );
        END LOOP;
    END IF;
    
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 3. UPDATE TRIGGER FUNCTION
-- =====================================================

-- Update the trigger function to use the new column name
CREATE OR REPLACE FUNCTION trigger_extract_analysis_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Extract all data types
    PERFORM extract_technical_indicators(NEW.id);
    PERFORM extract_sector_benchmarking(NEW.id);
    PERFORM extract_pattern_recognition(NEW.id);
    PERFORM extract_trading_levels(NEW.id);
    PERFORM extract_volume_analysis(NEW.id);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 4. VERIFICATION QUERIES
-- =====================================================

-- Verify the column structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'stock_analyses' 
AND column_name LIKE '%analysis_data%'
ORDER BY column_name;

-- =====================================================
-- 5. MIGRATION COMPLETE
-- =====================================================

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- Create a comment documenting the migration
COMMENT ON COLUMN stock_analyses.analysis_data_json IS 'JSON data containing complete analysis results - renamed from analysis_data to avoid column reference ambiguity'; 
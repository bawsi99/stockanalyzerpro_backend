-- Temporarily disable the trigger that's causing column reference ambiguity
ALTER TABLE stock_analyses DISABLE TRIGGER extract_analysis_data_trigger;

-- Verify the trigger is disabled
SELECT 
    trigger_name,
    event_manipulation,
    action_timing,
    action_orientation,
    action_statement
FROM information_schema.triggers 
WHERE event_object_table = 'stock_analyses'; 
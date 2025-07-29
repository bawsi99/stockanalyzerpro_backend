-- Quick Schema Fix: Rename analysis_data column to avoid conflicts
-- This is the fastest solution to resolve the column reference ambiguity

-- Step 1: Rename the column
ALTER TABLE stock_analyses RENAME COLUMN analysis_data TO analysis_data_json;

-- Step 2: Verify the change
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'stock_analyses' 
AND column_name LIKE '%analysis_data%';

-- Step 3: Test insert with new column name
-- (This will be done by the application after the schema change) 
-- SQL script to remove problematic constraints
-- This will allow the backend to work directly with the database

-- 1. Remove the foreign key constraint from profiles table that references users table
-- This constraint is preventing direct profile creation from the backend
ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_id_fkey;

-- 2. Check if there are any other foreign key constraints on profiles table
-- and remove them if they're causing issues
-- (We'll keep the primary key constraint)

-- 3. Verify the constraint was removed
SELECT 
    tc.constraint_type,
    tc.constraint_name,
    kcu.column_name,
    ccu.table_name as references_table,
    ccu.column_name as references_column
FROM 
    information_schema.table_constraints tc
    LEFT JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
    LEFT JOIN information_schema.constraint_column_usage ccu
        ON tc.constraint_name = ccu.constraint_name
WHERE 
    tc.table_name = 'profiles'
    AND tc.table_schema = 'public'
ORDER BY tc.constraint_type;

-- 4. Optional: If you want to make the profiles table completely independent
-- (remove all foreign key constraints), uncomment the following:

-- ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_user_id_fkey;
-- ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_created_by_fkey;
-- ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_updated_by_fkey;

-- 5. Check stock_analyses table constraints and remove problematic ones
-- Remove foreign key constraint that references profiles table if it exists
ALTER TABLE stock_analyses DROP CONSTRAINT IF EXISTS stock_analyses_user_id_fkey;

-- 6. Verify stock_analyses constraints after removal
SELECT 
    tc.constraint_type,
    tc.constraint_name,
    kcu.column_name,
    ccu.table_name as references_table,
    ccu.column_name as references_column
FROM 
    information_schema.table_constraints tc
    LEFT JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
    LEFT JOIN information_schema.constraint_column_usage ccu
        ON tc.constraint_name = ccu.constraint_name
WHERE 
    tc.table_name = 'stock_analyses'
    AND tc.table_schema = 'public'
ORDER BY tc.constraint_type;

-- 7. Summary of remaining constraints
SELECT 
    'REMAINING CONSTRAINTS' as section,
    tc.table_name,
    tc.constraint_type,
    tc.constraint_name,
    kcu.column_name
FROM 
    information_schema.table_constraints tc
    LEFT JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
WHERE 
    tc.table_name IN ('profiles', 'stock_analyses')
    AND tc.table_schema = 'public'
ORDER BY tc.table_name, tc.constraint_type; 
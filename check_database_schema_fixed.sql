-- Database Schema Inspection Script (Fixed)
-- This script will show you all tables, views, and their structures in your database

-- 1. List all tables in the database
SELECT 
    schemaname,
    tablename,
    tableowner,
    hasindexes,
    hasrules,
    hastriggers,
    rowsecurity
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- 2. List all views in the database
SELECT 
    schemaname,
    viewname,
    viewowner,
    definition
FROM pg_views 
WHERE schemaname = 'public'
ORDER BY viewname;

-- 3. Get detailed table information with column details
SELECT 
    t.table_name,
    c.column_name,
    c.data_type,
    c.is_nullable,
    c.column_default,
    c.character_maximum_length,
    c.numeric_precision,
    c.numeric_scale
FROM information_schema.tables t
JOIN information_schema.columns c ON t.table_name = c.table_name
WHERE t.table_schema = 'public' 
    AND t.table_type = 'BASE TABLE'
    AND c.table_schema = 'public'
ORDER BY t.table_name, c.ordinal_position;

-- 4. Get foreign key relationships
SELECT 
    tc.table_name, 
    kcu.column_name, 
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name 
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;

-- 5. Get table row counts (approximate)
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows
FROM pg_stat_user_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- 6. Get index information (simplified)
SELECT 
    indexname,
    tablename,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- 7. Summary of database objects
SELECT 
    'Tables' as object_type,
    COUNT(*) as count
FROM pg_tables 
WHERE schemaname = 'public'
UNION ALL
SELECT 
    'Views' as object_type,
    COUNT(*) as count
FROM pg_views 
WHERE schemaname = 'public'
UNION ALL
SELECT 
    'Indexes' as object_type,
    COUNT(*) as count
FROM pg_indexes 
WHERE schemaname = 'public';

-- 8. Simple table list (most reliable)
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name; 
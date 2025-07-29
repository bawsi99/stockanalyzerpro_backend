-- SQL script to identify all constraints in the database
-- This script will show all foreign keys, primary keys, unique constraints, and check constraints

-- 1. Get all foreign key constraints
SELECT 
    tc.table_schema,
    tc.table_name,
    kcu.column_name,
    ccu.table_schema AS foreign_table_schema,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    tc.constraint_name,
    rc.delete_rule,
    rc.update_rule
FROM 
    information_schema.table_constraints AS tc 
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
    JOIN information_schema.referential_constraints AS rc
      ON tc.constraint_name = rc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_schema, tc.table_name, kcu.column_name;

-- 2. Get all primary key constraints
SELECT 
    tc.table_schema,
    tc.table_name,
    kcu.column_name,
    tc.constraint_name
FROM 
    information_schema.table_constraints AS tc 
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY'
ORDER BY tc.table_schema, tc.table_name, kcu.column_name;

-- 3. Get all unique constraints
SELECT 
    tc.table_schema,
    tc.table_name,
    kcu.column_name,
    tc.constraint_name
FROM 
    information_schema.table_constraints AS tc 
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'UNIQUE'
ORDER BY tc.table_schema, tc.table_name, kcu.column_name;

-- 4. Get all check constraints
SELECT 
    tc.table_schema,
    tc.table_name,
    tc.constraint_name,
    cc.check_clause
FROM 
    information_schema.table_constraints AS tc 
    JOIN information_schema.check_constraints AS cc
      ON tc.constraint_name = cc.constraint_name
WHERE tc.constraint_type = 'CHECK'
ORDER BY tc.table_schema, tc.table_name, tc.constraint_name;

-- 5. Get all not null constraints (columns)
SELECT 
    table_schema,
    table_name,
    column_name,
    is_nullable
FROM 
    information_schema.columns
WHERE is_nullable = 'NO'
ORDER BY table_schema, table_name, column_name;

-- 6. Summary of all tables and their constraint counts
SELECT 
    t.table_schema,
    t.table_name,
    COUNT(DISTINCT pk.constraint_name) as primary_keys,
    COUNT(DISTINCT fk.constraint_name) as foreign_keys,
    COUNT(DISTINCT uq.constraint_name) as unique_constraints,
    COUNT(DISTINCT ck.constraint_name) as check_constraints
FROM 
    information_schema.tables t
    LEFT JOIN information_schema.table_constraints pk 
        ON t.table_name = pk.table_name 
        AND t.table_schema = pk.table_schema 
        AND pk.constraint_type = 'PRIMARY KEY'
    LEFT JOIN information_schema.table_constraints fk 
        ON t.table_name = fk.table_name 
        AND t.table_schema = fk.table_schema 
        AND fk.constraint_type = 'FOREIGN KEY'
    LEFT JOIN information_schema.table_constraints uq 
        ON t.table_name = uq.table_name 
        AND t.table_schema = uq.table_schema 
        AND uq.constraint_type = 'UNIQUE'
    LEFT JOIN information_schema.table_constraints ck 
        ON t.table_name = ck.table_name 
        AND t.table_schema = ck.table_schema 
        AND ck.constraint_type = 'CHECK'
WHERE 
    t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
    AND t.table_type = 'BASE TABLE'
GROUP BY 
    t.table_schema, t.table_name
ORDER BY 
    t.table_schema, t.table_name;

-- 7. Specific focus on profiles and stock_analyses tables
SELECT 
    'PROFILES TABLE CONSTRAINTS' as section,
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

SELECT 
    'STOCK_ANALYSES TABLE CONSTRAINTS' as section,
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
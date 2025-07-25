# UUID Fix for Supabase Storage

## Problem
The analysis service was failing with the error:
```
postgrest.exceptions.APIError: {'message': 'invalid input syntax for type uuid: "system"', 'code': '22P02', 'hint': None, 'details': None}
```

This occurred because the code was passing string values like 'system' and 'anonymous' as user_id when the Supabase database expected valid UUID format.

## Root Cause
The Supabase `stock_analyses` table has a `user_id` column of type UUID, but the application was passing string values:
- `'system'` in analysis_service.py and api.py
- `'anonymous'` in authentication functions

## Solution
1. **Added UUID import** to analysis_service.py
2. **Replaced string user_ids with proper UUIDs**:
   - `'system'` → `str(uuid.uuid4())`
   - `'anonymous'` → `str(uuid.uuid4())`
3. **Enhanced validation** in analysis_storage.py to ensure proper UUID format
4. **Updated all authentication functions** to generate UUIDs for anonymous users

## Files Modified

### backend/analysis_service.py
- Added `import uuid`
- Fixed user_id generation in `/analyze` endpoint
- Now generates proper UUID for anonymous/system users

### backend/data_service.py
- Fixed user_id generation for anonymous users (already updated)
- Updated `authenticate_websocket()` to return UUID for anonymous users (already updated)
- Updated `LiveDataPubSub.subscribe()` to use UUID for anonymous users (already updated)

**Note**: The deprecated `api.py` service has been removed from the architecture. The system now uses a split backend with Data Service (Port 8000) and Analysis Service (Port 8001).

### backend/analysis_storage.py
- Added comprehensive validation for user_id, symbol, and analysis data
- Added proper error handling and documentation
- Validates UUID format before attempting Supabase operations

## Testing
Created `test_uuid_fix.py` to verify:
- UUID generation works correctly
- UUID format validation passes
- Analysis storage function accepts proper UUIDs

## Benefits
1. **Fixed Supabase storage errors** - Analysis results can now be stored successfully
2. **Better data integrity** - Proper UUID format ensures database constraints are met
3. **Enhanced error handling** - Clear error messages for invalid data
4. **Future-proof** - UUIDs provide unique identification for anonymous users

## Notes
- Each anonymous user now gets a unique UUID
- UUIDs are generated using Python's `uuid.uuid4()` which creates version 4 UUIDs
- The fix maintains backward compatibility while ensuring proper data format 
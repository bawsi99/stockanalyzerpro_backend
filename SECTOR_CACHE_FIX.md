# Sector Cache Fix - Stock-Specific vs Sector-Agnostic Data

## Problem Identified

### Original Bug
The sector cache was keyed by **sector** but contained **stock-specific data**, causing wrong metrics to be returned for different stocks in the same sector.

```python
# Bug Example:
# User 1: Analyze HDFCBANK (NIFTY_BANK sector)
Cache["NIFTY_BANK"] = {
    "stock_symbol": "HDFCBANK",
    "beta": 1.15,  # HDFCBANK's beta
    "sector_benchmarking": {...}  # HDFCBANK vs NIFTY_BANK
}

# User 2: Analyze ICICIBANK (also NIFTY_BANK sector)
# Cache HIT! Returns HDFCBANK's data! ❌ WRONG!
```

---

## Solution Implemented

### Split Data into Two Categories

**1. Sector-Agnostic Data (CACHED)**
- Sector rotation (which sectors are leading/lagging)
- Sector correlation (how sectors move together)
- Sector synthesis (LLM description of sector)
- Can be reused for ALL stocks in the sector

**2. Stock-Specific Data (FRESH)**
- Stock vs market benchmarking
- Stock vs sector benchmarking
- Stock beta, correlation, returns
- MUST be calculated fresh for each stock

---

## Implementation Details

### Files Modified

#### 1. `services/analysis_service.py`

**Changed Cache Lookup (Lines 2236-2286):**
```python
# OLD: Cached entire comprehensive object (stock + sector data)
cached_analysis = SECTOR_CACHE.get_cached_analysis(sector)
comprehensive = cached_analysis  # ❌ Contains wrong stock's data

# NEW: Cache only sector-agnostic data
cached_sector_data = SECTOR_CACHE.get_cached_analysis(sector)
sector_rotation = cached_sector_data.get('sector_rotation')     # ✅ Reusable
sector_correlation = cached_sector_data.get('sector_correlation') # ✅ Reusable

# Always calculate fresh stock-specific benchmarking
comprehensive = await provider.get_optimized_comprehensive_sector_analysis(
    symbol=req.symbol,  # ← Stock-specific
    stock_data=stock_data,  # ← Stock-specific
    sector=sector,
    cached_rotation=sector_rotation,      # ← Pass cached data
    cached_correlation=sector_correlation  # ← Pass cached data
)
```

**Changed Cache Saving (Lines 2315-2344):**
```python
# OLD: Saved everything (stock + sector data)
SECTOR_CACHE.save_analysis(sector, comprehensive, current_price)  # ❌ Stock-specific

# NEW: Save only sector-agnostic data
sector_agnostic_cache = {
    'sector_rotation': comprehensive.get('sector_rotation'),      # ✅ Reusable
    'sector_correlation': comprehensive.get('sector_correlation'), # ✅ Reusable
    'sector_synthesis': synthesis_result,                          # ✅ Reusable
    # ❌ NOT including sector_benchmarking (stock-specific)
}
SECTOR_CACHE.save_analysis(sector, sector_agnostic_cache, current_price)
```

#### 2. `agents/sector/benchmarking.py`

**Updated Function Signature (Lines 2959-2989):**
```python
async def get_optimized_comprehensive_sector_analysis(
    self,
    symbol: str,
    stock_data: pd.DataFrame,
    sector: str,
    requested_period: int = None,
    use_all_sectors: bool = True,
    cached_rotation: Optional[Dict[str, Any]] = None,      # ← NEW
    cached_correlation: Optional[Dict[str, Any]] = None    # ← NEW
) -> Dict[str, Any]:
```

**Skip Sector Fetching When Cached (Lines 3069-3114):**
```python
# OPTIMIZATION: Skip fetching other sectors if we have cached data
if cached_rotation and cached_correlation:
    logging.info("✅ Skipping sector fetching - using cached rotation & correlation")
    # No need to fetch NIFTY_IT, NIFTY_PHARMA, etc.
else:
    # Fetch other sectors for rotation and correlation calculation
    relevant_sectors = self._get_relevant_sectors_for_analysis(sector, use_all_sectors)
    # Fetch sector data in parallel...
```

**Use Cached Data (Lines 3111-3133):**
```python
# ALWAYS calculate benchmarking (stock-specific)
benchmarking = self._calculate_optimized_benchmarking(symbol, stock_data, sector, sector_data, nifty_data)

# Use cached rotation if available
if cached_rotation:
    rotation = cached_rotation  # ✅ Reuse
else:
    rotation = self._calculate_optimized_rotation(...)  # Calculate fresh

# Use cached correlation if available  
if cached_correlation:
    correlation = cached_correlation  # ✅ Reuse
else:
    correlation = self._calculate_optimized_correlation(...)  # Calculate fresh
```

---

## Performance Impact

### API Call Reduction

**Before Fix:**
```
Request 1: HDFCBANK
- Fetch: HDFCBANK, NIFTY_50, NIFTY_BANK, NIFTY_IT, NIFTY_PHARMA, ... (8 fetches)
- Cache: Everything including HDFCBANK metrics

Request 2: ICICIBANK (5 mins later)
- Cache HIT! (0 fetches)
- ❌ Returns HDFCBANK metrics (WRONG!)
```

**After Fix:**
```
Request 1: HDFCBANK
- Fetch: HDFCBANK, NIFTY_50, NIFTY_BANK, NIFTY_IT, NIFTY_PHARMA, ... (8 fetches)
- Cache: Only rotation & correlation (sector-agnostic)

Request 2: ICICIBANK (5 mins later)
- Cache HIT for rotation & correlation! (partial cache)
- Fetch: ICICIBANK, NIFTY_50, NIFTY_BANK (3 fetches - stock-specific)
- Skip: NIFTY_IT, NIFTY_PHARMA, ... (saved 5 fetches!)
- ✅ Returns ICICIBANK metrics (CORRECT!)
```

### Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **First request (cache miss)** | 8 API calls | 8 API calls | Same |
| **Second request (same sector)** | 0 API calls ❌ Wrong data | 3 API calls ✅ Correct data | **Correctness fixed** |
| **API calls saved** | 8 calls saved ❌ | 5 calls saved ✅ | **5 calls saved** |
| **Data accuracy** | ❌ Wrong stock | ✅ Correct stock | **100% accurate** |

---

## What Gets Cached vs Fresh

| Data Type | Cached? | Why | Example |
|-----------|---------|-----|---------|
| **Stock beta** | ❌ Fresh | Stock-specific | HDFCBANK: 1.15, ICICIBANK: 1.08 |
| **Stock correlation** | ❌ Fresh | Stock-specific | HDFCBANK: 0.85, ICICIBANK: 0.82 |
| **Stock returns** | ❌ Fresh | Stock-specific | HDFCBANK: 12%, ICICIBANK: 10% |
| **Stock Sharpe ratio** | ❌ Fresh | Stock-specific | HDFCBANK: 0.45, ICICIBANK: 0.42 |
| **Sector rotation** | ✅ Cached | Sector-agnostic | NIFTY_IT #1, NIFTY_BANK #3 (same for all) |
| **Sector correlation** | ✅ Cached | Sector-agnostic | BANK-IT: 0.65 (same for all) |
| **Sector synthesis** | ✅ Cached | Sector-agnostic | "Banking sector showing momentum" |

---

## Cache Lifecycle

### Before Fix (WRONG):
```
Cache Key: "NIFTY_BANK"

First Stock: HDFCBANK
  ↓
Cache Miss → Fetch all data
  ↓
Store: {
  stock: "HDFCBANK",
  beta: 1.15,
  rotation: {...},
  correlation: {...}
}

Second Stock: ICICIBANK (same sector)
  ↓
Cache Hit! → Return cached data
  ↓
❌ Returns HDFCBANK's beta (1.15) instead of ICICIBANK's (1.08)
```

### After Fix (CORRECT):
```
Cache Key: "NIFTY_BANK"

First Stock: HDFCBANK
  ↓
Cache Miss → Fetch all data
  ↓
Store: {
  rotation: {...},      # ✅ Reusable
  correlation: {...}    # ✅ Reusable
  # ❌ NOT storing stock-specific data
}

Second Stock: ICICIBANK (same sector)
  ↓
Cache Partial Hit! → Reuse rotation & correlation
  ↓
Fresh Fetch: ICICIBANK, NIFTY_50, NIFTY_BANK (stock-specific)
  ↓
Calculate: ICICIBANK's beta, correlation, returns
  ↓
✅ Returns ICICIBANK's correct metrics (beta: 1.08)
```

---

## Testing

### Verification Steps

1. **Test different stocks in same sector:**
   ```bash
   # Request 1: HDFCBANK
   curl -X POST http://localhost:8002/agents/sector/analyze-all \
     -H "Content-Type: application/json" \
     -d '{"symbol": "HDFCBANK", "exchange": "NSE"}'
   
   # Check response: beta should be ~1.15
   
   # Request 2: ICICIBANK (5 seconds later)
   curl -X POST http://localhost:8002/agents/sector/analyze-all \
     -H "Content-Type: application/json" \
     -d '{"symbol": "ICICIBANK", "exchange": "NSE"}'
   
   # Check response: beta should be ~1.08 (different from HDFCBANK)
   ```

2. **Verify cache logs:**
   ```bash
   # Should see:
   # ✅ Using cached sector-agnostic data for NIFTY_BANK (rotation & correlation)
   # 🔄 Calculating fresh stock-specific benchmarking for ICICIBANK vs NIFTY_BANK
   # ✅ Skipping sector fetching - using cached rotation & correlation
   ```

3. **Verify API call reduction:**
   ```bash
   # First request: 8 API calls
   # Second request: 3 API calls (saved 5 calls)
   ```

---

## Edge Cases Handled

### 1. Partial Cache Hit
```python
# Rotation cached but correlation expired
if cached_rotation and cached_correlation:
    # Use both cached
elif cached_rotation:
    # Use cached rotation, calculate fresh correlation
else:
    # Calculate both fresh
```

### 2. Cache Expiration
```python
# Cache expires after 7 days or 2.5% price change
# Next request will regenerate sector-agnostic data
```

### 3. First Stock in Sector
```python
# No cache available
# Fetch all data, calculate everything
# Store only sector-agnostic data in cache
```

---

## Benefits

1. ✅ **Correctness**: Each stock gets its own metrics
2. ✅ **Performance**: Saves 5 API calls on cache hit (rotation/correlation)
3. ✅ **Simplicity**: Clear separation of concerns
4. ✅ **Maintainability**: Easy to understand what's cached vs fresh
5. ✅ **Scalability**: Cache benefits increase with more stocks per sector

---

## Future Enhancements

### Potential Improvements

1. **Two-Level Cache**: 
   - Sector-agnostic cache (7 days TTL)
   - Stock-specific cache (1 hour TTL)

2. **Smarter Invalidation**:
   - Invalidate stock cache on price change > 5%
   - Invalidate sector cache on rotation change > 10%

3. **Cache Warming**:
   - Pre-fetch sector data during off-peak hours
   - Background job to refresh sector caches

---

## Conclusion

This fix ensures that each stock receives accurate, stock-specific benchmarking metrics while still benefiting from cached sector-agnostic data (rotation and correlation). The cache now correctly separates what can be shared across stocks (sector rotation/correlation) from what must be unique per stock (benchmarking metrics).

**Result**: 100% data accuracy + performance optimization 🎯

# Phase 1 Diagnostic Summary: Parameter Comparison Analysis

## Executive Summary

**Root Cause Identified**: Our evaluation script is selecting the **first strategy [0]** from each window instead of the **optimal strategy** that matches the notebook's performance-based selection. This explains the trade count discrepancy (1,821 vs 846 trades).

## Key Findings

### Strategy Selection Accuracy
- **Correct selections**: 2/6 windows
- **Incorrect selections**: 4/6 windows
- **Overall accuracy**: 33.3%

### Detailed Window Analysis

| Window | Notebook Optimal Parameters | Current Script Uses | Should Use | Status |
|--------|----------------------------|-------------------|------------|---------|
| W1-5min | enter_long=0.004, enter_short=-0.005 | Strategy [0] | Strategy [0] | ✅ CORRECT |
| W2-5min | enter_long=0.002, enter_short=-0.001 | Strategy [0] | Strategy [1] | ❌ WRONG |
| W3-5min | enter_short=-0.006, exit_short=0.003 | Strategy [0] | Strategy [8] | ❌ WRONG |
| W4-5min | enter_long=0.002, enter_short=-0.005 | Strategy [0] | Strategy [1] | ❌ WRONG |
| W5-5min | enter_long=0.002, enter_short=-0.003 | Strategy [0] | Strategy [0] | ✅ CORRECT |
| W6-5min | enter_long=0.001, enter_short=-0.007 | Strategy [0] | Strategy [5] | ❌ WRONG |

### Parameter Discrepancies

**W2-5min (INCORRECT SELECTION)**:
- Current: enter_long=0.001, enter_short=-0.001
- Should be: enter_long=0.002, enter_short=-0.001

**W3-5min (INCORRECT SELECTION)**:
- Current: enter_long=0.004, enter_short=-0.006, exit_short=0.003
- Should be: enter_short=-0.006, exit_short=0.003 (no enter_long)

**W4-5min (INCORRECT SELECTION)**:
- Current: enter_long=0.007, enter_short=-0.005, exit_short=0.002
- Should be: enter_long=0.002, enter_short=-0.005 (no exit_short)

**W6-5min (INCORRECT SELECTION)**:
- Current: enter_long=0.001, exit_long=-0.003 (missing enter_short)
- Should be: enter_long=0.001, enter_short=-0.007

## Cache Analysis Results

### Cache Structure Verified
- Cache file: `cache/5min-best-strategies-v2.pkl`
- Contains 6 windows × 10 strategies each = 60 total strategies
- All notebook optimal parameter combinations **ARE PRESENT** in cache
- Cache contains both correct and incorrect strategy configurations

### Strategy Object Structure
```python
ModelGmadlPredictionsStrategy attributes:
- enter_long: float or None
- enter_short: float or None  
- exit_long: float or None
- exit_short: float or None
- predictions: DataFrame
- exchange_fee: 0.0003
```

## Impact Analysis

### Why This Causes 1,821 vs 846 Trade Discrepancy
1. **Different Parameters = Different Signals**: Wrong thresholds generate different buy/sell signals
2. **More Aggressive Thresholds**: Some incorrect selections use more aggressive entry/exit thresholds
3. **Additional Exit Rules**: Some wrong strategies have exit rules that shouldn't be active
4. **Compounding Effect**: Wrong parameters in multiple windows compound the trading differences

### Examples of Wrong Strategy Impact
- **W3-5min**: Using strategy [0] with enter_long=0.004 instead of strategy [8] with no enter_long rule
- **W6-5min**: Using strategy [0] missing the enter_short=-0.007 rule entirely

## Root Cause Summary

**Current Logic (WRONG)**:
```python
for window_idx, strategies in enumerate(cache_data['gmadl_model']):
    best_strategy = strategies[0]  # Always picks first strategy
```

**Should Be**:
```python
for window_idx, strategies in enumerate(cache_data['gmadl_model']):
    best_strategy = find_strategy_matching_notebook_optimal(strategies, window_idx)
```

## Next Steps (Phase 2)

1. **Implement Corrected Strategy Selection**
   - Create function to match strategies to notebook optimal parameters
   - Update evaluation script to use correct strategy selection logic

2. **Verification Testing**
   - Re-run evaluation with corrected strategies
   - Verify trade count matches 846
   - Verify performance metrics match notebook

3. **Performance Validation**  
   - Confirm Sharpe ratio and other metrics align with notebook results
   - Validate that the equity curve matches expected pattern

## Confidence Level

**High Confidence (95%+)** that this is the primary root cause:
- Clear parameter mismatches identified
- All optimal parameters found in cache
- Logical explanation for trade count difference
- Systematic pattern of incorrect selections (4/6 windows)

The fix should resolve the 1,821 vs 846 trade discrepancy and align our results with the notebook's expected performance metrics.
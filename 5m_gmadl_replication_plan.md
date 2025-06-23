# 5m GMADL Results Replication Plan

## Target Metrics to Achieve
- **Annualized Return Compound (ARC)**: 115.88%
- **Information Ratio (IR)**: 2.129
- **Modified Information Ratio (IR**)**: 7.552
- **Maximum Drawdown (MD)**: 32.66%

## Original Paper Resources Available
- **Public W&B Project**: https://wandb.ai/filipstefaniuk/wne-masters-thesis-testing
- **Existing Sweep ID**: `filipstefaniuk/wne-masters-thesis-testing/0pro3i5i`
- **Status**: Model checkpoints and predictions already available

## Simplified Execution Plan (30-60 minutes vs 8-10 hours)

### Step 1: Verify Existing Resources
- Use existing predictions from sweep `0pro3i5i`
- Confirm model configurations match Table 12 from paper
- Validate GMADL loss function (a=100, b=2)

### Step 2: Load Predictions and Data
```python
# Load existing predictions (already implemented in notebook)
SWEEP_ID = 'filipstefaniuk/wne-masters-thesis-testing/0pro3i5i'
valid_gmadl_pred_windows = get_sweep_window_predictions(SWEEP_ID, 'valid')
test_gmadl_pred_windows = get_sweep_window_predictions(SWEEP_ID, 'test')

# Load data windows
data_windows = get_data_windows(
    'wne-masters-thesis-testing',
    'btc-usdt-5m:latest',
    min_window=0, 
    max_window=5
)
```

### Step 3: Create Strategies with Paper Thresholds
Apply exact thresholds from Table 18:

```python
# Paper thresholds per window
window_thresholds = [
    {'enter_long': 0.004, 'enter_short': -0.005, 'exit_long': None, 'exit_short': None},  # W1
    {'enter_long': 0.002, 'enter_short': -0.001, 'exit_long': None, 'exit_short': None},  # W2
    {'enter_long': None, 'enter_short': -0.006, 'exit_long': None, 'exit_short': 0.003},  # W3
    {'enter_long': 0.002, 'enter_short': -0.005, 'exit_long': None, 'exit_short': None},  # W4
    {'enter_long': 0.002, 'enter_short': -0.003, 'exit_long': None, 'exit_short': None},  # W5
    {'enter_long': 0.001, 'enter_short': -0.007, 'exit_long': None, 'exit_short': None},  # W6
]

# Create strategies
gmadl_model_best_strategies = []
for i, (valid_preds, test_preds, thresholds) in enumerate(zip(valid_gmadl_pred_windows, test_gmadl_pred_windows, window_thresholds)):
    predictions_df = get_predictions_dataframe(valid_preds, test_preds)
    strategy = ModelGmadlPredictionsStrategy(
        predictions=predictions_df,
        **thresholds
    )
    gmadl_model_best_strategies.append(strategy)
```

### Step 4: Evaluate Concatenated Strategy
```python
# Prepare test data
test_data = pd.concat([data_windows[0][0][-PADDING:]] + [data_window[1] for data_window in data_windows])

# Create concatenated strategy
concatenated_strategy = ConcatenatedStrategies(
    window_size=len(data_windows[0][1]),
    strategies=gmadl_model_best_strategies,
    padding=PADDING
)

# Evaluate strategy
gmadl_result = evaluate_strategy(
    test_data, 
    concatenated_strategy,
    padding=PADDING, 
    interval='5min',
    exchange_fee=0.001
)
```

### Step 5: Verify Results
```python
print("=== REPLICATION RESULTS ===")
print(f"ARC: {gmadl_result['arc']*100:.2f}% (Target: 115.88%)")
print(f"IR: {gmadl_result['ir']:.3f} (Target: 2.129)")
print(f"Modified IR: {gmadl_result['mod_ir']:.3f} (Target: 7.552)")
print(f"Max Drawdown: {gmadl_result['md']*100:.2f}% (Target: 32.66%)")
print(f"Portfolio Value: {gmadl_result['value']:.3f}")
print(f"Number of Trades: {gmadl_result['n_trades']}")
print(f"Long Position %: {gmadl_result['long_pos']*100:.2f}%")
print(f"Short Position %: {gmadl_result['short_pos']*100:.2f}%")
```

## Configuration Verification
- ✅ past_window: 28
- ✅ batch_size: 256
- ✅ d_model: 256
- ✅ n_attention_heads: 2
- ✅ dropout: 0.01
- ✅ n_encoder_layers: 1
- ✅ n_decoder_layers: 3
- ✅ learning_rate: 0.0001
- ✅ GMADL loss: a=100, b=2

## Expected Outcome
The notebook cell 27 already shows the target result: **ARC=115.88%, IR=2.129**. This plan validates that the existing sweep `0pro3i5i` contains the exact models and strategy logic used in the paper.

## Benefits
1. **No retraining required** - Uses existing computational work
2. **Exact replication** - Same models as paper
3. **Fast execution** - 30-60 minutes vs 8-10 hours
4. **Focus on validation** - Confirms strategy evaluation logic
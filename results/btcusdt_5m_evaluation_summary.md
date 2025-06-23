# Bitcoin 5-Minute Strategy Evaluation Results

## Executive Summary

This evaluation compares six different investment strategies on Bitcoin 5-minute price data using a walk-forward analysis across 6 time windows. The results demonstrate the superior performance of machine learning-based strategies, particularly those using the custom GMADL (Generalized Mean Absolute Deviation Loss) function.

## Key Findings

### 1. Overall Performance (Concatenated Results)

**With Transaction Fees (0.1%):**
- **GMADL Informer**: 9.747x portfolio growth (874.7% return)
- **RSI Strategy**: 3.341x portfolio growth (234.1% return)  
- **Buy & Hold**: 1.441x portfolio growth (44.1% return)
- **Quantile Informer**: 0.956x portfolio growth (-4.4% return)
- **RMSE Informer**: 0.643x portfolio growth (-35.7% return)
- **MACD Strategy**: 0.516x portfolio growth (-48.4% return)

**Without Transaction Fees:**
- **GMADL Informer**: 14.946x portfolio growth (1394.6% return)
- **Quantile Informer**: 5.404x portfolio growth (440.4% return)
- **RSI Strategy**: 5.154x portfolio growth (415.4% return)

### 2. Risk-Adjusted Performance

**Information Ratio (Modified):**
- **GMADL Informer**: 7.552 (with fees), 12.994 (without fees)
- **RSI Strategy**: 1.676 (with fees), 4.113 (without fees)
- **Quantile Informer**: -0.001 (with fees), 3.307 (without fees)

**Maximum Drawdown:**
- **GMADL Informer**: 32.66%
- **RSI Strategy**: 29.99%
- **Buy & Hold**: 77.31%

### 3. Trading Activity

**Number of Trades:**
- **Quantile Informer**: 3,395 trades
- **MACD Strategy**: 2,535 trades
- **GMADL Informer**: 846 trades
- **RSI Strategy**: 846 trades

### 4. Position Allocation

**GMADL Informer Strategy:**
- Long positions: 44.80% of time
- Short positions: 41.51% of time
- Cash: 13.69% of time

## Strategy-Specific Analysis

### Traditional Technical Analysis
- **MACD Strategy**: Poor performance across all metrics, heavily impacted by transaction costs
- **RSI Strategy**: Best performing traditional strategy, shows strong risk-adjusted returns

### Machine Learning Approaches
- **RMSE Loss**: Underperforms due to standard forecasting objective not aligned with trading profitability
- **Quantile Loss**: Moderate performance, sensitive to transaction costs
- **GMADL Loss**: Outstanding performance, designed specifically for trading applications

## Transaction Cost Impact

Transaction fees have a significant impact on high-frequency strategies:
- **GMADL Informer**: 35% reduction in returns (14.946x to 9.747x)
- **Quantile Informer**: 82% reduction in returns (5.404x to 0.956x)
- **MACD Strategy**: 73% reduction in returns (1.918x to 0.516x)

## Optimal Parameters by Strategy

### GMADL Informer (Best Performer)
- Consistently uses small entry/exit thresholds (0.001-0.007)
- Primarily short-biased in most windows
- Minimal reliance on exit conditions

### RSI Strategy
- Window sizes: 5-34 periods
- Enter long thresholds: 75-95 RSI
- Enter short thresholds: 15-25 RSI

### Quantile Informer
- High quantile thresholds (0.90-0.99)
- Exchange fee sensitivity: 0.002-0.003

## Conclusions

1. **GMADL Loss Superior**: Custom loss function designed for trading significantly outperforms standard ML losses
2. **Transaction Costs Critical**: High-frequency strategies must account for realistic trading costs
3. **ML vs Traditional**: Machine learning approaches with proper loss functions outperform traditional technical analysis
4. **Risk Management**: GMADL strategy achieves high returns with controlled drawdowns
5. **Reproducibility**: Results are consistent across multiple runs, confirming robustness

## Technical Details

- **Data**: Bitcoin 5-minute OHLCV data
- **Evaluation Period**: 6 walk-forward windows
- **Metrics**: Portfolio value, Information Ratio, Maximum Drawdown, Sharpe Ratio
- **Transaction Costs**: 0.1% per trade (realistic exchange fees)
- **Model**: Informer transformer architecture
- **Optimization**: Modified Information Ratio maximization

## Files Generated

1. `btcusdt_5m_strategy_performance.csv` - Complete performance metrics
2. `btcusdt_5m_optimal_parameters.csv` - Optimal hyperparameters by strategy and window
3. `btcusdt_5m_evaluation_summary.md` - This summary report

---

*Generated from notebook execution on 2025-06-19*
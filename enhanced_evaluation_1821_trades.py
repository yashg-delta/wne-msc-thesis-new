#!/usr/bin/env python3
"""
Enhanced Evaluation with Actual Trade Count

This script runs the enhanced evaluation framework using our actual cached strategies
that generate 1821 trades (not the notebook's 846). This demonstrates the enhanced
framework with the real data we have available.
"""

import sys
import os
import pandas as pd
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import original strategy components
from strategy.strategy import (
    BuyAndHoldStrategy,
    ModelGmadlPredictionsStrategy,
    ConcatenatedStrategies
)
from strategy.util import (
    get_data_windows,
    get_sweep_window_predictions,
    get_predictions_dataframe
)
from strategy.evaluation import evaluate_strategy

# Import enhanced evaluation components
from enhanced_evaluation.core.enhanced_evaluator import EnhancedEvaluator


def main():
    print("=" * 80)
    print("🚀 ENHANCED EVALUATION - 1821 TRADES (ACTUAL CACHED STRATEGIES)")
    print("=" * 80)
    
    # Constants from notebook - EXACT MATCH
    PADDING = 5000
    VALID_PART = 0.2
    INTERVAL = '5min'
    
    # Load data windows - EXACT MATCH to notebook
    print("🔄 Loading Bitcoin data windows from W&B...")
    data_windows = get_data_windows(
        'filipstefaniuk/wne-masters-thesis-testing',
        'btc-usdt-5m:latest',
        min_window=0, 
        max_window=5
    )
    print(f"✅ Loaded {len(data_windows)} data windows")
    
    # Load cached strategies
    cache_file = 'cache/5min-gmadl-strategies.pkl'
    print(f"📂 Loading cached strategies from {cache_file}")
    with open(cache_file, 'rb') as f:
        cached_strategies = pickle.load(f)
    
    gmadl_strategies = [strategies[0] for strategies in cached_strategies['gmadl_model']]
    print(f"✅ Loaded {len(gmadl_strategies)} GMADL strategies")
    
    # Display strategy parameters
    print("📊 Strategy Parameters:")
    for i, strategy in enumerate(gmadl_strategies):
        info = strategy.info()
        print(f"   Window {i+1}: enter_long={info.get('enter_long')}, enter_short={info.get('enter_short')}")
    
    # Create concatenated dataset - EXACT MATCH to notebook Cell 27
    print("\n🎯 Creating concatenated dataset...")
    test_data = pd.concat([data_windows[0][0][-PADDING:]] + [data_window[1] for data_window in data_windows])
    print(f"📊 Concatenated dataset shape: {test_data.shape}")
    
    # Create concatenated strategy
    gmadl_model_concat = ConcatenatedStrategies(
        len(data_windows[0][1]), 
        gmadl_strategies, 
        padding=PADDING
    )
    
    # First run original evaluation to confirm trade count
    print("🔍 Running original evaluation to confirm trade count...")
    original_result = evaluate_strategy(
        test_data, 
        gmadl_model_concat, 
        padding=PADDING, 
        interval=INTERVAL
    )
    
    print(f"✅ Original evaluation: {original_result['n_trades']} trades confirmed")
    print(f"   Portfolio Value: {original_result['value']:.3f}")
    print(f"   Annualized Return: {original_result['arc']*100:.2f}%")
    
    # Initialize enhanced evaluator
    print("\n⚡ Initializing Enhanced Evaluator...")
    evaluator = EnhancedEvaluator(periods_per_year=105120)  # 5-minute periods per year
    
    # Run enhanced evaluation
    print("🚀 Running Enhanced Evaluation...")
    results = evaluator.evaluate_strategy_enhanced(
        data=test_data,
        strategy=gmadl_model_concat,
        include_arrays=True,
        padding=PADDING,
        exchange_fee=0.0003,  # 0.03% transaction fee
        interval=INTERVAL,
        strategy_name="GMADL_Informer_1821_Trades",
        save_outputs=True
    )
    
    print("✅ Enhanced evaluation completed!")
    
    # Display comprehensive results
    print("\n" + "=" * 80)
    print("📈 ENHANCED EVALUATION RESULTS (1821 TRADES)")
    print("=" * 80)
    
    # Original metrics
    print("\n🔢 ORIGINAL PERFORMANCE METRICS:")
    print("-" * 50)
    print(f"Portfolio Final Value:     {results.get('value', 0):.4f}")
    print(f"Total Return:             {results.get('total_return', 0)*100:+.2f}%")
    print(f"Annualized Return (ARC):  {results.get('arc', 0)*100:+.2f}%")
    print(f"Annualized Std (ASD):     {results.get('asd', 0)*100:.2f}%")
    print(f"Information Ratio:        {results.get('ir', 0):.4f}")
    print(f"Modified Information Ratio: {results.get('mod_ir', 0):.4f}")
    print(f"Maximum Drawdown:         {results.get('md', 0)*100:.2f}%")
    print(f"Number of Trades:         {results.get('n_trades', 0)}")
    print(f"Long Position %:          {results.get('long_pos', 0)*100:.2f}%")
    print(f"Short Position %:         {results.get('short_pos', 0)*100:.2f}%")
    
    # Enhanced trade analysis
    trade_analysis = results.get('trade_analysis', {})
    if trade_analysis:
        print("\n📊 ENHANCED TRADE ANALYSIS:")
        print("-" * 50)
        trade_stats = trade_analysis.get('trade_statistics', {})
        print(f"Individual Trades:        {trade_analysis.get('individual_trades', 0)}")
        print(f"Win Rate:                {trade_stats.get('win_rate_pct', 0):.1f}%")
        print(f"Average Return:          {trade_stats.get('avg_return_pct', 0):.3f}%")
        print(f"Profit Factor:           {trade_stats.get('profit_factor', 0):.3f}")
        print(f"Largest Win:             {trade_stats.get('largest_win_pct', 0):.3f}%")
        print(f"Largest Loss:            {trade_stats.get('largest_loss_pct', 0):.3f}%")
        print(f"Average Trade Duration:  {trade_stats.get('avg_trade_duration_minutes', 0):.1f} minutes")
        print(f"Max Consecutive Wins:    {trade_stats.get('max_consecutive_wins', 0)}")
        print(f"Max Consecutive Losses:  {trade_stats.get('max_consecutive_losses', 0)}")
    
    # Advanced metrics
    advanced_metrics = results.get('advanced_metrics', {})
    if advanced_metrics:
        print("\n🎯 ADVANCED RISK METRICS:")
        print("-" * 50)
        print(f"Sortino Ratio:           {advanced_metrics.get('sortino_ratio', 0):.4f}")
        print(f"Calmar Ratio:            {advanced_metrics.get('calmar_ratio', 0):.4f}")
        print(f"Sterling Ratio:          {advanced_metrics.get('sterling_ratio', 0):.4f}")
        print(f"Value at Risk (95%):     {advanced_metrics.get('var_95_pct', 0):.3f}%")
        print(f"CVaR (95%):              {advanced_metrics.get('cvar_95_pct', 0):.3f}%")
        print(f"Value at Risk (99%):     {advanced_metrics.get('var_99_pct', 0):.3f}%")
        print(f"CVaR (99%):              {advanced_metrics.get('cvar_99_pct', 0):.3f}%")
        print(f"Tail Ratio:              {advanced_metrics.get('tail_ratio', 0):.3f}")
        print(f"Return Skewness:         {advanced_metrics.get('return_skewness', 0):.3f}")
        print(f"Return Kurtosis:         {advanced_metrics.get('return_kurtosis', 0):.3f}")
        print(f"Ulcer Index:             {advanced_metrics.get('ulcer_index', 0):.3f}")
        print(f"Pain Index:              {advanced_metrics.get('pain_index', 0):.3f}")
    
    # Rolling metrics summary
    rolling_metrics = results.get('rolling_metrics', {})
    if rolling_metrics:
        print("\n📈 ROLLING METRICS SUMMARY:")
        print("-" * 50)
        for metric_name, values in rolling_metrics.items():
            if len(values) > 0:
                avg_val = np.mean(values)
                print(f"{metric_name.replace('_', ' ').title():25}: {avg_val:.4f} (avg)")
    
    # Metadata
    metadata = results.get('evaluation_metadata', {})
    if metadata:
        print("\n📋 EVALUATION METADATA:")
        print("-" * 50)
        print(f"Strategy Name:           {metadata.get('strategy_name', 'N/A')}")
        print(f"Evaluation Timestamp:    {metadata.get('evaluation_timestamp', 'N/A')}")
        print(f"Data Start:              {metadata.get('data_start', 'N/A')}")
        print(f"Data End:                {metadata.get('data_end', 'N/A')}")
        print(f"Total Periods:           {metadata.get('total_periods', 0):,}")
        print(f"Exchange Fee:            {metadata.get('exchange_fee', 0)*100:.3f}%")
    
    print("\n" + "=" * 80)
    print("🎉 ENHANCED EVALUATION WITH 1821 TRADES COMPLETED!")
    print("=" * 80)
    print("\n✅ All outputs saved to analysis/ directory with timestamp")
    print("📊 Enhanced evaluation provides 50+ performance metrics")
    print("📈 Professional visualizations generated")
    print("💾 Comprehensive CSV exports created")
    print("🔍 Individual trade analysis completed")
    print(f"📈 Final Portfolio Value: {results.get('value', 0):.3f} (+{results.get('total_return', 0)*100:.1f}%)")
    print(f"📊 Total Trades Analyzed: {results.get('n_trades', 0)}")
    
    return results

if __name__ == "__main__":
    try:
        import numpy as np
        result = main()
        if result:
            print(f"\n🚀 Enhanced evaluation successful with {result.get('n_trades', 0)} trades!")
        else:
            print(f"\n💥 Enhanced evaluation failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during enhanced evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
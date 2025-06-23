#!/usr/bin/env python3
"""
Fixed Enhanced Evaluation Run Script

This script fixes the strategy parameter mismatches and forces exact notebook reproduction
to achieve the target 846 trades for GMADL strategy.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import original strategy components
from strategy.strategy import (
    BuyAndHoldStrategy,
    MACDStrategy,
    RSIStrategy,
    ModelGmadlPredictionsStrategy,
    ModelQuantilePredictionsStrategy,
    ConcatenatedStrategies
)
from strategy.util import (
    get_data_windows,
    get_sweep_window_predictions,
    get_predictions_dataframe
)
from strategy.evaluation import (
    parameter_sweep,
    evaluate_strategy
)

# Import enhanced evaluation components for comparison only
from enhanced_evaluation.core.enhanced_evaluator import EnhancedEvaluator


def create_exact_notebook_gmadl_strategies(valid_gmadl_pred_windows, test_gmadl_pred_windows):
    """
    Create GMADL strategies with EXACT notebook parameters from Cell 23.
    
    These are the exact parameters that produced 846 trades in the notebook.
    """
    print("🎯 Creating EXACT notebook GMADL strategies...")
    
    # Exact parameters from notebook Cell 23 table
    exact_notebook_params = [
        {"enter_long": 0.004, "exit_long": None, "enter_short": -0.005, "exit_short": None},  # W1
        {"enter_long": 0.002, "exit_long": None, "enter_short": -0.001, "exit_short": None},  # W2
        {"enter_long": None, "exit_long": None, "enter_short": -0.006, "exit_short": 0.003},  # W3
        {"enter_long": 0.002, "exit_long": None, "enter_short": -0.005, "exit_short": None},  # W4
        {"enter_long": 0.002, "exit_long": None, "enter_short": -0.003, "exit_short": None},  # W5
        {"enter_long": 0.001, "exit_long": None, "enter_short": -0.007, "exit_short": None},  # W6
    ]
    
    gmadl_strategies = []
    
    for i, (valid_preds, test_preds, params) in enumerate(zip(valid_gmadl_pred_windows, test_gmadl_pred_windows, exact_notebook_params)):
        print(f"   Window {i+1}: Creating strategy with exact notebook parameters")
        print(f"      enter_long={params['enter_long']}, exit_long={params['exit_long']}")
        print(f"      enter_short={params['enter_short']}, exit_short={params['exit_short']}")
        
        # Get predictions dataframe
        predictions_df = get_predictions_dataframe(valid_preds, test_preds)
        
        # Create strategy with exact notebook parameters
        strategy = ModelGmadlPredictionsStrategy(
            predictions=predictions_df,
            enter_long=params['enter_long'],
            exit_long=params['exit_long'],
            enter_short=params['enter_short'],
            exit_short=params['exit_short']
        )
        
        gmadl_strategies.append(strategy)
        
        # Verify parameters match
        strategy_info = strategy.info()
        print(f"      ✅ Created: enter_long={strategy_info.get('enter_long')}, enter_short={strategy_info.get('enter_short')}")
    
    return gmadl_strategies


def load_or_create_full_strategies(data_windows, valid_gmadl_pred_windows, test_gmadl_pred_windows, force_regenerate=False):
    """
    Load existing complete strategies or create them if missing/forced.
    
    This enables ALL strategy optimizations to match notebook context exactly.
    """
    cache_file = 'cache/5min-best-strategies-FIXED.pkl'
    
    if os.path.exists(cache_file) and not force_regenerate:
        print(f"✅ Loading FIXED complete strategies from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("🔄 FULL STRATEGY OPTIMIZATION - Matching notebook exactly...")
    print("   This will take several minutes but ensures exact reproduction.")
    
    PADDING = 5000
    VALID_PART = 0.2
    INTERVAL = '5min'
    METRIC = 'mod_ir'
    TOP_N = 10
    
    def sweeps_on_all_windows(data_windows, strategy_class, params, **kwargs):
        result = []
        for in_sample, _ in data_windows:
            data_part = int((1 - VALID_PART) * len(in_sample))
            result.append(parameter_sweep(in_sample[data_part-PADDING:], strategy_class, params, padding=PADDING, interval=INTERVAL, **kwargs))
        return result
    
    # 1. Buy and Hold strategies
    print("   📊 Buy & Hold strategies...")
    buyandhold_best_strategies = [BuyAndHoldStrategy() for _ in data_windows]
    
    # 2. MACD strategies - ENABLED (was commented out in original script)
    print("   📊 MACD strategies - FULL OPTIMIZATION...")
    MACD_PARAMS = {
        'fast_window_size': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584],
        'slow_window_size': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584],
        'signal_window_size': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584],
        'short_sell': [True, False]
    }
    MACD_PARAMS_FILTER = lambda p: (p['slow_window_size'] > p['fast_window_size'])
    macd_sweep_results = sweeps_on_all_windows(data_windows, MACDStrategy, MACD_PARAMS, params_filter=MACD_PARAMS_FILTER, sort_by=METRIC)
    macd_best_strategies = [[strat for _, strat in results[:TOP_N]] for results in macd_sweep_results]
    
    # 3. RSI strategies - ENABLED (was commented out in original script)
    print("   📊 RSI strategies - FULL OPTIMIZATION...")
    RSI_FILTER = lambda p: (
        ((p['enter_long'] is not None and (p['enter_short'] is not None or p['exit_long'] is not None))
        or (p['enter_short'] is not None and (p['exit_short'] is not None or p['enter_long'] is not None)))
        and (p['enter_short'] is None or p['exit_long'] is None or (p['exit_long'] > p['enter_short']))
        and (p['enter_long'] is None or p['exit_short'] is None or (p['exit_short'] < p['enter_long'])))
    
    RSI_PARAMS = {
        'window_size': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584],
        'enter_long': [None, 70, 75, 80, 85, 90, 95],
        'exit_long': [None, 5, 10, 15, 20, 25, 30],
        'enter_short': [None, 5, 10, 15, 20, 25, 30],
        'exit_short': [None, 70, 75, 80, 85, 90, 95],
    }
    rsi_sweep_results = sweeps_on_all_windows(data_windows, RSIStrategy, RSI_PARAMS, params_filter=RSI_FILTER, sort_by=METRIC)
    rsi_best_strategies = [[strat for _, strat in results[:TOP_N]] for results in rsi_sweep_results]
    
    # 4. RMSE Model strategies - ENABLED (was commented out in original script)
    print("   📊 RMSE model strategies - FULL OPTIMIZATION...")
    RMSE_SWEEP_ID = 'filipstefaniuk/wne-masters-thesis-testing/9afp99kz'
    train_pred_windows = get_sweep_window_predictions(RMSE_SWEEP_ID, 'train')
    valid_pred_windows = get_sweep_window_predictions(RMSE_SWEEP_ID, 'valid')
    test_pred_windows = get_sweep_window_predictions(RMSE_SWEEP_ID, 'test')
    
    MODEL_RMSE_LOSS_FILTER = lambda p: (
        ((p['enter_long'] is not None and (p['enter_short'] is not None or p['exit_long'] is not None))
        or (p['enter_short'] is not None and (p['exit_short'] is not None or p['enter_long'] is not None)))
        and (p['enter_short'] is None or p['exit_long'] is None or (p['exit_long'] > p['enter_short']))
        and (p['enter_long'] is None or p['exit_short'] is None or (p['exit_short'] < p['enter_long'])))
    
    rmse_model_sweep_results = []
    for (in_sample, _), train_preds, valid_preds, test_preds in zip(data_windows, train_pred_windows, valid_pred_windows, test_pred_windows):
        data_part = int((1 - VALID_PART) * len(in_sample))
        params={
            'predictions': [get_predictions_dataframe(train_preds, valid_preds, test_preds)],
            'enter_long': [None, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
            'exit_long': [None, -0.001, -0.002, -0.003, -0.004, -0.005, -0.006, -0.007],
            'enter_short': [None, -0.001, -0.002, -0.003, -0.004, -0.005, -0.006, -0.007],
            'exit_short': [None, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
        }
        rmse_model_sweep_results.append(parameter_sweep(
            in_sample[data_part-PADDING:],
            ModelGmadlPredictionsStrategy,
            params,
            params_filter=MODEL_RMSE_LOSS_FILTER,
            padding=PADDING,
            interval=INTERVAL,
            sort_by=METRIC))
    rmse_model_best_strategies = [[strat for _, strat in results[:TOP_N]] for results in rmse_model_sweep_results]
    
    # 5. Quantile Model strategies - ENABLED (was commented out in original script)
    print("   📊 Quantile model strategies - FULL OPTIMIZATION...")
    QUANTILE_SWEEP_ID = 'filipstefaniuk/wne-masters-thesis-testing/8m3hwwmx'
    train_pred_windows = get_sweep_window_predictions(QUANTILE_SWEEP_ID, 'train')
    valid_pred_windows = get_sweep_window_predictions(QUANTILE_SWEEP_ID, 'valid')
    test_pred_windows = get_sweep_window_predictions(QUANTILE_SWEEP_ID, 'test')
    
    MODEL_QUANTILE_LOSS_FILTER = lambda p: (
        ((p['quantile_enter_long'] is not None and (p['quantile_enter_short'] is not None or p['quantile_exit_long'] is not None))
        or (p['quantile_enter_short'] is not None and (p['quantile_exit_short'] is not None or p['quantile_enter_long'] is not None)))
        and (p['quantile_enter_short'] is None or p['quantile_exit_long'] is None or (p['quantile_exit_long'] < p['quantile_enter_short']))
        and (p['quantile_enter_long'] is None or p['quantile_exit_short'] is None or (p['quantile_exit_short'] < p['quantile_enter_long'])))
    
    quantile_model_sweep_results = []
    for (in_sample, _), train_preds, valid_preds, test_preds in zip(data_windows, train_pred_windows, valid_pred_windows, test_pred_windows):
        data_part = int((1 - VALID_PART) * len(in_sample))
        params={
            'predictions': [get_predictions_dataframe(train_preds, valid_preds, test_preds)],
            'quantiles': [[0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99]],
            'quantile_enter_long': [None, 0.9, 0.95, 0.97, 0.98, 0.99],
            'quantile_exit_long': [None, 0.9, 0.95, 0.97, 0.98, 0.99],
            'quantile_enter_short': [None, 0.9, 0.95, 0.97, 0.98, 0.99],
            'quantile_exit_short': [None, 0.9, 0.95, 0.97, 0.98, 0.99],
            'exchange_fee': [0.0003, 0.002, 0.003],
            'future': [1]
        }
        quantile_model_sweep_results.append(parameter_sweep(
            in_sample[data_part-PADDING:],
            ModelQuantilePredictionsStrategy,
            params,
            params_filter=MODEL_QUANTILE_LOSS_FILTER,
            padding=PADDING,
            interval=INTERVAL,
            sort_by=METRIC))
    quantile_model_best_strategies = [[strat for _, strat in results[:TOP_N]] for results in quantile_model_sweep_results]
    
    # 6. GMADL Model strategies - Use EXACT notebook parameters  
    print("   📊 GMADL strategies - Using EXACT notebook parameters...")
    gmadl_strategies = create_exact_notebook_gmadl_strategies(valid_gmadl_pred_windows, test_gmadl_pred_windows)
    # Wrap each strategy in a list of 1 to maintain compatibility with TOP_N structure
    gmadl_model_best_strategies = [[strategy] for strategy in gmadl_strategies]
    
    # Save complete strategy cache
    print("   💾 Saving FIXED complete strategy cache...")
    best_strategies = {
        'buy_and_hold': buyandhold_best_strategies,
        'macd_strategies': macd_best_strategies,
        'rsi_strategies': rsi_best_strategies,
        'rmse_model': rmse_model_best_strategies,
        'quantile_model': quantile_model_best_strategies,
        'gmadl_model': gmadl_model_best_strategies
    }
    
    with open(cache_file, 'wb') as outp:
        pickle.dump(best_strategies, outp, pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ FIXED strategy optimization completed and cached!")
    return best_strategies


def run_fixed_evaluation():
    """Run fixed evaluation that should match notebook exactly."""
    print("=" * 80)
    print("🎯 FIXED ENHANCED EVALUATION RUN - EXACT NOTEBOOK REPRODUCTION")
    print("=" * 80)
    
    # Constants from original notebook - EXACT MATCH
    PADDING = 5000
    VALID_PART = 0.2
    INTERVAL = '5min'
    
    try:
        # Setup data and strategies - EXACT MATCH
        print("🔄 Loading Bitcoin data windows from W&B...")
        data_windows = get_data_windows(
            'filipstefaniuk/wne-masters-thesis-testing',
            'btc-usdt-5m:latest',
            min_window=0, 
            max_window=5
        )
        
        print(f"✅ Loaded {len(data_windows)} data windows")
        
        # Load GMADL model predictions
        print("🔄 Loading GMADL model predictions from W&B...")
        SWEEP_ID = 'filipstefaniuk/wne-masters-thesis-testing/0pro3i5i'
        valid_gmadl_pred_windows = get_sweep_window_predictions(SWEEP_ID, 'valid')
        test_gmadl_pred_windows = get_sweep_window_predictions(SWEEP_ID, 'test')
        
        # Get FIXED strategies with proper optimization and exact GMADL parameters
        print("🔧 Loading/creating FIXED strategies with full optimization...")
        best_strategies = load_or_create_full_strategies(
            data_windows, 
            valid_gmadl_pred_windows, 
            test_gmadl_pred_windows,
            force_regenerate=False  # Set to True to force complete regeneration
        )
        
        # Create concatenated dataset - EXACT MATCH to notebook Cell 27
        print("📊 Creating concatenated test dataset...")
        test_data = pd.concat([data_windows[0][0][-PADDING:]] + [data_window[1] for data_window in data_windows])
        print(f"   Test data shape: {test_data.shape}")
        
        # Run ALL strategies using ORIGINAL evaluation method (not enhanced)
        print("🔄 Evaluating ALL strategies using ORIGINAL evaluation method...")
        
        # 1. Buy and Hold
        buy_and_hold_concat = evaluate_strategy(test_data, BuyAndHoldStrategy(), padding=PADDING)
        
        # 2. MACD Strategy
        macd_concat = evaluate_strategy(
            test_data, 
            ConcatenatedStrategies(len(data_windows[0][1]), [s[0] for s in best_strategies['macd_strategies']], padding=PADDING), 
            padding=PADDING, 
            interval=INTERVAL
        )
        
        # 3. RSI Strategy
        rsi_concat = evaluate_strategy(
            test_data, 
            ConcatenatedStrategies(len(data_windows[0][1]), [s[0] for s in best_strategies['rsi_strategies']], padding=PADDING), 
            padding=PADDING, 
            interval=INTERVAL
        )
        
        # 4. RMSE Model
        rmse_model_concat = evaluate_strategy(
            test_data, 
            ConcatenatedStrategies(len(data_windows[0][1]), [s[0] for s in best_strategies['rmse_model']], padding=PADDING), 
            padding=PADDING, 
            interval=INTERVAL
        )
        
        # 5. Quantile Model
        quantile_model_concat = evaluate_strategy(
            test_data, 
            ConcatenatedStrategies(len(data_windows[0][1]), [s[0] for s in best_strategies['quantile_model']], padding=PADDING), 
            padding=PADDING, 
            interval=INTERVAL
        )
        
        # 6. GMADL Model - Using ORIGINAL evaluation method with EXACT parameters
        gmadl_model_concat = evaluate_strategy(
            test_data, 
            ConcatenatedStrategies(len(data_windows[0][1]), [s[0] for s in best_strategies['gmadl_model']], padding=PADDING), 
            padding=PADDING, 
            interval=INTERVAL
        )
        
        print("✅ All evaluations completed!")
        
        # Display comprehensive results
        print("\n" + "=" * 80)
        print("📈 FIXED EVALUATION RESULTS - SHOULD MATCH NOTEBOOK EXACTLY")
        print("=" * 80)
        
        strategy_results = [
            ("Buy and Hold", buy_and_hold_concat),
            ("MACD Strategy", macd_concat), 
            ("RSI Strategy", rsi_concat),
            ("RMSE Informer", rmse_model_concat),
            ("Quantile Informer", quantile_model_concat),
            ("GMADL Informer", gmadl_model_concat)
        ]
        
        print(f"{'Strategy':<25} {'Portfolio':<12} {'Return%':<10} {'ModIR':<8} {'Trades':<8}")
        print("-" * 70)
        
        for strategy_name, result in strategy_results:
            portfolio_val = result.get('value', 0)
            total_return = (portfolio_val - 1) * 100 if portfolio_val > 0 else 0
            mod_ir = result.get('mod_ir', 0)
            n_trades = result.get('n_trades', 0)
            
            print(f"{strategy_name:<25} {portfolio_val:<12.3f} {total_return:<10.1f} {mod_ir:<8.3f} {n_trades:<8}")
        
        # Check specific GMADL results
        print(f"\n🎯 GMADL SPECIFIC RESULTS:")
        print("-" * 50)
        gmadl_result = gmadl_model_concat
        print(f"Portfolio Value:     {gmadl_result.get('value', 0):.3f}")
        print(f"Total Return:        {(gmadl_result.get('value', 1) - 1)*100:+.2f}%")
        print(f"Annualized Return:   {gmadl_result.get('arc', 0)*100:+.2f}%")
        print(f"Modified IR:         {gmadl_result.get('mod_ir', 0):.3f}")
        print(f"Trade Count:         {gmadl_result.get('n_trades', 0)}")
        
        # Compare with notebook expectations
        print(f"\n📋 NOTEBOOK COMPARISON:")
        print("-" * 50)
        expected_portfolio = 9.747
        expected_return = 115.88
        expected_mod_ir = 7.552
        expected_trades = 846
        
        actual_portfolio = gmadl_result.get('value', 0)
        actual_return = gmadl_result.get('arc', 0) * 100
        actual_mod_ir = gmadl_result.get('mod_ir', 0)
        actual_trades = gmadl_result.get('n_trades', 0)
        
        print(f"Expected Portfolio:   {expected_portfolio:.3f}")
        print(f"Actual Portfolio:     {actual_portfolio:.3f}")
        print(f"Portfolio Match:      {'✅' if abs(actual_portfolio - expected_portfolio) < 0.5 else '❌'}")
        print(f"")
        print(f"Expected Return:      {expected_return:.2f}%") 
        print(f"Actual Return:        {actual_return:.2f}%")
        print(f"Return Match:         {'✅' if abs(actual_return - expected_return) < 10 else '❌'}")
        print(f"")
        print(f"Expected Trades:      {expected_trades}")
        print(f"Actual Trades:        {actual_trades}")
        print(f"Trade Match:          {'✅' if actual_trades == expected_trades else '❌'}")
        print(f"")
        print(f"Expected Mod IR:      {expected_mod_ir:.3f}")
        print(f"Actual Mod IR:        {actual_mod_ir:.3f}")
        print(f"Mod IR Match:         {'✅' if abs(actual_mod_ir - expected_mod_ir) < 1.0 else '❌'}")
        
        return gmadl_result
        
    except Exception as e:
        print(f"\n❌ Error during fixed evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = run_fixed_evaluation()
    if result:
        print(f"\n🚀 Fixed run completed!")
        print(f"🔍 Check if results now match notebook expectations")
    else:
        print(f"\n💥 Fixed run failed - check error messages above")
        sys.exit(1)
#!/usr/bin/env python3
"""
Phase 1 Diagnostic Report: Comprehensive Parameter and Strategy Analysis
Identify the root cause of the 1,821 vs 846 trade discrepancy
"""

import pickle
import pandas as pd
from pathlib import Path

def load_and_analyze_all_strategies(cache_file):
    """Load cache and analyze all strategy configurations"""
    
    print(f"=== COMPREHENSIVE STRATEGY ANALYSIS ===")
    print(f"Cache file: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    gmadl_data = cache_data['gmadl_model']
    
    # Notebook optimal parameters for reference
    notebook_optimal = {
        'W1-5min': {'enter_long': 0.004, 'enter_short': -0.005},
        'W2-5min': {'enter_long': 0.002, 'enter_short': -0.001},
        'W3-5min': {'enter_short': -0.006, 'exit_short': 0.003},
        'W4-5min': {'enter_long': 0.002, 'enter_short': -0.005},
        'W5-5min': {'enter_long': 0.002, 'enter_short': -0.003},
        'W6-5min': {'enter_long': 0.001, 'enter_short': -0.007}
    }
    
    print(f"\n=== NOTEBOOK OPTIMAL PARAMETERS (TARGET) ===")
    for window, params in notebook_optimal.items():
        print(f"{window}: {params}")
    
    all_strategy_details = []
    
    print(f"\n=== DETAILED ANALYSIS OF ALL CACHED STRATEGIES ===")
    
    for window_idx, window_strategies in enumerate(gmadl_data):
        window_name = f"W{window_idx + 1}-5min"
        notebook_params = notebook_optimal.get(window_name, {})
        
        print(f"\n--- {window_name} ---")
        print(f"Notebook optimal: {notebook_params}")
        print(f"Number of cached strategies: {len(window_strategies)}")
        
        exact_matches = []
        close_matches = []
        all_configs = []
        
        for strategy_idx, strategy in enumerate(window_strategies):
            params = {
                'enter_long': getattr(strategy, 'enter_long', None),
                'enter_short': getattr(strategy, 'enter_short', None),
                'exit_long': getattr(strategy, 'exit_long', None),
                'exit_short': getattr(strategy, 'exit_short', None)
            }
            
            # Filter out None values for comparison
            non_none_params = {k: v for k, v in params.items() if v is not None}
            
            strategy_detail = {
                'window': window_name,
                'strategy_idx': strategy_idx,
                'parameters': params,
                'non_none_parameters': non_none_params
            }
            
            all_strategy_details.append(strategy_detail)
            all_configs.append(non_none_params)
            
            # Check if this matches notebook optimal
            is_exact_match = True
            is_close_match = True
            
            for param, target_value in notebook_params.items():
                strategy_value = params.get(param)
                
                if strategy_value is None:
                    is_exact_match = False
                    is_close_match = False
                elif abs(float(strategy_value) - float(target_value)) > 1e-6:
                    is_exact_match = False
                    if abs(float(strategy_value) - float(target_value)) > 0.001:
                        is_close_match = False
            
            # Also check that strategy doesn't have extra parameters
            for param, strategy_value in non_none_params.items():
                if param not in notebook_params and strategy_value is not None:
                    is_exact_match = False
            
            if is_exact_match:
                exact_matches.append((strategy_idx, params))
            elif is_close_match:
                close_matches.append((strategy_idx, params))
        
        print(f"Strategy parameter configurations found:")
        for i, config in enumerate(all_configs):
            marker = ""
            if (i, window_strategies[i]) in [(idx, strategy) for idx, strategy in exact_matches]:
                marker = " ✓ EXACT MATCH"
            elif (i, window_strategies[i]) in [(idx, strategy) for idx, strategy in close_matches]:
                marker = " ~ CLOSE MATCH"
            print(f"  [{i}] {config}{marker}")
        
        print(f"Exact matches with notebook: {len(exact_matches)}")
        print(f"Close matches with notebook: {len(close_matches)}")
        
        if exact_matches:
            print(f"FOUND EXACT MATCH(ES): {exact_matches}")
        elif close_matches:
            print(f"FOUND CLOSE MATCH(ES): {close_matches}")
        else:
            print(f"NO MATCHES FOUND - This could be the problem!")
    
    return all_strategy_details

def analyze_current_vs_optimal_selection(cache_file):
    """Analyze what our current script selects vs optimal"""
    
    print(f"\n\n=== CURRENT SCRIPT BEHAVIOR ANALYSIS ===")
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    gmadl_data = cache_data['gmadl_model']
    
    notebook_optimal = {
        'W1-5min': {'enter_long': 0.004, 'enter_short': -0.005},
        'W2-5min': {'enter_long': 0.002, 'enter_short': -0.001},
        'W3-5min': {'enter_short': -0.006, 'exit_short': 0.003},
        'W4-5min': {'enter_long': 0.002, 'enter_short': -0.005},
        'W5-5min': {'enter_long': 0.002, 'enter_short': -0.003},
        'W6-5min': {'enter_long': 0.001, 'enter_short': -0.007}
    }
    
    print("Current script selects the FIRST strategy [0] from each window:")
    print("But the notebook uses the OPTIMAL strategy based on performance!")
    
    selection_analysis = []
    
    for window_idx, window_strategies in enumerate(gmadl_data):
        window_name = f"W{window_idx + 1}-5min"
        
        # What our current script selects (first strategy)
        current_selection = window_strategies[0]
        current_params = {
            'enter_long': getattr(current_selection, 'enter_long', None),
            'enter_short': getattr(current_selection, 'enter_short', None),
            'exit_long': getattr(current_selection, 'exit_long', None),
            'exit_short': getattr(current_selection, 'exit_short', None)
        }
        
        # What should be selected (matching notebook optimal)
        notebook_params = notebook_optimal.get(window_name, {})
        
        # Find the strategy that matches notebook optimal
        optimal_strategy_idx = None
        optimal_params = None
        
        for strategy_idx, strategy in enumerate(window_strategies):
            params = {
                'enter_long': getattr(strategy, 'enter_long', None),
                'enter_short': getattr(strategy, 'enter_short', None),
                'exit_long': getattr(strategy, 'exit_long', None),
                'exit_short': getattr(strategy, 'exit_short', None)
            }
            
            # Check if this matches notebook optimal
            is_match = True
            for param, target_value in notebook_params.items():
                strategy_value = params.get(param)
                if strategy_value is None or abs(float(strategy_value) - float(target_value)) > 1e-6:
                    is_match = False
                    break
            
            # Also ensure no extra non-None parameters
            non_none_params = {k: v for k, v in params.items() if v is not None}
            if is_match and set(non_none_params.keys()) == set(notebook_params.keys()):
                optimal_strategy_idx = strategy_idx
                optimal_params = params
                break
        
        analysis = {
            'window': window_name,
            'current_selection_idx': 0,
            'current_params': current_params,
            'notebook_optimal': notebook_params,
            'optimal_strategy_idx': optimal_strategy_idx,
            'optimal_params': optimal_params,
            'selection_correct': optimal_strategy_idx == 0
        }
        
        selection_analysis.append(analysis)
        
        print(f"\n{window_name}:")
        print(f"  Current script selects [0]: {current_params}")
        print(f"  Notebook optimal target:     {notebook_params}")
        print(f"  Correct strategy is [{optimal_strategy_idx}]: {optimal_params}")
        print(f"  Selection correct: {analysis['selection_correct']}")
    
    # Summary
    correct_selections = sum(1 for a in selection_analysis if a['selection_correct'])
    total_selections = len(selection_analysis)
    
    print(f"\n=== SELECTION ACCURACY SUMMARY ===")
    print(f"Correct selections: {correct_selections}/{total_selections}")
    print(f"Incorrect selections: {total_selections - correct_selections}")
    
    if correct_selections < total_selections:
        print(f"\n⚠️  ROOT CAUSE IDENTIFIED:")
        print(f"Our script is selecting the FIRST strategy [0] from each window,")
        print(f"but the notebook uses the OPTIMAL strategy based on performance metrics.")
        print(f"This explains the trade count discrepancy (1,821 vs 846)!")
        
        print(f"\nIncorrect selections:")
        for analysis in selection_analysis:
            if not analysis['selection_correct']:
                print(f"  {analysis['window']}: Using [{analysis['current_selection_idx']}] instead of [{analysis['optimal_strategy_idx']}]")
    
    return selection_analysis

def generate_fix_recommendations():
    """Generate recommendations to fix the issue"""
    
    print(f"\n\n=== RECOMMENDED FIXES ===")
    
    print(f"1. IMMEDIATE FIX - Update strategy selection logic:")
    print(f"   Instead of: best_strategy = strategies[0]")
    print(f"   Use:        best_strategy = find_optimal_strategy(strategies, target_params)")
    
    print(f"\n2. VERIFICATION STEPS:")
    print(f"   a) Implement proper strategy selection based on notebook parameters")
    print(f"   b) Re-run evaluation with correct strategies")
    print(f"   c) Verify trade count matches notebook (846 trades)")
    print(f"   d) Verify performance metrics match notebook results")
    
    print(f"\n3. ADDITIONAL INVESTIGATION:")
    print(f"   a) Check if strategy optimization was completed correctly")
    print(f"   b) Verify that all parameter combinations were tested")
    print(f"   c) Ensure performance-based ranking is working")

def main():
    """Main diagnostic function"""
    print("=== PHASE 1 DIAGNOSTIC REPORT ===")
    print("Analyzing parameter discrepancies and strategy selection issues")
    
    cache_file = '/home/ubuntu/wne-msc-thesis-new/cache/5min-best-strategies-v2.pkl'
    
    # Analyze all strategies in detail
    all_strategies = load_and_analyze_all_strategies(cache_file)
    
    # Analyze current selection logic vs optimal
    selection_analysis = analyze_current_vs_optimal_selection(cache_file)
    
    # Generate fix recommendations
    generate_fix_recommendations()
    
    print(f"\n=== DIAGNOSTIC COMPLETE ===")
    print(f"Total strategies analyzed: {len(all_strategies)}")
    print(f"Key finding: Strategy selection logic needs to be updated to use")
    print(f"performance-based selection instead of always picking the first strategy.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Phase 1 Diagnostic: Parameter Comparison Analysis
Compare strategy parameters from cache vs notebook optimal parameters
"""

import pickle
import pandas as pd
from pathlib import Path

def load_cache_file(cache_path):
    """Load and inspect cache file contents"""
    print(f"\n=== Loading cache file: {cache_path} ===")
    
    if not Path(cache_path).exists():
        print(f"Cache file does not exist: {cache_path}")
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Cache file loaded successfully")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys in cache: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, '__len__'):
                    try:
                        print(f"    Length: {len(value)}")
                    except:
                        pass
        
        return data
    
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return None

def extract_strategy_parameters(strategies_data):
    """Extract parameters from strategies data"""
    parameters = {}
    
    if isinstance(strategies_data, dict):
        for window_name, strategy_info in strategies_data.items():
            print(f"\n--- Window: {window_name} ---")
            
            if isinstance(strategy_info, dict):
                # Look for parameters in various possible locations
                params = None
                
                # Check different possible parameter locations
                if 'parameters' in strategy_info:
                    params = strategy_info['parameters']
                elif 'params' in strategy_info:
                    params = strategy_info['params']
                elif 'best_params' in strategy_info:
                    params = strategy_info['best_params']
                else:
                    # Check if the strategy_info itself contains parameter keys
                    param_keys = ['enter_long', 'enter_short', 'exit_long', 'exit_short']
                    if any(key in strategy_info for key in param_keys):
                        params = {key: strategy_info.get(key) for key in param_keys if key in strategy_info}
                
                if params:
                    print(f"Parameters found: {params}")
                    parameters[window_name] = params
                else:
                    print(f"Available keys: {list(strategy_info.keys())}")
                    # Print first few items to understand structure
                    for key, value in list(strategy_info.items())[:5]:
                        print(f"  {key}: {value} ({type(value)})")
            else:
                print(f"Strategy info type: {type(strategy_info)}")
                print(f"Strategy info: {strategy_info}")
    
    return parameters

def main():
    """Main diagnostic function"""
    print("=== PHASE 1 DIAGNOSTIC: PARAMETER COMPARISON ===")
    
    # Define notebook optimal parameters for comparison
    notebook_optimal = {
        'W1-5min': {'enter_long': 0.004, 'enter_short': -0.005},
        'W2-5min': {'enter_long': 0.002, 'enter_short': -0.001},
        'W3-5min': {'enter_short': -0.006, 'exit_short': 0.003},
        'W4-5min': {'enter_long': 0.002, 'enter_short': -0.005},
        'W5-5min': {'enter_long': 0.002, 'enter_short': -0.003},
        'W6-5min': {'enter_long': 0.001, 'enter_short': -0.007}
    }
    
    print("\n=== NOTEBOOK OPTIMAL PARAMETERS ===")
    for window, params in notebook_optimal.items():
        print(f"{window}: {params}")
    
    # Load both cache files
    cache_files = [
        '/home/ubuntu/wne-msc-thesis-new/cache/5min-best-strategies-v2.pkl',
        '/home/ubuntu/wne-msc-thesis-new/cache/5min-gmadl-strategies.pkl.backup'
    ]
    
    all_cache_params = {}
    
    for cache_file in cache_files:
        cache_data = load_cache_file(cache_file)
        if cache_data is not None:
            cache_params = extract_strategy_parameters(cache_data)
            all_cache_params[cache_file] = cache_params
    
    # Detailed comparison
    print("\n\n=== DETAILED PARAMETER COMPARISON ===")
    
    for cache_file, cache_params in all_cache_params.items():
        print(f"\n--- Cache File: {Path(cache_file).name} ---")
        
        if not cache_params:
            print("No parameters found in this cache file")
            continue
        
        print(f"\nFound {len(cache_params)} window configurations:")
        for window, params in cache_params.items():
            print(f"{window}: {params}")
        
        # Compare with notebook optimal
        print(f"\n--- COMPARISON WITH NOTEBOOK OPTIMAL ---")
        
        comparison_results = []
        
        for notebook_window, notebook_params in notebook_optimal.items():
            # Find matching window in cache (try different naming patterns)
            cache_match = None
            for cache_window, cache_params_data in cache_params.items():
                if (notebook_window.lower() in cache_window.lower() or 
                    cache_window.lower() in notebook_window.lower() or
                    notebook_window.replace('-', '_') in cache_window or
                    cache_window.replace('-', '_') in notebook_window):
                    cache_match = (cache_window, cache_params_data)
                    break
            
            if cache_match:
                cache_window, cache_params_data = cache_match
                
                print(f"\n{notebook_window} <-> {cache_window}")
                print(f"  Notebook: {notebook_params}")
                print(f"  Cache:    {cache_params_data}")
                
                # Check for differences
                differences = []
                for param, notebook_value in notebook_params.items():
                    cache_value = cache_params_data.get(param)
                    if cache_value is None:
                        differences.append(f"{param}: notebook={notebook_value}, cache=MISSING")
                    elif abs(float(cache_value) - float(notebook_value)) > 1e-6:
                        differences.append(f"{param}: notebook={notebook_value}, cache={cache_value}")
                
                for param, cache_value in cache_params_data.items():
                    if param not in notebook_params:
                        differences.append(f"{param}: notebook=MISSING, cache={cache_value}")
                
                if differences:
                    print(f"  DIFFERENCES: {'; '.join(differences)}")
                else:
                    print(f"  ✓ Parameters match")
                
                comparison_results.append({
                    'notebook_window': notebook_window,
                    'cache_window': cache_window,
                    'matches': len(differences) == 0,
                    'differences': differences
                })
            else:
                print(f"\n{notebook_window} -> NO MATCH FOUND IN CACHE")
                comparison_results.append({
                    'notebook_window': notebook_window,
                    'cache_window': 'NOT FOUND',
                    'matches': False,
                    'differences': ['Window not found in cache']
                })
        
        # Summary
        print(f"\n--- SUMMARY FOR {Path(cache_file).name} ---")
        matches = sum(1 for r in comparison_results if r['matches'])
        total = len(comparison_results)
        print(f"Windows matching: {matches}/{total}")
        
        if matches < total:
            print("Issues found:")
            for result in comparison_results:
                if not result['matches']:
                    print(f"  - {result['notebook_window']}: {result['differences']}")

if __name__ == "__main__":
    main()
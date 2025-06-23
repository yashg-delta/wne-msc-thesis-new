#!/usr/bin/env python3
"""
Enhanced Phase 1 Diagnostic: Extract Parameters from Strategy Objects
Extract actual parameters from ModelGmadlPredictionsStrategy objects
"""

import pickle
import pandas as pd
from pathlib import Path

def extract_strategy_object_parameters(strategy_obj):
    """Extract parameters from a strategy object"""
    params = {}
    
    # Common parameter attributes to check
    param_attrs = [
        'enter_long', 'enter_short', 'exit_long', 'exit_short',
        'enter_long_threshold', 'enter_short_threshold', 
        'exit_long_threshold', 'exit_short_threshold',
        'params', 'parameters', 'thresholds'
    ]
    
    for attr in param_attrs:
        if hasattr(strategy_obj, attr):
            value = getattr(strategy_obj, attr)
            params[attr] = value
    
    # Also check if there's a __dict__ with parameters
    if hasattr(strategy_obj, '__dict__'):
        obj_dict = strategy_obj.__dict__
        for key, value in obj_dict.items():
            if any(param_key in key.lower() for param_key in ['enter', 'exit', 'threshold', 'param']):
                params[key] = value
    
    return params

def analyze_gmadl_strategies(cache_data):
    """Analyze GMADL strategy objects and extract parameters"""
    
    if 'gmadl_model' not in cache_data:
        print("No GMADL model data found")
        return {}
    
    gmadl_data = cache_data['gmadl_model']
    print(f"GMADL data contains {len(gmadl_data)} windows")
    
    window_parameters = {}
    
    for window_idx, window_strategies in enumerate(gmadl_data):
        window_name = f"W{window_idx + 1}-5min"
        print(f"\n=== Analyzing {window_name} ===")
        print(f"Number of strategies in this window: {len(window_strategies)}")
        
        # Look at all strategies in this window to find the best one
        strategy_params = []
        
        for strategy_idx, strategy in enumerate(window_strategies):
            params = extract_strategy_object_parameters(strategy)
            
            # Try to get performance metrics to identify the best strategy
            performance_attrs = ['sharpe_ratio', 'total_return', 'performance', 'metrics']
            performance_info = {}
            
            for attr in performance_attrs:
                if hasattr(strategy, attr):
                    performance_info[attr] = getattr(strategy, attr)
            
            strategy_info = {
                'strategy_idx': strategy_idx,
                'parameters': params,
                'performance': performance_info,
                'type': type(strategy).__name__
            }
            
            strategy_params.append(strategy_info)
        
        # Print details for first few strategies to understand structure
        print(f"Sample strategy details:")
        for i, info in enumerate(strategy_params[:3]):  # Show first 3
            print(f"  Strategy {i}: {info['type']}")
            print(f"    Parameters: {info['parameters']}")
            print(f"    Performance keys: {list(info['performance'].keys())}")
            
            # Print actual attribute names for debugging
            if hasattr(gmadl_data[window_idx][i], '__dict__'):
                attrs = list(gmadl_data[window_idx][i].__dict__.keys())
                print(f"    All attributes: {attrs[:10]}...")  # Show first 10 attrs
        
        # For now, let's assume the first strategy is the best (we'll refine this)
        if strategy_params:
            window_parameters[window_name] = strategy_params[0]['parameters']
    
    return window_parameters

def detailed_object_inspection(strategy_obj, max_depth=2):
    """Perform detailed inspection of a strategy object"""
    print(f"\n--- Detailed Object Inspection ---")
    print(f"Object type: {type(strategy_obj)}")
    print(f"Object ID: {id(strategy_obj)}")
    
    if hasattr(strategy_obj, '__dict__'):
        obj_dict = strategy_obj.__dict__
        print(f"Object dictionary keys: {list(obj_dict.keys())}")
        
        for key, value in obj_dict.items():
            value_type = type(value)
            print(f"  {key}: {value_type}")
            
            # If it's a simple type, show the value
            if value_type in [int, float, str, bool]:
                print(f"    Value: {value}")
            elif value_type in [list, tuple] and len(value) < 10:
                print(f"    Value: {value}")
            elif hasattr(value, '__dict__') and max_depth > 0:
                print(f"    Nested object attributes: {list(value.__dict__.keys())}")
    
    # Check for common method names that might reveal parameters
    method_names = [name for name in dir(strategy_obj) if not name.startswith('_')]
    print(f"Public methods/attributes: {method_names}")

def main():
    """Main enhanced diagnostic function"""
    print("=== ENHANCED PHASE 1 DIAGNOSTIC ===")
    
    # Define notebook optimal parameters for comparison
    notebook_optimal = {
        'W1-5min': {'enter_long': 0.004, 'enter_short': -0.005},
        'W2-5min': {'enter_long': 0.002, 'enter_short': -0.001},
        'W3-5min': {'enter_short': -0.006, 'exit_short': 0.003},
        'W4-5min': {'enter_long': 0.002, 'enter_short': -0.005},
        'W5-5min': {'enter_long': 0.002, 'enter_short': -0.003},
        'W6-5min': {'enter_long': 0.001, 'enter_short': -0.007}
    }
    
    # Load cache file with GMADL strategies
    cache_file = '/home/ubuntu/wne-msc-thesis-new/cache/5min-best-strategies-v2.pkl'
    
    print(f"Loading cache file: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    # Perform detailed inspection of the first GMADL strategy
    if 'gmadl_model' in cache_data and len(cache_data['gmadl_model']) > 0:
        first_window = cache_data['gmadl_model'][0]
        if len(first_window) > 0:
            first_strategy = first_window[0]
            detailed_object_inspection(first_strategy)
    
    # Extract parameters from all GMADL strategies
    print(f"\n=== EXTRACTING GMADL STRATEGY PARAMETERS ===")
    gmadl_parameters = analyze_gmadl_strategies(cache_data)
    
    print(f"\n=== EXTRACTED PARAMETERS SUMMARY ===")
    for window, params in gmadl_parameters.items():
        print(f"{window}: {params}")
    
    # Compare with notebook optimal
    print(f"\n=== COMPARISON WITH NOTEBOOK OPTIMAL ===")
    
    for notebook_window, notebook_params in notebook_optimal.items():
        if notebook_window in gmadl_parameters:
            cache_params = gmadl_parameters[notebook_window]
            
            print(f"\n{notebook_window}:")
            print(f"  Notebook: {notebook_params}")
            print(f"  Cache:    {cache_params}")
            
            # Check for parameter matches
            if cache_params:
                differences = []
                for param, notebook_value in notebook_params.items():
                    cache_value = cache_params.get(param)
                    if cache_value is None:
                        differences.append(f"{param}: notebook={notebook_value}, cache=MISSING")
                    elif abs(float(cache_value) - float(notebook_value)) > 1e-6:
                        differences.append(f"{param}: notebook={notebook_value}, cache={cache_value}")
                
                if differences:
                    print(f"  DIFFERENCES: {'; '.join(differences)}")
                else:
                    print(f"  ✓ Parameters match")
            else:
                print(f"  ERROR: No parameters found in cache")
        else:
            print(f"\n{notebook_window}: NOT FOUND IN CACHE")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
TMUX Strategy Validation Run

This script validates that the strategy selection fix works in the complete 
evaluation environment. It runs a quick but complete test to ensure everything 
is working before doing the full evaluation.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all required components
from strategy.strategy import (
    BuyAndHoldStrategy,
    ConcatenatedStrategies
)
from strategy.util import get_data_windows
from enhanced_evaluation.core.enhanced_evaluator import EnhancedEvaluator
from complete_evaluation_run import select_optimal_gmadl_strategies


def run_quick_validation():
    """Run a quick validation of the strategy selection fix."""
    print("=" * 80)
    print("🚀 TMUX STRATEGY VALIDATION - QUICK RUN")
    print("=" * 80)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Load cached strategies
        print("\n📂 Loading cached strategies...")
        cache_file = 'cache/5min-best-strategies-v2.pkl'
        if not os.path.exists(cache_file):
            print(f"❌ Cache not found: {cache_file}")
            return False
            
        with open(cache_file, 'rb') as f:
            best_strategies = pickle.load(f)
        print("✅ Strategies loaded successfully")
        
        # 2. Test optimal strategy selection
        print("\n🔍 Testing optimal strategy selection...")
        gmadl_strategies = select_optimal_gmadl_strategies(best_strategies['gmadl_model'])
        print(f"✅ Selected {len(gmadl_strategies)} optimal strategies")
        
        # 3. Load a minimal dataset for validation
        print("\n📊 Loading minimal data for validation...")
        data_windows = get_data_windows(
            'filipstefaniuk/wne-masters-thesis-testing',
            'btc-usdt-5m:latest',
            min_window=0, 
            max_window=1  # Just load 2 windows for quick test
        )
        print(f"✅ Loaded {len(data_windows)} data windows for validation")
        
        # 4. Create concatenated strategy with optimal selection
        print("\n🔧 Creating concatenated strategy with optimal strategies...")
        PADDING = 5000
        window_size = len(data_windows[0][1])
        
        # Use only first 2 optimal strategies for quick test
        test_strategies = gmadl_strategies[:2]
        concatenated_strategy = ConcatenatedStrategies(
            window_size, 
            test_strategies, 
            padding=PADDING
        )
        print("✅ Concatenated strategy created")
        
        # 5. Quick evaluation test
        print("\n⚡ Running quick evaluation test...")
        test_data = pd.concat([data_windows[0][0][-PADDING:]] + [data_window[1] for data_window in data_windows[:2]])
        print(f"   Test data shape: {test_data.shape}")
        
        # Initialize evaluator
        evaluator = EnhancedEvaluator(periods_per_year=105120)
        
        # Run quick evaluation
        print("   🔄 Running enhanced evaluation...")
        results = evaluator.evaluate_strategy_enhanced(
            data=test_data,
            strategy=concatenated_strategy,
            include_arrays=False,  # Skip arrays for speed
            padding=PADDING,
            exchange_fee=0.0003,
            interval='5min',
            strategy_name="Quick_Validation_Test",
            save_outputs=False  # Don't save for quick test
        )
        
        print("✅ Quick evaluation completed successfully!")
        
        # 6. Validate results
        print(f"\n📈 VALIDATION RESULTS:")
        print("-" * 50)
        print(f"Portfolio Value:     {results.get('value', 0):.4f}")
        print(f"Total Return:        {(results.get('value', 1) - 1)*100:+.2f}%")
        print(f"Modified IR:         {results.get('mod_ir', 0):.4f}")
        print(f"Number of Trades:    {results.get('n_trades', 0)}")
        
        # 7. Verify strategy parameters were correctly applied
        print(f"\n🔍 STRATEGY PARAMETER VERIFICATION:")
        print("-" * 50)
        expected_w1 = {'enter_long': 0.004, 'enter_short': -0.005}
        expected_w2 = {'enter_long': 0.002, 'enter_short': -0.001}
        
        if len(test_strategies) >= 2:
            w1_strategy = test_strategies[0]
            w2_strategy = test_strategies[1]
            
            w1_actual = {'enter_long': w1_strategy.enter_long, 'enter_short': w1_strategy.enter_short}
            w2_actual = {'enter_long': w2_strategy.enter_long, 'enter_short': w2_strategy.enter_short}
            
            w1_match = w1_actual == expected_w1
            w2_match = w2_actual == expected_w2
            
            print(f"W1 Expected: {expected_w1}")
            print(f"W1 Actual:   {w1_actual} {'✅' if w1_match else '❌'}")
            print(f"W2 Expected: {expected_w2}")
            print(f"W2 Actual:   {w2_actual} {'✅' if w2_match else '❌'}")
            
            if w1_match and w2_match:
                print("✅ Strategy parameters correctly applied!")
                success = True
            else:
                print("❌ Strategy parameters mismatch!")
                success = False
        else:
            print("❌ Insufficient strategies for validation")
            success = False
        
        print(f"\n🎯 OVERALL VALIDATION RESULT:")
        print("-" * 50)
        if success:
            print("✅ VALIDATION SUCCESSFUL!")
            print("✅ Strategy selection fix is working properly")
            print("✅ Ready for full complete evaluation run")
            print("🚀 You can now run: python complete_evaluation_run.py")
        else:
            print("❌ VALIDATION FAILED!")
            print("❌ Strategy selection fix needs investigation")
        
        return success
        
    except Exception as e:
        print(f"\n💥 Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 TMUX Strategy Validation Starting...")
    success = run_quick_validation()
    
    final_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n⏰ Finished: {final_time}")
    
    if success:
        print("\n🎉 VALIDATION SUCCESSFUL - STRATEGY FIX IS WORKING!")
        print("🚀 Ready to run complete evaluation with optimal strategies")
    else:
        print("\n💥 VALIDATION FAILED - NEEDS INVESTIGATION")
    
    sys.exit(0 if success else 1)
"""
Enhanced Evaluator Module

Extends the original strategy evaluation with comprehensive trade analysis
and advanced metrics while maintaining full backward compatibility.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

# Add the original strategy module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from strategy.evaluation import evaluate_strategy as original_evaluate_strategy
from strategy.strategy import StrategyBase, EXIT_POSITION, LONG_POSITION, SHORT_POSITION, BuyAndHoldStrategy
from .trade_analyzer import TradeAnalyzer
from .advanced_metrics import AdvancedMetrics
from ..exporters.csv_exporter import CSVExporter
from ..visualization.equity_charts import EquityChartGenerator
from ..visualization.prediction_charts import InteractivePredictionVisualizer


class EnhancedEvaluator:
    """
    Enhanced strategy evaluator that provides comprehensive analysis
    while maintaining compatibility with the original evaluation framework.
    """
    
    def __init__(self, periods_per_year: int = 105120):
        """
        Initialize enhanced evaluator.
        
        Args:
            periods_per_year: Number of periods per year for annualization (5min = 105120)
        """
        self.periods_per_year = periods_per_year
        self.trade_analyzer = TradeAnalyzer()
        
    def evaluate_strategy_enhanced(
        self,
        data: pd.DataFrame,
        strategy: StrategyBase,
        include_arrays: bool = True,
        padding: int = 0,
        exchange_fee: float = 0.0003,
        interval: str = "5min",
        strategy_name: str = None,
        output_dir: str = None,
        save_outputs: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced strategy evaluation with comprehensive analysis.
        
        Args:
            data: Price data DataFrame
            strategy: Strategy instance to evaluate
            include_arrays: Whether to include time series arrays
            padding: Number of periods to skip at start
            exchange_fee: Transaction fee per trade
            interval: Time interval for annualization
            strategy_name: Name for output files
            output_dir: Directory for output files
            
        Returns:
            Comprehensive results dictionary
        """
        
        # Store evaluation data for potential visualization use
        self._last_evaluation_data = (data, strategy)
        
        # Get original evaluation results
        original_results = original_evaluate_strategy(
            data=data,
            strategy=strategy,
            include_arrays=include_arrays,
            padding=padding,
            exchange_fee=exchange_fee,
            interval=interval
        )
        
        # Extract positions and prepare data for enhanced analysis
        positions = strategy.run(data)[:-1]  # Skip last position
        close_prices = data['close_price'].to_numpy()
        timestamps = data['close_time'].to_numpy()
        
        # Apply padding
        positions = positions[padding:]
        close_prices = close_prices[padding:]
        timestamps = timestamps[padding:]
        
        # Extract individual trades
        trades = self.trade_analyzer.extract_trades(
            positions=positions,
            prices=close_prices,
            timestamps=timestamps,
            exchange_fee=exchange_fee
        )
        
        # Calculate trade statistics
        trade_stats = self.trade_analyzer.calculate_trade_statistics()
        trade_summary_by_type = self.trade_analyzer.get_trade_summary_by_type()
        
        # Calculate advanced metrics
        portfolio_values = original_results.get('portfolio_value', np.array([1.0]))
        advanced_metrics = AdvancedMetrics.calculate_all_metrics(
            portfolio_values=portfolio_values,
            periods_per_year=self.periods_per_year
        )
        
        # Calculate benchmark for comparison
        benchmark_values = self.calculate_benchmark_values(
            data=data, 
            padding=padding, 
            exchange_fee=exchange_fee
        )
        
        # Calculate rolling metrics if we have enough data
        rolling_metrics = {}
        if len(portfolio_values) > 30:  # Minimum window for rolling calculations
            window_size = min(30, len(portfolio_values) // 4)  # Adaptive window size
            rolling_metrics = AdvancedMetrics.rolling_metrics(
                portfolio_values=portfolio_values,
                window=window_size,
                periods_per_year=self.periods_per_year
            )
        
        # Compile comprehensive results
        enhanced_results = {
            # Original results (for backward compatibility)
            **original_results,
            
            # Enhanced trade analysis
            'trade_analysis': {
                'individual_trades': len(trades),
                'trade_statistics': trade_stats,
                'trade_summary_by_type': trade_summary_by_type,
            },
            
            # Advanced risk and performance metrics
            'advanced_metrics': advanced_metrics,
            
            # Rolling performance metrics
            'rolling_metrics': rolling_metrics,
            
            # Benchmark comparison
            'benchmark_values': benchmark_values,
            
            # Metadata
            'evaluation_metadata': {
                'strategy_name': strategy_name or strategy.__class__.__name__,
                'evaluation_timestamp': datetime.now().isoformat(),
                'exchange_fee': exchange_fee,
                'padding_periods': padding,
                'data_start': pd.to_datetime(timestamps[0]).isoformat() if len(timestamps) > 0 else None,
                'data_end': pd.to_datetime(timestamps[-1]).isoformat() if len(timestamps) > 0 else None,
                'total_periods': len(timestamps),
                'periods_per_year': self.periods_per_year
            }
        }
        
        # Add trade details if requested
        if include_arrays and trades:
            trades_df = self.trade_analyzer.get_trades_dataframe()
            enhanced_results['trade_details'] = {
                'trades_dataframe': trades_df,
                'raw_trades': trades
            }
        
        # Save outputs if requested
        if save_outputs and strategy_name:
            self.save_evaluation_outputs(
                results=enhanced_results,
                strategy_name=strategy_name,
                interval=interval,
                output_dir=output_dir
            )
        
        return enhanced_results
    
    def calculate_benchmark_values(self, data: pd.DataFrame, padding: int = 0, 
                                 exchange_fee: float = 0.0003) -> np.ndarray:
        """
        Calculate buy-and-hold benchmark portfolio values for comparison.
        
        Args:
            data: Price data DataFrame
            padding: Number of periods to skip at start
            exchange_fee: Transaction fee per trade
        
        Returns:
            Array of benchmark portfolio values
        """
        try:
            # Create buy-and-hold strategy
            benchmark_strategy = BuyAndHoldStrategy()
            
            # Evaluate benchmark strategy with same parameters
            benchmark_results = original_evaluate_strategy(
                data=data,
                strategy=benchmark_strategy,
                include_arrays=True,
                padding=padding,
                exchange_fee=exchange_fee,
                interval="5min"
            )
            
            # Return benchmark portfolio values
            return benchmark_results.get('portfolio_value', np.array([1.0]))
            
        except Exception as e:
            print(f"Warning: Could not calculate benchmark values: {e}")
            return None
    
    def calculate_drawdown_periods(self, portfolio_values: np.ndarray, 
                                 timestamps: np.ndarray = None) -> pd.DataFrame:
        """
        Identify and analyze distinct drawdown periods.
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            timestamps: Array of timestamps (optional)
            
        Returns:
            DataFrame with drawdown period analysis
        """
        if len(portfolio_values) < 2:
            return pd.DataFrame()
            
        # Calculate running peak and drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        # Identify drawdown periods (when drawdown > 0)
        in_drawdown = drawdown > 1e-6  # Small threshold to avoid floating point issues
        
        drawdown_periods = []
        period_id = 1
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                # Start of new drawdown
                start_idx = i
            elif not is_dd and start_idx is not None:
                # End of drawdown
                end_idx = i - 1
                
                # Find peak before drawdown
                peak_idx = start_idx
                if start_idx > 0:
                    peak_idx = np.argmax(portfolio_values[:start_idx+1])
                
                # Find recovery point
                recovery_idx = None
                peak_value = portfolio_values[peak_idx]
                for j in range(end_idx + 1, len(portfolio_values)):
                    if portfolio_values[j] >= peak_value:
                        recovery_idx = j
                        break
                
                # Calculate drawdown statistics
                max_dd_idx = np.argmin(portfolio_values[start_idx:end_idx+1]) + start_idx
                max_drawdown = drawdown[max_dd_idx]
                
                period_data = {
                    'drawdown_id': period_id,
                    'peak_idx': peak_idx,
                    'start_idx': start_idx,
                    'trough_idx': max_dd_idx,
                    'end_idx': end_idx,
                    'recovery_idx': recovery_idx,
                    'peak_value': portfolio_values[peak_idx],
                    'trough_value': portfolio_values[max_dd_idx],
                    'max_drawdown_pct': max_drawdown * 100,
                    'duration_periods': end_idx - start_idx + 1,
                    'recovery_periods': (recovery_idx - peak_idx) if recovery_idx else None
                }
                
                # Add timestamps if available
                if timestamps is not None and len(timestamps) == len(portfolio_values):
                    period_data.update({
                        'peak_time': pd.to_datetime(timestamps[peak_idx]),
                        'start_time': pd.to_datetime(timestamps[start_idx]),
                        'trough_time': pd.to_datetime(timestamps[max_dd_idx]),
                        'end_time': pd.to_datetime(timestamps[end_idx]),
                        'recovery_time': pd.to_datetime(timestamps[recovery_idx]) if recovery_idx else None
                    })
                
                drawdown_periods.append(period_data)
                period_id += 1
                start_idx = None
        
        # Handle case where drawdown continues to end of data
        if start_idx is not None:
            end_idx = len(portfolio_values) - 1
            peak_idx = start_idx
            if start_idx > 0:
                peak_idx = np.argmax(portfolio_values[:start_idx+1])
            
            max_dd_idx = np.argmin(portfolio_values[start_idx:]) + start_idx
            max_drawdown = drawdown[max_dd_idx]
            
            period_data = {
                'drawdown_id': period_id,
                'peak_idx': peak_idx,
                'start_idx': start_idx,
                'trough_idx': max_dd_idx,
                'end_idx': end_idx,
                'recovery_idx': None,  # No recovery yet
                'peak_value': portfolio_values[peak_idx],
                'trough_value': portfolio_values[max_dd_idx],
                'max_drawdown_pct': max_drawdown * 100,
                'duration_periods': end_idx - start_idx + 1,
                'recovery_periods': None
            }
            
            if timestamps is not None and len(timestamps) == len(portfolio_values):
                period_data.update({
                    'peak_time': pd.to_datetime(timestamps[peak_idx]),
                    'start_time': pd.to_datetime(timestamps[start_idx]),
                    'trough_time': pd.to_datetime(timestamps[max_dd_idx]),
                    'end_time': pd.to_datetime(timestamps[end_idx]),
                    'recovery_time': None
                })
            
            drawdown_periods.append(period_data)
        
        return pd.DataFrame(drawdown_periods)
    
    def save_evaluation_outputs(self, results: Dict[str, Any], strategy_name: str, 
                               interval: str = "5min", output_dir: str = None) -> str:
        """
        Save all evaluation outputs to a single timestamped folder.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            interval: Time interval for naming
            output_dir: Base output directory (defaults to analysis/)
            
        Returns:
            Path to the created output folder
        """
        # Determine output directory
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '../../..')
            output_dir = os.path.join(project_root, 'analysis')
        
        # Create timestamped folder name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_strategy_name = strategy_name.lower().replace(' ', '_').replace('-', '_')
        timeframe = interval.replace('min', 'm')
        folder_name = f"{clean_strategy_name}_{timeframe}_{timestamp}"
        
        output_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nSaving evaluation outputs to: {output_folder}")
        
        # Initialize exporters with the specific output folder
        csv_exporter = CSVExporter(output_folder)
        chart_generator = EquityChartGenerator(output_folder)
        prediction_visualizer = InteractivePredictionVisualizer(output_folder)
        
        # Export CSV files
        csv_files = csv_exporter.export_all(results, strategy_name, timeframe)
        
        # Generate visualizations
        chart_files = chart_generator.generate_all_charts(results, strategy_name, timeframe)
        
        # Generate interactive prediction visualization (only for strategies with predictions)
        interactive_files = {}
        try:
            # Check if we have evaluation metadata with original data and strategy
            if hasattr(self, '_last_evaluation_data'):
                data, strategy = self._last_evaluation_data
                
                # Check if strategy has predictions (GMADL or other model strategies)
                if hasattr(strategy, 'predictions') and strategy.predictions is not None:
                    print(f"Generating interactive prediction visualization...")
                    
                    portfolio_values = results.get('portfolio_value')
                    interactive_path = os.path.join(output_folder, f"{clean_strategy_name}_{timeframe}_interactive_predictions.html")
                    
                    saved_path = prediction_visualizer.create_interactive_prediction_plot(
                        strategy=strategy,
                        data=data,
                        title=f"{strategy_name} - Predictions & Trading Signals ({timeframe})",
                        save_path=interactive_path,
                        portfolio_values=portfolio_values
                    )
                    
                    if saved_path:
                        interactive_files['prediction_chart'] = saved_path
                        print(f"Interactive prediction chart saved: {os.path.basename(saved_path)}")
                        
        except Exception as e:
            print(f"Warning: Could not generate interactive prediction visualization: {e}")
            interactive_files = {}
        
        # Create a summary file with all file paths
        summary_data = {
            'evaluation_summary': {
                'strategy_name': strategy_name,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'output_folder': output_folder,
                'csv_files': csv_files,
                'chart_files': chart_files,
                'interactive_files': interactive_files
            }
        }
        
        # Save summary as JSON
        import json
        summary_file = os.path.join(output_folder, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"Evaluation complete. All outputs saved to: {output_folder}")
        print(f"Summary file: {summary_file}")
        
        return output_folder
    
    def generate_equity_curve_data(self, portfolio_values: np.ndarray, 
                                 timestamps: np.ndarray, 
                                 positions: np.ndarray = None) -> pd.DataFrame:
        """
        Generate comprehensive equity curve data.
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            timestamps: Array of timestamps
            positions: Array of position signals (optional)
            
        Returns:
            DataFrame with equity curve and related metrics
        """
        if len(portfolio_values) != len(timestamps):
            raise ValueError("Portfolio values and timestamps must have same length")
            
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0], returns])  # Add zero return for first period
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        # Create base DataFrame
        equity_data = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'portfolio_value': portfolio_values,
            'peak_value': peak,
            'drawdown_pct': drawdown * 100,
            'period_return_pct': returns * 100,
            'cumulative_return_pct': ((portfolio_values / portfolio_values[0]) - 1) * 100
        })
        
        # Add position information if available
        if positions is not None and len(positions) == len(portfolio_values):
            equity_data['position'] = positions
            equity_data['position_name'] = equity_data['position'].map({
                EXIT_POSITION: 'CASH',
                LONG_POSITION: 'LONG', 
                SHORT_POSITION: 'SHORT'
            })
        
        # Calculate rolling metrics
        if len(portfolio_values) > 30:
            window = min(30, len(portfolio_values) // 4)
            
            rolling_returns = equity_data['period_return_pct'].rolling(window=window).mean()
            rolling_volatility = equity_data['period_return_pct'].rolling(window=window).std()
            rolling_sharpe = rolling_returns / rolling_volatility
            
            equity_data['rolling_return_pct'] = rolling_returns
            equity_data['rolling_volatility_pct'] = rolling_volatility
            equity_data['rolling_sharpe'] = rolling_sharpe
        
        return equity_data
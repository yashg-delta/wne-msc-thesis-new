"""
CSV Exporter Module

Generates comprehensive CSV exports for strategy evaluation results.
Handles all output file creation with proper naming conventions.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class CSVExporter:
    """Handles CSV export of enhanced strategy evaluation results."""
    
    def __init__(self, output_base_dir: str = None):
        """
        Initialize CSV exporter.
        
        Args:
            output_base_dir: Base directory for outputs (single folder per run)
        """
        if output_base_dir is None:
            # Default to analysis directory relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '../../..')
            output_base_dir = os.path.join(project_root, 'analysis')
        
        self.output_base_dir = os.path.abspath(output_base_dir)
        # Create the base directory (single folder for all outputs)
        os.makedirs(self.output_base_dir, exist_ok=True)
    
    
    def _generate_filename(self, strategy_name: str, timeframe: str, 
                          analysis_type: str, extension: str = 'csv') -> str:
        """
        Generate standardized filename.
        
        Format: {strategy_name}_{timeframe}_{date}_{analysis_type}.{extension}
        """
        date_str = datetime.now().strftime('%Y%m%d')
        # Clean strategy name
        clean_strategy_name = strategy_name.lower().replace(' ', '_').replace('-', '_')
        
        filename = f"{clean_strategy_name}_{timeframe}_{date_str}_{analysis_type}.{extension}"
        return filename
    
    def export_trade_ledger(self, results: Dict[str, Any], strategy_name: str, 
                           timeframe: str = '5m') -> str:
        """
        Export detailed trade-by-trade ledger.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            timeframe: Data timeframe
            
        Returns:
            Path to exported file
        """
        trade_details = results.get('trade_details', {})
        trades_df = trade_details.get('trades_dataframe')
        
        if trades_df is None or trades_df.empty:
            print(f"No trade data available for {strategy_name}")
            return None
        
        # Enhance trade data with additional columns
        enhanced_trades = trades_df.copy()
        
        # Add running portfolio value if available
        if 'portfolio_value' in results:
            portfolio_values = results['portfolio_value']
            if len(portfolio_values) > 0:
                # Match trade exit times to portfolio values (simplified)
                enhanced_trades['running_portfolio_value'] = np.nan
                for i, row in enhanced_trades.iterrows():
                    enhanced_trades.loc[i, 'running_portfolio_value'] = portfolio_values[min(i+1, len(portfolio_values)-1)]
        
        # Generate filename and path
        filename = self._generate_filename(strategy_name, timeframe, 'trade_ledger')
        filepath = os.path.join(self.output_base_dir, filename)
        
        # Export to CSV
        enhanced_trades.to_csv(filepath, index=False)
        print(f"Trade ledger exported to: {filepath}")
        
        return filepath
    
    def export_performance_summary(self, results: Dict[str, Any], strategy_name: str,
                                 timeframe: str = '5m') -> str:
        """
        Export comprehensive performance summary.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            timeframe: Data timeframe
            
        Returns:
            Path to exported file
        """
        # Combine all performance metrics
        performance_data = {}
        
        # Original metrics
        original_metrics = {
            'portfolio_final_value': results.get('value', 0),
            'total_return_pct': results.get('total_return', 0) * 100,
            'annualized_return_pct': results.get('arc', 0) * 100,
            'annualized_std_pct': results.get('asd', 0) * 100,
            'information_ratio': results.get('ir', 0),
            'modified_information_ratio': results.get('mod_ir', 0),
            'max_drawdown_pct': results.get('md', 0) * 100,
            'num_trades': results.get('n_trades', 0),
            'long_position_pct': results.get('long_pos', 0) * 100,
            'short_position_pct': results.get('short_pos', 0) * 100
        }
        performance_data.update(original_metrics)
        
        # Trade analysis metrics
        trade_stats = results.get('trade_analysis', {}).get('trade_statistics', {})
        performance_data.update(trade_stats)
        
        # Advanced metrics
        advanced_metrics = results.get('advanced_metrics', {})
        performance_data.update(advanced_metrics)
        
        # Metadata
        metadata = results.get('evaluation_metadata', {})
        performance_data.update({
            'strategy_name': strategy_name,
            'timeframe': timeframe,
            'evaluation_date': metadata.get('evaluation_timestamp', ''),
            'exchange_fee': metadata.get('exchange_fee', 0),
            'data_start': metadata.get('data_start', ''),
            'data_end': metadata.get('data_end', ''),
            'total_periods': metadata.get('total_periods', 0)
        })
        
        # Convert to DataFrame for easy export
        performance_df = pd.DataFrame([performance_data])
        
        # Generate filename and path
        filename = self._generate_filename(strategy_name, timeframe, 'performance_detailed')
        filepath = os.path.join(self.output_base_dir, filename)
        
        # Export to CSV
        performance_df.to_csv(filepath, index=False)
        print(f"Performance summary exported to: {filepath}")
        
        return filepath
    
    def export_equity_curve(self, results: Dict[str, Any], strategy_name: str,
                           timeframe: str = '5m') -> str:
        """
        Export equity curve data.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            timeframe: Data timeframe
            
        Returns:
            Path to exported file
        """
        # Extract time series data
        portfolio_values = results.get('portfolio_value')
        timestamps = results.get('time')
        positions = results.get('strategy_positions')
        
        if portfolio_values is None or timestamps is None:
            print(f"No equity curve data available for {strategy_name}")
            return None
        
        # Create equity curve DataFrame
        equity_data = {
            'timestamp': pd.to_datetime(timestamps),
            'portfolio_value': portfolio_values,
            'cumulative_return_pct': ((portfolio_values / portfolio_values[0]) - 1) * 100
        }
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0], returns])
        equity_data['period_return_pct'] = returns * 100
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        equity_data['peak_value'] = peak
        equity_data['drawdown_pct'] = drawdown * 100
        
        # Add position data if available
        if positions is not None:
            equity_data['position'] = positions
            equity_data['position_name'] = pd.Series(positions).map({
                0: 'CASH',
                1: 'LONG',
                -1: 'SHORT'
            })
        
        # Add rolling metrics if available
        rolling_metrics = results.get('rolling_metrics', {})
        if rolling_metrics:
            # Pad rolling metrics to match length
            for metric_name, metric_values in rolling_metrics.items():
                if len(metric_values) > 0:
                    # Pad with NaN at the beginning
                    padded_values = np.full(len(portfolio_values), np.nan)
                    start_idx = len(portfolio_values) - len(metric_values)
                    padded_values[start_idx:] = metric_values
                    equity_data[metric_name] = padded_values
        
        equity_df = pd.DataFrame(equity_data)
        
        # Generate filename and path
        filename = self._generate_filename(strategy_name, timeframe, 'equity_curve')
        filepath = os.path.join(self.output_base_dir, filename)
        
        # Export to CSV
        equity_df.to_csv(filepath, index=False)
        print(f"Equity curve exported to: {filepath}")
        
        return filepath
    
    def export_drawdown_periods(self, results: Dict[str, Any], strategy_name: str,
                               timeframe: str = '5m') -> str:
        """
        Export drawdown period analysis.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            timeframe: Data timeframe
            
        Returns:
            Path to exported file
        """
        # This would use the drawdown analysis from enhanced evaluator
        # For now, create a basic version
        portfolio_values = results.get('portfolio_value')
        timestamps = results.get('time')
        
        if portfolio_values is None or len(portfolio_values) < 2:
            print(f"No drawdown data available for {strategy_name}")
            return None
        
        # Calculate basic drawdown info
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        # Create simple drawdown summary
        drawdown_data = {
            'max_drawdown_pct': [max_drawdown * 100],
            'max_drawdown_date': [pd.to_datetime(timestamps[max_dd_idx]) if timestamps is not None else None],
            'max_drawdown_value': [portfolio_values[max_dd_idx]],
            'peak_before_max_dd': [peak[max_dd_idx]],
            'avg_drawdown_pct': [np.mean(drawdown[drawdown > 0]) * 100 if np.any(drawdown > 0) else 0]
        }
        
        drawdown_df = pd.DataFrame(drawdown_data)
        
        # Generate filename and path
        filename = self._generate_filename(strategy_name, timeframe, 'drawdown_analysis')
        filepath = os.path.join(self.output_base_dir, filename)
        
        # Export to CSV
        drawdown_df.to_csv(filepath, index=False)
        print(f"Drawdown analysis exported to: {filepath}")
        
        return filepath
    
    def export_all(self, results: Dict[str, Any], strategy_name: str, 
                   timeframe: str = '5m') -> Dict[str, str]:
        """
        Export all available data to CSV files.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            timeframe: Data timeframe
            
        Returns:
            Dictionary mapping export type to file path
        """
        exported_files = {}
        
        # Export trade ledger
        trade_file = self.export_trade_ledger(results, strategy_name, timeframe)
        if trade_file:
            exported_files['trade_ledger'] = trade_file
        
        # Export performance summary
        perf_file = self.export_performance_summary(results, strategy_name, timeframe)
        if perf_file:
            exported_files['performance_summary'] = perf_file
        
        # Export equity curve
        equity_file = self.export_equity_curve(results, strategy_name, timeframe)
        if equity_file:
            exported_files['equity_curve'] = equity_file
        
        # Export drawdown analysis
        drawdown_file = self.export_drawdown_periods(results, strategy_name, timeframe)
        if drawdown_file:
            exported_files['drawdown_analysis'] = drawdown_file
        
        print(f"\nExported {len(exported_files)} files for {strategy_name} ({timeframe}):")
        for export_type, filepath in exported_files.items():
            print(f"  {export_type}: {os.path.basename(filepath)}")
        
        return exported_files
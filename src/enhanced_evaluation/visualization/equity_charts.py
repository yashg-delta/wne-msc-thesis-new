"""
Equity Charts Module

Creates professional equity curve and drawdown visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List
import os


class EquityChartGenerator:
    """Generates professional equity curve and related charts."""
    
    def __init__(self, output_dir: str = None, style: str = 'default', figsize: tuple = (12, 8)):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory for saving charts
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.output_dir = output_dir
        self.style = style
        self.figsize = figsize
        
        # Create output directory if provided
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Professional color scheme
        self.colors = {
            'equity': '#2E86AB',
            'drawdown': '#A23B72', 
            'peak': '#F18F01',
            'benchmark': '#C73E1D',
            'positive': '#4CAF50',
            'negative': '#F44336',
            'neutral': '#FFC107'
        }
    
    def plot_equity_curve_with_drawdown(self, portfolio_values: np.ndarray,
                                      timestamps: np.ndarray,
                                      title: str = "Strategy Performance",
                                      save_path: str = None,
                                      benchmark_values: np.ndarray = None,
                                      benchmark_name: str = "Buy & Hold") -> str:
        """
        Create equity curve with drawdown subplot.
        
        Args:
            portfolio_values: Array of portfolio values
            timestamps: Array of timestamps
            title: Chart title
            save_path: Path to save chart
            benchmark_values: Optional buy-and-hold benchmark values
            benchmark_name: Name for benchmark line
            
        Returns:
            Path to saved chart
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)
        
        # Convert timestamps
        dates = pd.to_datetime(timestamps)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        # Plot equity curve
        ax1.plot(dates, portfolio_values, color=self.colors['equity'], 
                linewidth=2, label='Portfolio Value')
        ax1.plot(dates, peak, color=self.colors['peak'], 
                linewidth=1, alpha=0.7, linestyle='--', label='Peak Value')
        
        # Plot benchmark if provided
        if benchmark_values is not None and len(benchmark_values) == len(dates):
            ax1.plot(dates, benchmark_values, color=self.colors['benchmark'], 
                    linewidth=2, alpha=0.8, linestyle='-', label=benchmark_name)
        
        ax1.set_ylabel('Portfolio Value', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot drawdown
        ax2.fill_between(dates, 0, -drawdown * 100, 
                        color=self.colors['drawdown'], alpha=0.6)
        ax2.plot(dates, -drawdown * 100, color=self.colors['drawdown'], linewidth=1)
        
        ax2.set_ylabel('Drawdown %', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_underwater_curve(self, portfolio_values: np.ndarray,
                             timestamps: np.ndarray,
                             title: str = "Underwater Curve",
                             save_path: str = None) -> str:
        """
        Create underwater curve (continuous drawdown view).
        
        Args:
            portfolio_values: Array of portfolio values
            timestamps: Array of timestamps
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        dates = pd.to_datetime(timestamps)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        # Plot underwater curve
        ax.fill_between(dates, 0, -drawdown * 100, 
                       color=self.colors['drawdown'], alpha=0.6, label='Drawdown')
        ax.plot(dates, -drawdown * 100, color=self.colors['drawdown'], linewidth=2)
        
        ax.set_ylabel('Drawdown %', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=1)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Underwater curve saved to: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_rolling_metrics(self, timestamps: np.ndarray,
                           rolling_metrics: Dict[str, np.ndarray],
                           title: str = "Rolling Performance Metrics",
                           save_path: str = None) -> str:
        """
        Plot rolling performance metrics.
        
        Args:
            timestamps: Array of timestamps
            rolling_metrics: Dictionary of rolling metric arrays
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Path to saved chart
        """
        num_metrics = len(rolling_metrics)
        if num_metrics == 0:
            return None
            
        fig, axes = plt.subplots(num_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * 0.7 * num_metrics),
                                sharex=True)
        
        if num_metrics == 1:
            axes = [axes]
        
        dates = pd.to_datetime(timestamps)
        colors = plt.cm.Set3(np.linspace(0, 1, num_metrics))
        
        for i, (metric_name, values) in enumerate(rolling_metrics.items()):
            if len(values) > 0:
                # Align with timestamps (rolling metrics start later)
                metric_dates = dates[-len(values):]
                
                axes[i].plot(metric_dates, values, color=colors[i], linewidth=2)
                axes[i].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
                axes[i].grid(True, alpha=0.3)
                
                # Add horizontal reference lines for some metrics
                if 'sharpe' in metric_name.lower():
                    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    axes[i].axhline(y=1, color='green', linestyle='--', alpha=0.5)
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=12)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rolling metrics chart saved to: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_returns_distribution(self, returns: np.ndarray,
                                title: str = "Returns Distribution",
                                save_path: str = None) -> str:
        """
        Plot returns distribution histogram with statistics.
        
        Args:
            returns: Array of period returns
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Path to saved chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(returns * 100, bins=50, alpha=0.7, color=self.colors['equity'], edgecolor='black')
        ax1.axvline(np.mean(returns) * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns)*100:.2f}%')
        ax1.axvline(np.median(returns) * 100, color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(returns)*100:.2f}%')
        ax1.set_xlabel('Return %', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Returns Histogram', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Returns distribution chart saved to: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def generate_all_charts(self, results: Dict[str, Any], strategy_name: str, 
                           timeframe: str = '5m') -> Dict[str, str]:
        """
        Generate all charts for a strategy evaluation.
        
        Args:
            results: Enhanced evaluation results
            strategy_name: Name of the strategy
            timeframe: Data timeframe
            
        Returns:
            Dictionary mapping chart type to file path
        """
        if not self.output_dir:
            print("No output directory specified for chart generation")
            return {}
        
        chart_files = {}
        
        # Extract data
        portfolio_values = results.get('portfolio_value')
        timestamps = results.get('time')
        rolling_metrics = results.get('rolling_metrics', {})
        benchmark_values = results.get('benchmark_values')
        
        if portfolio_values is None or timestamps is None:
            print(f"No chart data available for {strategy_name}")
            return chart_files
        
        # Clean strategy name for filenames
        clean_name = strategy_name.lower().replace(' ', '_').replace('-', '_')
        
        # Generate equity curve with drawdown
        equity_path = os.path.join(self.output_dir, f"{clean_name}_{timeframe}_equity_curve.png")
        if self.plot_equity_curve_with_drawdown(
            portfolio_values, timestamps, 
            title=f"{strategy_name} - Equity Curve ({timeframe})",
            save_path=equity_path,
            benchmark_values=benchmark_values
        ):
            chart_files['equity_curve'] = equity_path
        
        # Generate underwater curve
        underwater_path = os.path.join(self.output_dir, f"{clean_name}_{timeframe}_underwater_curve.png")
        if self.plot_underwater_curve(
            portfolio_values, timestamps,
            title=f"{strategy_name} - Underwater Curve ({timeframe})", 
            save_path=underwater_path
        ):
            chart_files['underwater_curve'] = underwater_path
        
        # Generate rolling metrics if available
        if rolling_metrics:
            rolling_path = os.path.join(self.output_dir, f"{clean_name}_{timeframe}_rolling_metrics.png")
            if self.plot_rolling_metrics(
                timestamps, rolling_metrics,
                title=f"{strategy_name} - Rolling Metrics ({timeframe})",
                save_path=rolling_path
            ):
                chart_files['rolling_metrics'] = rolling_path
        
        # Generate returns distribution
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns_path = os.path.join(self.output_dir, f"{clean_name}_{timeframe}_returns_distribution.png")
            if self.plot_returns_distribution(
                returns,
                title=f"{strategy_name} - Returns Distribution ({timeframe})",
                save_path=returns_path
            ):
                chart_files['returns_distribution'] = returns_path
        
        print(f"\nGenerated {len(chart_files)} charts for {strategy_name}:")
        for chart_type, filepath in chart_files.items():
            print(f"  {chart_type}: {os.path.basename(filepath)}")
        
        return chart_files
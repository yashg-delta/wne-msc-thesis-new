"""
Advanced Metrics Module

Provides sophisticated risk and performance metrics beyond basic statistics.
Includes Sortino ratio, VaR, CVaR, Calmar ratio, and other institutional metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats


class AdvancedMetrics:
    """Collection of advanced risk and performance metrics."""
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, target_return: float = 0.0, 
                     periods_per_year: int = 105120) -> float:
        """
        Calculate Sortino ratio (excess return / downside deviation).
        
        Args:
            returns: Array of period returns
            target_return: Minimum acceptable return (default 0)
            periods_per_year: Number of periods per year for annualization
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 0.0
            
        avg_excess_return = np.mean(excess_returns)
        annualized_excess = avg_excess_return * periods_per_year
        annualized_downside_dev = downside_deviation * np.sqrt(periods_per_year)
        
        return annualized_excess / annualized_downside_dev
    
    @staticmethod
    def calmar_ratio(portfolio_values: np.ndarray, 
                    periods_per_year: int = 105120) -> float:
        """
        Calculate Calmar ratio (annualized return / maximum drawdown).
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        if len(portfolio_values) < 2:
            return 0.0
            
        # Calculate annualized return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        periods = len(portfolio_values) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
        
        # Calculate maximum drawdown
        max_dd = AdvancedMetrics.max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
            
        return annualized_return / max_dd
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown as a percentage."""
        if len(portfolio_values) < 2:
            return 0.0
            
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    @staticmethod
    def ulcer_index(portfolio_values: np.ndarray) -> float:
        """
        Calculate Ulcer Index (measure of downside risk).
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            
        Returns:
            Ulcer Index
        """
        if len(portfolio_values) < 2:
            return 0.0
            
        peak = np.maximum.accumulate(portfolio_values)
        drawdown_pct = ((peak - portfolio_values) / peak) * 100
        ulcer = np.sqrt(np.mean(drawdown_pct ** 2))
        
        return ulcer
    
    @staticmethod
    def var_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> tuple:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            returns: Array of period returns
            confidence_level: Risk level (0.05 = 95% confidence)
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        if len(returns) == 0:
            return 0.0, 0.0
            
        sorted_returns = np.sort(returns)
        index = int(confidence_level * len(sorted_returns))
        
        if index >= len(sorted_returns):
            return sorted_returns[-1], sorted_returns[-1]
            
        var = sorted_returns[index]
        cvar = np.mean(sorted_returns[:index+1]) if index >= 0 else sorted_returns[0]
        
        return var, cvar
    
    @staticmethod
    def recovery_factor(portfolio_values: np.ndarray) -> float:
        """
        Calculate recovery factor (total return / maximum drawdown).
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            
        Returns:
            Recovery factor
        """
        if len(portfolio_values) < 2:
            return 0.0
            
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        max_dd = AdvancedMetrics.max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return float('inf') if total_return > 0 else 0.0
            
        return total_return / max_dd
    
    @staticmethod
    def sterling_ratio(portfolio_values: np.ndarray, 
                      periods_per_year: int = 105120) -> float:
        """
        Calculate Sterling ratio (annualized return / average drawdown).
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            periods_per_year: Number of periods per year
            
        Returns:
            Sterling ratio
        """
        if len(portfolio_values) < 2:
            return 0.0
            
        # Calculate annualized return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        periods = len(portfolio_values) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
        
        # Calculate average drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        avg_drawdown = np.mean(drawdown[drawdown > 0])
        
        if avg_drawdown == 0 or np.isnan(avg_drawdown):
            return float('inf') if annualized_return > 0 else 0.0
            
        return annualized_return / avg_drawdown
    
    @staticmethod
    def pain_index(portfolio_values: np.ndarray) -> float:
        """
        Calculate Pain Index (average drawdown over the entire period).
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            
        Returns:
            Pain Index as percentage
        """
        if len(portfolio_values) < 2:
            return 0.0
            
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        return np.mean(drawdown) * 100
    
    @staticmethod
    def tail_ratio(returns: np.ndarray, tail_percentile: float = 5) -> float:
        """
        Calculate tail ratio (average of top tail / average of bottom tail).
        
        Args:
            returns: Array of period returns
            tail_percentile: Percentile for tail definition (default 5%)
            
        Returns:
            Tail ratio
        """
        if len(returns) == 0:
            return 0.0
            
        top_percentile = 100 - tail_percentile
        
        top_threshold = np.percentile(returns, top_percentile)
        bottom_threshold = np.percentile(returns, tail_percentile)
        
        top_tail_returns = returns[returns >= top_threshold]
        bottom_tail_returns = returns[returns <= bottom_threshold]
        
        if len(bottom_tail_returns) == 0:
            return float('inf')
            
        avg_top_tail = np.mean(top_tail_returns)
        avg_bottom_tail = np.mean(np.abs(bottom_tail_returns))
        
        if avg_bottom_tail == 0:
            return float('inf')
            
        return avg_top_tail / avg_bottom_tail
    
    @staticmethod
    def rolling_metrics(portfolio_values: np.ndarray, window: int, 
                       periods_per_year: int = 105120) -> Dict[str, np.ndarray]:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            window: Rolling window size
            periods_per_year: Number of periods per year
            
        Returns:
            Dictionary of rolling metrics
        """
        if len(portfolio_values) < window:
            return {}
            
        rolling_returns = []
        rolling_volatility = []
        rolling_sharpe = []
        rolling_max_dd = []
        
        for i in range(window, len(portfolio_values)):
            window_values = portfolio_values[i-window:i+1]
            
            # Calculate returns for this window
            window_returns = np.diff(window_values) / window_values[:-1]
            
            # Rolling return (annualized)
            total_return = (window_values[-1] / window_values[0]) - 1
            annualized_return = (1 + total_return) ** (periods_per_year / window) - 1
            rolling_returns.append(annualized_return)
            
            # Rolling volatility (annualized)
            if len(window_returns) > 1:
                vol = np.std(window_returns) * np.sqrt(periods_per_year)
                rolling_volatility.append(vol)
                
                # Rolling Sharpe ratio
                if vol > 0:
                    sharpe = annualized_return / vol
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0.0)
            else:
                rolling_volatility.append(0.0)
                rolling_sharpe.append(0.0)
            
            # Rolling max drawdown
            rolling_max_dd.append(AdvancedMetrics.max_drawdown(window_values))
        
        return {
            'rolling_returns': np.array(rolling_returns),
            'rolling_volatility': np.array(rolling_volatility),
            'rolling_sharpe': np.array(rolling_sharpe),
            'rolling_max_drawdown': np.array(rolling_max_dd)
        }
    
    @staticmethod
    def calculate_all_metrics(portfolio_values: np.ndarray, 
                            periods_per_year: int = 105120) -> Dict[str, Any]:
        """
        Calculate all advanced metrics for a portfolio.
        
        Args:
            portfolio_values: Array of cumulative portfolio values
            periods_per_year: Number of periods per year (5min = 105120)
            
        Returns:
            Dictionary containing all advanced metrics
        """
        if len(portfolio_values) < 2:
            return {}
            
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate VaR and CVaR at different confidence levels
        var_95, cvar_95 = AdvancedMetrics.var_cvar(returns, 0.05)
        var_99, cvar_99 = AdvancedMetrics.var_cvar(returns, 0.01)
        
        metrics = {
            # Risk-adjusted performance ratios
            'sortino_ratio': AdvancedMetrics.sortino_ratio(returns, periods_per_year=periods_per_year),
            'calmar_ratio': AdvancedMetrics.calmar_ratio(portfolio_values, periods_per_year),
            'sterling_ratio': AdvancedMetrics.sterling_ratio(portfolio_values, periods_per_year),
            'recovery_factor': AdvancedMetrics.recovery_factor(portfolio_values),
            
            # Risk metrics
            'max_drawdown_pct': AdvancedMetrics.max_drawdown(portfolio_values) * 100,
            'ulcer_index': AdvancedMetrics.ulcer_index(portfolio_values),
            'pain_index': AdvancedMetrics.pain_index(portfolio_values),
            
            # Value at Risk metrics
            'var_95_pct': var_95 * 100,
            'cvar_95_pct': cvar_95 * 100,
            'var_99_pct': var_99 * 100,
            'cvar_99_pct': cvar_99 * 100,
            
            # Tail risk
            'tail_ratio': AdvancedMetrics.tail_ratio(returns),
            
            # Return distribution metrics
            'return_skewness': stats.skew(returns) if len(returns) > 0 else 0,
            'return_kurtosis': stats.kurtosis(returns) if len(returns) > 0 else 0,
            
            # Volatility metrics
            'annualized_volatility_pct': np.std(returns) * np.sqrt(periods_per_year) * 100 if len(returns) > 0 else 0,
            
            # Additional metrics
            'total_return_pct': ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100,
            'periods_analyzed': len(portfolio_values),
        }
        
        return metrics
"""
Trade Analysis Module

Extracts and analyzes individual trades from strategy position arrays.
Provides detailed trade-level statistics and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta


class Trade:
    """Represents a single trade with all relevant metrics."""
    
    def __init__(self, trade_id: int, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                 position_type: int, entry_price: float, exit_price: float,
                 return_pct: float, commission: float = 0.0):
        self.trade_id = trade_id
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.position_type = position_type  # 1 for long, -1 for short
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.return_pct = return_pct
        self.commission = commission
        
        # Calculate derived metrics
        self.duration = exit_time - entry_time
        self.duration_minutes = self.duration.total_seconds() / 60
        self.gross_pnl = return_pct
        self.net_pnl = return_pct - commission
        self.is_winner = self.net_pnl > 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for export."""
        return {
            'trade_id': self.trade_id,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'duration_minutes': self.duration_minutes,
            'position_type': 'LONG' if self.position_type == 1 else 'SHORT',
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'return_pct': self.return_pct * 100,
            'gross_pnl_pct': self.gross_pnl * 100,
            'commission_pct': self.commission * 100,
            'net_pnl_pct': self.net_pnl * 100,
            'is_winner': self.is_winner
        }


class TradeAnalyzer:
    """Analyzes trading strategy performance at the individual trade level."""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.position_constants = {
            'EXIT': 0,
            'LONG': 1, 
            'SHORT': -1
        }
        
    def extract_trades(self, positions: np.ndarray, prices: np.ndarray, 
                      timestamps: np.ndarray, exchange_fee: float = 0.0003) -> List[Trade]:
        """
        Extract individual trades from position and price arrays.
        
        Args:
            positions: Array of position signals (0=exit, 1=long, -1=short)
            prices: Array of prices corresponding to each position
            timestamps: Array of timestamps for each position
            exchange_fee: Transaction fee per trade
            
        Returns:
            List of Trade objects
        """
        trades = []
        trade_id = 1
        
        # Track current position state
        current_position = 0
        entry_time = None
        entry_price = None
        entry_position = None
        
        for i in range(len(positions)):
            position = positions[i]
            price = prices[i]
            timestamp = pd.to_datetime(timestamps[i])
            
            # Position change detected
            if position != current_position:
                
                # If we had an open position, close it
                if current_position != 0 and entry_time is not None:
                    # Calculate return based on position type
                    if entry_position == 1:  # Long position
                        return_pct = (price - entry_price) / entry_price
                    else:  # Short position
                        return_pct = (entry_price - price) / price
                    
                    # Create trade object
                    trade = Trade(
                        trade_id=trade_id,
                        entry_time=entry_time,
                        exit_time=timestamp,
                        position_type=entry_position,
                        entry_price=entry_price,
                        exit_price=price,
                        return_pct=return_pct,
                        commission=exchange_fee * 2  # Entry + exit fees
                    )
                    trades.append(trade)
                    trade_id += 1
                
                # If entering a new position
                if position != 0:
                    entry_time = timestamp
                    entry_price = price
                    entry_position = position
                else:
                    entry_time = None
                    entry_price = None
                    entry_position = None
                    
                current_position = position
        
        self.trades = trades
        return trades
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trade-level statistics."""
        if not self.trades:
            return {}
            
        # Basic counts
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.is_winner]
        losing_trades = [t for t in self.trades if not t.is_winner]
        long_trades = [t for t in self.trades if t.position_type == 1]
        short_trades = [t for t in self.trades if t.position_type == -1]
        
        # Win/Loss statistics
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        win_rate = num_winners / total_trades if total_trades > 0 else 0
        
        # Return statistics
        all_returns = [t.net_pnl for t in self.trades]
        winning_returns = [t.net_pnl for t in winning_trades]
        losing_returns = [t.net_pnl for t in losing_trades]
        
        avg_return = np.mean(all_returns) if all_returns else 0
        avg_winning_return = np.mean(winning_returns) if winning_returns else 0
        avg_losing_return = np.mean(losing_returns) if losing_returns else 0
        
        # Profit factor
        gross_profit = sum(winning_returns) if winning_returns else 0
        gross_loss = abs(sum(losing_returns)) if losing_returns else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Duration statistics
        durations = [t.duration_minutes for t in self.trades]
        avg_duration = np.mean(durations) if durations else 0
        median_duration = np.median(durations) if durations else 0
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_outcomes(True)
        consecutive_losses = self._calculate_consecutive_outcomes(False)
        
        # Largest wins/losses
        largest_win = max(winning_returns) if winning_returns else 0
        largest_loss = min(losing_returns) if losing_returns else 0
        
        # Expectancy
        expectancy = (win_rate * avg_winning_return) - ((1 - win_rate) * abs(avg_losing_return))
        
        return {
            # Basic counts
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            
            # Performance metrics
            'win_rate_pct': win_rate * 100,
            'avg_return_pct': avg_return * 100,
            'avg_winning_return_pct': avg_winning_return * 100,
            'avg_losing_return_pct': avg_losing_return * 100,
            'profit_factor': profit_factor,
            'expectancy_pct': expectancy * 100,
            
            # Extremes
            'largest_win_pct': largest_win * 100,
            'largest_loss_pct': largest_loss * 100,
            'best_trade_pct': max(all_returns) * 100 if all_returns else 0,
            'worst_trade_pct': min(all_returns) * 100 if all_returns else 0,
            
            # Duration statistics
            'avg_trade_duration_minutes': avg_duration,
            'median_trade_duration_minutes': median_duration,
            'min_trade_duration_minutes': min(durations) if durations else 0,
            'max_trade_duration_minutes': max(durations) if durations else 0,
            
            # Consecutive outcomes
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            
            # Additional metrics
            'gross_profit_pct': gross_profit * 100,
            'gross_loss_pct': gross_loss * 100,
            'total_commission_pct': sum(t.commission for t in self.trades) * 100
        }
    
    def _calculate_consecutive_outcomes(self, target_outcome: bool) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not self.trades:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.is_winner == target_outcome:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to pandas DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
            
        trade_data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(trade_data)
        
        # Add cumulative metrics
        df['cumulative_pnl_pct'] = df['net_pnl_pct'].cumsum()
        df['cumulative_commission_pct'] = df['commission_pct'].cumsum()
        df['running_win_rate'] = (df['is_winner'].cumsum() / (df.index + 1)) * 100
        
        return df
    
    def get_trade_summary_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics broken down by position type."""
        long_trades = [t for t in self.trades if t.position_type == 1]
        short_trades = [t for t in self.trades if t.position_type == -1]
        
        def analyze_trade_subset(trades: List[Trade]) -> Dict[str, Any]:
            if not trades:
                return {}
                
            returns = [t.net_pnl for t in trades]
            winners = [t for t in trades if t.is_winner]
            
            return {
                'count': len(trades),
                'win_rate_pct': (len(winners) / len(trades)) * 100 if trades else 0,
                'avg_return_pct': np.mean(returns) * 100 if returns else 0,
                'total_return_pct': sum(returns) * 100,
                'avg_duration_minutes': np.mean([t.duration_minutes for t in trades]) if trades else 0
            }
        
        return {
            'long_positions': analyze_trade_subset(long_trades),
            'short_positions': analyze_trade_subset(short_trades),
            'all_positions': analyze_trade_subset(self.trades)
        }
import pandas as pd
import numpy as np
from strategy import metrics
from strategy.strategy import LONG_POSITION, SHORT_POSITION, EXIT_POSITION
from strategy.strategy import StrategyBase


def evaluate_strategy(
        data: pd.DataFrame,
        strategy: StrategyBase,
        exchange_fee: float = 0.001,
        interval: str = "5min"):
    """Evaluates a trading strategy."""

    # Get strategy positions,
    # position at time t is used at t+1.
    # Skip last position as it cannot be evaluated.
    positions = strategy.run(data)[:-1]

    # Compute returns for long and short positions.
    close_price = data['close_price'].to_numpy()
    long_returns = (
        (close_price[1:] - close_price[:-1]) / close_price[:-1])
    short_returns = (
        (close_price[:-1] - close_price[1:]) / close_price[1:])
    assert positions.shape == long_returns.shape
    assert positions.shape == short_returns.shape

    # timestamps = data['close_time'].astype('datetime64[s]').to_numpy()
    timestamps = data['close_time'].to_numpy()
    assert positions.shape[0] == timestamps.shape[0] - 1

    # Compute returns of the strategy.
    strategy_returns = np.zeros_like(positions, dtype=np.float64)
    strategy_returns[positions == LONG_POSITION] = \
        long_returns[positions == LONG_POSITION]
    strategy_returns[positions == SHORT_POSITION] = \
        short_returns[positions == SHORT_POSITION]

    # Include exchange fees
    positions_changed = np.append([EXIT_POSITION], positions[:-1]) != positions
    strategy_returns[positions_changed] = (
        strategy_returns[positions_changed] + 1.0) * (1.0 - exchange_fee) - 1.0

    strategy_returns = np.append([0.], strategy_returns)
    portfolio_value = np.cumprod(strategy_returns + 1)

    # Compute all the metrics
    result = {
        'value': portfolio_value[-1],
        'total_return': portfolio_value[-1] - 1,
        'arc': metrics.arc(portfolio_value, interval=interval),
        'asd': metrics.asd(portfolio_value, interval=interval),
        'ir': metrics.ir(portfolio_value, interval=interval),
        'md': metrics.max_drawdown(portfolio_value),
        'n_trades': np.sum(np.append([EXIT_POSITION], positions[:-1]) !=
                           np.append(positions[1:], [EXIT_POSITION])),
        'long_pos': np.sum(positions == LONG_POSITION) / positions.size,
        'short_pos': np.sum(positions == SHORT_POSITION) / positions.size,
        # Arrays
        'portfolio_value': portfolio_value,
        'strategy_returns': strategy_returns,
        'strategy_positions': np.append([EXIT_POSITION], positions),
        'time': timestamps
    }

    return result

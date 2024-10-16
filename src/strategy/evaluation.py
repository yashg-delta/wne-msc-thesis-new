from typing import Dict, List, Any, Optional, Callable
import itertools
import pandas as pd
import numpy as np
import functools
from tqdm import tqdm
from multiprocessing import Pool
from strategy import metrics
from strategy.strategy import LONG_POSITION, SHORT_POSITION, EXIT_POSITION
from strategy.strategy import StrategyBase


def parameter_sweep(
        data: pd.DataFrame,
        strategy_class: StrategyBase.__class__,
        params: Dict[str, List[Any]],
        num_workers: int = 4,
        params_filter: Optional[Callable] = None,
        log_every: int = 200,
        exchange_fee: float = 0.001,
        padding: int = 0,
        sort_by: str = 'mod_ir',
        interval: str = '5min') -> pd.DataFrame:
    """Evaluates the strategy on a different sets of hyperparameters."""

    # Obtain sets of parameters to evaluate
    param_sets = list(filter(params_filter, map(lambda p: dict(
        zip(params.keys(), p)), itertools.product(*params.values()))))

    result = []
    total = len(param_sets)

    # Evaluate sets of different hyperparameters in parallel
    with Pool(num_workers) as pool, tqdm(total=total) as pbar:
        for chunk in (param_sets[i:i + log_every]
                      for i in range(0, total, log_every)):
            tmp = list(
                pool.map(
                    functools.partial(
                        evaluate_strategy,
                        data,
                        exchange_fee=exchange_fee,
                        interval=interval,
                        padding=padding,
                        include_arrays=False),
                    map(
                        lambda p: strategy_class(
                            **p), chunk)))
            pbar.update(len(tmp))
            result += list(zip(tmp, map(
                lambda p: strategy_class(
                    **p), chunk)))

    return sorted(result, key=lambda x: x[0][sort_by], reverse=True)


def evaluate_strategy(
        data: pd.DataFrame,
        strategy: StrategyBase,
        include_arrays: bool = True,
        padding: int = 0,
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

    # Pad the results
    positions = positions[padding:]
    timestamps = timestamps[padding:]
    long_returns = long_returns[padding:]
    short_returns = short_returns[padding:]

    # Compute returns of the strategy.
    strategy_returns = np.zeros_like(positions, dtype=np.float64)
    strategy_returns[positions == LONG_POSITION] = \
        long_returns[positions == LONG_POSITION]
    strategy_returns[positions == SHORT_POSITION] = \
        short_returns[positions == SHORT_POSITION]

    # Include exchange fees
    strategy_returns = (strategy_returns + 1.0) * (
        1.0 - exchange_fee * np.abs(np.append(
            [EXIT_POSITION], positions[:-1]) - positions)) - 1.0

    strategy_returns = np.append([0.], strategy_returns)
    portfolio_value = np.cumprod(strategy_returns + 1)

    # Compute all the metrics
    result = {
        'value': portfolio_value[-1],
        'total_return': portfolio_value[-1] - 1,
        'arc': metrics.arc(portfolio_value, interval=interval),
        'asd': metrics.asd(portfolio_value, interval=interval),
        'ir': metrics.ir(portfolio_value, interval=interval),
        'mod_ir': metrics.modified_ir(portfolio_value, interval=interval),
        'md': metrics.max_drawdown(portfolio_value),
        'n_trades': np.sum(np.abs(np.append([EXIT_POSITION], positions[:-1]) -
                           np.append(positions[1:], [EXIT_POSITION]))),
        'long_pos': np.sum(positions == LONG_POSITION) / positions.size,
        'short_pos': np.sum(positions == SHORT_POSITION) / positions.size,
    }

    result |= strategy.info()

    if include_arrays:
        result |= {
            'portfolio_value': portfolio_value,
            'strategy_returns': strategy_returns,
            'strategy_positions': np.append([EXIT_POSITION], positions),
            'time': timestamps
        }

    return result

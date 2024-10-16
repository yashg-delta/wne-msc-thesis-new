from typing import Any
import numpy as np
from numpy.typing import NDArray


NUM_INTERVALS = {
    'min': 365 * 24 * 60,
    '5min': 365 * 24 * 12,
    '15min': 365 * 24 * 4,
    '30min': 365 * 24 * 2,
    'hour': 365 * 24,
    'day': 365
}


def investment_return(array: NDArray[Any]):
    """Return at the end of the investment period."""
    return (array[-1] - array[0]) / array[0]


def arc(array: NDArray[Any], interval: str = '5min'):
    """Annualised Return Compounded for the investment period."""
    return np.power(array[-1] / array[0],
                    NUM_INTERVALS[interval] / array.size) - 1


def asd(array: NDArray[Any], interval: str = '5min'):
    """Annualised Standard Deviation for the investment period."""
    simple_returns = (array[1:] - array[:-1]) / array[:-1]
    avg_simple_return = np.mean(simple_returns)
    return np.sqrt(
        (NUM_INTERVALS[interval] /
         array.size) *
        np.sum(
            np.power(
                simple_returns -
                avg_simple_return,
                2)))


def ir(array: NDArray[Any], interval: str = '5min'):
    """Information Ratio, the amount of return for a given unit of risk."""
    std = asd(array, interval=interval)
    return arc(array, interval=interval) / std if std else 0.0


def max_drawdown(array: NDArray[Any]):
    """The maximum percentage drawdown during the investment period."""
    cummax = np.maximum.accumulate(array)
    return np.max((cummax - array) / cummax)


def modified_ir(array: NDArray[Any], interval: str = '5min'):
    ret = (ir(array, interval=interval)
           * np.abs(arc(array, interval=interval)))
    md = max_drawdown(array)

    if md > 0:
        ret = ret / md

    return ret

import numpy as np
import pandas as pd
from typing import Dict, Any

EXIT_POSITION = 0
LONG_POSITION = 1
SHORT_POSITION = 2


class StrategyBase:
    """Base class for investment strategies."""

    def info(self) -> Dict[str, Any]:
        """Returns general informaiton about the strategy."""
        raise NotImplementedError

    def run(self, data: pd.DataFrame):
        """Run strategy on data."""
        raise NotImplementedError()


class BuyAndHoldStrategy(StrategyBase):
    """Simple benchmark strategy, always long position"""

    NAME = "BUY_AND_HOLD"

    def info(self) -> Dict[str, Any]:
        return {'strategy_name': BuyAndHoldStrategy.NAME}

    def run(self, data: pd.DataFrame):
        return np.full_like(
            data['close_price'].to_numpy(),
            LONG_POSITION,
            dtype=np.int32)


class ModelReturnsPredictionStrategy(StrategyBase):
    """Strategy that selects position based on returns predictions."""

    def __init__(
            self,
            predictions,
            threshold=0.001,
            name=None):
        self.predictions = predictions
        assert 'time_index' in self.predictions.columns
        assert 'group_id' in self.predictions.columns
        assert 'prediction' in self.predictions.columns

        self.name = name or "ML Returns prediction"
        self.threshold = threshold

    def info(self) -> Dict[str, Any]:
        return {'strategy_name': self.name}

    def run(self, data):
        arr = pd.merge(
            data, self.predictions, on=['time_index', 'group_id'],
            how='left')['prediction'].to_numpy()

        positions = []
        for i in range(len(arr)):
            if arr[i] > self.threshold:
                positions.append(LONG_POSITION)
            elif arr[i] < -self.threshold:
                positions.append(EXIT_POSITION)
            elif not len(positions):
                positions.append(EXIT_POSITION)
            else:
                positions.append(positions[-1])

        return np.array(positions, dtype=np.int32)

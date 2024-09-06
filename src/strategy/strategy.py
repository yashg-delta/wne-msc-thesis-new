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


class ModelPredictionsStrategyBase(StrategyBase):
    """Base class for strategies based on model predictions."""
    def __init__(self,
                 predictions,
                 name: str = None):
        self.predictions = predictions
        assert 'time_index' in self.predictions.columns
        assert 'group_id' in self.predictions.columns
        assert 'prediction' in self.predictions.columns

        self.name = name


    def info(self):
        return {'strategy_name': self.name or 'Unknown model'}

    def run(self, data):
        # Adds predictions to data, if prediction is unknown for a given
        # item it will be nan.
        merged_data = pd.merge(
            data, self.predictions, on=['time_index', 'group_id'],
            how='left')
        
        return self.get_positions(merged_data)

    def get_positions(self, data):
        raise NotImplementedError()


class ReturnsPredictionStrategy(ModelPredictionsStrategyBase):
    """Strategy that selects position based on returns predictions."""

    def __init__(
            self,
            predictions,
            threshold=0.001,
            name=None):
        super().__init__(predictions, name=name)
        self.threshold = threshold

    def get_positions(self, data):
        arr = data['prediction']
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


class PriceQuantilePredictionStrategy(ModelPredictionsStrategyBase):
    def __init__(
        self,
        predictions,
        name=None):
        super().__init__(predictions, name=name)

    def info(self):
        return {'strategy_name': self.name}

    def get_positions(self, data):

        arr_preds = data['prediction'].to_numpy()
        arr_close_price = data['close_price'].to_numpy()

        positions = []
        for i in range(len(arr_preds)):
            if not np.isnan(arr_preds[i]).any():
                price = arr_close_price[i]
                pred_low = arr_preds[i][0]
                pred_high = arr_preds[i][-1]
                if (pred_low - price) / price > 0.001:
                    positions.append(LONG_POSITION)
                    continue
                elif (pred_high - price) / price < -0.001:
                    positions.append(EXIT_POSITION)
                    continue
            
            if not len(positions):
                positions.append(EXIT_POSITION)
            else:
                positions.append(positions[-1])

        return np.array(positions, dtype=np.int32)
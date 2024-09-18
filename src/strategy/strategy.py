import numpy as np
import pandas as pd
# import logging
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
                 name: str = None,
                 future: int = 1,
                 exchange_fee: int = 0.001,
                 target: str = 'close_price'):
        self.predictions = predictions
        assert 'time_index' in self.predictions.columns
        assert 'group_id' in self.predictions.columns
        assert 'prediction' in self.predictions.columns

        self.name = name
        self.future = future
        self.target = target
        self.exchange_fee = exchange_fee

    def info(self):
        return {
            'strategy_name': self.name or 'Unknown model',
            'future': self.future,
            'target': self.target
        }

    def run(self, data):
        # Adds predictions to data, if prediction is unknown for a given
        # item it will be nan.
        merged_data = pd.merge(
            data, self.predictions, on=['time_index', 'group_id'],
            how='left')

        return self.get_positions(merged_data)

    def get_positions(self, data):
        raise NotImplementedError()


class ModelQuantilePredictionsStrategy(ModelPredictionsStrategyBase):
    def __init__(
        self,
        predictions,
        quantiles,
        quantile_enter_long=None,
        quantile_exit_long=None,
        quantile_enter_short=None,
        quantile_exit_short=None,
        name: str = None,
        future: int = 1,
        target: str = 'close_price',
        exchange_fee: int = 0.001
    ):
        super().__init__(
            predictions,
            name=name,
            future=future,
            target=target,
            exchange_fee=exchange_fee)

        self.quantiles = quantiles
        self.quantile_enter_long = quantile_enter_long
        self.quantile_exit_long = quantile_exit_long
        self.quantile_enter_short = quantile_enter_short
        self.quantile_exit_short = quantile_exit_short

    def info(self):
        return super().info() | {
            'quantiles': self.quantiles,
            'quantile_enter_long': self.quantile_enter_long,
            'quantile_exit_long': self.quantile_exit_long,
            'quantile_enter_short': self.quantile_enter_short,
            'quantile_exit_short': self.quantile_exit_short
        }

    def get_positions(self, data):
        arr_preds = data['prediction'].to_numpy()
        arr_target = data[self.target].to_numpy()

        positions = [EXIT_POSITION]
        for i in range(len(arr_preds)):

            # If strategy does not have prediction
            # keep the current position.
            if np.isnan(arr_preds[i]).any():
                # logging.warning(f"Missing value for time index {i}.")
                positions.append(positions[-1])
                continue

            target = arr_target[i]
            prediction = arr_preds[i][self.future - 1]

            # Enter long position
            if (self.quantile_enter_long and
                    (prediction[self.get_quantile_idx(
                        round(1 - self.quantile_enter_long, 2)
                    )] - target)
                    / target > self.exchange_fee):
                positions.append(LONG_POSITION)

            # Enter short position
            elif (self.quantile_enter_short and
                  (prediction[self.get_quantile_idx(
                    self.quantile_enter_short)] - target)
                    / target < -self.exchange_fee):
                positions.append(SHORT_POSITION)

            # Exit long position
            elif (self.quantile_exit_long and
                  (prediction[self.get_quantile_idx(
                    self.quantile_exit_long)] - target)
                    / target < -self.exchange_fee):
                positions.append(EXIT_POSITION)

            # Exit short postion
            elif (self.quantile_exit_short and
                  (prediction[self.get_quantile_idx(
                    round(1 - self.quantile_exit_short, 2)
                  )] - target) / target > self.exchange_fee):
                positions.append(EXIT_POSITION)

            else:
                positions.append(positions[-1])

        return np.array(positions[1:], dtype=np.int32)

    def get_quantile_idx(self, quantile):
        return self.quantiles.index(quantile)


class ConcatenatedStrategies(StrategyBase):
    """
    Evaluates multiple strategies,
    each on the next `window_size` data points.
    """

    def __init__(self, window_size, strategies, name='Concatenated Strategy'):
        self.window_size = window_size
        self.strategies = strategies
        self.name = name

    def info(self):
        return {'strategy_name': self.name}

    def run(self, data):
        chunks = [data[i:i+self.window_size].copy()
                  for i in range(0, data.shape[0], self.window_size)]
        assert len(chunks) <= len(self.strategies)

        positions = []
        for chunk, strategy in zip(chunks, self.strategies):
            positions.append(strategy.run(chunk))

        return np.concatenate(positions)

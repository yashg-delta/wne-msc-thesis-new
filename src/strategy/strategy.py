import talib
import numpy as np
import pandas as pd
# import logging
from typing import Dict, Any
# from strategy.util import rsi_obos

EXIT_POSITION = 0
LONG_POSITION = 1
SHORT_POSITION = -1


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


class MACDStrategy(StrategyBase):
    """Strategy based on Moving Average Convergence / Divergence."""

    NAME = "MACD"

    def __init__(
            self,
            fast_window_size: int = 12,
            slow_window_size: int = 26,
            signal_window_size: int = 9,
            short_sell: bool = False):

        if (fast_window_size == 1 or
                slow_window_size == 1 or
                signal_window_size == 1 or
                fast_window_size >= slow_window_size):
            raise ValueError

        self.fast_window_size = fast_window_size
        self.slow_window_size = slow_window_size
        self.signal_window_size = signal_window_size
        self.short_sell = short_sell
        self.name = MACDStrategy.NAME
        # f"{MACDStrategy.NAME}" +\
        #     "(fast={self.fast_window_size}," +\
        #     " slow={self.slow_window_size}," +\
        #     " signal={self.signal_window_size})"

    def info(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.name,
            'fast_window_size': self.fast_window_size,
            'slow_window_size': self.slow_window_size,
            'signal_window_size': self.signal_window_size,
            'short_sell': self.short_sell
        }

    def run(self, data: pd.DataFrame):
        array = data['close_price'].to_numpy()
        macd, signal, _ = talib.MACD(
            array,
            fastperiod=self.fast_window_size,
            slowperiod=self.slow_window_size,
            signalperiod=self.signal_window_size
        )

        result = np.full_like(array, EXIT_POSITION, dtype=np.int32)
        result[macd > signal] = LONG_POSITION

        if self.short_sell:
            result[macd < signal] = SHORT_POSITION

        # run_info = {
        #     'macd': macd,
        #     'signal': signal
        # }
        return result  # , run_info


class RSIStrategy(StrategyBase):
    """Strategy based on RSI."""

    NAME = "RSI"

    def __init__(self,
                 window_size: int = 14,
                 enter_long=None,
                 exit_long=None,
                 enter_short=None,
                 exit_short=None):
        self.window_size = window_size
        self.enter_long = enter_long
        self.exit_long = exit_long
        self.enter_short = enter_short
        self.exit_short = exit_short
        self.name = RSIStrategy.NAME
        # f"{RSIStrategy.NAME}(" +\
        #     "window={self.window_size}," +\
        #     "[{self.oversold}, {self.overbought}])"

    def info(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.name,
            'window_size': self.window_size,
            'enter_long': self.enter_long,
            'exit_long': self.exit_long,
            'enter_short': self.enter_short,
            'exit_short': self.exit_short
        }

    def run(self, data: pd.DataFrame):
        array = data['close_price'].to_numpy()

        rsi = talib.RSI(array, timeperiod=self.window_size)
        enter_long = rsi > (self.enter_long or np.infty)
        exit_long = rsi < (self.exit_long or -np.infty)
        enter_short = rsi < (
            self.enter_short or -np.infty)
        exit_short = rsi > (self.exit_short or np.infty)

        positions = np.full(rsi.shape, np.nan)
        positions[exit_long | exit_short] = EXIT_POSITION
        positions[enter_long] = LONG_POSITION
        positions[enter_short] = SHORT_POSITION

        # Fix the first position
        if np.isnan(positions[0]):
            positions[0] = EXIT_POSITION

        mask = np.isnan(positions)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        positions[mask] = positions[idx[mask]]

        return positions.astype(np.int32)
        # result = rsi_obos(rsi, self.oversold, self.overbought)

        # run_info = {
        #     'rsi': rsi
        # }

        # return result  # , run_info


class BaselineReturnsStrategy(StrategyBase):
    def __init__(
            self,
            enter_long,
            exit_long,
            enter_short,
            exit_short):
        self.enter_long = enter_long
        self.exit_long = exit_long
        self.enter_short = enter_short
        self.exit_short = exit_short

    def info(self):
        return {
            'strategy_name': 'Baseline predictions',
            'enter_long': self.enter_long,
            'exit_long': self.exit_long,
            'enter_short': self.enter_short,
            'exit_short': self.exit_short
        }

    def run(self, data):

        ret = data['returns'].to_numpy()
        enter_long = ret > (self.enter_long or np.infty)
        exit_long = ret < (self.exit_long or -np.infty)
        enter_short = ret < (
            self.enter_short or -np.infty)
        exit_short = ret > (self.exit_short or np.infty)

        positions = np.full(ret.shape, np.nan)
        positions[exit_long | exit_short] = EXIT_POSITION
        positions[enter_long] = LONG_POSITION
        positions[enter_short] = SHORT_POSITION

        # Fix the first position
        if np.isnan(positions[0]):
            positions[0] = EXIT_POSITION

        mask = np.isnan(positions)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        positions[mask] = positions[idx[mask]]

        return positions.astype(np.int32)


class ModelPredictionsStrategyBase(StrategyBase):
    """Base class for strategies based on model predictions."""

    def __init__(self,
                 predictions,
                 name: str = None,
                 future: int = 1,
                 exchange_fee: int = 0.0003,
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


class ModelGmadlPredictionsStrategy(ModelPredictionsStrategyBase):
    def __init__(
            self,
            predictions,
            enter_long=None,
            exit_long=None,
            enter_short=None,
            exit_short=None,
            future=1,
            name: str = None,
    ):
        super().__init__(
            predictions,
            name=name
        )

        self.enter_long = enter_long
        self.exit_long = exit_long
        self.enter_short = enter_short
        self.exit_short = exit_short
        self.future = future

    def info(self):
        return super().info() | {
            'enter_long': self.enter_long,
            'exit_long': self.exit_long,
            'enter_short': self.enter_short,
            'exit_short': self.exit_short
        }

    def get_positions(self, data):
        # bfill() is a hack to make it work with non predicted data
        arr_preds = np.stack(data['prediction'].ffill().bfill().to_numpy())
        arr_preds = arr_preds[:, self.future, 0]

        enter_long = arr_preds > (self.enter_long or np.infty)
        exit_long = arr_preds < (self.exit_long or -np.infty)
        enter_short = arr_preds < (
            self.enter_short or -np.infty)
        exit_short = arr_preds > (self.exit_short or np.infty)

        positions = np.full(arr_preds.shape, np.nan)
        positions[exit_long | exit_short] = EXIT_POSITION
        positions[enter_long] = LONG_POSITION
        positions[enter_short] = SHORT_POSITION

        # Fix the first position
        if np.isnan(positions[0]):
            positions[0] = EXIT_POSITION

        mask = np.isnan(positions)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        positions[mask] = positions[idx[mask]]

        return positions.astype(np.int32)


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
        exchange_fee: int = 0.0003,
        new_impl=True
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
        self.new_impl = new_impl

    def info(self):
        return super().info() | {
            'quantiles': self.quantiles,
            'exchange_fee': self.exchange_fee,
            'quantile_enter_long': self.quantile_enter_long,
            'quantile_exit_long': self.quantile_exit_long,
            'quantile_enter_short': self.quantile_enter_short,
            'quantile_exit_short': self.quantile_exit_short
        }

    def get_positions(self, data):
        if self.new_impl:
            return self.get_positions2(data)
        return self.get_positions1(data)

    def get_positions2(self, data):
        arr_target = data[self.target].to_numpy()
        arr_preds = np.stack(
            # bfill() is a hack to make it work with non predicted data
            data['prediction'].ffill().bfill().to_numpy())

        enter_long = (((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                round(1 - self.quantile_enter_long, 2))]
            if self.quantile_enter_long
            else np.full(arr_target.shape, -np.infty)))
            - arr_target) / arr_target > self.exchange_fee
        enter_short = ((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                self.quantile_enter_short)]
            if self.quantile_enter_short
            else np.full(arr_target.shape, np.infty))
            - arr_target) / arr_target < -self.exchange_fee
        exit_long = ((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                self.quantile_exit_long)]
            if self.quantile_exit_long
            else np.full(arr_target.shape, np.infty))
            - arr_target) / arr_target < -self.exchange_fee
        exit_short = ((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                round(1 - self.quantile_exit_short, 2))]
            if self.quantile_exit_short
            else np.full(arr_target.shape, -np.infty))
            - arr_target) / arr_target > self.exchange_fee

        positions = np.full(arr_target.shape, np.nan)
        positions[exit_long | exit_short] = EXIT_POSITION
        positions[enter_long] = LONG_POSITION
        positions[enter_short] = SHORT_POSITION

        # Fix the first position
        if np.isnan(positions[0]):
            positions[0] = EXIT_POSITION

        mask = np.isnan(positions)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        positions[mask] = positions[idx[mask]]

        return positions.astype(np.int32)

    def get_positions1(self, data):
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

    def __init__(
            self,
            window_size,
            strategies,
            name='Concatenated Strategy',
            padding=0):
        self.window_size = window_size
        self.strategies = strategies
        self.name = name
        self.padding = padding

    def info(self):
        return {'strategy_name': self.name}

    def run(self, data):
        chunks = [data[i-self.padding:i+self.window_size].copy()
                  for i in range(
                      self.padding, data.shape[0], self.window_size)]
        assert len(chunks) <= len(self.strategies)

        positions = []
        for chunk, strategy in zip(chunks, self.strategies):
            positions.append(strategy.run(chunk))

        positions = [
            pos if not i else pos[self.padding:]
            for i, pos in enumerate(positions)
        ]

        return np.concatenate(positions)


class ModelQuantileReturnsPredictionsStrategy(ModelPredictionsStrategyBase):
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
        target: str = 'returns',
        exchange_fee: int = 0.0003,
        new_impl=True
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
        self.new_impl = new_impl

    def info(self):
        return super().info() | {
            'quantiles': self.quantiles,
            'exchange_fee': self.exchange_fee,
            'quantile_enter_long': self.quantile_enter_long,
            'quantile_exit_long': self.quantile_exit_long,
            'quantile_enter_short': self.quantile_enter_short,
            'quantile_exit_short': self.quantile_exit_short
        }

    def get_positions(self, data):
        arr_target = data[self.target].to_numpy()
        arr_preds = np.stack(
            # bfill() is a hack to make it work with non predicted data
            data['prediction'].ffill().bfill().to_numpy())

        enter_long = (((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                round(1 - self.quantile_enter_long, 2))]
            if self.quantile_enter_long
            else np.full(arr_target.shape, -np.infty)))
            > self.exchange_fee)
        enter_short = ((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                self.quantile_enter_short)]
            if self.quantile_enter_short
            else np.full(arr_target.shape, np.infty))
            < -self.exchange_fee)
        exit_long = ((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                self.quantile_exit_long)]
            if self.quantile_exit_long
            else np.full(arr_target.shape, np.infty))
            < -self.exchange_fee)
        exit_short = ((arr_preds[
            :, self.future - 1, self.get_quantile_idx(
                round(1 - self.quantile_exit_short, 2))]
            if self.quantile_exit_short
            else np.full(arr_target.shape, -np.infty))
            > self.exchange_fee)

        positions = np.full(arr_target.shape, np.nan)
        positions[exit_long | exit_short] = EXIT_POSITION
        positions[enter_long] = LONG_POSITION
        positions[enter_short] = SHORT_POSITION

        # Fix the first position
        if np.isnan(positions[0]):
            positions[0] = EXIT_POSITION

        mask = np.isnan(positions)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        positions[mask] = positions[idx[mask]]

        return positions.astype(np.int32)

    def get_quantile_idx(self, quantile):
        return self.quantiles.index(quantile)

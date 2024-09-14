import torch

from pytorch_forecasting import QuantileLoss
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


def get_loss(config):
    loss_name = config['loss']['name']

    if loss_name == 'Quantile':
        return QuantileLoss(config['loss']['quantiles'])

    if loss_name == 'GMADL':
        return GMADL(
            a=config['loss']['a'],
            b=config['loss']['b']
        )

    raise ValueError("Unknown loss")


class GMADL(MultiHorizonMetric):
    """GMADL loss function."""

    def __init__(self, a=1000, b=2, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def loss(self, y_pred, target):
        return -1 * \
            (1 / (1 + torch.exp(-self.a * self.to_prediction(y_pred) * target)
                  ) - 0.5) * torch.pow(torch.abs(target), self.b)

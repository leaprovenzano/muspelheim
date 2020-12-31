from abc import ABC
from typing import Tuple
import torch
from torch import Tensor


class Metric(ABC):
    """abstract base class for all metrics.

    Note:
        Metrics should inherit from this method and define a forward method (for compatability
        with torch losses and modules)
    """

    def forward(self, inputs: Tensor, target: Tensor, **kwargs) -> Tensor:
        """given call this metric given an input and target and optional keyword arguments.

        Args:
            inp: the input tensor... generally some form of prediciton.
            target: the target tensor.

        Returns:
            a scalar tensor
        """
        return NotImplemented

    def _mask(self, inputs, targets) -> Tuple[Tensor, Tensor]:
        return inputs, targets

    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        return inputs, targets

    def _aggregate(self, result):
        return result

    def __call__(self, inp: Tensor, target: Tensor, **kwargs) -> Tensor:
        with torch.no_grad():
            return self._aggregate(self.forward(*self._prepare(*self._mask(inp, target)), **kwargs))

from typing import Tuple, Union
import torch
from torch import Tensor
from dataclasses import dataclass
from hearth.metrics.base import Metric
from hearth.metrics.functional import accuracy, precision, recall, f1, fbeta
from hearth.metrics._utils import onehot_inputs_and_targets


@dataclass
class BinaryMixin(Metric):
    """mixin for binary metrics with logit and masking support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.
    """

    from_logits: bool = False

    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        inputs, targets = super()._prepare(inputs, targets)
        targets = targets.squeeze(-1)
        inputs = inputs.reshape_as(targets)
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        return inputs, targets


@dataclass
class HardBinaryMixin(BinaryMixin):
    """This mixin rounds binary inputs."""

    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        inputs, targets = super()._prepare(inputs, targets)
        return inputs.round(), targets


@dataclass
class MaskingMixin(Metric):
    """mixin for masking inputs and targets based on a flag value in the target.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
    """

    mask_target: int = -1

    def _mask(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        valid = targets != self.mask_target
        return inputs[valid], targets[valid]


class ArgmaxMixin(Metric):
    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        indices = torch.argmax(inputs, dim=-1)
        return indices, targets


class OneHotMixin(Metric):
    def _prepare(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        return onehot_inputs_and_targets(inputs, targets)


class AverageMixin(Metric):
    def _aggregate(self, result):
        return result.mean()


class AccuracyMixin(Metric):
    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        return accuracy(inputs, targets)


@dataclass
class RecallMixin(Metric):

    eps: float = 1e-8
    dim: Union[Tuple[int], int] = 0

    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        return recall(inputs, targets, eps=self.eps, dim=self.dim)


@dataclass
class PrecisionMixin(Metric):

    eps: float = 1e-8
    dim: Union[Tuple[int], int] = 0

    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        return precision(inputs, targets, eps=self.eps, dim=self.dim)


@dataclass
class F1Mixin(Metric):

    eps: float = 1e-8
    dim: Union[Tuple[int], int] = 0

    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        return f1(inputs, targets, eps=self.eps, dim=self.dim)


@dataclass
class FBetaMixin(Metric):

    beta: float = 1
    eps: float = 1e-8
    dim: Union[Tuple[int], int] = 0

    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        return fbeta(inputs, targets, beta=self.beta, eps=self.eps, dim=self.dim)

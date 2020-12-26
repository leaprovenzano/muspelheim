from typing import Union
from dataclasses import dataclass
from torch import nn
from hearth.callbacks import Callback
from hearth.grad import trainable_parameters


@dataclass
class ClipGradNorm(Callback):
    """clips gradient norm of all trainable paramters of the ``loop.model`` on backward end.

    Args:
        max_norm: max norm of the gradients.
        norm_type: type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """

    max_norm: float
    norm_type: Union[float, int] = 2

    def on_backward_end(self, loop):
        nn.utils.clip_grad_norm_(
            trainable_parameters(loop.model), max_norm=self.max_norm, norm_type=self.norm_type
        )


@dataclass
class ClipGradValue(Callback):
    """clips gradient of all trainable paramters of the ``loop.model`` on backward end.

    Args:
        clip_value: value to clip at.
    """

    clip_value: float

    def on_backward_end(self, loop):
        nn.utils.clip_grad_value_(trainable_parameters(loop.model), clip_value=self.clip_value)

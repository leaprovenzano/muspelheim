import torch
from torch import nn


def mish(x: torch.Tensor) -> torch.Tensor:
    """functional version of mish activation see :class:`Mish` for more info."""
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Mish(nn.Module):
    """Applies the Mish activation function element-wise

    :math:`\\text{Mish}(x)=x\\tanh(softplus(x))`

    .. image:: ../images/mish.png

    **Reference**:
        `Diganta Misra: Mish: A Self Regularized Non-Monotonic Activation Function
        <https://arxiv.org/abs/1908.08681>`_

    Example:
        >>> import torch
        >>> from hearth.activations import Mish
        >>>
        >>> activation = Mish()
        >>> x = torch.linspace(-2, 2, 8)
        >>> activation(x)
        tensor([-0.2525, -0.3023, -0.2912, -0.1452,  0.1969,  0.7174,  1.3256,  1.9440])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)

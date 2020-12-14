import torch
from torch import nn
from hearth.modules import BaseModule


class Residual(BaseModule):
    """wraps a block in a residual connection :math:`y = block(x) + x`.

    Args:
        block: the module to wrap.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.modules import Residual
        >>> _ = torch.manual_seed(0)
        >>>
        >>> res = Residual(nn.Linear(4, 4))
        >>>
        >>> x = torch.rand(2, 4) # (batch, feats)
        >>> res(x)
        tensor([[ 0.6371,  1.5493,  0.0031, -0.0379],
                [ 0.3584,  0.8512,  0.5208, -0.7607]], grad_fn=<AddBackward0>)
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward padd for ``Residual`` wrapper."""
        y = self.block(x)
        return y + x

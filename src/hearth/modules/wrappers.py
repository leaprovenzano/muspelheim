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
        return x + y


class ReZero(Residual):
    """Implements ReZero residual connection around a block with dropout (as in transformer\
     implementation).

    **Reference**:
        `Bachlechner et al: ReZero is All You Need: Fast Convergence at Large Depth
        <https://arxiv.org/abs/2003.04887>`_

    Example:
        >>> import torch
        >>> from hearth.modules import ReZero
        >>>
        >>> transformation = nn.Sequential(nn.Linear(10, 10),
        ...                                nn.ReLU(),
        ...                                nn.Dropout(.1),
        ...                                nn.Linear(10, 10)
        ...                                )
        >>> re_zero = ReZero(transformation, dropout=.1)
        >>>
        >>> x = torch.normal(0, 1, size=(5, 10))
        >>> y = re_zero(x)
        >>> y.shape
        torch.Size([5, 10])

        since ``re_zero``'s weight parameter has not actually been trained ``y`` its going to be
        equal to ``x`` and nothing from the transformation will be added to the input...
        as training goes on this should change.

        >>> (y == x).all()
        tensor(True)
    """

    def __init__(self, block: nn.Module, dropout: float = 0.0):
        super().__init__(block=block)
        self.dropout = nn.Dropout(dropout)
        self.res_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward for ReZero."""
        y = self.block(x)
        y = y * self.res_weight
        return x + self.dropout(y)

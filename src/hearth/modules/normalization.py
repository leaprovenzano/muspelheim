from typing import Union, List
import torch
from torch import nn
from hearth.modules import BaseModule


class LayerNormSimple(BaseModule):
    """Layer norm without learnable bias and gain.

    This basically just wraps the standard torch LayerNorm so it has no elementwise
    affine by default.

    Args:
        normalized_shape: input shape from an expected input. If a single integer is used, it is
             treated as a singleton list, and this module will normalize over the last dimension
             which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5

    **Reference**:
        `Xu et al: Understanding and Improving Layer Normalization
        <https://arxiv.org/abs/1911.07013>`_

    Example:
        >>> import torch
        >>> from hearth.modules import LayerNormSimple
        >>> _ = torch.manual_seed(1)
        >>>
        >>> layer = LayerNormSimple(5) # 5 feats
        >>> layer
        LayerNormSimple(5, eps=1e-05)

        >>> x = torch.rand(3, 5) * 10 - 5
        >>> y = layer(x)
        >>> y.mean(dim=1)
        tensor([ 0.0000e+00, -1.7881e-08,  0.0000e+00])

        >>> y.std(dim=1)
        tensor([1.1180, 1.1180, 1.1180])

        >>> y.shape
        torch.Size([3, 5])
    """

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def __repr__(self) -> str:
        shape_or_feats = self.norm.normalized_shape
        shape_or_feats = shape_or_feats[0] if len(shape_or_feats) == 1 else shape_or_feats
        return f'{self.__class__.__name__}({shape_or_feats}, eps={self.norm.eps})'

from typing import Optional, Union, Dict, Mapping

import torch
from torch import nn
from hearth.containers import TensorDict, NumberDict
from hearth._multihead import _MultiHeadFunc


class MultiHeadLoss(nn.Module, _MultiHeadFunc):
    """wrapper for losses for models with multiple output heads.

    Args:
        weights: a mapping of key to scalar weight, will be multiplied with losses before summing
            to create the aggregate value at `aggregate_key`, need not sum to 1. Defaults to None
            (all losses weighted evenly).
        aggregate_key: the key to use for the aggregate loss. Defaults to 'weighted_sum'.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.losses import MultiHeadLoss
        >>>
        >>> _ =torch.manual_seed(0)
        >>>
        >>> loss = MultiHeadLoss(a=nn.BCEWithLogitsLoss(),
        ...                      b=nn.CrossEntropyLoss())
        >>> loss
        MultiHeadLoss(a=BCEWithLogitsLoss(),
                      b=CrossEntropyLoss(),
                      weights=NumberDict({'a': 1.0, 'b': 1.0}),
                      aggregate_key=weighted_sum)

        multihead loss expects inputs and targets to be dicts with containing
        the it's keys (in this case 'a' and 'b'):

        >>> batch_size = 10
        >>> inputs = {'a': torch.rand(batch_size, 1),
        ...           'b': torch.normal(batch_size, 1, size=(batch_size, 4))}
        >>> targets = {'a': torch.rand(batch_size, 1).round(),
        ...            'b': torch.randint(4, size=(batch_size,))}
        >>>
        >>> loss(inputs, targets)
        TensorDict({'a': tensor(0.5791), 'b': tensor(1.2425), 'weighted_sum': tensor(1.8216)})

        ``weighted_sum`` is the default aggregate key. this is the bit you should call backward on.
        you can change the default aggregate key if you like by specifying it at init.

        >>> loss = MultiHeadLoss(a=nn.BCEWithLogitsLoss(),
        ...                      b=nn.CrossEntropyLoss(),
        ...                      aggregate_key='sally')
        >>> loss(inputs, targets)
        TensorDict({'a': tensor(0.5791), 'b': tensor(1.2425), 'sally': tensor(1.8216)})

        you can aslo specify ``weights`` at init to weight contribution losses differently:

        >>> loss = MultiHeadLoss(a=nn.BCEWithLogitsLoss(),
        ...                      b=nn.CrossEntropyLoss(),
        ...                      weights={'a': .2, 'b':.8})
        >>> loss(inputs, targets)
        TensorDict({'a': tensor(0.5791), 'b': tensor(1.2425), 'weighted_sum': tensor(1.1098)})
    """

    def __init__(
        self,
        *,
        weights: Optional[Union[NumberDict, Dict[str, int]]] = None,
        aggregate_key: str = 'weighted_sum',
        **kwargs,
    ):
        nn.Module.__init__(self)
        self._fns = nn.ModuleDict(kwargs)
        self.aggregate_key = aggregate_key
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: Optional[Union[NumberDict, Dict[str, int]]] = None):
        if weights is not None:
            if weights is not NumberDict:
                weights = NumberDict(weights)
            if set(weights.keys()) != set(self.keys()):
                raise ValueError('weight keys must match keys for loss functions!')
        else:
            weights = NumberDict({k: 1.0 for k in self.keys()})
        self._weights = weights

    def _aggregate(self, out):
        out[self.aggregate_key] = (out * self.weights).sum()
        return out

    def forward(
        self, inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], **kwargs
    ) -> TensorDict:
        out = TensorDict()
        for k, func in self.items():
            out[k] = func(inputs[k], targets[k], **kwargs)
        return self._aggregate(out)

    def _argrepr(self):
        return (
            super()._argrepr() + f', weights={self.weights!r}, aggregate_key={self.aggregate_key}'
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self._argrepr()})'

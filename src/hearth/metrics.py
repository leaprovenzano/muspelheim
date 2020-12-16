from abc import ABC, abstractmethod
import torch
from torch import Tensor
from dataclasses import dataclass


class Metric(ABC):
    """abstract base class for all metrics.

    Note:
        Metrics should inherit from this method and define a forward method (for compatability
        with torch losses and modules)
    """

    @abstractmethod
    def forward(self, inp: Tensor, target: Tensor, **kwargs) -> Tensor:
        """given call this metric given an input and target and optional keyword arguments.

        Args:
            inp: the input tensor... generally some form of prediciton.
            target: the target tensor.

        Returns:
            a scalar tensor
        """
        return NotImplemented

    def __call__(self, inp: Tensor, target: Tensor, **kwargs) -> Tensor:
        with torch.no_grad():
            return self.forward(inp, target, **kwargs)


@dataclass
class BinaryAccuracy(Metric):
    """binary accruracy over possibly unnormalized values (see ``from_logits``) with target masking\
     support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>>
        >>>
        >>> metric = BinaryAccuracy()
        >>> metric
        BinaryAccuracy(mask_target=-1, from_logits=False)

        by default ``from_logits`` is false so we expect inputs to be normalized.

        >>> targets = torch.randint(1, size=(100,)) # (batch, 1)
        >>> predictions = torch.normal(0, 1, size=(100, 1)) # (batch, 1)
        >>>
        >>> metric(predictions.sigmoid(), targets)
        tensor(0.4400)

        alternatively we could specify ``from_logits=True`` in which case inputs
        will be sigmoided for us before comparing with targets.

        >>> metric = BinaryAccuracy(from_logits=True)
        >>> metric(predictions, targets) # no sigmoid!
        tensor(0.4400)

        this metric also supports masked targets in the case of variable length sequence data
        which should be **batch first** ``(batch, time)`` or ``(batch, time, 1)``.

        >>> metric = BinaryAccuracy()
        >>> targets = torch.tensor([[1, 1, 1, -1], [1, 1, -1, -1], [1, 1, 1, 0]]) # (batch, time)
        >>> predictions = torch.ones(3, 4)
        >>> metric(predictions, targets)
        tensor(0.8889)
    """

    mask_target: int = -1
    from_logits: bool = False

    def forward(self, inp: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        targets = targets.squeeze(-1)
        inp = inp.reshape_as(targets)
        if self.from_logits:
            inp = torch.sigmoid(inp)
        valid = targets != self.mask_target
        masked_eq = (torch.round(inp[valid]) == targets[valid]) * 1.0
        return masked_eq.mean()


@dataclass
class ClassAccuracy(Metric):
    """classification accuracy over possibly unnormalized scores given target indices.

    Built to have a similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.

    Args:
        mask_index: mask targets to this value. defaults to ``-1``


    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>>
        >>>
        >>> metric = ClassAccuracy()
        >>> metric
        ClassAccuracy(ignore_index=-1)


        >>> targets = torch.randint(4, size=(100,)) # (bath,)
        >>> predictions = torch.normal(0, 1, size=(100, 4)) # (batch, classes)
        >>> metric(predictions, targets)
        tensor(0.2200)


        mask targets using the ``ignore_index``, particularly useful for mixed length
        sequence predicions....

        >>> targets = torch.tensor([[0, 5, 9, -1], [2, 3, -1, -1], [1, 6, 3, 4]]) # (batch, time)
        >>> predictions = torch.normal(0, 1, size=(3, 4,  9)) # (batch, time, classes)
        >>> metric(predictions, targets)
        tensor(0.1111)
    """

    ignore_index: int = -1

    def forward(self, inp: Tensor, targets: Tensor, **kwargs) -> Tensor:  # type: ignore
        _, indices = torch.max(inp, dim=-1)
        valid = targets != self.ignore_index
        masked_eq = (indices[valid] == targets[valid]) * 1.0
        return masked_eq.mean()

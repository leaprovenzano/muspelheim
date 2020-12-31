from dataclasses import dataclass
from hearth.metrics.mixins import (
    HardBinaryMixin,
    MaskingMixin,
    AccuracyMixin,
    ArgmaxMixin,
    RecallMixin,
    PrecisionMixin,
    F1Mixin,
    FBetaMixin,
    BinaryMixin,
)


@dataclass
class BinaryAccuracy(HardBinaryMixin, MaskingMixin, AccuracyMixin):
    """binary accruracy over possibly unnormalized values (see ``from_logits``) with target masking\
     support.

    Args:
        mask_target: mask targets equal to this value. defaults to ``-1``.
        from_logits: if ``True`` inputs are expected to be unnormalized and a sigmoid
            function will be applied before comparison to targets. defaults to ``False``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from hearth.metrics import BinaryAccuracy
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

    pass


@dataclass
class CategoricalAccuracy(MaskingMixin, ArgmaxMixin, AccuracyMixin):
    """categorical accuracy over possibly unnormalized scores given target indices.

    Built to have a similar interface to\
     `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/data.html#torch.nn.CrossEntropyLoss>`_.

    Args:
        mask_target: mask this value if seen in the targets to this value. defaults to ``-1``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from hearth.metrics import CategoricalAccuracy
        >>>
        >>> metric = CategoricalAccuracy()
        >>> metric
        CategoricalAccuracy(mask_target=-1)


        >>> targets = torch.randint(4, size=(100,)) # (bath,)
        >>> predictions = torch.normal(0, 1, size=(100, 4)) # (batch, classes)
        >>> metric(predictions, targets)
        tensor(0.2200)


        mask targets using the ``mask_target``, particularly useful for mixed length
        sequence predicions....

        >>> targets = torch.tensor([[0, 5, 9, -1], [2, 3, -1, -1], [1, 6, 3, 4]]) # (batch, time)
        >>> predictions = torch.normal(0, 1, size=(3, 4,  9)) # (batch, time, classes)
        >>> metric(predictions, targets)
        tensor(0.1111)
    """

    pass


@dataclass
class BinaryRecall(HardBinaryMixin, MaskingMixin, RecallMixin):
    pass


@dataclass
class BinaryPrecision(HardBinaryMixin, MaskingMixin, PrecisionMixin):
    pass


@dataclass
class SoftBinaryRecall(BinaryMixin, MaskingMixin, RecallMixin):
    pass


@dataclass
class SoftBinaryPrecision(BinaryMixin, MaskingMixin, PrecisionMixin):
    pass


@dataclass
class BinaryF1(HardBinaryMixin, MaskingMixin, F1Mixin):
    pass


@dataclass
class BinaryFBeta(HardBinaryMixin, MaskingMixin, FBetaMixin):
    pass

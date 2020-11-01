from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Union, Generic

import torch
import numpy as np

InT = TypeVar('InT')
OutT = TypeVar('OutT')
TensorApplicable = Union[torch.Tensor, np.ndarray, int, float]


class Transform(ABC, Generic[InT, OutT]):
    """Abstract base class for all transforms."""

    def _repr_args(self):
        return ''

    def __repr__(self):
        return f'{self.__class__.__name__}({self._repr_args()})'

    @abstractmethod
    def __call__(self, x: InT) -> OutT:
        return NotImplemented


class Tensorize(Transform[InT, torch.Tensor]):
    """Tensorizes the given input with optional dtype and device

    Args:
        dtype : an optional string or torch.dtype. Defaults to None.
        device : [description]. Defaults to 'cpu'.

    Example:
        >>> import torch
        >>> from muspelheim.transforms import Tensorize
        >>>
        >>> transform = Tensorize(dtype='float32')
        >>> transform([1.1, 2.2, 3.3])
        tensor([1.1000, 2.2000, 3.3000])
    """

    def __init__(
        self,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Union[str, torch.device] = 'cpu',
    ):
        self._dtype = self._get_dtype(dtype)
        self._device = device

    def _get_dtype(self, dtype):
        if dtype is None:
            return dtype

        try:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)
            return dtype
        except (AttributeError, AssertionError):
            raise TypeError(
                'dtype must be a valid torch.dtype or a string'
                f' corresponding to a valid torch.dtype got {dtype}.'
            )

    def _repr_args(self):
        return f'dtype={self._dtype}, device={self._device}'

    def __call__(self, x: InT) -> torch.Tensor:
        return torch.tensor(x, dtype=self._dtype, device=self._device)


class Normalize(Transform):
    """Normalize a tensor or array from a fixed mean and std

    Example:
        >>> import torch
        >>> from muspelheim.transforms import Normalize
        >>>
        >>> transform = Normalize(mean=1.5, std=1.1859)
        >>> x = torch.linspace(0, 3, 5)
        >>> transform(x)
        tensor([-1.2649, -0.6324,  0.0000,  0.6324,  1.2649])

        >>> channel_transform = Normalize(mean=torch.tensor([7.6596, 8.0000, 8.3404]),
        ...                                 std=torch.tensor([4.8622, 4.8622, 4.8622]))
        >>> x= torch.linspace(0, 16, 48).reshape(4, 4, 3)
        >>> y = channel_transform(x)
        >>> y.shape
        torch.Size([4, 4, 3])

        >>> y.mean(dim=(0, 1))
        tensor([-5.1707e-06,  3.7253e-08,  5.3197e-06])

        >>> y.std(dim=(0, 1))
        tensor([1.0000, 1.0000, 1.0000])
    """

    def __init__(self, mean: TensorApplicable, std: TensorApplicable):
        self.mean = mean
        self.std = std

    def _repr_args(self):
        return f'mean={self.mean}, std={self.std}'

    def __call__(self, x):
        return (x - self.mean) / self.std

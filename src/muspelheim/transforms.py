from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Union, Generic

import torch

InT = TypeVar('InT')
OutT = TypeVar('OutT')


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

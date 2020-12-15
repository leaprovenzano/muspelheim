from typing import Type
import torch
from torch import nn
from hearth._internals import Registry

_activation_registry: Registry[Type[nn.Module]] = Registry('activations')
_activation_registry.register(nn.ReLU)
_activation_registry.register(nn.RReLU)
_activation_registry.register(nn.Hardtanh)
_activation_registry.register(nn.ReLU6)
_activation_registry.register(nn.Sigmoid)
_activation_registry.register(nn.Hardsigmoid)
_activation_registry.register(nn.Tanh)
_activation_registry.register(nn.SiLU)
_activation_registry.register(nn.Hardswish)
_activation_registry.register(nn.ELU)
_activation_registry.register(nn.CELU)
_activation_registry.register(nn.SELU)
_activation_registry.register(nn.GLU)
_activation_registry.register(nn.GELU)
_activation_registry.register(nn.Hardshrink)
_activation_registry.register(nn.LeakyReLU)
_activation_registry.register(nn.LogSigmoid)
_activation_registry.register(nn.Softplus)
_activation_registry.register(nn.Softshrink)
_activation_registry.register(nn.PReLU)
_activation_registry.register(nn.Softsign)
_activation_registry.register(nn.Tanhshrink)


def mish(x: torch.Tensor) -> torch.Tensor:
    """functional version of mish activation see :class:`Mish` for more info."""
    return x * torch.tanh(torch.nn.functional.softplus(x))


@_activation_registry.register
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


def get_activation(name: str, **kwargs) -> nn.Module:
    """get an new instance activiation by its name, optionally passing any extra kwargs.

    Args:
        name: string name of the activation. will be best effort normalized.

    Returns:
        nn.Module: instance of an activation module

    Example:
        >>> from torch import nn
        >>> from hearth.activations import get_activation
        >>>
        >>> get_activation('relu')
        ReLU()

        you can also pass kwargs... although this is slightly silly
        since you may as well use the actual module then...

        >>> get_activation('prelu', num_parameters=6)
        PReLU(num_parameters=6)

        name will be normalized so dont worry about casing:

        >>> get_activation('mish')
        Mish()

        >>> get_activation('MISH')
        Mish()
    """
    return _activation_registry[name](**kwargs)  # type: ignore

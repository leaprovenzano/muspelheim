from typing import Iterator
from torch import nn


def freeze(model: nn.Module):
    """freeze a model in place

    Args:
        model : model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module):
    """unfreeze a model in place

    Args:
        model : model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def requires_grad(param: nn.Parameter) -> bool:
    """return true if the parameter requires grad"""
    return param.requires_grad


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """yields trainable parameters from a model."""
    yield from filter(requires_grad, model.parameters())

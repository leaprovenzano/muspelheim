from typing import Iterator, Tuple
from torch import nn


def freeze(model: nn.Module) -> nn.Module:
    """freeze a model in place

    Args:
        model : model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model: nn.Module) -> nn.Module:
    """unfreeze a model in place

    Args:
        model : model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def allgrad(model: nn.Module) -> bool:
    """return true if all parameters in the model require grad."""
    return all(p.requires_grad for p in model.parameters())


def anygrad(model: nn.Module) -> bool:
    """return true if any parameters in the model require grad."""
    return any(p.requires_grad for p in model.parameters())


def requires_grad(param: nn.Parameter) -> bool:
    """return true if the parameter requires grad"""
    return param.requires_grad


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """yields trainable parameters from a model."""
    yield from filter(requires_grad, model.parameters())


def named_trainable_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    """yields named trainable params from a model

    Args:
        model: model to get named params from

    Yields:
        tuples of name (str), param (nn.Parameter)
    """
    yield from filter(lambda x: requires_grad(x[-1]), model.named_parameters())  # type: ignore

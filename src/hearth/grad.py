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

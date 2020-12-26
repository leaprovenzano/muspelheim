import torch
from torch import nn
import pytest
from hearth.grad import freeze, unfreeze, trainable_parameters


def py_model():
    return nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))


def script_model():
    model = py_model()
    return torch.jit.script(model)


def partially_frozen_model(freeze_n: int = 1):
    model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Linear(3, 3))
    for i in range(freeze_n):
        freeze(model[i])
        assert not model[i].weight.requires_grad
        assert not model[i].bias.requires_grad
    return model


@pytest.mark.parametrize(
    'model,',
    [py_model(), script_model()],
)
def test_freeze(model):
    freeze(model)
    for param in model.parameters():
        assert not param.requires_grad


@pytest.mark.parametrize(
    'model,',
    [py_model(), script_model()],
)
def test_unfreeze(model):
    freeze(model)
    unfreeze(model)
    for param in model.parameters():
        assert param.requires_grad


@pytest.mark.parametrize(
    'model,',
    [py_model(), partially_frozen_model(1), partially_frozen_model(2)],
)
def test_trainable_params(model):
    for param in trainable_parameters(model):
        assert param.requires_grad

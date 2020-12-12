import torch
from torch import nn
import pytest
from hearth.grad import freeze, unfreeze


def py_model():
    return nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))


def script_model():
    model = py_model()
    return torch.jit.script(model)


@pytest.mark.parametrize(
    'model,', [py_model(), script_model()],
)
def test_freeze(model):
    freeze(model)
    for param in model.parameters():
        assert not param.requires_grad


@pytest.mark.parametrize(
    'model,', [py_model(), script_model()],
)
def test_unfreeze(model):
    freeze(model)
    unfreeze(model)
    for param in model.parameters():
        assert param.requires_grad

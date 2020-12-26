from dataclasses import dataclass
import pytest
import torch
import hearth
from torch import nn
from hearth.callbacks import ClipGradNorm, ClipGradValue


@dataclass
class FakeLoop:
    model: nn.Module


@pytest.fixture
def loop():
    model = nn.Sequential(nn.Linear(2, 5), nn.Linear(5, 1))
    return FakeLoop(model)


def test_clipgradnorm(mocker, loop):
    patched_trainable_params = mocker.patch.object(
        hearth.callbacks.grad_clipping,
        'trainable_parameters',
        autospec=True,
        return_value=torch.tensor(1.0),
    )
    spy = mocker.spy(torch.nn.utils, 'clip_grad_norm_')
    callback = ClipGradNorm(max_norm=0.5)
    callback.on_backward_end(loop)
    patched_trainable_params.assert_called_once_with(loop.model)
    spy.assert_called_once_with(torch.tensor(1.0), max_norm=0.5, norm_type=2)


def test_clipgradvalue(mocker, loop):
    patched_trainable_params = mocker.patch.object(
        hearth.callbacks.grad_clipping,
        'trainable_parameters',
        autospec=True,
        return_value=torch.tensor(1.0),
    )
    spy = mocker.spy(torch.nn.utils, 'clip_grad_value_')

    callback = ClipGradValue(clip_value=5.0)
    callback.on_backward_end(loop)
    patched_trainable_params.assert_called_once_with(loop.model)
    spy.assert_called_once_with(torch.tensor(1.0), clip_value=5.0)

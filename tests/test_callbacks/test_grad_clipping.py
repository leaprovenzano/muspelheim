from dataclasses import dataclass
import pytest
import torch
from torch import nn
from hearth.callbacks import ClipGradNorm, ClipGradValue


@dataclass
class FakeLoop:
    model: nn.Module


@pytest.fixture
def loop():
    model = nn.Sequential(nn.Linear(2, 5), nn.Linear(5, 1))
    loop = FakeLoop(model)
    return loop


def test_clipgradnorm(mocker, loop):
    spy = mocker.spy(torch.nn.utils, 'clip_grad_norm_')
    callback = ClipGradNorm(max_norm=0.5)
    callback.on_backward_end(loop)
    spy.assert_called_once()
    call_args = spy.call_args_list[0]
    print(call_args.args[0])
    assert call_args.args[0].__name__ == 'trainable_parameters'
    assert call_args.kwargs['max_norm'] == 0.5
    assert call_args.kwargs['norm_type'] == 2


def test_clipgradvalue(mocker, loop):
    spy = mocker.spy(torch.nn.utils, 'clip_grad_value_')
    callback = ClipGradValue(clip_value=5.0)
    callback.on_backward_end(loop)
    spy.assert_called_once()
    call_args = spy.call_args_list[0]
    print(call_args.args[0])
    assert call_args.args[0].__name__ == 'trainable_parameters'
    assert call_args.kwargs['clip_value'] == 5.0

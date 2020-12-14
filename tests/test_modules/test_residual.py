import pytest
import torch
from torch import nn
from hearth.modules import Residual


def test_script():
    model = Residual(nn.Linear(4, 4))
    model.freeze()

    try:
        scripted = model.script()
    except RuntimeError as err:
        pytest.fail(f'failed to script model with script method! raised {err}')

    x = torch.normal(0, 1, (4, 4))
    model_y = model(x)
    scripted_y = scripted(x)
    torch.testing.assert_allclose(scripted_y, model_y)
    assert scripted_y.requires_grad == model_y.requires_grad


def test_with_identity():
    res = Residual(nn.Identity())
    x = torch.rand(3, 3)
    expected = 2 * x
    with torch.no_grad():
        y = res(x)
    torch.testing.assert_allclose(y, expected)

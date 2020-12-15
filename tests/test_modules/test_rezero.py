import pytest
import torch
from torch import nn
from hearth.modules import ReZero


def test_script():
    model = ReZero(nn.Linear(10, 10), dropout=0.1)
    model.freeze()
    model.eval()

    try:
        scripted = model.script()
    except RuntimeError as err:
        pytest.fail(f'failed to script model with script method! raised {err}')

    x = torch.normal(0, 1, (5, 10))
    model_y = model(x)
    scripted_y = scripted(x)
    torch.testing.assert_allclose(scripted_y, model_y)
    assert scripted_y.requires_grad == model_y.requires_grad

import pytest
import torch
from hearth._internals import MissingFromRegistryError
from hearth.activations import get_activation
from hearth.activations import Mish


@pytest.mark.parametrize('mod,', [Mish(), get_activation('mish'), get_activation('relu')])
def test_script(mod):
    try:
        scripted = torch.jit.script(mod)
    except RuntimeError as err:
        pytest.fail(f'failed to script activation {mod.__name__} with script method! raised {err}')

    x = torch.normal(0, 10, size=(100,))
    orig_y = mod(x)
    scripted_y = scripted(x)
    torch.testing.assert_allclose(scripted_y, orig_y)


def test_unregistered_activation():
    with pytest.raises(MissingFromRegistryError, match='boop is not in activations registry!'):
        get_activation('boop')

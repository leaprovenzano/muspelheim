import pytest
from torch import nn
from hearth.grad import anygrad, allgrad, freeze


@pytest.mark.parametrize(
    'model, expected',
    [
        pytest.param(nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)), True, id="unfrozen"),
        pytest.param(
            nn.Sequential(freeze(nn.Linear(5, 5)), nn.Linear(5, 5)), True, id="partially_frozen"
        ),
        pytest.param(
            freeze(nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))), False, id="all_frozen"
        ),
    ],
)
def test_anygrad(model, expected):
    return anygrad(model) is expected


@pytest.mark.parametrize(
    'model, expected',
    [
        pytest.param(nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)), True, id="unfrozen"),
        pytest.param(
            nn.Sequential(freeze(nn.Linear(5, 5)), nn.Linear(5, 5)), False, id="partially_frozen"
        ),
        pytest.param(
            freeze(nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))), False, id="all_frozen"
        ),
    ],
)
def test_allgrad(model, expected):
    return allgrad(model) is expected

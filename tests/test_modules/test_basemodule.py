import pytest
import json
import torch
from torch import nn
from hearth.modules import BaseModule
from hearth._config import SupportsConfig


class SimpleModel(BaseModule):
    def __init__(self, in_feats: int, out_feats: int, dropout=0.5):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        return self.dropout(x)


class Subclassed(SimpleModel):
    def __init__(self, *args, extra='boop', **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = extra


class Container(BaseModule):
    def __init__(self, *modules):
        super().__init__()
        self.wrapped = nn.Sequential(*modules)

    def forward(self, x):
        return self.wrapped(x)


class ModelWithModuleArg(BaseModule):
    def __init__(self, in_feats: int, hidden_feats: int, hidden: nn.Module, dropout=0.5):
        super().__init__()
        self.in_layer = nn.Linear(in_feats, hidden_feats)
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        x = self.hidden(x)
        x = self.dropout(x)
        return self.out_layer(x)


@pytest.mark.parametrize(
    'model_type,',
    [
        BaseModule,
        SimpleModel,
        Subclassed,
        Container,
        ModelWithModuleArg,
    ],
)
def test_supports_config(model_type):
    assert issubclass(model_type, SupportsConfig)


@pytest.mark.parametrize(
    'model,',
    [
        SimpleModel(4, 5),
        Subclassed(4, 5, dropout=0.2),
        Container(nn.Linear(2, 2), nn.Linear(2, 2)),
        ModelWithModuleArg(3, 4, hidden=nn.Linear(4, 4)),
        ModelWithModuleArg(3, 4, hidden=nn.Sequential(nn.Linear(4, 4), nn.ReLU())),
        ModelWithModuleArg(3, 4, hidden=Subclassed(4, 4, dropout=0.1)),
    ],
)
def test_config_is_jsonable(model):
    try:
        config = model.config()
        json.dumps(config)
    except (TypeError, json.JSONDecodeError) as err:
        pytest.fail(f'config {config} is not jsonable! got error {err}')

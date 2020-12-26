import json
import pytest
import torch
import os
from torch import nn
from hearth.grad import freeze
from hearth.modules import BaseModule
from hearth._config import SupportsConfig


class Dummy(BaseModule):
    def __init__(self, in_feats: int, hidden: int, out_feats: int):
        super().__init__()
        self.lin1 = nn.Linear(in_feats, hidden)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden, out_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


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


def test_freeze():
    model = Dummy(4, 5, 2)
    model.freeze()
    for param in model.parameters():
        assert not param.requires_grad


def test_unfreeze():
    model = Dummy(4, 5, 2)
    model.freeze()
    model.unfreeze()
    for param in model.parameters():
        assert param.requires_grad


def test_script():
    model = Dummy(4, 5, 2)
    model.freeze()

    try:
        scripted = model.script()
    except RuntimeError as err:
        pytest.fail(f'failed to script model with script method! raised {err}')

    x = torch.normal(0, 1, (3, 4))
    model_y = model(x)
    scripted_y = scripted(x)
    torch.testing.assert_allclose(scripted_y, model_y)
    assert scripted_y.requires_grad == model_y.requires_grad


def test_trainable_parameters():
    model = Dummy(4, 5, 2)
    freeze(model.lin1)
    assert not model.lin1.weight.requires_grad
    assert not model.lin1.bias.requires_grad

    params = list(model.trainable_parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)
    assert (params[0] == model.lin2.weight).all()
    assert (params[1] == model.lin2.bias).all()


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


@pytest.mark.parametrize(
    'model,',
    [
        Dummy(3, 5, 2),
        SimpleModel(3, 5),
        Subclassed(3, 5, dropout=0.2),
        Container(nn.Linear(3, 2), nn.Linear(2, 2)),
        ModelWithModuleArg(3, 4, hidden=nn.Linear(4, 4)),
        ModelWithModuleArg(3, 4, hidden=nn.Sequential(nn.Linear(4, 4), nn.ReLU())),
        ModelWithModuleArg(3, 4, hidden=Subclassed(4, 4, dropout=0.1)),
    ],
)
def test_save_load(model, tmpdir):

    # make temp model dir
    model_dir = str(tmpdir.mkdir('model'))
    model.save(model_dir)

    assert os.path.exists(os.path.join(model_dir, 'state.pt'))
    assert os.path.exists(os.path.join(model_dir, 'config.json'))

    loaded_model = model.__class__.load(model_dir)
    assert isinstance(loaded_model, model.__class__)
    assert loaded_model.config() == model.config()

    dummy_inp = torch.normal(0, 1, size=(5, 3))

    with torch.no_grad():
        orig_out = model.eval()(dummy_inp)
        loaded_out = loaded_model.eval()(dummy_inp)

    torch.testing.assert_allclose(loaded_out, orig_out)

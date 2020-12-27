from dataclasses import dataclass
import os
import torch
from torch import nn
import pytest
from hearth.callbacks import ModelCheckpoint
from hearth.events import Improvement, Stagnation, ModelSaved
from hearth.modules import BaseModule


@dataclass
class DummyLoop:

    model: nn.Module

    def __post_init__(self):
        self._event_log = []

    def fire(self, event):
        self._event_log.append(event)


class HearthModel(BaseModule):
    def __init__(self, in_feats: int = 3, out_feats: int = 6):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.mark.parametrize(
    'callback, event',
    [
        (ModelCheckpoint('fake'), Improvement('blah', 'blah', 0, 0, 0)),
        (ModelCheckpoint('fake', field='loss'), Improvement('loss', 'blah', 0, 0, 0)),
        (ModelCheckpoint('fake', field='loss', stage='val'), Improvement('loss', 'val', 0, 0, 0)),
        (ModelCheckpoint('fake', stage='val'), Improvement('blah', 'val', 0, 0, 0)),
    ],
)
def test_should_save(callback, event):
    assert not callback._should_save
    callback.on_event(loop=None, event=event)
    assert callback._should_save


@pytest.mark.parametrize(
    'callback, event',
    [
        (ModelCheckpoint('fake'), Stagnation('blah', 'blah', 0, 0)),
        (ModelCheckpoint('fake', field='loss', stage='val'), Stagnation('loss', 'val', 0, 0)),
        (ModelCheckpoint('fake', field='loss'), Improvement('metric', 'val', 0, 0, 0)),
        (
            ModelCheckpoint('fake', field='loss', stage='val'),
            Improvement('metric', 'train', 0, 0, 0),
        ),
    ],
)
def test_should_not_save(callback, event):
    assert not callback._should_save
    callback.on_event(loop=None, event=event)
    assert not callback._should_save


def test_not_a_dir_on_registration(tmpdir):
    # make a file
    p = tmpdir.mkdir("models").join("hello.txt")
    p.write("hello")
    callback = ModelCheckpoint(model_dir=str(p))

    fakeloop = DummyLoop(model=HearthModel())
    with pytest.raises(
        NotADirectoryError, match='ModelCheckpoint expects model_dir to be a directory!'
    ):
        callback.on_registration(fakeloop)


def test_not_a_basemodule_on_registration(tmpdir):
    # make a file
    p = tmpdir.mkdir('modeldir')
    callback = ModelCheckpoint(model_dir=str(p))

    fakeloop = DummyLoop(model=nn.Linear(5, 3))
    with pytest.raises(TypeError, match='ModelCheckpoint callback only supports hearth.BaseModule'):
        callback.on_registration(fakeloop)


def test_on_registration_when_everythings_fine(tmpdir):
    # make a file
    base = tmpdir.mkdir('models')
    model_dir = os.path.join(str(base), 'boop')
    callback = ModelCheckpoint(model_dir=model_dir)
    fakeloop = DummyLoop(model=HearthModel())
    callback.on_registration(fakeloop)
    assert os.path.exists(model_dir)
    assert os.path.isdir(model_dir)


def test_full_save_on_event(tmpdir):
    # make a file
    base = tmpdir.mkdir('models')
    model_dir = os.path.join(str(base), 'dummy')

    callback = ModelCheckpoint(model_dir=model_dir)
    loop = DummyLoop(model=HearthModel())

    # sim registration
    callback.on_registration(loop)
    assert not callback._should_save

    # sim event
    event = Improvement(field='loss', stage='val', steps=1, best=0.1, last_best=0.2)
    callback.on_event(loop, event)
    assert callback._should_save

    # sim epoch end...
    callback.on_epoch_end(loop)
    # now model should be saved
    assert os.path.exists(os.path.join(model_dir, 'state.pt'))
    assert os.path.exists(os.path.join(model_dir, 'config.json'))
    # and should be able to be reloaded
    loaded = HearthModel.load(model_dir)
    assert loaded.config() == loop.model.config()
    assert (loaded.linear.weight == loop.model.linear.weight).all()
    assert (loaded.linear.bias == loop.model.linear.bias).all()
    # and _should_save should be false
    assert not callback._should_save

    # and the loop should have seen an event...
    assert loop._event_log == [ModelSaved(model_dir)]

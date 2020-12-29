from typing import Optional
from dataclasses import dataclass
import os
import torch
from torch import nn
import pytest
from hearth.callbacks import Checkpoint, History
from hearth.events import Improvement, Stagnation, CheckpointSaved
from hearth.modules import BaseModule


@dataclass
class DummyLoop:

    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    history: Optional[History] = None

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
        (Checkpoint('fake'), Improvement('blah', 'blah', 0, 0, 0)),
        (Checkpoint('fake', field='loss'), Improvement('loss', 'blah', 0, 0, 0)),
        (Checkpoint('fake', field='loss', stage='val'), Improvement('loss', 'val', 0, 0, 0)),
        (Checkpoint('fake', stage='val'), Improvement('blah', 'val', 0, 0, 0)),
    ],
)
def test_should_save(callback, event):
    assert not callback._should_save
    callback.on_event(loop=None, event=event)
    assert callback._should_save


@pytest.mark.parametrize(
    'callback, event',
    [
        (Checkpoint('fake'), Stagnation('blah', 'blah', 0, 0)),
        (Checkpoint('fake', field='loss', stage='val'), Stagnation('loss', 'val', 0, 0)),
        (Checkpoint('fake', field='loss'), Improvement('metric', 'val', 0, 0, 0)),
        (
            Checkpoint('fake', field='loss', stage='val'),
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
    callback = Checkpoint(model_dir=str(p))

    fakeloop = DummyLoop(model=HearthModel())
    with pytest.raises(NotADirectoryError, match='Checkpoint expects model_dir to be a directory!'):
        callback.on_registration(fakeloop)


def test_not_a_basemodule_on_registration(tmpdir):
    # make a file
    p = tmpdir.mkdir('modeldir')
    callback = Checkpoint(model_dir=str(p))

    fakeloop = DummyLoop(model=nn.Linear(5, 3))
    with pytest.raises(TypeError, match='Checkpoint callback only supports hearth.BaseModule'):
        callback.on_registration(fakeloop)


def test_on_registration_when_everythings_fine(tmpdir):
    # make a file
    base = tmpdir.mkdir('models')
    model_dir = os.path.join(str(base), 'boop')
    callback = Checkpoint(model_dir=model_dir)
    fakeloop = DummyLoop(model=HearthModel())
    callback.on_registration(fakeloop)
    assert os.path.exists(model_dir)
    assert os.path.isdir(model_dir)


def test_full_save_on_event(tmpdir):
    # make a file
    base = tmpdir.mkdir('models')
    model_dir = os.path.join(str(base), 'dummy')
    history = History(
        {
            'epoch': 0,
            'lrs': {'group0': 0.001},
            'train': {'loss': 0.53, 'metric': 0.85},
            'val': {'loss': 0.20, 'metric': 0.93},
        }
    )
    model = HearthModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    callback = Checkpoint(
        model_dir=model_dir,
    )
    loop = DummyLoop(model=model, history=history, optimizer=optimizer)

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
    loaded_model = HearthModel.load(model_dir)
    assert loaded_model.config() == loop.model.config()
    assert (loaded_model.linear.weight == loop.model.linear.weight).all()
    assert (loaded_model.linear.bias == loop.model.linear.bias).all()

    # test saved history
    assert os.path.exists(os.path.join(model_dir, 'history.json'))
    # and should be able to be reloaded
    loaded_history = History.load(model_dir)
    assert loaded_history == loop.history

    # test saved optimizer state
    assert os.path.exists(os.path.join(model_dir, 'optimizer_state.pt'))
    loaded_opt_state = torch.load(os.path.join(model_dir, 'optimizer_state.pt'))
    assert loaded_opt_state == loop.optimizer.state_dict()

    # and _should_save should be false
    assert not callback._should_save
    # and the loop should have seen an event...
    assert loop._event_log == [CheckpointSaved(model_dir)]

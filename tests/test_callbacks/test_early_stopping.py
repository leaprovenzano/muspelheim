from dataclasses import dataclass
import pytest
from hearth.callbacks import EarlyStopping
from hearth.events import Stagnation, Improvement, EarlyStop


@dataclass
class DummyLoop:

    epoch: int = 0

    def __post_init__(self):
        self.should_stop = False
        self._event_log = []

    def fire(self, event):
        self._event_log.append(event)


@pytest.mark.parametrize(
    'callback, event',
    [
        (EarlyStopping(), Stagnation(field='blah', stage='blah', steps=6, best=0.1)),
        (EarlyStopping(field='loss'), Stagnation(field='loss', stage='blah', steps=6, best=0.1)),
        (EarlyStopping(stage='val'), Stagnation(field='metric', stage='val', steps=6, best=0.1)),
        (EarlyStopping(patience=2), Stagnation(field='loss', stage='val', steps=3, best=0.1)),
    ],
)
def test_should_stop(callback, event):
    loop = DummyLoop(epoch=0)
    assert not loop.should_stop
    callback.on_event(loop, event)
    assert loop.should_stop
    callback.on_epoch_end(loop)
    assert loop._event_log == [EarlyStop(0)]


@pytest.mark.parametrize(
    'callback, event',
    [
        (
            EarlyStopping(field='loss', stage='val'),
            Improvement(field='loss', stage='val', steps=6, best=0.1, last_best=0.2),
        ),
        (EarlyStopping(patience=5), Stagnation(field='loss', stage='val', steps=3, best=0.1)),
        (EarlyStopping(patience=100), Stagnation(field='loss', stage='val', steps=5, best=0.1)),
        (EarlyStopping(stage='val'), Stagnation(field='loss', stage='train', steps=6, best=0.1)),
        (EarlyStopping(field='loss'), Stagnation(field='metric', stage='val', steps=6, best=0.1)),
        (
            EarlyStopping(field='loss', stage='val'),
            Stagnation(field='metric', stage='val', steps=6, best=0.1),
        ),
        (
            EarlyStopping(field='loss', stage='val'),
            Stagnation(field='loss', stage='train', steps=6, best=0.1),
        ),
    ],
)
def test_should_not_stop(callback, event):
    loop = DummyLoop(epoch=0)
    assert not loop.should_stop
    callback.on_event(loop, event)
    assert not loop.should_stop
    callback.on_epoch_end(loop)
    assert loop._event_log == []

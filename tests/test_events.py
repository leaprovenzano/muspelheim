import pytest
from hearth.events import Improvement, Stagnation, ModelSaved, EarlyStop


@pytest.mark.parametrize(
    'event, msg',
    [
        (
            Stagnation(field='loss', stage='val', steps=2, best=0.01),
            'Stagnation[val.loss] stagnant for 2 steps no improvement from 0.0100.',
        ),
        (
            Improvement(field='metric', stage='val', steps=2, best=0.9, last_best=0.8),
            'Improvement[val.metric] improved from : 0.8000 to 0.9000 in 2 steps.',
        ),
        (ModelSaved('some/dir'), 'ModelSaved model checkpoint saved to some/dir.'),
        (EarlyStop(1), 'EarlyStop triggered at epoch 1.'),
    ],
)
def test_logmsg(event, msg):
    assert event.logmsg() == msg

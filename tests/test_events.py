import pytest
from hearth.events import Improvement, Stagnation


@pytest.mark.parametrize(
    'event, msg',
    [
        (
            Stagnation(field='loss', steps=2, best=0.01),
            'Stagnation[loss] loss stagnant for 2 steps no improvement from 0.0100.',
        ),
        (
            Improvement(field='metric', steps=2, best=0.9, last_best=0.8),
            'Improvement[metric] metric improved from : 0.8000 to 0.9000 in 2 steps.',
        ),
    ],
)
def test_logmsg(event, msg):
    assert event.logmsg() == msg

import pytest
from hearth.callbacks import Callback, CallbackManager


class Bim(Callback):
    pass


class Bam(Callback):
    pass


class Boop(Callback):
    pass


def test_valid_init():
    CallbackManager(Bim(), Bam())


def test_len():
    callbacks = CallbackManager(Bim(), Bam(), Bam())
    assert len(callbacks) == 3


def test_getitem():
    a, b, c, d = Bim(), Bam(), Boop(), Bam()
    callbacks = CallbackManager(a, b, c, d)
    assert callbacks[0] is a
    assert callbacks[-1] is d

    assert isinstance(callbacks[:2], CallbackManager)
    assert len(callbacks[:2]) == 2
    assert callbacks[:2][-1] is b


@pytest.mark.parametrize(
    'method,',
    [
        'on_registration',
        'on_stage_start',
        'on_stage_end',
        'on_epoch_start',
        'on_epoch_end',
        'on_batch_start',
        'on_batch_end',
        'on_loss_start',
        'on_loss_end',
        'on_step_start',
        'on_step_end',
        'on_metric_start',
        'on_metric_end',
        'on_backward_start',
        'on_backward_end',
    ],
)
def test_standard_callback_methods(mocker, method):
    bim, bam = Bim(), Bam()
    bim_spy = mocker.spy(bim, method)
    bam_spy = mocker.spy(bam, method)

    callbacks = CallbackManager(bim, bam)
    method_ = getattr(callbacks, method)

    method_(1)  # not actually a loop... but whatever

    bim_spy.assert_called_once_with(1)
    bam_spy.assert_called_once_with(1)


def test_on_event(mocker):
    bim, bam = Bim(), Bam()
    bim_spy = mocker.spy(bim, 'on_event')
    bam_spy = mocker.spy(bam, 'on_event')

    callbacks = CallbackManager(bim, bam)

    fake_loop = 'loop'
    fake_event = 'fake'

    callbacks.on_event(fake_loop, fake_event)

    bim_spy.assert_called_once_with(fake_loop, fake_event)
    bam_spy.assert_called_once_with(fake_loop, fake_event)


def test_bad_init():
    expected_msg = 'expected Callback type but got <class \'str\'>.'
    with pytest.raises(TypeError, match=expected_msg):
        CallbackManager(Bim(), 'bad')

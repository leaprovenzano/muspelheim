from dataclasses import dataclass
from hearth.callbacks import Callback
from hearth.callbacks.utils import if_active, if_inactive


class FakeLoop:
    pass


@dataclass
class ActiveSwitchCallback(Callback):

    active: bool = False
    on_epoch_start_calls = 0
    on_epoch_end_calls = 0

    @if_inactive
    def on_epoch_start(self, loop):
        self.on_epoch_start_calls += 1

    @if_active
    def on_epoch_end(self, loop):
        self.on_epoch_end_calls += 1


def test_if_active_deco():
    loop = FakeLoop()
    callback = ActiveSwitchCallback()
    assert not callback.active
    # should not increase calls since not active
    callback.on_epoch_end(loop)
    assert callback.on_epoch_end_calls == 0

    callback.active = True
    # ok now it should do something

    callback.on_epoch_end(loop)
    assert callback.on_epoch_end_calls == 1


def test_if_inactive_deco():
    loop = FakeLoop()
    callback = ActiveSwitchCallback()
    assert not callback.active
    # should not increase calls since not active
    callback.on_epoch_start(loop)
    assert callback.on_epoch_start_calls == 1

    callback.active = True
    # ok now it should do not call any more

    callback.on_epoch_start(loop)
    assert callback.on_epoch_start_calls == 1

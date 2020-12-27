from typing import Sequence, Type, Optional
from collections import deque
from hearth.callbacks import Callback
from dataclasses import dataclass
from hearth.events import MonitoringEvent, Stagnation, EarlyStop


@dataclass
class EarlyStopping(Callback):
    """This callback stops the training loop early if it recieves a monitoring event for enough \
    steps.

    Args:
        patience: wait for this many steps (epochs) before triggering stopping.
        event_types: Types of monitoring events to checkpoint on.
            Default is (:class:`hearth.events.Stagnation`, )
        prepare_model: Optional callable which accepts and returns
             a :class:`BaseModule` for preparing the model to be saved.
             This function will always recieve a **copy** of the model on the loop just for
             safety. Defaults to None.
        field: If provided only stop on events where field matches this field.
        stage: If provided only stop on events where stage matches this stage.

    **Active On:**
        - event
        - epoch_end

    **Events Listened For:**
        - :class:`hearth.events.Stagnation` (default)

    **Events Emitted:**
        - :class:`hearth.events.EarlyStop`

    **Accesses Loop Attributes:**
        - should_stop

    **Modifies Loop Attributes:**
        - should_stop

    **Accesses Event Attributes:**
        - field
        - stage
        - steps
    """

    patience: int = 5
    event_types: Sequence[Type[MonitoringEvent]] = (Stagnation,)
    field: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self):
        self._pending_event = deque([], maxlen=1)

    def _should_stop(self, event):
        if isinstance(event, self.event_types):
            if self.field:
                if event.field != self.field:
                    return False
            if self.stage:
                if event.stage != self.stage:
                    return False
            return event.steps > self.patience
        return False

    def on_epoch_end(self, loop):
        if self._pending_event:
            event = self._pending_event.pop()
            loop.fire(event)

    def on_event(self, loop, event):
        if self._should_stop(event):
            loop.should_stop = True
            self._pending_event.append(EarlyStop(epoch=loop.epoch))

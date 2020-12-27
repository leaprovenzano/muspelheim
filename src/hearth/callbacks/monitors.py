import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal
from dataclasses import dataclass
import operator
from functools import reduce, partial
from hearth.callbacks.base import Callback
from hearth.events import Improvement, Stagnation


def dotted_attrgetter(path: str, obj):
    return reduce(getattr, path.split('.'), obj)


@dataclass
class ImprovementMonitor(Callback):
    """This callback monitors a metric or loss on the specified stage for improvement and \
    stagnation.

    When improvement or stagnation is detected on the metric it emits \
     :class:`hearth.events.Improvement` and :class:`hearth.events.Stagnation` events which can be \
    used in other callbacks.

    Args:
        field: the field to monitor on the loop... may be a dotted path string for nested objects.
            Defaults to 'loss'.
        improvement_on: string operator specifier ie (lt is less_than and gt is greater than).
            Defaults to 'lt' (for loss measurement).
        stage: named stage to measure improvement on. this should correspond to stages on your
            loop obviously. Defaults to 'val'.
        stagnant_after: wait this number of steps before you start issuing stagnation events.
            Defaults to 1.

    **Active On:**
        - stage_end

    **Events Emitted:**
        - :class:`hearth.events.Improvement`
        - :class:`hearth.events.Stagnation`

    **Accesses Loop Attributes:**
        - metric or loss
        - stage
        - epoch
    """

    field: str = 'loss'
    improvement_on: Literal['gt', 'lt'] = 'lt'
    stage: str = 'val'
    stagnant_after: int = 1

    def __post_init__(self):
        if self.improvement_on not in ('gt', 'lt'):
            raise ValueError(
                f'improvement_on must be one of ["gt", "lt"] but got {self.improvement_on}'
            )
        self._get_value = partial(dotted_attrgetter, self.field)
        self._is_improvement = getattr(operator, self.improvement_on)
        self._last_best = float('inf') if self.improvement_on == 'lt' else -float('inf')
        self._best_step = -1

    def on_stage_end(self, loop):
        if loop.stage == self.stage:
            this_value = self._get_value(loop)
            steps = loop.epoch - self._best_step
            event = None
            if self._is_improvement(this_value, self._last_best):
                event = Improvement(
                    self.field,
                    stage=self.stage,
                    steps=steps,
                    best=this_value,
                    last_best=self._last_best,
                )
                self._best_step = loop.epoch
                self._last_best = this_value
            elif steps > self.stagnant_after:
                event = Stagnation(
                    field=self.field, stage=self.stage, steps=steps, best=self._last_best
                )
            if event:
                loop.fire(event)

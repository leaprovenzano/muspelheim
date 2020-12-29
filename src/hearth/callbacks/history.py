import os
from typing import Dict
from pprint import pformat
from collectionish import AttyDict
from collections import deque
from hearth.callbacks import Callback
from hearth._file_utils import load_json, save_json


class History(Callback):
    """This callback tracks history across epochs.

    Note:
        this callback is auto-registered :class:`hearth.loop.Loop` by default so you
        don't need to list it in callbacks. It should be accessable at ``loop.history``

    **Active On:**
        - epoch_start
        - stage_end
        - epoch_end

    **Accesses Loop Attributes:**
        - loss
        - metrics
        - epoch
        - optimizer
        - stage
    """

    @classmethod
    def load(cls, model_dir) -> 'History':
        """load this history object from the model_dir"""
        path = os.path.join(model_dir, 'history.json')
        hist = load_json(path)
        return cls(*hist)

    def __init__(self, *history):
        self._history = [AttyDict(row) for row in history]
        self._current_step_buffer = deque(maxlen=1)

    @property
    def current_step(self):
        """the current step if tracking is in the middle of an epoch."""
        return self._current_step_buffer[0]

    @property
    def last_epoch(self) -> int:
        """the last epoch in the complete history"""
        if len(self):
            return self[-1].epoch
        return -1

    def _gather_lrs(self, optimizer) -> Dict[str, float]:
        return {f'group{i}': group['lr'] for i, group in enumerate(optimizer.param_groups)}

    def on_epoch_start(self, loop):
        current_step = AttyDict(epoch=loop.epoch, lrs=self._gather_lrs(loop.optimizer))
        self._current_step_buffer.append(current_step)

    def on_stage_end(self, loop):
        self.current_step[loop.stage] = AttyDict(loss=loop.loss, metric=loop.metric)

    def on_epoch_end(self, loop):
        self._history.append(self._current_step_buffer.pop())

    def __len__(self) -> int:
        return len(self._history)

    def __getitem__(self, i):
        return self._history[i]

    def save(self, model_dir: str):
        """save this history to a file history.json in ``model_dir```."""
        path = os.path.join(model_dir, 'history.json')
        save_json(self._history, path)

    def _iter_pformat_args(self):
        if not self._history:
            yield ''
        else:
            padding = len(self.__class__.__name__)
            lines = pformat(self._history).split('\n')
            lines[-1] = lines[-1][:-1]
            yield lines.pop(0)[1:]
            for line in lines:
                yield ' ' * padding + line

    def __repr__(self):
        reprargs = '\n'.join(self._iter_pformat_args())
        return f'{self.__class__.__name__}({reprargs})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._history == other._history
        return NotImplemented

from typing import Sequence, Type, Optional, Callable
import os
from dataclasses import dataclass
from copy import deepcopy
import torch
from hearth.events import MonitoringEvent
from hearth.callbacks import Callback
from hearth.events import Improvement, CheckpointSaved
from hearth.modules import BaseModule
from hearth._file_utils import mkdirs_if_not_exist


@dataclass
class Checkpoint(Callback):
    """This callback saves checkpoints on certain events.

    Note:
        only models derived from :class:`hearth.modules.BaseModule` are currently supported.

    Args:
        model_dir: directory to save model checkpoint in. If the directory does not exist
            it will be created on registration.
        event_types: Types of monitoring events to checkpoint on.
            Default is (:class:`hearth.events.Improvement`, )
        prepare_model: Optional callable which accepts and returns
             a :class:`BaseModule` for preparing the model to be saved.
             This function will always recieve a **copy** of the model on the loop just for
             safety. Defaults to None.
        field: If provided only save on events where field matches this field.
        stage: If provided only save on events where stage matches this stage.
        save_history: if True save the loop history at this step to the model dir. Defaults to True
        save_optimizer: if True save the optimizer state dict to the model dir. Defaults to True.

    **Active On:**
        - registration
        - event
        - epoch_end

    **Events Listened For:**
        - :class:`hearth.events.Improvement` (default)

    **Events Emitted:**
        - :class:`hearth.events.ModelSaved`

    **Accesses Loop Attributes:**
        - model
        - history (if save_history is True)
        - optimizer (if save_optimizer is True)

    **Accesses Event Attributes:**
        - field
        - stage
    """

    model_dir: str
    event_types: Sequence[Type[MonitoringEvent]] = (Improvement,)
    prepare_model: Optional[Callable[[BaseModule], BaseModule]] = None
    field: Optional[str] = None
    stage: Optional[str] = None
    save_history: bool = True
    save_optimizer: bool = True

    def __post_init__(self):
        self._should_save = False

    def on_registration(self, loop):
        # check model
        if not isinstance(loop.model, BaseModule):
            raise TypeError(f'{self.__class__.__name__} callback only supports hearth.BaseModule')
        if os.path.exists(self.model_dir):
            if not os.path.isdir(self.model_dir):
                raise NotADirectoryError(
                    f'{self.__class__.__name__} expects model_dir to be a directory!'
                )
        else:
            mkdirs_if_not_exist(self.model_dir, verbose=True)

    def _is_save_event(self, event):
        if isinstance(event, self.event_types):
            if self.field:
                if event.field != self.field:
                    return False
            if self.stage:
                if event.stage != self.stage:
                    return False
            return True
        return False

    def _save_model(self, model):
        save_model = deepcopy(model)
        if self.prepare_model is not None:
            save_model = self.prepare_model(save_model)
        save_model.save(self.model_dir)

    def _save_history(self, history):
        history.save(self.model_dir)

    def _save_optimizer(self, optimizer):
        path = os.path.join(self.model_dir, 'optimizer_state.pt')
        torch.save(optimizer.state_dict(), path)

    def save_checkpoint(self, loop):
        self._save_model(loop.model)
        if self.save_history:
            self._save_history(loop.history)
        if self.save_optimizer:
            self._save_optimizer(loop.optimizer)

    def on_epoch_end(self, loop):
        if self._should_save:
            self.save_checkpoint(loop)
            event = CheckpointSaved(self.model_dir)
            loop.fire(event)
            self._should_save = False

    def on_event(self, loop, event):
        if self._is_save_event(event):
            self._should_save = True

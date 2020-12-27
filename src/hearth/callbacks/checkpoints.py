from typing import Sequence, Type, Optional, Callable
import os
from dataclasses import dataclass
from copy import deepcopy
from hearth.events import MonitoringEvent
from hearth.callbacks import Callback
from hearth.events import Improvement, ModelSaved
from hearth.modules import BaseModule
from hearth._file_utils import mkdirs_if_not_exist


@dataclass
class ModelCheckpoint(Callback):
    """This callback saves model checkpoints on certain events.

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

    **Accesses Event Attributes:**
        - field
        - stage
    """

    model_dir: str
    event_types: Sequence[Type[MonitoringEvent]] = (Improvement,)
    prepare_model: Optional[Callable[[BaseModule], BaseModule]] = None
    field: Optional[str] = None
    stage: Optional[str] = None

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

    def on_epoch_end(self, loop):
        if self._should_save:
            self._save_model(loop.model)
            event = ModelSaved(self.model_dir)
            loop.fire(event)
            self._should_save = False

    def on_event(self, loop, event):
        if self._is_save_event(event):
            self._should_save = True

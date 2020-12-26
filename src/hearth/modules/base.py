import os
from typing import Iterator
import torch
from torch import nn
from hearth.grad import freeze, unfreeze, trainable_parameters
from hearth._config import _init_wrapper, from_config
from hearth._file_utils import save_json, load_json


class BaseModule(nn.Module):
    """A base class like nn.Module but with a few extra useful bits.

    **features include**:
        - auto config generation based on init args.
        - standardized loading and saving.
        - torchscripting via the :meth:`script` method.
        - whole model freezing via the :meth:`freeze` method.
        - and more...
    """

    @classmethod
    def from_config(cls, config):
        """given a valid config return a new instance of this Module."""
        return from_config(cls, config)

    @classmethod
    def load(cls, model_dir: str, strict: bool = True):
        """create a new instance of this with config and parameters loaded from  ``model_dir``.

        This method expects model dir to have the following files:
             - state.pt: the state dict for this model
             - config.json :  the config of this model, nessisary to reinstantiate a new model \
                 with the :meth:`from_config`  method.

         you can use the :meth:`save` method on an instance of this module to create them.

         Args:
             model_dir: directory to load from.
             strict: if we shold be strict about loading the state dict. Defaults to True.
        """
        config = load_json(os.path.join(model_dir, 'config.json'))
        state_dict = torch.load(os.path.join(model_dir, 'state.pt'))

        instance = cls.from_config(config)
        instance.load_state_dict(state_dict, strict=strict)
        return instance

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__init__ = _init_wrapper(cls.__init__)

    def freeze(self):
        """freeze this model in place."""
        freeze(self)

    def unfreeze(self):
        """unfreeze this model in place."""
        unfreeze(self)

    def script(self) -> torch.jit.RecursiveScriptModule:
        """torchscript this model using jit.script."""
        with torch.jit.optimized_execution(True):
            scripted = torch.jit.script(self)
        return scripted

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """yields trainable parameters from this model."""
        yield from trainable_parameters(self)

    def config(self):
        """get the config for this module.

        This config can be passed to the :meth:`from_config` class method to create a new
        instance of this module.
        """
        return dict(self.__config__)

    def save(self, model_dir: str):
        """save this models state and config to a ``model_dir`` so it can be re-created later.

        This method will generate the following files in the ``model_dir``:
            - state.pt: the state dict for this model
            - config.json :  the config of this model, nessisary to reinstantiate a new model \
                with the :meth:`from_config`  method.

        Args:
            model_dir: directory to save stuff in.
        """
        torch.save(self.state_dict(), os.path.join(model_dir, 'state.pt'))
        save_json(self.config(), path=os.path.join(model_dir, 'config.json'))

from typing import Iterator
import torch
from torch import nn
from hearth.grad import freeze, unfreeze, trainable_parameters
from hearth._config import _init_wrapper, from_config


class BaseModule(nn.Module):
    """A base class like nn.Module but with a few extra useful bits.

    features include:
        - auto config generation based on init args.

    """

    @classmethod
    def from_config(cls, config):
        """given a valid config return a new instance of this Module."""
        return from_config(cls, config)

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

        This config can be passed to the `from_config` class method to create a new instance of \
        this module.
        """
        return self.__config__

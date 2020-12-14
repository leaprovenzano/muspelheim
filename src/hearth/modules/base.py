import torch
from torch import nn
from hearth.grad import freeze, unfreeze


class BaseModule(nn.Module):
    """A base class like nn.Module but with a few extra useful bits."""

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
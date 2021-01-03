from typing import Union, Sized
from dataclasses import dataclass
from torch.utils.data import Dataset

from hearth.containers import TensorDict


@dataclass
class XYDataset(Dataset):
    """basic dataset that returns a tuple of inputs and targets.

    supports :class:`hearth.containers.TensorDataset`
    """

    x: Union[Sized, TensorDict]
    y: Union[Sized, TensorDict]

    def __len__(self) -> int:
        if isinstance(self.x, TensorDict):
            return len(next(iter(self.x.values())))  # type: ignore
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

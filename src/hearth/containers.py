from collectionish import NumAttyDict
import torch


class NumberDict(NumAttyDict):
    """just like
    `collectionish.NumAttyDict\
    <https://collectionish.readthedocs.io/en/stable/_autosummary/collectionish.NumAttyDict.html>`_ \
     but with a special format method for better handling in logging.
    """

    def __format__(self, *args, **kwargs):
        return " ".join(f'{k}: {format(v, *args, **kwargs)}' for k, v in self.items())


class TensorDict(NumAttyDict):
    """a very basic keyed attribute accessable container for tensors.

    Can handle basic python math operators since it comes from :class:`NumAttyDict`
    """

    def to(self, device):
        """move all tensors in this Tensordict to device"""
        for v in self.values():
            v.to(device)
        return self

    def __setitem__(self, k, v):
        if not isinstance(v, (TensorDict, torch.Tensor)):
            raise TypeError('values must be either tensors or TensorDicts')
        self[k] = v

    def item(self) -> 'NumAttyDict':
        """get python numbers out of this Tensordict by calling item on all tensors in it.

        Note:
            this will only work if all values in this tensordict are scalar tensors!

        Returns:
            :class:`NumberDict`
        """
        return NumberDict({k: v.item() for k, v in self.items()})  # type: ignore

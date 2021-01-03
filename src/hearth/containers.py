from collectionish import NumAttyDict


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

    def _idx_tensors(self, idx):
        return self.__class__({k: v[idx] for k, v in self.items()})

    def __getitem__(self, idx_or_key):
        if isinstance(idx_or_key, str):
            return dict.__getitem__(self, idx_or_key)
        return self._idx_tensors(idx_or_key)

    def item(self) -> 'NumAttyDict':
        """get python numbers out of this Tensordict by calling item on all tensors in it.

        Note:
            this will only work if all values in this tensordict are scalar tensors!

        Returns:
            :class:`NumberDict`
        """
        return NumberDict({k: v.item() for k, v in self.items()})  # type: ignore

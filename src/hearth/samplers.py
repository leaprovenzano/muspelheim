"""Samplers for sampling batch indices (or example indexes) for use with \
`torch.utils.data.DataLoader\
<https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
"""
from typing import Sized, List, Iterator
import torch
from torch.utils.data import Sampler


class SubsequenceSampler(Sampler):
    """a batch sampler that keeps sequences aligned within batches but shuffles batches.

    Often when training on a dataset that represents a full sequence we may want to
    window that sequence into multiple subsequences and shuffle those. Additionally
    when batch_size is not divisible by dataset length we choose a different short sequence
    on each iteration which means batches are not always starting and ending at the same
    position.

    Args:
        dataset: something we can call ``len()`` on that represents the size of the dataset being
            batched. It can be the dataset itself, a tensor, list, range etc...
        batch_size: the desired batch size
        drop_shortest: if ``True`` drop the shortest batch on each iteration if the size of the
            dataset is not divisible by ``batch_size``. The short batch will be chosen randomly
            on each iteration (providing a little extra noise and ensuring we dont see exactly
            the same subsequences on each iteration). Defaults to False.

    Example:

        >>> import torch
        >>> from hearth.samplers import SubsequenceSampler
        >>> _ = torch.manual_seed(0)
        >>>
        >>> # subsequence sampler only dataset
        >>> # so it could be a range object or a Dataset, a tensor etc...
        >>> sampler = SubsequenceSampler(range(15), batch_size=4)
        >>> sampler
        SubsequenceSampler(dataset=range(0, 15), batch_size=4, drop_shortest=False)

        by default we will get one short batch of 3, since ``batch_size=4`` and  \
        ``drop_shortest=False``:

        >>> for batch in sampler:
        ...     print(batch)
        [11, 12, 13, 14]
        [0, 1, 2]
        [7, 8, 9, 10]
        [3, 4, 5, 6]


        between runs we shuffle and choose a new short batch:

        >>> for batch in sampler:
        ...     print(batch)
        [12, 13, 14]
        [8, 9, 10, 11]
        [0, 1, 2, 3]
        [4, 5, 6, 7]


        when ``drop_shortest=True`` the randomly chosen short batch will be dropped...

        >>> for batch in SubsequenceSampler(range(15), batch_size=4, drop_shortest=True):
        ...     print(batch)
        [11, 12, 13, 14]
        [0, 1, 2, 3]
        [7, 8, 9, 10]

        and again this will be different for each iteration:

        >>> for batch in SubsequenceSampler(range(15), batch_size=4, drop_shortest=True):
        ...     print(batch)
        [11, 12, 13, 14]
        [7, 8, 9, 10]
        [3, 4, 5, 6]


        when ``batch_size`` is divisible by the number of examples in the dataset
        behavior is a little more deterministic, since the indexes in each batch will always
        be the same:

        >>> for batch in SubsequenceSampler(range(12), batch_size=4):
        ...     print(batch)
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [8, 9, 10, 11]

        but batches will still be shuffled between iterations:

        >>> for batch in SubsequenceSampler(range(12), batch_size=4):
        ...     print(batch)
        [8, 9, 10, 11]
        [0, 1, 2, 3]
        [4, 5, 6, 7]

    """

    def __init__(self, dataset: Sized, batch_size: int, drop_shortest: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_shortest = drop_shortest

    def _get_lengths(self, total_samples: int) -> List[int]:
        short_batch_size = total_samples % self.batch_size
        n_batches = total_samples // self.batch_size + (short_batch_size > 0)
        lengths = [self.batch_size] * n_batches

        if short_batch_size != 0:
            short_batch_idx: int = torch.randint(high=n_batches, size=(1,)).item()  # type: ignore
            lengths[short_batch_idx] = short_batch_size

        return lengths

    def __iter__(self) -> Iterator[List[int]]:
        total_samples = len(self.dataset)
        batches = torch.split_with_sizes(
            torch.arange(total_samples), self._get_lengths(total_samples)
        )
        batch_idxes = torch.randperm(len(batches))
        for i in batch_idxes:
            batch = batches[i]
            if not self.drop_shortest or (len(batch) == self.batch_size):
                yield batch.tolist()

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_shortest:
            return n // self.batch_size  # type: ignore
        else:
            return (n + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = (
            f'dataset={self.dataset},'
            f'batch_size={self.batch_size},'
            f'drop_shortest={self.drop_shortest}'
        )
        return f'{name}({args})'

from hearth.samplers import SubsequenceSampler
from itertools import chain


def test_subsequence_sampler():

    sampler = SubsequenceSampler(range(100), batch_size=12)
    batches = []
    for batch in sampler:
        assert batch == list(range(batch[0], batch[-1] + 1))
        batches.append(batch)

    assert len(batches) == len(sampler)
    assert sum(map(lambda x: len(x) == 12, batches)) == 8
    assert sorted(chain.from_iterable(batches)) == list(range(100))

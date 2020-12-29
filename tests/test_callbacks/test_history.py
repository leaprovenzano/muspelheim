import os
import pytest
from hearth.callbacks import History


@pytest.fixture
def history():
    return History(
        {
            'epoch': 0,
            'lrs': {'group0': 0.001},
            'train': {'loss': 0.53, 'metric': 0.85},
            'val': {'loss': 0.20, 'metric': 0.93},
        },
        {
            'epoch': 1,
            'lrs': {'group0': 0.001},
            'train': {'loss': 0.27, 'metric': 0.92},
            'val': {'loss': 0.14, 'metric': 0.95},
        },
        {
            'epoch': 2,
            'lrs': {'group0': 0.001},
            'train': {'loss': 0.22, 'metric': 0.93},
            'val': {'loss': 0.12, 'metric': 0.96},
        },
    )


def test_len(history):
    assert len(history) == 3


def test_attydict_features(history):
    assert history[0].epoch == 0
    assert history[0].train.loss == 0.53


def test_last_epoch(history):
    assert history.last_epoch == 2


def test_load_save(history, tmpdir):
    expected_path = f'{str(tmpdir)}/history.json'
    history.save(tmpdir)
    assert os.path.exists(expected_path)
    loaded = History.load(tmpdir)
    assert isinstance(loaded, History)
    assert len(loaded) == len(history)
    assert loaded == history

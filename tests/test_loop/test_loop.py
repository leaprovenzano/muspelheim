import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from hearth.metrics import BinaryAccuracy
from hearth.loop import Loop


def gen_xor_dataset(n: int, val_split: float = 0.2):
    x = torch.rand(n, 2)
    y = np.logical_xor(x[:, 0] >= 0.5, x[:, 1] >= 0.5).unsqueeze(-1) * 1.0
    x = x * 2 - 1

    n_train = n - int(round(n * val_split))

    train = torch.utils.data.TensorDataset(x[:n_train], y[:n_train])
    val = torch.utils.data.TensorDataset(x[n_train:], y[n_train:])

    return train, val


def test_full_loop_on_simple_xor():
    train, val = gen_xor_dataset(10000)

    train_batches = DataLoader(train, batch_size=32, shuffle=True, drop_last=False)
    val_batches = DataLoader(val, batch_size=32, shuffle=True, drop_last=False)

    model = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
    )

    loop = Loop(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        loss_fn=nn.BCELoss(),
        metric_fn=BinaryAccuracy(),
    )

    loop(train_batches, val_batches, 4)
    assert loop.epoch == 4
    assert loop.stage == 'val'
    assert loop.metric >= 0.9

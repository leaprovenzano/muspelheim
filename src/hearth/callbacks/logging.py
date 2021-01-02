from dataclasses import dataclass
from hearth.callbacks import Callback
from hearth.events import Event


DEFAULT_BATCH_FMT = (
    'epoch: {loop.epoch} stage: [{loop.stage}]'
    ' batch: {loop.batches_seen}/{loop.n_batches}'
    ' loss: {loop.loss:0.4f}'
)

DEFAULT_METRIC_FMT = " {loop.metric:0.4f}"


@dataclass
class PrintLogger(Callback):
    """a very simple logging callback that just prints stuff to sdout.

    Args:
        batch_format: format string which will printed (single line) for each batch
            and will passed a single argument ``loop``. Defaults to \
            `hearth.callbacks.logging.DEFAULT_BATCH_FMT`.
        epoch_delim: single char delimiter that will be used to seperate epochs. Defaults to ``-``.
        epoch_delim_width: width of epoch delimiter. Defaults to 80.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> _ = torch.manual_seed(0)
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import PrintLogger
        >>> from hearth.metrics import BinaryAccuracy
        >>>
        >>> train = TensorDataset(torch.normal(0, 2, size=(5000, 64)),
        ...                       torch.randint(2, size=(5000, 1))*1.0)
        >>> val = TensorDataset(torch.normal(0, 2, size=(3000, 64)),
        ...                     torch.randint(2, size=(3000, 1))*1.0)
        >>>
        >>> train_batches = DataLoader(train, batch_size=32, shuffle=True, drop_last=False)
        >>> val_batches = DataLoader(val, batch_size=32, shuffle=True, drop_last=False)
        >>>
        >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        >>>
        >>> loop = Loop(model=model,
        ...         optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        ...        loss_fn = nn.BCELoss(),
        ...        metrics = BinaryAccuracy(),
        ...        callbacks= [PrintLogger()]
        ...       )
        >>> loop(train_batches, val_batches, 2) # doctest: +SKIP
        epoch: 0 stage: [train] batch: 157/157 loss: 0.7036 metric: 0.5054
        epoch: 0 stage: [val] batch: 94/94 loss: 0.7036 metric: 0.4933
        --------------------------------------------------------------------------------
        epoch: 1 stage: [train] batch: 157/157 loss: 0.6799 metric: 0.5660
        epoch: 1 stage: [val] batch: 94/94 loss: 0.7097 metric: 0.4860
        --------------------------------------------------------------------------------
    """

    batch_format: str = DEFAULT_BATCH_FMT
    metric_format: str = DEFAULT_METRIC_FMT
    epoch_delim: str = '-'
    epoch_delim_width: int = 80

    def __post_init__(self):
        self._end_cursor = None

    def print_msg(self, msg: str):
        print(msg, end=self._end_cursor, flush=True, sep='')

    def on_registration(self, loop):
        if loop._has_metrics:
            self.batch_format = self.batch_format + self.metric_format

    def get_batch_msg(self, loop) -> str:
        return self.batch_format.format(loop=loop)

    def on_epoch_end(self, loop):
        print(self.epoch_delim * self.epoch_delim_width)

    def on_batch_start(self, loop):
        if loop.n_batches and (loop.batches_seen + 1 == loop.n_batches):
            self._end_cursor = None
        else:
            self._end_cursor = '\r'

    def on_batch_end(self, loop):
        return self.print_msg(self.get_batch_msg(loop))

    def on_stage_end(self, loop):
        self._end_cursor = None

    def on_event(self, loop, event: Event):
        print(f'EVENT: {event.logmsg()}')

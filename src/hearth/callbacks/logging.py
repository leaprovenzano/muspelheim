from dataclasses import dataclass
from callbacks import Callback

_default_batch_fmt = (
    'epoch: {loop.epoch} stage: [{loop.stage}]'
    ' batch: {loop.batches_seen}/{loop.n_batches}'
    ' loss: {loop.loss:0.4}'
    ' metric: {loop.metric: 0.4}'
)


@dataclass
class PrintLogger(Callback):

    batch_format: str = _default_batch_fmt
    epoch_delim: str = '-'
    epoch_delim_width: int = 80

    def __post_init__(self):
        self._end_cursor = None

    def print_msg(self, msg: str):
        print(msg, end=self._end_cursor)

    def get_batch_msg(self, loop) -> str:
        return self.batch_format.format(loop=loop)

    def on_epoch_end(self, loop):
        print(self.epoch_delim * self.epoch_delim_width)

    def on_batch_start(self, loop):
        if loop.n_batches and (loop.batches_seen == loop.n_batches):
            self._end_cursor = None
        else:
            self._end_cursor = '\r'

    def on_batch_end(self, loop):
        return self.print_msg(self.get_batch_msg(loop))

    def on_stage_end(self, loop):
        self._end_cursor = None
        print()

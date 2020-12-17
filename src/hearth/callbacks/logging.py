from dataclasses import dataclass
from hearth.callbacks import Callback

DEFAULT_BATCH_FMT = (
    'epoch: {loop.epoch} stage: [{loop.stage}]'
    ' batch: {loop.batches_seen}/{loop.n_batches}'
    ' loss: {loop.loss:0.4}'
    ' metric: {loop.metric:0.4}'
)


@dataclass
class PrintLogger(Callback):
    """a very simple logging callback that just prints stuff to sdout.

    Args:
        batch_format: format string which will printed (single line) for each batch
            and will passed a single argument ``loop``. Defaults to \
            `hearth.callbacks.logging.DEFAULT_BATCH_FMT`.
        epoch_delim: single char delimiter that will be used to seperate epochs. Defaults to ``-``.
        epoch_delim_width: width of epoch delimiter. Defaults to 80.
    """

    batch_format: str = DEFAULT_BATCH_FMT
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

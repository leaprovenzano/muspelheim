from typing import Sequence, Callable, Optional
import torch
from torch import nn
from contextlib import nullcontext
from hearth.callbacks import Callback, CallbackManager
from hearth.metrics import Running


class Loop:

    """The simplest kind of loop for basic supervised learning.

    Note:
        If you have more custom things you'd like to to that cant be handled
        in callbacks it's recommended to subclass this and overide the  ``handle_batch`` method.
    """

    stages = ('train', 'val')

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        metric_fn: Optional[Callable],
        callbacks: Sequence[Callback] = (),
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = Running(loss_fn)
        self.metric_fn = Running(metric_fn)  # type: ignore
        self.n_batches = 0
        self.batches_seen = 0
        self.stage = self.stages[0]
        self.epoch = 0
        self.callbacks = CallbackManager(*callbacks)
        self.callbacks.on_registration(self)

    def grad_context(self):
        if self.stage == 'train':
            return nullcontext()
        return torch.no_grad()

    def _requires_backward(self) -> bool:
        return self.stage == 'train'

    @property
    def loss(self):
        return self.loss_fn.average

    @property
    def metric(self):
        return self.metric_fn.average

    def optimizer_step(self):
        self.callbacks.on_step_start(self)
        self.optimizer.step()
        self.callbacks.on_step_end(self)

    def compute_loss(self, *args, **kwargs):
        self.callbacks.on_loss_start(self)
        loss = self.loss_fn(*args, **kwargs)
        self.callbacks.on_loss_end(self)
        return loss

    def compute_metric(self, *args, **kwargs):
        with torch.no_grad():
            self.callbacks.on_metric_start(self)
            metric = self.metric_fn(*args, **kwargs)
            self.callbacks.on_metric_end(self)
        return metric

    def backward(self, loss, **kwargs):
        self.callbacks.on_backward_start(self)
        loss.backward()
        self.callbacks.on_backward_end(self)

    def forward(self, *args, **kwargs):
        with self.grad_context():
            return self.model(*args, **kwargs)

    def handle_batch(self, batch):
        # do forward pass and get loss
        self.optimizer.zero_grad()
        # unpack the batch... you can override this if your batch differs...
        x, y = batch

        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)
        if self._requires_backward():
            self.backward(loss)
            self.optimizer_step()

        # compute metrics (grad management is already handled in compute metrics fn)
        self.compute_metric(y_hat, y)

    def handle_batches(self, batches):
        self.n_batches = len(batches)
        self.batches_seen = 0
        for batch in batches:
            self.callbacks.on_batch_start(self)
            self.handle_batch(batch)
            self.batches_seen += 1
            self.callbacks.on_batch_end(self)

    def handle_stage(self, stage, batches):
        self.stage = stage
        if self.stage == 'train':
            self.model.train()
        else:
            self.model.eval()
        self.metric_fn.reset()
        self.loss_fn.reset()
        self.callbacks.on_stage_start(self)

        self.handle_batches(batches)
        self.callbacks.on_stage_end(self)

    def fire(self, event):
        self.callbacks.on_event(event, self)

    def __call__(self, train, val, epochs: int = 1):
        for _ in range(epochs):
            self.callbacks.on_epoch_start(self)
            for stage, batches in zip(self.stages, (train, val)):
                self.handle_stage(stage, batches)
            self.callbacks.on_epoch_end(self)
            self.epoch += 1

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(model={self.model},'
            f' optimizer={self.optimizer}'
            f' loss_fn={self.loss_fn}'
            f' metric_fn={self.metric_fn}'
            f' callbacks={self.callbacks})'
        )

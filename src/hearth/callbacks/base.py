from typing import Union


class Callback:
    """Base class for all callbacks.

    Child callbacks should subclass this and override methods they want to actually do something on.
    all methods will be passed the loop  and the ``on_event`` method will additonally be passed an
    event.
    """

    active: bool = True

    def on_registration(self, loop):
        """this will be called when the loop sets up the callbacks before any training starts."""
        pass

    def on_stage_start(self, loop):
        """This will be called on the start of a stage.

        an identifier such as `train`, `val` etc will be accessable at ``loop.stage``.
        """
        pass

    def on_stage_end(self, loop):
        """This will be called on the start of a stage.

        an identifier such as `train`, `val` etc will be accessable at ``loop.stage``.
        """
        pass

    def on_epoch_start(self, loop):
        """This will be called on when each epoch starts."""
        pass

    def on_epoch_end(self, loop):
        """This will be called on when each epoch ends."""
        pass

    def on_batch_start(self, loop):
        """this will be called before the batch is passed to the model on each stage"""
        pass

    def on_batch_end(self, loop):
        """this will be called after each batch is proccessed (for each stage)"""
        pass

    def on_loss_start(self, loop):
        """this will be called for each batch before loss is calculated (for each stage)"""
        pass

    def on_loss_end(self, loop):
        """this will be called for each batch after the loss is calculated (for each stage)"""
        pass

    def on_step_start(self, loop):
        """this will be called for before the optimizer step (for training stage only)"""
        pass

    def on_step_end(self, loop):
        """this will be called after the optimizer step (for training stage only)"""
        pass

    def on_metric_start(self, loop):
        """this will be called just before metrics are calculated (for each batch)"""
        pass

    def on_metric_end(self, loop):
        """this will be called just after are metrics calculated (for each batch)"""
        pass

    def on_backward_start(self, loop):
        """this will be called just before backward is called on the loss (for each batch)"""
        pass

    def on_backward_end(self, loop):
        """this will be called just after backward is called (for each batch)"""
        pass

    def on_event(self, loop, event):
        """this will be called whenever an event is fired"""
        pass

    def _repr_args(self) -> str:
        return ''

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._repr_args()})'


class CallbackManager(Callback):
    """Manages multiple callbacks and calls them in order."""

    def __init__(self, *callbacks: Callback):
        self._callbacks = []
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError(f'expected Callback type but got {type(callback)}.')

            self._callbacks.append(callback)

    def on_registration(self, loop):
        for callback in self:
            callback.on_registration(loop)

    def on_stage_start(self, loop):
        for callback in self:
            callback.on_stage_start(loop)

    def on_stage_end(self, loop):
        for callback in self:
            callback.on_stage_end(loop)

    def on_epoch_start(self, loop):
        for callback in self:
            callback.on_epoch_start(loop)

    def on_epoch_end(self, loop):
        for callback in self:
            callback.on_epoch_end(loop)

    def on_batch_start(self, loop):
        for callback in self:
            callback.on_batch_start(loop)

    def on_batch_end(self, loop):
        for callback in self:
            callback.on_batch_end(loop)

    def on_loss_start(self, loop):
        for callback in self:
            callback.on_loss_start(loop)

    def on_loss_end(self, loop):
        for callback in self:
            callback.on_loss_end(loop)

    def on_step_start(self, loop):
        for callback in self:
            callback.on_step_start(loop)

    def on_step_end(self, loop):
        for callback in self:
            callback.on_step_end(loop)

    def on_metric_start(self, loop):
        for callback in self:
            callback.on_metric_start(loop)

    def on_metric_end(self, loop):
        for callback in self:
            callback.on_metric_end(loop)

    def on_backward_start(self, loop):
        for callback in self:
            callback.on_backward_start(loop)

    def on_backward_end(self, loop):
        for callback in self:
            callback.on_backward_end(loop)

    def on_event(self, loop, event):
        for callback in self:
            callback.on_event(loop, event)

    def _repr_args(self) -> str:
        return ','.join(f'{callback!r}' for callback in self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def __getitem__(self, i: Union[int, slice]) -> Callback:
        if isinstance(i, slice):
            return CallbackManager(*self._callbacks[i])
        return self._callbacks[i]  # type: ignore

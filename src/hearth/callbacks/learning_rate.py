import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal
from typing import Type, ClassVar, Optional, Union, Callable, List
from dataclasses import dataclass
from itertools import starmap
from functools import reduce, partial
import operator

from torch.optim.lr_scheduler import (
    _LRScheduler,
    LambdaLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from torch.optim.lr_scheduler import MultiplicativeLR  # type: ignore

from hearth.callbacks import Callback
from hearth.callbacks.utils import if_active


def _dotted_attrgetter(path: str, obj):
    return reduce(getattr, path.split('.'), obj)


class _LRSchedulerCallback(Callback):
    """Base class for lr scheduler callbacks.

    all lr scheduler callbacks and bases should inherit from this class.
    """

    _scheduler_cls: ClassVar[Type[_LRScheduler]] = NotImplemented

    start_epoch: int = 0
    end_epoch: Optional[int] = None

    def __init_subclass__(cls, scheduler_cls=None, *args, **kwargs):
        if scheduler_cls:
            cls._scheduler_cls = scheduler_cls
            # add start epoch and end epoch as optional
            # keyword only dataclass fields
            cls.__annotations__['start_epoch'] = int
            cls.__annotations__['end_epoch'] = Optional[int]
            cls.start_epoch = 0
            cls.end_epoch = None
            # autodataclass
            cls = dataclass(cls)
        return super().__init_subclass__(*args, **kwargs)

    def __post_init__(self):
        self.active = False

    def _exclude_from_scheduler_kwargs(self):
        return ('start_epoch', 'end_epoch')

    def _get_scheduler_kwargs(self):
        exclude = self._exclude_from_scheduler_kwargs()
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in exclude
        }

    def _build_scheduler(self, optimizer):
        self.scheduler = self._scheduler_cls(optimizer, **self._get_scheduler_kwargs())

    def _on_activation(self, loop):
        self._build_scheduler(loop.optimizer)
        self.active = True

    def on_epoch_start(self, loop):
        if loop.epoch == self.start_epoch:
            self._on_activation(loop)

        elif loop.epoch == self.end_epoch:
            self.active = False


class EpochLRSchedulerCallback(_LRSchedulerCallback):
    """base class for lr scheduler callbacks that modify learning rate based on epoch."""

    @if_active
    def on_epoch_end(self, loop):
        self.scheduler.step()


class LambdaLRCallback(EpochLRSchedulerCallback, scheduler_cls=LambdaLR):
    """Sets the learning rate of each parameter group to the initial lr times a given function.

    Args:
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        start_epoch: this callback will start at this epoch. Defaults to 0
        end_epoch: : this callback will stop at this epoch. Defaults to None

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import LambdaLRCallback
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ... we'll stop early at epoch 3
        >>> callback = LambdaLRCallback(lambda x: x**2, end_epoch=3)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 5 epochs
        >>> loop(train, val, epochs=5)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.1}
        2 {'group0': 0.4}
        3 {'group0': 0.9}
        4 {'group0': 0.9}
    """

    lr_lambda: Union[Callable, List[Callable]]


class MultiplicativeLRCallback(EpochLRSchedulerCallback, scheduler_cls=MultiplicativeLR):
    """Multiply the learning rate of each parameter group by the factor given in the specified \
    function.

    Args:
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        start_epoch: this callback will start at this epoch. Defaults to 0
        end_epoch: : this callback will stop at this epoch. Defaults to None

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import MultiplicativeLRCallback
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ... we'll start the scheduler rolling at epoch2
        >>> callback = MultiplicativeLRCallback(lambda x: .8, start_epoch=2)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 6 epochs
        >>> loop(train, val, epochs=6)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.1}
        2 {'group0': 0.1}
        3 {'group0': 0.08}
        4 {'group0': 0.064}
        5 {'group0': 0.0512}
    """

    lr_lambda: Union[Callable, List[Callable]]


class StepLRCallback(EpochLRSchedulerCallback, scheduler_cls=StepLR):
    """Decays the learning rate of each parameter group by gamma every step_size epochs. Notice \
    that such decay can happen simultaneously with other changes to the learning rate from outside \
    this scheduler.

    Args:
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay. Defaults to 0.1.
        start_epoch: this callback will start at this epoch. Defaults to 0
        end_epoch: : this callback will stop at this epoch. Defaults to None

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import StepLRCallback
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ...
        >>> callback = StepLRCallback(step_size=2, gamma=.9)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 6 epochs
        >>> loop(train, val, epochs=6)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.1}
        2 {'group0': 0.09}
        3 {'group0': 0.09}
        4 {'group0': 0.081}
        5 {'group0': 0.081}
    """

    step_size: int
    gamma: float = 0.1


class MultiStepLRCallback(EpochLRSchedulerCallback, scheduler_cls=MultiStepLR):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches \
    one of the milestones. Notice that such decay can happen simultaneously with other changes to \
    the learning rate from outside this scheduler.

    Args:
        milestones (list): List of epoch indices. Must be increasing. Also note that these values
            are epochs are from the perspective of the scheduler. so they will start when
            the scheduler starts not nessisarily based on the epoch in loop.
        gamma (float): Multiplicative factor of learning rate decay. Defaults to 0.1.
        start_epoch: this callback will start at this epoch. Defaults to 0
        end_epoch: : this callback will stop at this epoch. Defaults to None

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import MultiStepLRCallback
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ...
        >>> callback = MultiStepLRCallback(milestones=[1, 3], gamma=.9)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 6 epochs
        >>> loop(train, val, epochs=6)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.09}
        2 {'group0': 0.09}
        3 {'group0': 0.081}
        4 {'group0': 0.081}
        5 {'group0': 0.081}
    """

    milestones: List[int]
    gamma: float = 0.1

    def __post_init__(self):
        # check milestones is properly increasing
        if not all(starmap(operator.lt, zip(self.milestones[:-1], self.milestones[1:]))):
            raise ValueError('milestones must be an increasing list.')
        super().__post_init__()


class ExponentialLRCallback(EpochLRSchedulerCallback, scheduler_cls=ExponentialLR):
    """Decays the learning rate of each parameter group by gamma every epoch.

    Args:
        gamma (float): Multiplicative factor of learning rate decay. Defaults to 0.1.
        start_epoch: this callback will start at this epoch. Defaults to 0
        end_epoch: : this callback will stop at this epoch. Defaults to None.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import ExponentialLRCallback
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ...
        >>> callback = ExponentialLRCallback(gamma=.8)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 5 epochs
        >>> loop(train, val, epochs=5)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.08}
        2 {'group0': 0.064}
        3 {'group0': 0.0512}
        4 {'group0': 0.04096}
    """

    gamma: float


class CosineAnnealingLRCallback(EpochLRSchedulerCallback, scheduler_cls=CosineAnnealingLR):
    """Set the learning rate of each parameter group using a cosine annealing schedule.

    Args:
        T_max: Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        end_epoch: : this callback will stop at this epoch. Defaults to None.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import CosineAnnealingLRCallback
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ...
        >>> callback = CosineAnnealingLRCallback(T_max=4, eta_min=.0001)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 6 epochs
        >>> loop(train, val, epochs=7)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.0853}
        2 {'group0': 0.0500}
        3 {'group0': 0.0147}
        4 {'group0': 0.0001}
        5 {'group0': 0.0147}
        6 {'group0': 0.0500}
    """

    T_max: int
    eta_min: float = 0.0


class ReduceLROnPlateauCallback(EpochLRSchedulerCallback, scheduler_cls=ReduceLROnPlateau):
    """Reduce learning rate when the metric specified by ``field`` on the loop has stopped \
    improving.

    Models often benefit from reducing the learning rate by a factor of 2-10 once learning \
    stagnates. This scheduler reads a metrics quantity and if no improvement is seen for a \
    'patience' number of epochs, the learning rate is reduced.

    Args:
        field: the name of the field to access on the loop, may be dotted path.
            Defaults to 'loss'.
        mode: if the metric is being minimized or maximized . Default: 'min'.
        factor: Factor by which the learning rate will be reduced. Default: 0.1.
        patience: number of stagnant epochs to wait before reducting lr. Defaults to 10.
        threshold: Threshold for measuring the new optimum to only focus on significant \
            changes. Default: 1e-4.
        threshold_mode: One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or \
            best * ( 1 - threshold ) in `min` mode. In `abs` mode, \
            dynamic_threshold = best + threshold in `max` mode or best - threshold \
            in `min` mode. Default: 'rel'.
        cooldown: Number of epochs to wait before resuming normal operation after \
            lr has been reduced. Default: 0.
        min_lr: lower bound on the learning rate of all param groups or each group \
            respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference between new and old lr is \
            smaller than eps, the update is ignored. Default: 1e-8.
        start_epoch: this callback will start at this epoch. Defaults to 0
        end_epoch: : this callback will stop at this epoch. Defaults to None

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>>
        >>> from hearth.loop import Loop
        >>> from hearth.callbacks import ReduceLROnPlateauCallback
        >>> _ = torch.manual_seed(0)
        >>>
        >>> # make fakey train data and model
        >>> x, y = torch.rand(130, 5), torch.rand(130, 1).round()
        >>> train = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16)
        >>> val = DataLoader(TensorDataset(x[-30:], y[-30:]), batch_size=16)
        >>> model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        >>>
        >>>
        >>> # make callback ...
        >>> callback = ReduceLROnPlateauCallback(field='loss', patience=1)
        >>> # setup the loop
        >>> loop = Loop(model,
        ...             optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ...             loss_fn = nn.BCELoss(),
        ...             callbacks = [callback]
        ...            )
        >>>
        >>> # run for 6 epochs
        >>> loop(train, val, epochs=7)
        >>> for row in loop.history:
        ...     print(row.epoch, row.lrs)
        0 {'group0': 0.1}
        1 {'group0': 0.1}
        2 {'group0': 0.1}
        3 {'group0': 0.01}
        4 {'group0': 0.01}
        5 {'group0': 0.001}
        6 {'group0': 0.001}
    """

    field: str = 'loss'
    mode: Literal['min', 'max'] = 'min'
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: Literal['rel', 'abs'] = 'rel'
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        self._get_value = partial(_dotted_attrgetter, self.field)

    def _exclude_from_scheduler_kwargs(self):
        return super()._exclude_from_scheduler_kwargs() + ('field',)

    @if_active
    def on_epoch_end(self, loop):
        self.scheduler.step(self._get_value(loop))

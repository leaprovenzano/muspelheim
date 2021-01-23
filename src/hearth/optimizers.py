from typing import ClassVar, Type
from copy import deepcopy
from torch import nn
from torch import optim
from hearth.grad import trainable_parameters, named_trainable_parameters
from more_itertools import partition


def _is_bias_namedparam(x):
    return x[0].endswith('bias')


def _has_weight_decay(defaults, **kwargs) -> bool:
    return bool(kwargs.get('weight_decay', 0) or defaults.get('weight_decay', 0))


class LazyOptimizer:
    """A base wrapper class for torch optimizers that is initilized without model parameters.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models \
    (see :meth:`add_model`).
    """

    _optimizer_type: ClassVar[Type[optim.Optimizer]] = NotImplemented

    def __init__(self, *, decay_bias: bool = False, **kwargs):
        self.initialized = False
        self._optimizer_args = kwargs
        self.decay_bias = decay_bias

    def _init_optimizer(self, params):
        self.optimizer = self._optimizer_type(params, **self._optimizer_args)
        self.initialized = True

    def __getattr__(self, attr: str):
        if self.initialized:
            return getattr(self.optimizer, attr)
        try:
            return self._optimizer_args[attr]
        except KeyError:
            raise AttributeError(attr)

    def _params_from_model(self, model, **kwargs):
        return {'params': trainable_parameters(model), **kwargs}

    def _params_by_weight_decay(self, model, **kwargs):
        no_decay_kwargs = deepcopy(kwargs)
        no_decay_kwargs['weight_decay'] = 0.0
        named_weights, named_biases = partition(
            _is_bias_namedparam, named_trainable_parameters(model)
        )
        yield {'params': map(lambda x: x[-1], named_weights), **kwargs}
        if named_biases:
            yield {'params': map(lambda x: x[-1], named_biases), **no_decay_kwargs}

    def _init_optimizer_from_model(self, model):
        # if we dont use weight decay or are decaying biases same as weight
        # then just use trainable params from model
        if not _has_weight_decay(self._optimizer_args) or self.decay_bias:
            params = trainable_parameters(model)

        else:
            # otherwise get params by weight decay
            params = self._params_by_weight_decay(model)

        self._init_optimizer(params)

    def _update_from_model(self, model, **kwargs):
        # if we dont use weight decay or are decaying biases same as weight
        # then just use trainable params from model
        param_groups = []
        if not _has_weight_decay(self._optimizer_args, **kwargs) or self.decay_bias:
            param_groups.append(self._params_from_model(model, **kwargs))

        else:
            # otherwise get params by weight decay
            for g in self._params_by_weight_decay(model, **kwargs):
                param_groups.append(g)

        for group in param_groups:
            self.optimizer.add_param_group(group)

    def add_model(self, model: nn.Module, **kwargs):
        """add all trainable parameters from this model to the optimizer

        if extra kwargs are provided and the optimizer is already initilized these
        will be used to create param groups. If ::attr::`decay_bias` is False
        and :attr:`weight_decay` is non-zero, or a non-zero `weight_decay`
        value is passed to kwargs then weights and biases will be split into different
        parameter groups
        """
        if not self.initialized:
            self._init_optimizer_from_model(model)

        else:
            self._update_from_model(model, **kwargs)

    def __repr__(self) -> str:
        if self.initialized:
            return self.optimizer.__repr__()

        name = self.__class__.__name__
        kwargrepr = ', '.join(f'{k}={v!r}' for k, v in self._optimizer_args.items())
        return f'{name}({kwargrepr}, decay_bias={self.decay_bias})'

    def get_lrs(self):
        return [grp['lr'] for grp in self.param_groups]


class AdamW(LazyOptimizer):
    r"""Lazy version of AdamW.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square
         (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 1e-2)
        amsgrad: whether to use the AMSGrad variant of this algorithm from the paper
            `On the Convergence of Adam and Beyond`_  (default: False)
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import AdamW
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = AdamW(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze
        a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _=freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1, 0.1]

        later we may unfreeze the first layer of our model
        and add that to our optimizer with a different learning
        rate:

        >>> _=unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.1, 0.01, 0.01]

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    _optimizer_type = optim.AdamW

    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
        decay_bias: bool = False,
    ):
        super().__init__(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            decay_bias=decay_bias,
        )


class Adadelta(LazyOptimizer):
    """Lazy version of Adadelta.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Args:
        rho: coefficient used for computing a running average of squared gradients (default: 0.9)
        eps: term added to the denominator to improve numerical stability (default: 1e-6)
        lr: coefficient that scale delta before it is applied to the parameters (default: 1.0)
        weight_decay: weight decay (L2 penalty) (default: 0) __ https://arxiv.org/abs/1212.5701
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import Adadelta
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = Adadelta(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _ =freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1]

        later we may unfreeze the first layer of our model and add that to our optimizer with a
        different learning rate:

        >>> _ = unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.01]

    """

    _optimizer_type = optim.Adadelta

    def __init__(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, decay_bias: bool = False):
        super().__init__(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, decay_bias=decay_bias)


class Adagrad(LazyOptimizer):
    """Lazy version of Adagrad.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Args:
        lr: learning rate (default: 1e-2)
        lr_decay: learning rate decay (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        eps: term added to the denominator to improve numerical stability (default: 1e-10)
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import Adagrad
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = Adagrad(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _ =freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1]

        later we may unfreeze the first layer of our model and add that to our optimizer with a
        different learning rate:

        >>> _ =unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.01]

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html

    """

    _optimizer_type = optim.Adagrad

    def __init__(
        self,
        lr=0.01,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        decay_bias: bool = False,
    ):
        super().__init__(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            decay_bias=decay_bias,
        )


class Adam(LazyOptimizer):
    r"""Lazy version of Adam.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square
            (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant of this algorithm from the paper
            `On the Convergence of Adam and Beyond`_ (default: False)
            .. _Adam\: A Method for Stochastic Optimization:
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import Adam
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = Adam(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _ = freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1]

        later we may unfreeze the first layer of our model and add that to our optimizer with a
        different learning rate:

        >>> _ =unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.01]

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    _optimizer_type = optim.Adam

    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
        decay_bias: bool = False,
    ):
        super().__init__(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            decay_bias=decay_bias,
        )


class ASGD(LazyOptimizer):
    """Lazy version of ASGD.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    It has been proposed in `Acceleration of stochastic approximation by averaging`_.

    Args:
        lr: learning rate (default: 1e-2)
        lambd: decay term (default: 1e-4)
        alpha: power for eta update (default: 0.75)
        t0: point at which to start averaging (default: 1e6)
        weight_decay: weight decay (L2 penalty) (default: 0)
            .. _Acceleration of stochasticapproximation by averaging:
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import ASGD
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = ASGD(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _ =freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1]

        later we may unfreeze the first layer of our model and add that to our optimizer with a
        different learning rate:

        >>> _ =unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.01]

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098

    """

    _optimizer_type = optim.ASGD

    def __init__(
        self,
        lr=0.01,
        lambd=0.0001,
        alpha=0.75,
        t0=1000000.0,
        weight_decay=0,
        decay_bias: bool = False,
    ):
        super().__init__(
            lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay, decay_bias=decay_bias
        )


class RMSprop(LazyOptimizer):
    """Lazy version of RMSprop.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\\sqrt{v} + \\epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        lr: learning rate (default: 1e-2)
        momentum: momentum factor (default: 0)
        alpha: smoothing constant (default: 0.99)
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        centered: if ``True``, compute the centered RMSProp, the gradient is normalized by an
            estimation of its variance (Defaults to None)
        weight_decay: weight decay (L2 penalty) (default: 0)
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import RMSprop
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = RMSprop(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _ = freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1]

        later we may unfreeze the first layer of our model and add that to our optimizer with a
        different learning rate:

        >>> _= unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.01]
    """

    _optimizer_type = optim.RMSprop

    def __init__(
        self,
        lr=0.01,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False,
        decay_bias: bool = False,
    ):
        super().__init__(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            decay_bias=decay_bias,
        )


class SGD(LazyOptimizer):
    """Lazy version of SGD.

    Supports instantiation without model parameters, exclusion of biases from weight decay
    where applicable and adding only trainable parameters directly from models
    (see :meth:`add_model`).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        lr: learning rate
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        dampening: dampening for momentum (default: 0)
        nesterov: enables Nesterov momentum (default: False)
        decay_bias: if True include biases in weight_decay. (default: False)

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from hearth.optimizers import SGD
        >>> from hearth.grad import freeze, unfreeze
        >>>
        >>> # no need to tell it about the model
        >>> opt = SGD(lr=.1)
        >>> opt.initialized
        False

        define a model at some point, we'll also freeze a single layer:

        >>> model = nn.Sequential(nn.Linear(4, 5),
        ...                       nn.Linear(5, 6),
        ...                       nn.Linear(6, 7))
        >>> _=freeze(model[0])
        >>> opt.add_model(model)
        >>> opt.get_lrs()
        [0.1]

        later we may unfreeze the first layer of our model and add that to our optimizer with a
        different learning rate:

        >>> _=unfreeze(model[0])
        >>> opt.add_model(model[0], lr=.01)
        >>> opt.get_lrs()
        [0.1, 0.01]
    """

    _optimizer_type = optim.SGD

    def __init__(
        self,
        lr: float,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        decay_bias: bool = False,
    ):
        super().__init__(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            decay_bias=decay_bias,
        )

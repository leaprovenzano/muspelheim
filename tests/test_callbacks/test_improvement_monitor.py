from typing import List
from dataclasses import dataclass
from hearth.callbacks import CallbackManager, Callback, ImprovementMonitor
from hearth.events import Stagnation, Improvement


@dataclass
class FakeLoop:
    callbacks: List[Callback]
    loss: float = float('inf')
    metric: float = -float('inf')
    epoch: int = 0
    stage: str = 'train'

    def __post_init__(self):
        self.event_log = []
        self.callbacks = CallbackManager(*self.callbacks)

    def handle_stage(self, stage):
        pass

    def fire(self, event):
        self.callbacks.on_event(self, event)
        self.event_log.append(event)

    def __call__(self, epochs):
        for _ in range(epochs):
            self.callbacks.on_epoch_start(self)
            for stage in ['train', 'val']:
                self.stage = stage
                self.callbacks.on_stage_start(self)
                self.handle_stage(stage)
                self.callbacks.on_stage_end(self)
            self.callbacks.on_epoch_end(self)
            self.epoch += 1


@dataclass
class ImprovingLossLoop(FakeLoop):
    def handle_stage(self, stage):
        if stage == 'val':
            self.loss = 2 - self.epoch * 0.1


@dataclass
class ImprovingMetricLoop(FakeLoop):
    def handle_stage(self, stage):
        if stage == 'val':
            self.metric = (self.epoch + 1) * 0.25


def test_stagnation():
    monitor = ImprovementMonitor(stagnant_after=2)
    loop = FakeLoop(callbacks=[monitor])

    expected_log = [
        Stagnation(field='loss', stage='val', steps=3, best=float('inf')),
        Stagnation(field='loss', stage='val', steps=4, best=float('inf')),
        Stagnation(field='loss', stage='val', steps=5, best=float('inf')),
    ]
    loop(5)
    assert loop.event_log == expected_log


def test_loss_improvement():
    monitor = ImprovementMonitor(stagnant_after=2)
    loop = ImprovingLossLoop(callbacks=[monitor])

    expected_log = [
        Improvement(field='loss', stage='val', steps=1, best=2.0, last_best=float('inf')),
        Improvement(field='loss', stage='val', steps=1, best=1.9, last_best=2.0),
        Improvement(field='loss', stage='val', steps=1, best=1.8, last_best=1.9),
    ]
    loop(3)
    assert loop.event_log == expected_log


def test_metric_improvement():
    monitor = ImprovementMonitor(field='metric', improvement_on='gt')
    loop = ImprovingMetricLoop(callbacks=[monitor])
    expected_log = [
        Improvement(field='metric', stage='val', steps=1, best=0.25, last_best=-float('inf')),
        Improvement(field='metric', stage='val', steps=1, best=0.5, last_best=0.25),
        Improvement(field='metric', stage='val', steps=1, best=0.75, last_best=0.5),
    ]

    loop(3)
    assert loop.event_log == expected_log

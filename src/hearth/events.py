from dataclasses import dataclass


@dataclass
class Event:
    def fire(self, loop):
        loop.fire(self)

    @property
    def msg(self) -> str:
        return ''

    def logmsg(self) -> str:
        return f'{self.__class__.__name__}{self.msg}'


@dataclass
class MonitoringEvent(Event):
    field: str
    stage: str
    steps: int

    def _get_stepmsg(self) -> str:
        return f'{self.steps} step{"s" if self.steps > 1 else ""}'

    def _get_fieldmsg(self) -> str:
        return f'{self.stage}.{self.field}'

    def logmsg(self) -> str:
        return f'{self.__class__.__name__}[{self._get_fieldmsg()}] {self.msg}'


@dataclass
class Improvement(MonitoringEvent):

    best: float
    last_best: float
    name: str = 'improvement'

    @property
    def msg(self) -> str:
        return (
            f'improved from : {self.last_best:0.4f} to {self.best:0.4f} in {self._get_stepmsg()}.'
        )


@dataclass
class Stagnation(MonitoringEvent):

    best: float
    name: str = 'stagnation'

    @property
    def msg(self) -> str:
        return f'stagnant for {self._get_stepmsg()}' f' no improvement from {self.best:0.4f}.'

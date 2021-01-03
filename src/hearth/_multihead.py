class _MultiHeadFunc:
    """base mixin for multiheaded losses and metrics"""

    def __init__(self, **kwargs):
        self._fns = {}
        for k, v in kwargs.items():
            if not callable(v):
                raise TypeError(
                    f'functions passed to {self.__class__.__name__} must be callable'
                    f' but got type {type(v)} for key {k}!'
                )
            self._fns[k] = v

    def items(self):
        return self._fns.items()

    def keys(self):
        return self._fns.keys()

    def values(self):
        return self._fns.values()

    def _argrepr(self):
        return ', '.join(f'{k}={v}' for k, v in self.items())

    def __repr__(self):
        return f'{self.__class__.__name__}({self._argrepr()})'

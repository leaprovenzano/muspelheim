import functools


def on_stage(stage: str):
    """decorator for callback methods so that they only run on a certain stage"""

    def wrapper(f):
        @functools.wraps(f)
        def wrapped(self, loop, *args, **kwargs):
            if loop.stage == stage:
                return f(self, loop, *args, **kwargs)

        return wrapped

    return wrapper


def if_active(f):
    """decorator for callback methods so that they are only called when active"""

    @functools.wraps(f)
    def inner(self, loop, *args, **kwargs):
        if self.active:
            return f(self, loop, *args, **kwargs)

    return inner


def if_inactive(f):
    """decorator for callback methods so that they are only called when inactive"""

    @functools.wraps(f)
    def inner(self, loop, *args, **kwargs):
        if not self.active:
            return f(self, loop, *args, **kwargs)

    return inner

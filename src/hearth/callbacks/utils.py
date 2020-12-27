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

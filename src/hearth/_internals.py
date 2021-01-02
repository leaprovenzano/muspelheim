from typing import Mapping, TypeVar, Union, Dict
import re

T = TypeVar('T')

_TITLE_BOUNDRY = re.compile(r'(?=[A-Z])')


def to_snakecase(s: str):
    if len(s) > 1 and not s.islower():
        return f'{s[0]}{re.sub(_TITLE_BOUNDRY, "_", s[1:])}'.lower()
    return s


class MissingFromRegistryError(KeyError):
    pass


class Registry(Mapping[str, T]):
    """a mapping for key to class or function."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._data: Dict[str, T] = {}
        for k, v in kwargs.items():
            self._register(k, v)

    def _normalize_key(self, k: str) -> str:
        return k.lower().strip()

    def _register(self, k: str, v: T):
        self._data[self._normalize_key(k)] = v

    def __getitem__(self, k: str) -> T:
        k = self._normalize_key(k)
        try:
            return self._data[k]
        except KeyError:
            raise MissingFromRegistryError(f'normalized key: {k} is not in {self.name} registry!')

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def register(self, key_or_obj: Union[str, T]):
        """decorator for registering a class or function."""
        # check if this was called with a
        # key arg or not...
        if isinstance(key_or_obj, str):
            key = key_or_obj
            # make a wrapper to extend as decorator

            def inner(decorated):
                self._register(key, decorated)
                return decorated

            return inner
        # otherwise its a decorated function or class...
        # register it with its name
        key = key_or_obj.__name__  # type: ignore
        self._register(key, key_or_obj)
        return key_or_obj

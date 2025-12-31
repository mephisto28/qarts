import inspect
import functools
import typing as T
import numpy as np


def context_cache(func):
    sig = inspect.signature(func)
    default_name = sig.parameters['name'].default

    @functools.wraps(func)
    def wrapper(self, field: T.Union[str, T.Tuple[str, ...]], name: str = default_name, **kwargs):
        if isinstance(field, str):
            field_key = (field,)
        else:
            field_key = field
        key = (name, *field_key, *kwargs.values())
        if key in self.context.intermediate_cache:
            return self.context.get_cache(key)
        else:
            values = func(self, field, name, **kwargs)
            cache = self.context.get_cache(key, shape=values.shape, dtype=np.float32, scope=None)
            cache[:] = values
            return values
    return wrapper


def expand_tdim_on_batch(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        values = func(self, *args, **kwargs)
        if self.is_online:
            return values
        else:
            if isinstance(values, tuple):
                return tuple(v[..., np.newaxis] for v in values)
            else:
                return values[..., np.newaxis]
    return wrapper
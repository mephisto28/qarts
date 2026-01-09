from qarts.modeling.factors import FactorNames, FactorSpec, ContextSrc

__all__ = ['get_factor_group']


_factor_groups = {}
def register_factor_group(name: str):
    def wrapper(func):
        _factor_groups[name] = func
        return func
    return wrapper


def get_factor_group(name: str) -> list[FactorSpec]:
    return _factor_groups[name]()
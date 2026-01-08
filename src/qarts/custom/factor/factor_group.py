
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


@register_factor_group('default')
def generate_default_group() -> list[FactorSpec]:
    factors = []
    windows = [1, 2, 5, 10, 21, 63, 126]
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_MA, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_VWAP, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close', 'volume'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_POSITION, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_high', 'adjusted_low'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0.5, 'scale': 5})
        factors.append(spec)
    for i in windows:
        spec = FactorSpec(name=FactorNames.DAILY_MOM, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 50})
        factors.append(spec)
    for i in windows[1:]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLATILITY, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.025, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.DAILY_SKEWNESS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.0, 'scale': 1})
        factors.append(spec)
    for i in windows[3:]:
        spec = FactorSpec(name=FactorNames.DAILY_KURTOSIS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.0, 'scale': 0.5})
        factors.append(spec)
    for i in windows[1:5]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLVOL, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.01, 'scale': 200, 'window2': i*3})
        factors.append(spec)
    
    # ---- residual features ----
    for i in windows[1:]:
        spec = FactorSpec(name=FactorNames.DAILY_MOM_SUM, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total'],
        }, window=i, params={'shift': 0.0, 'scale': 50})
        factors.append(spec)
    for i in windows[1:]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLATILITY, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
        }, window=i, params={'shift': 0.02, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.DAILY_SKEWNESS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
        }, window=i, params={'shift': 0.0, 'scale': 1})
        factors.append(spec)
    return factors


@register_factor_group('filters')
def generate_filter_factors_group() -> list[FactorSpec]:
    factors = [FactorSpec(name=FactorNames.IS_TRADABLE, input_fields={
        ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
        ContextSrc.INTRADAY_QUOTATION: ['bid_price1', 'ask_volume1', 'bid_volume1']
    }, window=1)]
    for i in [1, 3, 5]:
        spec = FactorSpec(name=FactorNames.DAILY_RECENT_VACANCY, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
        }, window=i)
        factors.append(spec)
    return factors


@register_factor_group('targets_with_costs_10m_3D_with_range')
def generate_targets_with_costs_10m_3D_group() -> list[FactorSpec]:
    factors = []
    for i in [3, 2, 1, 0]:
        spec = FactorSpec(name=FactorNames.FUTURE_DAY_TARGETS, input_fields={
            ContextSrc.FUTURE_DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['ask_price1', 'mid_price']
        }, window=i)
        factors.append(spec)
    for i in [60, 30, 10]:
        spec = FactorSpec(name=FactorNames.TODAY_TARGETS, input_fields={
            ContextSrc.INTRADAY_QUOTATION: ['ask_price1', 'bid_price1', 'mid_price']
        }, window=i)
        factors.append(spec)
    for i in [3, 2, 1]:
        spec = FactorSpec(name=FactorNames.FUTURE_DAY_RANGE_TARGETS, input_fields={
            ContextSrc.FUTURE_DAILY_QUOTATION: ['adjusted_high', 'adjusted_low'],
            ContextSrc.INTRADAY_QUOTATION: ['ask_price1', 'mid_price']
        }, window=i)
        factors.append(spec)
        spec = FactorSpec(name=FactorNames.FUTURE_DAY_UP_RANGE_TARGETS, input_fields={
            ContextSrc.FUTURE_DAILY_QUOTATION: ['adjusted_high'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i)
        factors.append(spec)
        spec = FactorSpec(name=FactorNames.FUTURE_DAY_DOWN_RANGE_TARGETS, input_fields={
            ContextSrc.FUTURE_DAILY_QUOTATION: ['adjusted_low'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i)
        factors.append(spec)
    return factors
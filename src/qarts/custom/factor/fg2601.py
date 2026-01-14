from qarts.modeling.factors import register_factor_group, FactorNames, FactorSpec, ContextSrc, get_factor_group


@register_factor_group('fg260112')
def generate_group_260112() -> list[FactorSpec]:
    factors = []
    windows = [1, 2, 5, 10, 21, 63, 126]
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_MA, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 100})
        factors.append(spec)
        spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
            ContextSrc.FACTOR_CACHE: [f'{spec.name}_{i}']
        }, window=i, params={'shift': 0, 'scale': 1})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_VWAP, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close', 'volume'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 100})
        factors.append(spec)
        spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
            ContextSrc.FACTOR_CACHE: [f'{spec.name}_{i}']
        }, window=i, params={'shift': 0, 'scale': 1})
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
        spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
            ContextSrc.FACTOR_CACHE: [f'{spec.name}_{i}']
        }, window=i, params={'shift': 0, 'scale': 1})
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
        spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
            ContextSrc.FACTOR_CACHE: [f'{spec.name}_{i}']
        }, window=i, params={'shift': 0, 'scale': 1})
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
    for i in windows[1:5]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLVOL, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
        }, window=i, params={'shift': 0.01, 'scale': 200, 'window2': i*3})
        factors.append(spec)

    # ---- intraday features ----
    factors.append(FactorSpec(name=FactorNames.TODAY_MOM, input_fields={
        ContextSrc.FACTOR_CACHE: ['daily_mom_1']
    }, window=0, params={'shift': 0.0, 'scale': 50}))
    factors.append(FactorSpec(name=FactorNames.TODAY_VOLATILITY, input_fields={
        ContextSrc.FACTOR_CACHE: ['daily_mom_1']
    }, window=0, params={'shift': 0.0, 'scale': 200}))
    factors.append(FactorSpec(name=FactorNames.TODAY_SKEWNESS, input_fields={
        ContextSrc.FACTOR_CACHE: ['daily_mom_1']
    }, window=0, params={'shift': 0.0, 'scale': 1}))
    factors.append(FactorSpec(name=FactorNames.TODAY_POSITION, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['mid_price', 'high', 'low']
    }, window=0, params={'shift': 0.5, 'scale': 2}))

    factors.append(FactorSpec(name=FactorNames.TODAY_MOM, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
    }, window=0, params={'shift': 0.0, 'scale': 50}))
    factors.append(FactorSpec(name=FactorNames.TODAY_VOLATILITY, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
    }, window=0, params={'shift': 0.0, 'scale': 200}))
    factors.append(FactorSpec(name=FactorNames.TODAY_SKEWNESS, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
    }, window=0, params={'shift': 0.0, 'scale': 1}))
    factors.append(FactorSpec(name=FactorNames.TODAY_POSITION, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total', '1min_v4_barra4_total', '1min_v4_barra4_total']
    }, window=0, params={'shift': 0.5, 'scale': 2}))
    return factors


@register_factor_group('fg260113')
def generate_group_260113() -> list[FactorSpec]:
    factors = generate_group_260112()
    windows = [5, 10, 21, 63, 126]
    for i in windows:
        spec = FactorSpec(name=FactorNames.VOLUME_RATIO, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_volume'],
            ContextSrc.INTRADAY_QUOTATION: ['total_volume']
        }, window=i, params={'shift': 0.0, 'scale': 1})
        factors.append(spec)

    for i in windows:
        spec = FactorSpec(name=FactorNames.RV_CORR, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return', 'adjusted_volume'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1'],
            ContextSrc.INTRADAY_QUOTATION: ['total_volume']
        }, window=i, params={'shift': 0.0, 'scale': 5})
        factors.append(spec)

    return factors


@register_factor_group('fg260114')
def generate_group_260114() -> list[FactorSpec]:
    factors = generate_group_260113()
    windows = [5, 10, 21, 63, 126]
    for i in windows:
        spec = FactorSpec(name=FactorNames.RV_CORR, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha', 'adjusted_volume'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total', 'total_volume']
        }, window=i, params={'shift': 0.0, 'scale': 5})
        factors.append(spec)

    spec = FactorSpec(name=FactorNames.ID_TRANSFORM, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total_rank']
    }, window=0, params={'shift': 0.5, 'scale': 4})
    factors.append(spec)
    spec = FactorSpec(name=FactorNames.TODAY_MOM, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total_rank']
    }, window=0, params={'shift': 0.0, 'scale': 1})
    factors.append(spec)
    spec = FactorSpec(name=FactorNames.TODAY_STD, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['volume']
    }, window=0, params={'shift': 0.0, 'scale': 1})
    factors.append(spec)
    spec = FactorSpec(name=FactorNames.TODAY_LOG_SKEWNESS, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['volume']
    }, window=0, params={'shift': 0.0, 'scale': 1})
    factors.append(spec)
    spec = FactorSpec(name=FactorNames.TODAY_AMOUNT_RATIO, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['buy_amount', 'amount']
    }, window=0, params={'shift': 0.0, 'scale': 1})
    factors.append(spec)
    spec = FactorSpec(name=FactorNames.TODAY_AMOUNT_RATIO, input_fields={
        ContextSrc.INTRADAY_QUOTATION: ['sell_amount', 'amount']
    }, window=0, params={'shift': 0.0, 'scale': 1})
    factors.append(spec)

    for i in windows:
        spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
            ContextSrc.FACTOR_CACHE: [f'{FactorNames.VOLUME_RATIO}_{i}']
        }, window=0, params={'shift': 0, 'scale': 50})

    spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
        ContextSrc.FACTOR_CACHE: [FactorNames.TODAY_MOM]
    }, window=0, params={'shift': 0, 'scale': 50})
    factors.append(spec)
    spec = FactorSpec(name=FactorNames.ABS_TRANSFORM, input_fields={
        ContextSrc.FACTOR_CACHE: [f'1min_v4_barra4_total_{FactorNames.TODAY_MOM}']
    }, window=0, params={'shift': 0, 'scale': 50})
    factors.append(spec)
    return factors
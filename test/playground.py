

def beta_estimator_test():
    import matplotlib.pyplot as plt

    from qarts.loader import ParquetPanelLoader
    from qarts.pipelines.beta_estimator import CrossSectionalBetaEstimator

    i = 1705
    i += 1

    loader = ParquetPanelLoader()
    estimator = CrossSectionalBetaEstimator()
    dates = sorted(loader.list_available_dates('quotation'))
    date = dates[i]

    pblock = loader.load_intraday('prediction', model='518.c', epoch=1, date=date, use_ema=False)
    qblock = loader.load_intraday('quotation', date=date)
    results = estimator.predict(
        inputs=qblock.get_fields(['1min_v4_barra4_total']), 
        target=pblock.get_field('pred_0')
    )
    residual_df = results['residual_df']
    plt.plot(results['datetimes'], results['betas'])
    plt.grid()


def generate_factor_specs():
    from qarts.modeling.factors import FactorNames, FactorSpec, ContextSrc

    factors = []
    windows = [1, 2, 5, 10, 21, 63, 126]
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_MA, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i)
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_VWAP, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close', 'volume'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i)
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name='price_position', input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_high', 'adjusted_low'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i)
        factors.append(spec)
    for i in windows:
        spec = FactorSpec(name='daily_mom', input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i)
        factors.append(spec)
    for i in windows[1:]:
        spec = FactorSpec(name='daily_volatility', input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i)
        factors.append(spec)
    return factors


if __name__ == '__main__':
    from tqdm import tqdm
    from qarts.loader import ParquetPanelLoader
    from qarts.modeling.factors.engine import IntradayBatchProcessingEngine, FactorSpec, ContextSrc
    factors = generate_factor_specs()
    loader = ParquetPanelLoader()
    engine = IntradayBatchProcessingEngine(loader, factors)
    for date, factors_block in tqdm(engine.iterate_tasks()):
        print(date, factors_block.data.shape)
        breakpoint()


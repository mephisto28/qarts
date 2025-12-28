

if __name__ == '__main__':
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

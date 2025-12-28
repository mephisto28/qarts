from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from qarts.core import PanelBlockIndexed, PanelSeriesIndexed
from qarts.loader import PanelLoader


@dataclass
class IntradayVariableSpec:

    var_type: str
    fields: list[str]
    load_kwargs: dict = field(default_factory=dict)


@dataclass
class IntradayNeutralizeSpec:
    name: str
    use_intercept: bool = True
    neutralize_coef: float = 1.0


@dataclass
class IntradayNeutralizeOutputSpec:
    var_type: str
    specs: list[IntradayNeutralizeSpec]
    load_kwargs: dict = field(default_factory=dict)


class CrossSectionalBetaEstimator:

    def __init__(self, use_intercept: bool = True, min_samples: int = 500, neutralize_coef: float = 1.0):
        self.min_samples = min_samples
        self.use_intercept = use_intercept
        self.neutralize_coef = neutralize_coef

    def _compute_beta(self, df: pd.DataFrame, target: str, inputs: list[str]):
        X = df[inputs.fields].to_numpy(dtype=np.float32)
        y = df[target.field].to_numpy(dtype=np.float32)
        if self.use_intercept:
            X1 = np.column_stack([np.ones(len(y)), X])
        else:
            X1 = X
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        corrs = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        residual = y - X1 @ beta * self.neutralize_coef
        return beta, corrs, residual

    def predict(
        self,
        inputs: PanelBlockIndexed,
        target: PanelSeriesIndexed,
    ):
        block = inputs.merge(target)
        df_grouped = block.get_cross_sectional_groups()
        betas = []
        corrs_list = []
        residuals = []
        datetimes = []
        datetime_indices = []
        instrument_indices = []
        for dt, group_df in df_grouped:
            group_df.dropna(inplace=True)
            if len(group_df) < self.min_samples:
                continue
            beta, corrs, residual = self._compute_beta(group_df, target, inputs)
            datetimes.append(dt)
            betas.append(beta)
            corrs_list.append(corrs)
            residuals.append(residual)
            datetime_indices.extend([dt] * len(group_df))
            instrument_indices.extend(group_df.index.get_level_values('instrument'))
        residual_df = pd.Series(
            data=np.concatenate(residuals, axis=0),
            index=[datetime_indices, instrument_indices],
            name=f'{target.field}'
        )
        residual_df.index.names = ['datetime', 'instrument']

        columns = [f'beta_{f}' for f in inputs.fields]
        if self.use_intercept:
            columns = ['intercept'] + columns
        beta_df = pd.DataFrame(betas, index=datetimes, columns=[f'beta_{f}' for f in columns])
        beta_df.index = beta_df.index.time
        corr_df = pd.DataFrame(corrs_list, index=datetimes, columns=inputs.fields)
        corr_df.index = corr_df.index.time
        results = {
            'betas': beta_df,
            'residuals': residual_df,
            'corrs': corr_df,
        }
        return results


class IntradayNeutralizer:

    def __init__(self, loader: PanelLoader):
        self.loader = loader

    def _load_variables(self, specs: list[IntradayVariableSpec], **load_kwargs) -> list[PanelSeriesIndexed]:
        variables = []
        for spec in specs:
            spec.load_kwargs.update(load_kwargs)
            block = self.loader.load_intraday(spec)
            for field in spec.fields:
                series = block.get_field(field)
                variables.append(series)
        return variables

    def run(
        self, 
        indeps: list[IntradayVariableSpec],
        deps: list[IntradayVariableSpec], 
        output_spec: IntradayNeutralizeOutputSpec
    ):
        available_dates = self.loader.list_available_dates(deps + indeps)
        existing_date = self.loader.list_available_dates(output_spec)
        required_dates = set(available_dates) - set(existing_date)
        logger.info(f"Remaining/Available dates: {len(required_dates)} / {len(available_dates)}")

        for date in sorted(required_dates):
            logger.info(f"Processing date: {date}")
            targets_fields = [f for dep in deps for f in dep.fields]
            assert len(targets_fields) == len(output_spec.specs), "Number of fields in output spec must match number of fields in dependencies"
            
            inputs = self._load_variables(indeps, date=date)
            inputs = PanelBlockIndexed.from_series_list(inputs)
            targets = self._load_variables(deps, date=date)
            
            series_lst = []
            beta_stats = {}
            for out_spec, target in zip(output_spec.specs, targets):
                beta_estimator = CrossSectionalBetaEstimator(
                    use_intercept=out_spec.use_intercept,
                    neutralize_coef=out_spec.neutralize_coef,
                )
                results = beta_estimator.predict(inputs, target)
                residual_series = PanelSeriesIndexed(results['residuals'], field=out_spec.name)
                series_lst.append(residual_series)
                beta_stats[out_spec.name] = float(results["betas"].mean().iloc[0])
            logger.info(f'{date} beta stats: {", ".join([f"{f}: {beta:.2f}" for f, beta in beta_stats.items()])}')
            block = PanelBlockIndexed.from_series_list(series_lst)
            output_spec.load_kwargs.update({'date': date})
            self.loader.save_intraday(block, output_spec)



if __name__ == '__main__':
    from qarts.loader import ParquetPanelLoader
    
    loader = ParquetPanelLoader()
    intraday_neutralizer = IntradayNeutralizer(loader)

    inputs = IntradayVariableSpec(var_type='quotation', fields=['1min_v4_barra4_total'])
    targets = IntradayVariableSpec(var_type='prediction', fields=[f'pred_{i}' for i in range(7)] + [f'gt_{i}' for i in range(7)], load_kwargs={'model': '518.c', 'epoch': 1})
    specs = [IntradayNeutralizeSpec(name=f'pred_{i}', use_intercept=False, neutralize_coef=0.7) for i in range(7)] + \
        [IntradayNeutralizeSpec(name=f'gt_{i}', use_intercept=False, neutralize_coef=0) for i in range(7)]
    output_spec = IntradayNeutralizeOutputSpec(var_type='prediction', specs=specs, load_kwargs={'model': '518.c.n', 'epoch': 1})
    intraday_neutralizer.run([inputs], [targets], output_spec)

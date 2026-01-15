import os
import pickle
import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata
import matplotlib.pyplot as plt

from qarts.modeling.objectives import eval_utils
from .base import Processor, GlobalContext


class EvaluatorProcessor(Processor):
    _name: str = 'evaluator'
    _input_fields: list[str] = ['preds', 'factor_context_future']

    def __init__(
        self, 
        # model_name: str, 
        # epoch: int,
        pred_name: str,
        targets_name: str, 
        output_name: str,
        pred_fields: list[str], 
        target_fields: list[str],
        sample_stride: int = 2,
        quantiles: int = 10,
        min_count_xs: int = 200,
        execution_period: str = 'close',
        output_dir: str = None,
    ):
        self.pred_name = pred_name
        self.targets_name = targets_name
        self.output_name = output_name
        self.pred_fields = pred_fields
        self.target_fields = target_fields
        self.sample_stride = sample_stride
        self.quantiles = quantiles
        self.execution_period = execution_period
        self.min_count_xs = min_count_xs
        self.metrics = []
        
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, self.output_name)
        else:
            d = os.path.dirname
            project_dir = d(d(d(d(os.path.abspath(__file__)))))
            self.output_dir =  os.path.join(project_dir, 'experiments/output', self.output_name)
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def input_fields(self) -> list[str]:
        return [self.pred_name, self.targets_name]

    @property
    def output_fields(self) -> list[str]:
        return [self.output_name]

    def process(self, context: GlobalContext):
        preds_block = context.get(self.input_fields[0])
        targets_block = context.get(self.input_fields[1])
        preds = preds_block.get_copy(self.pred_fields) # F, N, T
        targets = targets_block.get_copy(self.target_fields)
        pred_s = preds.data.transpose(1, 2, 0)[preds_block.is_valid_instruments, ::self.sample_stride]
        gt_s = targets.data.transpose(1, 2, 0)[targets_block.is_valid_instruments, ::self.sample_stride]
        pred_s = self.sample_execution_period(pred_s)
        gt_s = self.sample_execution_period(gt_s)
        if pred_s.shape[-1] == 1:
            pred_s = np.tile(pred_s, (1, 1, gt_s.shape[-1]))
        N, T, D = pred_s.shape

        ic_td = eval_utils.nan_corr_xs(pred_s, gt_s, axis=0, min_count=self.min_count_xs)
        rp = rankdata(pred_s, axis=0, method="average", nan_policy="omit")
        rg = rankdata(gt_s, axis=0, method="average", nan_policy="omit")
        rankic_td = eval_utils.nan_corr_xs(rp, rg, axis=0, min_count=self.min_count_xs)
        rankic_long_tdh = eval_utils.nan_corr_xs_long(rp, rg, quantiles=0.5, axis=0, min_count=self.min_count_xs)
        rankic_short_tdh = eval_utils.nan_corr_xs_long(-rp, rg, quantiles=0.5, axis=0, min_count=self.min_count_xs)

        hit_td = eval_utils.nan_hit_rate_xs(pred_s, gt_s, axis=0, min_count=self.min_count_xs)
        auc_td = np.full((T, D), np.nan)
        for t in range(T):
            for h in range(D):
                auc_td[t, h] = eval_utils.auc_binary(gt_s[:, t, h], pred_s[:, t, h])

        qret_tdh = np.full((T, D, self.quantiles), np.nan)
        qspread_td = np.full((T, D), np.nan)
        qmono_td = np.full((T, D), np.nan)
        tail_td = np.full((T, D), np.nan)
        for t in range(T):
            for h in range(D):
                qret_tdh[t, h, :], qspread_td[t, h], qmono_td[t, h], tail_td[t, h] = \
                    eval_utils.quantile_stats_one_column(pred_s[:, t, h], gt_s[:, t, h], q=self.quantiles)

        # TODO: add more metrics here

        self.metrics.append({
            'date': context.current_datetime,
            'eval_ic': ic_td.mean(0),
            'eval_rankic': rankic_td.mean(0),
            'eval_rankic_long': rankic_long_tdh.mean(0),
            'eval_rankic_short': rankic_short_tdh.mean(0),
            'eval_hit': hit_td.mean(0),
            'eval_auc': auc_td.mean(0),
            'eval_qret': qret_tdh.mean(0),
            'eval_qret_top': qret_tdh.mean(0)[:, -1],
            'eval_qret_bot': qret_tdh.mean(0)[:, 0],
            'eval_qspread': qspread_td.mean(0),
            'eval_qmono': qmono_td.mean(0),
            'eval_tail': tail_td.mean(0),
            # TODO: add more metrics here
        })
        logger.info(
            f'{self.metrics[-1]["date"]} {self.output_name} metrics'
            f'\n{self.format("eval_rankic")}\n{self.format("eval_rankic_long")}\n{self.format("eval_auc")}\n{self.format("eval_qret_top")}\n{self.format("eval_qret_bot")}'
        )

    def format(self, name: str):
        return f'{name:<18}:\t' + ' '.join([f'{v:.4f}' for v in self.metrics[-1][name]])

    def sample_execution_period(self, arr: np.ndarray) -> np.ndarray:
        t = arr.shape[1]
        if self.execution_period == 'close':
            near_close_idx = int(t // 8 * 7)
            return arr[:, near_close_idx:]
        elif self.execution_period == 'open':
            near_open_idx = int(t // 8)
            return arr[:, :near_open_idx]
        elif self.execution_period == 'all':
            return arr
        else:
            raise ValueError(f'Invalid execution period: {self.execution_period}')

    def finalize(self) -> T.Dict[str, T.Any]:
        """Return results as pandas objects + numpy arrays."""
        dates = pd.DatetimeIndex([m['date'] for m in self.metrics], name="date")
        ic = np.vstack([m['eval_ic'] for m in self.metrics]) 
        ric = np.vstack([m['eval_rankic'] for m in self.metrics]) 
        ric_long = np.vstack([m['eval_rankic_long'] for m in self.metrics]) 
        ric_short = np.vstack([m['eval_rankic_short'] for m in self.metrics]) 
        hit = np.vstack([m['eval_hit'] for m in self.metrics]) 
        auc = np.vstack([m['eval_auc'] for m in self.metrics]) 

        qret = np.stack([m['eval_qret'] for m in self.metrics]) 
        qspread = np.vstack([m['eval_qspread'] for m in self.metrics]) 
        qmono = np.vstack([m['eval_qmono'] for m in self.metrics]) 
        tail = np.vstack([m['eval_tail'] for m in self.metrics])

        # bucket = np.stack(self.daily_bucket_ret, axis=0) if self.daily_bucket_ret else np.empty((0, self.D, len(self.cfg.value_bins)-1))

        # bt_ret = np.vstack(self.daily_bt_ret) if self.daily_bt_ret else np.empty((0, self.D))
        # bt_to = np.vstack(self.daily_bt_turnover) if self.daily_bt_turnover else np.empty((0, self.D))

        df_ic = pd.DataFrame(ic, index=dates, columns=self.target_fields)
        df_rankic = pd.DataFrame(ric, index=dates, columns=self.target_fields)
        df_rankic_long = pd.DataFrame(ric_long, index=dates, columns=self.target_fields)
        df_rankic_short = pd.DataFrame(ric_short, index=dates, columns=self.target_fields)
        df_hit = pd.DataFrame(hit, index=dates, columns=self.target_fields)
        df_auc = pd.DataFrame(auc, index=dates, columns=self.target_fields)
        # df_bt_ret = pd.DataFrame(bt_ret, index=dates, columns=self.h_names)
        # df_bt_to = pd.DataFrame(bt_to, index=dates, columns=self.h_names)

        # summary table
        summary_rows = []
        for j, f in enumerate(self.target_fields):
            x = df_ic.iloc[:, j].values
            mean = float(np.nanmean(x))
            std = float(np.nanstd(x, ddof=1))
            # lag = max(0, int(hs.horizon_days - 1))
            # t_nw = eval_utils.newey_west_tstat(x, lag=lag)

            xr = df_rankic.iloc[:, j].values
            rmean = float(np.nanmean(xr))
            rstd = float(np.nanstd(xr, ddof=1))

            s = {
                "horizon": f,
                "IC_mean": mean,
                "IC_std": std,
                # "IC_NW_tstat": t_nw,
                "RankIC_mean": rmean,
                "RankIC_std": rstd,
                "RankIC_long_mean": float(np.nanmean(df_rankic_long.iloc[:, j].values)),
                "RankIC_long_std": float(np.nanstd(df_rankic_long.iloc[:, j].values, ddof=1)),
                "RankIC_short_mean": float(np.nanmean(df_rankic_short.iloc[:, j].values)),
                "RankIC_short_std": float(np.nanstd(df_rankic_short.iloc[:, j].values, ddof=1)),
                "HitRate_mean": float(np.nanmean(df_hit.iloc[:, j].values)),
                "AUC_mean": float(np.nanmean(df_auc.iloc[:, j].values)),
                "TopBottomSpread_mean": float(np.nanmean(qspread[:, j])),
                "QuantileMonotonicity_mean": float(np.nanmean(qmono[:, j])),
                "Tail1pctSpread_mean": float(np.nanmean(tail[:, j])),
                # "BT_entryReturn_mean": float(np.nanmean(df_bt_ret.iloc[:, j].values)),
                # "BT_turnover_mean": float(np.nanmean(df_bt_to.iloc[:, j].values)),
            }
            summary_rows.append(s)
        df_summary = pd.DataFrame(summary_rows).set_index("horizon")

        # decay peak
        peak_ic_h = df_summary["IC_mean"].abs().idxmax() if len(df_summary) else None
        peak_rankic_h = df_summary["RankIC_mean"].abs().idxmax() if len(df_summary) else None
        peak_rankic_long_h = df_summary["RankIC_long_mean"].abs().idxmax() if len(df_summary) else None

        intraday = None

        final_results = {
            "dates": dates,
            "df_ic": df_ic,
            "df_rankic": df_rankic,
            "df_rankic_long": df_rankic_long,
            "df_hit": df_hit,
            "df_auc": df_auc,
            "qret": qret,          # (Nd,D,Q)
            "qspread": qspread,    # (Nd,D)
            "qmono": qmono,        # (Nd,D)
            "tail_spread": tail,   # (Nd,D)
            # "bucket_centers": self.bucket_centers,
            # "bucket_ret": bucket,  # (Nd,D,K)
            # "df_bt_ret": df_bt_ret,
            # "df_bt_turnover": df_bt_to,
            "df_summary": df_summary,
            "peak_ic_horizon": peak_ic_h,
            "peak_rankic_horizon": peak_rankic_h,
            "peak_rankic_long_horizon": peak_rankic_long_h,
            "intraday": intraday,
            "horizons": self.pred_fields,
            # "config": self.cfg,
        }
        with open(os.path.join(self.output_dir, "eval_results.pkl"), "wb") as f:
            pickle.dump(final_results, f)
        return final_results
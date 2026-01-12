import os
import io
import base64
import pickle

import fire
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class HTMLReport:
    def __init__(self, title: str):
        self.title = title
        self.sections: list[str] = []

    @staticmethod
    def _fig_to_base64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def add_markdown_like(self, html: str):
        self.sections.append(html)

    def add_figure(self, fig, caption: str = ""):
        b64 = self._fig_to_base64(fig)
        cap = f"<div style='margin:6px 0 18px 0;color:#444;font-size:13px'>{caption}</div>" if caption else ""
        self.sections.append(
            f"<div><img style='max-width:100%;border:1px solid #ddd' src='data:image/png;base64,{b64}'/></div>{cap}"
        )

    def add_table(self, df: pd.DataFrame, caption: str = "", floatfmt: str = "{:.6f}", max_rows: int = 50):
        dfx = df.copy()
        if len(dfx) > max_rows:
            dfx = dfx.head(max_rows)
        # format floats
        for c in dfx.columns:
            if np.issubdtype(dfx[c].dtype, np.floating):
                dfx[c] = dfx[c].map(lambda v: "" if not np.isfinite(v) else floatfmt.format(v))
        html_table = dfx.to_html(escape=False, border=0)
        cap = f"<div style='margin:6px 0 10px 0;color:#444;font-size:13px'>{caption}</div>" if caption else ""
        self.sections.append(cap + html_table)

    def write(self, path: str):
        style = """
        <style>
            body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; color: #111; }
            h1 { font-size: 22px; margin-bottom: 8px; }
            h2 { font-size: 16px; margin-top: 26px; }
            table { border-collapse: collapse; width: 100%; margin: 8px 0 22px 0; }
            th, td { border-bottom: 1px solid #eee; padding: 6px 8px; font-size: 12px; text-align: right; }
            th { background: #fafafa; position: sticky; top: 0; }
            td:first-child, th:first-child { text-align: left; }
            .note { color:#444; font-size:13px; margin: 6px 0 10px 0; }
            .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        </style>
        """
        html = f"<!doctype html><html><head><meta charset='utf-8'><title>{self.title}</title>{style}</head><body>"
        html += f"<h1>{self.title}</h1>"
        html += "\n".join(self.sections)
        html += "</body></html>"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


# ---------------------------
# Plot helpers
# ---------------------------

def plot_ic_decay(df_summary: pd.DataFrame, title: str = "IC / RankIC decay by horizon"):
    fig = plt.figure()
    x = np.arange(len(df_summary))
    plt.plot(x, df_summary["IC_mean"].values, marker="o", label="IC_mean")
    plt.plot(x, df_summary["RankIC_mean"].values, marker="o", label="RankIC_mean")
    plt.plot(x, df_summary["RankIC_long_mean"].values, marker="o", label="RankIC_long_mean")
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, df_summary.index.tolist(), rotation=30, ha="right")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_daily_series(df: pd.DataFrame, title: str, ylabel: str, base_value: float = 0.0):
    fig = plt.figure()
    for c in df.columns:
        s_acc = (df[c] - base_value).cumsum() / len(df)
        plt.plot(df.index, s_acc, label=c, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    return fig


def plot_intraday_heatmap(mat: np.ndarray, horizons: list[str], title: str, minute_stride: int):
    """
    mat: (T2,D) mean-by-minute
    """
    fig = plt.figure()
    img = plt.imshow(mat.T, aspect="auto", interpolation="nearest")
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.yticks(np.arange(len(horizons)), horizons)
    plt.xlabel(f"minute index (stride={minute_stride})")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_quantile_curve(qret_mean: np.ndarray, horizons: list[str], title: str):
    """
    qret_mean: (D,Q) average quantile return
    """
    fig = plt.figure()
    Q = qret_mean.shape[1]
    x = np.arange(1, Q + 1)
    for i, h in enumerate(horizons):
        plt.plot(x, qret_mean[i], marker="o", label=h)
    plt.title(title)
    plt.xlabel("Quantile (low -> high)")
    plt.axhline(0.0, linewidth=1)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_bucket_curve(bucket_centers: np.ndarray, bucket_mean: np.ndarray, horizons: list[str], title: str):
    """
    bucket_mean: (D,K)
    """
    fig = plt.figure()
    for i, h in enumerate(horizons):
        plt.plot(bucket_centers, bucket_mean[i], marker="o", label=h)
    plt.title(title)
    plt.xlabel("Signal z-score bucket center")
    plt.axhline(0.0, linewidth=1)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_equity_curve(entry_ret: pd.Series, title: str):
    """
    entry_ret: daily "entry-time" return series for one horizon.
    """
    fig = plt.figure()
    r = entry_ret.fillna(0.0).values
    eq = np.cumprod(1.0 + r)  # not annualized; purely illustrative
    plt.plot(entry_ret.index, eq, linewidth=1.2)
    plt.title(title)
    plt.ylabel("Cumulative (1 + entry_return) product")
    plt.grid(True)
    plt.tight_layout()
    return fig


def build_report(eval_dir: str):
    with open(os.path.join(eval_dir, "eval_results.pkl"), "rb") as f:
        res = pickle.load(f)
    rpt = HTMLReport(title="Multi-horizon Alpha Model Evaluation Report")

    rpt.add_markdown_like("<h2>Executive Summary</h2>")
    note = (
        "<div class='note'>"
        "This report aggregates minute-level cross-sectional metrics to daily series (mean over minutes). "
        "IC t-stats use Newey-West on the daily IC series with lag=max(horizon_days-1,0). "
        "Backtest metrics are simplified 'entry-time' long-short returns based on forward returns; "
        "they are useful for comparability across horizons but are not a full execution simulator."
        "</div>"
    )
    rpt.add_markdown_like(note)
    rpt.add_table(res["df_summary"].round(6), caption="Summary metrics by horizon (selected key indicators).", max_rows=200)

    rpt.add_markdown_like("<h2>Predictive Power</h2>")
    rpt.add_figure(plot_ic_decay(res["df_summary"]), caption="IC and RankIC mean across horizons (decay curve).")
    rpt.add_figure(plot_daily_series(res["df_ic"], "Daily IC (mean over minutes)", "IC"), caption="Daily IC time series by horizon.")
    rpt.add_figure(plot_daily_series(res["df_rankic"], "Daily RankIC (mean over minutes)", "RankIC"), caption="Daily RankIC time series by horizon.")
    rpt.add_figure(plot_daily_series(res["df_rankic_long"], "Daily RankIC Long (mean over minutes)", "RankIC Long"), caption="Daily RankIC Long time series by horizon.")
    rpt.add_figure(plot_daily_series(res["df_hit"], "Daily Hit Rate (mean over minutes)", "Hit Rate", base_value=0.5), caption="Directional hit rate (gt>0 vs pred>0).")
    rpt.add_figure(plot_daily_series(res["df_auc"], "Daily AUC (mean over minutes)", "AUC", base_value=0.5), caption="AUC by day and horizon (gt>0 as positive label).")

    # Quantile curves
    qret_mean = np.nanmean(res["qret"], axis=0) if res["qret"].size else None  # (D,Q)
    if qret_mean is not None:
        rpt.add_figure(plot_quantile_curve(qret_mean, res["horizons"], "Average quantile return curve (cross-section)"),
                    caption="Mean forward return per signal quantile (low->high), averaged across all minutes and days.")
    # bucket_mean = np.nanmean(res["bucket_ret"], axis=0) if res["bucket_ret"].size else None  # (D,K)
    # if bucket_mean is not None:
    #     rpt.add_figure(plot_bucket_curve(res["bucket_centers"], bucket_mean, res["horizons"], "Average value-bucket return curve"),
    #                 caption="Mean forward return in fixed z-score buckets of the signal, averaged across minutes and days.")

    # Intraday heatmap
    intraday = res.get("intraday", None)
    if intraday is not None:
        rpt.add_markdown_like("<h2>Intraday Pattern</h2>")
        rpt.add_figure(plot_intraday_heatmap(intraday["ic_mean_by_minute"], res["horizons"],
                                            "Mean IC by minute-of-day (aggregated across days)", intraday["minute_stride"]),
                    caption="Intraday IC seasonality (mean across days).")
        rpt.add_figure(plot_intraday_heatmap(intraday["rankic_mean_by_minute"], res["horizons"],
                                            "Mean RankIC by minute-of-day (aggregated across days)", intraday["minute_stride"]),
                    caption="Intraday RankIC seasonality (mean across days).")

    # Tradability section
    # rpt.add_markdown_like("<h2>Tradability (Simplified)</h2>")
    # rpt.add_table(res["df_bt_ret"].add_prefix("EntryReturn_").join(res["df_bt_turnover"].add_prefix("Turnover_")).round(6).tail(60),
    #             caption="Recent daily simplified backtest outputs (entry-time return, turnover).")

    # Equity curves for top 2 horizons by |BT_entryReturn_mean|
    df_sum = res["df_summary"]
    if "BT_entryReturn_mean" in df_sum.columns and len(df_sum) > 0:
        top2 = df_sum["BT_entryReturn_mean"].abs().sort_values(ascending=False).head(2).index.tolist()
        for h in top2:
            s = res["df_bt_ret"][h]
            rpt.add_figure(plot_equity_curve(s, f"Simplified equity curve: {h}"),
                        caption=f"Cumulative product of (1 + daily entry-time return) for horizon {h}.")

    report_path = os.path.join(eval_dir, "report.html")
    rpt.write(report_path)
    logger.info(f"Evaluation report saved to {report_path}")
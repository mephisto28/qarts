import numpy as np
import pandas as pd

from ..base import register_stats, DailyStats
from .. import ContextOps, ContextSrc


@register_stats('vol_dist')
class VolDist(DailyStats):
    def __init__(
            self, 
            input_fields: dict[str, list[str]], 
            decay_factor: float = 0.95,
            sample_stride: int = [1, 2, 5, 2, 1],
            output_count: int = 2000,
            **kwargs
        ):
        super().__init__(input_fields=input_fields, **kwargs)
        self.vol_field = input_fields[ContextSrc.INTRADAY_QUOTATION][0]
        self.decay_factor = decay_factor
        self.status = {}

    def output_fields(self) -> list[str]:
        return [f'cum_vol_ratio_{i}' for i in range(self.timesteps)]
    
    @staticmethod
    def sample_timestamp_indices(timestamps: np.ndarray,) -> np.ndarray:
        ts = np.asarray(timestamps)

        # 尝试统一为 datetime64[ns]
        if not np.issubdtype(ts.dtype, np.datetime64):
            ts = ts.astype("datetime64[ns]")
        else:
            ts = ts.astype("datetime64[ns]")

        # 如果存在 NaT，先剔除（保持 index 映射）
        valid_mask = ~np.isnat(ts)
        ts_valid = ts[valid_mask]
        idx_valid = np.nonzero(valid_mask)[0]

        if ts_valid.size == 0:
            return np.array([], dtype=int)

        # 以“自然日”分组（按 ts 的日期部分）
        dates = ts_valid.astype("datetime64[D]")
        unique_dates, start_pos = np.unique(dates, return_index=True)
        # 为了得到每个日期块的范围
        # end_pos 是下一块的 start，最后一块到末尾
        order = np.argsort(start_pos)
        unique_dates = unique_dates[order]
        start_pos = start_pos[order]
        end_pos = np.r_[start_pos[1:], ts_valid.size]

        # 分段定义：[(start_hm, end_hm, step_minutes), ...]
        segments = [
            ("09:30", "09:40", 1),
            ("09:40", "10:00", 2),
            ("10:00", "11:20", 5),
            ("11:20", "11:30", 2),
            ("13:00", "13:10", 1),
            ("13:10", "14:40", 5),
            ("14:40", "14:50", 2),
            ("14:50", "15:00", 1),
        ]

        def hm_to_min(hm: str) -> int:
            h, m = hm.split(":")
            return int(h) * 60 + int(m)

        seg_minutes = [(hm_to_min(a), hm_to_min(b), step) for a, b, step in segments]

        sampled_indices = []

        for d, s, e in zip(unique_dates, start_pos, end_pos):
            day_ts = ts_valid[s:e]
            day_idx = idx_valid[s:e]

            if day_ts.size == 0:
                continue

            # 交易日的 00:00 基准
            day0 = d.astype("datetime64[ns]")

            # 用 searchsorted 做“网格点 -> 第一个 >= 网格点”的索引映射
            for start_m, end_m, step in seg_minutes:
                start_t = day0 + np.timedelta64(start_m, "m")
                end_t = day0 + np.timedelta64(end_m, "m")

                # 构造采样网格（左闭右开），比如 09:30 到 09:40 取 09:30,...,09:39
                grid = np.arange(
                    start_t.astype("datetime64[m]"),
                    end_t.astype("datetime64[m]"),
                    np.timedelta64(step, "m"),
                    dtype="datetime64[m]",
                ).astype("datetime64[ns]")

                if grid.size == 0:
                    continue

                # 在 day_ts 中找每个 grid 点对应的第一条 >= grid 的记录
                pos = np.searchsorted(day_ts, grid, side="left")

                # 过滤越界（当天该段可能没数据）
                pos = pos[pos < day_ts.size]
                if pos.size == 0:
                    continue

                # 去重：多个 grid 可能映射到同一个成交记录
                pos = np.unique(pos)

                sampled_indices.append(day_idx[pos])

        if not sampled_indices:
            return np.array([], dtype=int)

        out = np.unique(np.concatenate(sampled_indices)).astype(int)
        return out

    def compute_from_context(self, ops: ContextOps):
        intraday_src = ContextSrc.INTRADAY_QUOTATION

        amount_cumsum = ops.today_cumsum(self.vol_field) # (N, T)
        amount_sum = amount_cumsum[:, -1:]
        amount_cum_ratio = amount_cumsum / amount_sum # (N, T)
        
        timestamps = ops.get_src(intraday_src).timestamps
        if getattr(self, '_sample_indices', None) is None:
            self._sample_indices = self.sample_timestamp_indices(timestamps)
            sampled_timestamps = timestamps[self._sample_indices]
            sampled_timestamps = [t.strftime('%H%M') for t in sampled_timestamps]
            self._output_fields = [f'cum_vol_ratio_{t}' for t in sampled_timestamps]
        amount_cum_ratio = amount_cum_ratio[:, self._sample_indices]
 
        for i, inst in enumerate(ops.get_src(intraday_src).instruments):
            amount_cum_ratio_inst = amount_cum_ratio[i, :]
            if inst not in self.status:
                self.status[inst] = amount_cum_ratio_inst
            else:
                self.status[inst] = self.status[inst] * self.decay_factor + (1 - self.decay_factor) * amount_cum_ratio_inst
        
        df = pd.DataFrame(self.status, index=timestamps[self._sample_indices])
        df.transpose(inplace=True)
        return df

        
import numpy as np
import numba as nb
import pandas as pd


def encode_instruments(daily_inst: np.ndarray, intra_inst: np.ndarray = None):
    if intra_inst is None:
        intra_inst = np.array([], dtype=object)
    cats = pd.Index(np.concatenate([daily_inst, intra_inst])).unique().sort_values()
    d = pd.Categorical(daily_inst, categories=cats, ordered=True).codes.astype(np.int32)
    i = pd.Categorical(intra_inst, categories=cats, ordered=True).codes.astype(np.int32)
    return d, i, cats

@nb.jit
def build_ranges(ids_sorted: np.ndarray, n_inst: int):
    start = np.full(n_inst, -1, dtype=np.int64)
    end = np.full(n_inst, -1, dtype=np.int64)
    if ids_sorted.size == 0:
        return start, end
    cur = int(ids_sorted[0])
    if cur >= 0:
        start[cur] = 0
    for k in range(1, ids_sorted.size):
        x = int(ids_sorted[k])
        if x != cur:
            if cur != -1:
                end[cur] = k
            cur = x
            if x == -1:
                continue
            if start[x] == -1:
                start[x] = k

    if end[cur] == -1:
        end[cur] = ids_sorted.size
    return start, end


def make_time_grid(
    trading_date: str | pd.Timestamp,
    frequency: str = "1min",
    sessions: tuple[tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "14:57")),
    include_call_auction_0925: bool = True,
) -> np.ndarray:
    """
    生成交易时间网格 (int64 nanoseconds)。
    grid 中的每个点代表一个 Bar 的开始时间。
    """
    d = pd.Timestamp(trading_date).normalize()
    parts = []

    # 1. 集合竞价 (通常作为一个独立的时刻)
    if include_call_auction_0925:
        parts.append(pd.DatetimeIndex([d + pd.Timedelta("09:25:00")]))

    # 2. 连续竞价时段
    for st, et in sessions:
        # 默认左闭右开区间生成网格，例如 09:30:00 是第一分钟的开始
        # 注意：pandas date_range 如果 end 不能被 freq 整除的处理逻辑
        st_ts = d + pd.Timedelta(st + ":00")
        et_ts = d + pd.Timedelta(et + ":00")
        
        # 生成时间序列，注意 closed='left' 这里生成的是每个 bar 的起始时间
        rng = pd.date_range(start=st_ts, end=et_ts, freq=frequency, inclusive='left')
        parts.append(rng)

    if not parts:
        raise ValueError("Empty time grid configuration")
        
    grid = parts[0]
    for p in parts[1:]:
        grid = grid.union(p) # union 会自动排序并去重
        
    # 转换为 numpy int64 (nanoseconds)，方便 Numba 处理
    return grid.values#.astype("datetime64[ns]").view("int64")


def compute_grid_pos(time_ns: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    time_ns = time_ns.astype("datetime64[ns]").view("i8")
    if grid_ns.size == 0:
        return np.full(time_ns.size, -1, dtype=np.int32)

    idx = np.searchsorted(grid_ns, time_ns)
    valid = (idx >= 0) & (idx < grid_ns.size)
    idx2 = np.clip(idx, 0, grid_ns.size - 1)
    valid &= (grid_ns[idx2] == time_ns)
    out = idx2.astype(np.int32)
    out[~valid] = -1
    return out


def densify_features_from_df(
    df: pd.DataFrame, 
    start: np.ndarray,
    end: np.ndarray,
    grid_ns: np.ndarray,
    inst_categories: np.ndarray,
    required_columns: list[str], 
    fill_methods: list[int], # 0 fill zero, 1 ffill, 2 no fill (nan)
    backward_fill: bool = False
) -> np.ndarray:
    # 1. 准备数据
    # dt = df.index.get_level_values("datetime").to_numpy(dtype="datetime64[ns]")
    dt = df.index.get_level_values("datetime").to_numpy(dtype="datetime64[ns]")
    feat = np.ascontiguousarray(df[required_columns].to_numpy(dtype=np.float32)) # 建议转 float32 节省显存
    
    # 2. 计算位置
    time_ns = dt.view("i8")
    pos = compute_grid_pos(time_ns, grid_ns)
    
    # 3. 准备参数
    B = len(inst_categories)
    T = int(grid_ns.size)
    F = feat.shape[1]
    
    # 提取需要 ffill 的列索引
    fill_method_arr = np.array(fill_methods, dtype=np.int32)
    cols_zero = np.where(fill_method_arr == 0)[0]
    cols_ffill = np.where(fill_method_arr == 1)[0].astype(np.int64)        
    out = np.full((B, T, F), np.nan, dtype=np.float32)        
    for col in cols_zero:
        out[:, :, col] = 0.0

    densify_fill(
        pos, 
        feat, 
        start, 
        end, 
        T, 
        cols_ffill, 
        np.nan,
        out
    )

    out = np.ascontiguousarray(np.transpose(out, (2, 0, 1))) # speed up feature access in C-order
    if backward_fill:
        backward_fill_3d(out, axis=-1)
    return out



@nb.njit(parallel=True)
def densify_fill(
    pos: np.ndarray,          # int32, len=N, minute position per row, -1 ignore
    feat: np.ndarray,         # float64/float32, shape (N, F)
    start: np.ndarray,        # int64, shape (B,), start idx per instrument, -1 missing
    end: np.ndarray,          # int64, shape (B,), end idx per instrument
    T: int,
    ffill_cols_idx: np.ndarray, # col index for ffill (others are zero filled)
    init_val: float,
    out: np.ndarray,          # shape (B, T, F)
) -> None:
    """
    Optimized densify function using Gap-Filling strategy.
    """
    B = start.shape[0]
    F = feat.shape[1]
    n_ffill = ffill_cols_idx.shape[0]
    
    # Pre-allocate a buffer for last values for each thread? 
    # No, allocate inside loop to avoid race conditions, strict loop isolation.
    
    for b in nb.prange(B):
        s = start[b]
        e = end[b]
        
        # 如果该股票完全没有数据，直接跳过 (out 已经是 0)
        if s < 0 or s >= e:
            continue
            
        # --- 1. 初始化 FFill 状态 ---
        # 仅维护需要 ffill 的列的状态
        last_vals = np.empty(n_ffill, dtype=out.dtype)
        for i in range(n_ffill):
            last_vals[i] = init_val
            
        prev_t = -1  # 上一次观测的时间点
        
        # --- 2. 遍历该股票的每一条实际数据 (Observation Loop) ---
        for k in range(s, e):
            curr_t = pos[k]
            
            # 边界保护
            if curr_t < 0 or curr_t >= T:
                continue
            
            # [核心优化]: 填充间隙 (Fill Gap)
            # 仅对 ffill 列，填充 (prev_t, curr_t) 之间的时间段
            # 如果 curr_t == prev_t + 1，说明时间连续，无间隙，循环不会执行
            if curr_t > prev_t + 1:
                # 这是一个小循环，填充空缺的分钟
                for t in range(prev_t + 1, curr_t):
                    for i in range(n_ffill):
                        col_idx = ffill_cols_idx[i]
                        out[b, t, col_idx] = last_vals[i]
            
            # [核心优化]: 写入当前观测值 (Write Current)
            # 对所有列（包括 ffill 和 zero_fill）直接从 feat 复制
            # 这里利用 feat 的连续性，通常编译器会优化
            for f in range(F):
                val = feat[k, f]
                out[b, curr_t, f] = val
                
            # 更新 ffill 的缓存值
            for i in range(n_ffill):
                col_idx = ffill_cols_idx[i]
                last_vals[i] = feat[k, col_idx]
                
            prev_t = curr_t
            
        # --- 3. 填充尾部 (Fill Tail) ---
        # 如果最后一条数据不在收盘时刻，需要将 ffill 列延续到 T
        if prev_t < T - 1:
            for t in range(prev_t + 1, T):
                for i in range(n_ffill):
                    col_idx = ffill_cols_idx[i]
                    out[b, t, col_idx] = last_vals[i]


@nb.njit(parallel=True)
def _bfill_core_axis0(arr):
    d0, d1, d2 = arr.shape
    for j in nb.prange(d1):
        for k in nb.prange(d2):
            next_valid = np.nan
            for i in range(d0 - 1, -1, -1):
                val = arr[i, j, k]
                
                if np.isnan(val):
                    if not np.isnan(next_valid):
                        arr[i, j, k] = next_valid
                else:
                    next_valid = val


def backward_fill_3d(arr, axis=0):
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError("Input array must be of floating point type supporting NaN.")
        
    if axis == 0:
        arr_view = arr
    else:
        arr_view = np.moveaxis(arr, axis, 0)
    
    _bfill_core_axis0(arr_view)
    return arr
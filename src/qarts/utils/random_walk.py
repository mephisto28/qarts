import numpy as np


def simulate_random_walk(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1.0,
    n_paths: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    模拟几何布朗运动(GBM)价格路径。

    dS_t = mu * S_t dt + sigma * S_t dW_t
    离散精确解:
    S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    参数
    - s0: 初始价格
    - mu: 漂移项(年化/对应dt尺度的期望收益率)
    - sigma: 波动率(年化/对应dt尺度)
    - n_steps: 步数
    - dt: 时间步长
    - n_paths: 路径条数
    - seed: 随机种子

    返回
    - prices: shape (n_paths, n_steps+1)，包含t=0的初始点
    """
    if sigma < 0 or dt <= 0 or n_steps < 1 or n_paths < 1:
        raise ValueError("sigma>=0, dt>0, n_steps>=1, n_paths>=1。")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    prices = np.empty((n_paths, n_steps + 1), dtype=float)
    prices[:, 0] = s0
    prices[:, 1:] = prices[:, :1] * np.exp(np.cumsum(increments, axis=1))
    return prices


def simulate_noisy_random_walk(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1.0,
    n_paths: int = 1,
    noise_std: float = 0.01,
    noise_mode: str = "lognormal",  # "lognormal" 或 "additive"
    seed: int | None = None,
    return_latent: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    模拟“真实价格=GBM” + “观测噪声”的价格序列。

    噪声模式:
    - lognormal: 观测价 = 真实价 * exp(eps), eps~N(0, noise_std^2)
      解释：乘性噪声，保证观测价为正，常见于相对误差/报价噪声
    - additive: 观测价 = max(真实价 + eps, tiny), eps~N(0, noise_std^2)
      解释：加性噪声，更像绝对误差；会做截断确保为正

    参数
    - noise_std: 噪声标准差（lognormal时为对数空间标准差；additive时为价格空间标准差）
    - return_latent: 是否返回(观测价, 真实价)

    返回
    - observed 或 (observed, latent)
      shape均为 (n_paths, n_steps+1)
    """
    if noise_std < 0:
        raise ValueError("noise_std 必须 >= 0。")

    latent = simulate_random_walk(s0, mu, sigma, n_steps, dt=dt, n_paths=n_paths, seed=seed)
    rng = np.random.default_rng(seed + 1 if seed is not None else None)

    if noise_mode.lower() == "lognormal":
        eps = rng.normal(loc=0.0, scale=noise_std, size=latent.shape)
        observed = latent * np.exp(eps)
    elif noise_mode.lower() == "additive":
        eps = rng.normal(loc=0.0, scale=noise_std, size=latent.shape)
        observed = latent + eps
        # 防止出现非正价格
        tiny = np.finfo(float).tiny
        observed = np.maximum(observed, tiny)
    else:
        raise ValueError("noise_mode 只能是 'lognormal' 或 'additive'。")

    return (observed, latent) if return_latent else observed


def generate_mock_context(seed: int = 42):
    from qarts.modeling.factors.context import FactorContext, ContextSrc
    daily_prices = simulate_random_walk(s0=100, mu=0.00, sigma=0.02, n_steps=252, n_paths=10, seed=seed)
    intraday_prices = simulate_noisy_random_walk(s0=daily_prices[:, -1], mu=0.00, sigma=0.001, n_steps=240, n_paths=10, seed=seed)

    return daily_prices, intraday_prices

if __name__ == '__main__':
    from qarts.utils.random_walk import simulate_random_walk, simulate_noisy_random_walk
    daily_prices = simulate_random_walk(s0=100, mu=0.00, sigma=0.02, n_steps=252, n_paths=10, seed=42)
    intraday_prices = simulate_noisy_random_walk(s0=daily_prices[:, -1], mu=0.00, sigma=0.001, n_steps=240, n_paths=10, seed=42)
    

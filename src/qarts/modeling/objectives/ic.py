import torch
import torch.nn as nn

from .registry import register_metric


@register_metric('ic')
class RankIC(nn.Module):
    """
    计算截面 Rank IC。
    支持 (N, D) 或 (N, T, D)。N 为截面维度 (dim=0)。
    """
    def __init__(self, method='pearson'):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        # 1. Convert to ranks (0 to N-1) along cross-section
        p_rank = preds.argsort(dim=0).argsort(dim=0).float()
        t_rank = targets.argsort(dim=0).argsort(dim=0).float()
        
        # 2. Standardize ranks (Zero mean, Unit variance)
        p_std = (p_rank - p_rank.mean(dim=0, keepdim=True)) / (p_rank.std(dim=0, keepdim=True) + 1e-8)
        t_std = (t_rank - t_rank.mean(dim=0, keepdim=True)) / (t_rank.std(dim=0, keepdim=True) + 1e-8)
        
        # 3. Compute Correlation (mean product)
        ic = (p_std * t_std).mean(dim=0) # Shape: (D,) or (T, D)
        
        # 4. Aggregate over Time if exists
        return ic.mean(dim=0) if ic.dim() > 1 else ic

@register_metric('long_ic')
class LongRankIC(RankIC):
    """
    多头截面 Rank IC (只关注预测值 Top K% 的样本的排序能力)。
    """
    def __init__(self, top_ratio=0.5):
        super().__init__()
        self.top_ratio = top_ratio

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        # Mask out non-top items
        k = int(preds.shape[0] * self.top_ratio)
        if k < 2: return torch.zeros(preds.shape[-1], device=preds.device)
        
        # Get indices of top k preds
        _, indices = torch.topk(preds, k, dim=0)
        
        # Gather data
        # Note: Handling (N, T, D) gathering is complex, simplified for (N, D) logic per column
        # To maintain compact code, we apply a mask approach or simple loop if dims represent independent tasks
        ics = []
        # Flatten T into D for computation if T exists, effectively treating (T, D) as independent features
        p_flat = preds.flatten(1) if preds.dim() == 3 else preds
        t_flat = targets.flatten(1) if targets.dim() == 3 else targets
        
        for i in range(p_flat.shape[1]):
            col_p = p_flat[:, i]
            col_t = t_flat[:, i]
            top_idx = torch.topk(col_p, k).indices
            
            p_sub = col_p[top_idx]
            t_sub = col_t[top_idx]
            
            # Rank logic inside the subset
            p_r = p_sub.argsort().argsort().float()
            t_r = t_sub.argsort().argsort().float()
            
            # Covariance
            cov = ((p_r - p_r.mean()) * (t_r - t_r.mean())).mean()
            std = p_r.std() * t_r.std()
            ics.append(cov / (std + 1e-8))
            
        res = torch.tensor(ics, device=preds.device)
        return res.view(preds.shape[1:]) if preds.dim() == 3 else res
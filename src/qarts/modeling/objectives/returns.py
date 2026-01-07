import torch
import torch.nn as nn
from .registry import register_metric


@register_metric('tiered_returns')
class TieredReturns(nn.Module):
    """
    分层收益率。
    mode='long': 返回 Top K% 组的平均 Target (收益率)。
    mode='spread': 返回 Top K% - Bottom K%。
    mode='short': 返回 Bottom K% 组的平均 Target (收益率)。
    """
    def __init__(self, top_ratio=0.1, mode='long'):
        super().__init__()
        self.top_ratio = top_ratio
        self.mode = mode

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        k = int(preds.shape[0] * self.top_ratio)
        if k == 0: return torch.zeros(preds.shape[1:], device=preds.device)

        if self.mode == 'long':
            _, top_indices = torch.topk(preds, k, dim=0)
            ret = torch.gather(targets, 0, top_indices).mean(dim=0)
        elif self.mode == 'spread':
            _, top_indices = torch.topk(preds, k, dim=0)
            top_ret = torch.gather(targets, 0, top_indices).mean(dim=0)
            _, bot_indices = torch.topk(preds, k, dim=0, largest=False)
            bot_ret = torch.gather(targets, 0, bot_indices).mean(dim=0)
            ret = top_ret - bot_ret
        elif self.mode == 'short':
            _, bot_indices = torch.topk(preds, k, dim=0, largest=False)
            bot_ret = torch.gather(targets, 0, bot_indices).mean(dim=0)
            ret = bot_ret
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return ret.mean(dim=0) if ret.dim() > 1 else ret


@register_metric('long_precision')
class LongPrecision(nn.Module):
    """
    多头精准率：Top K% 预测中，Target > 0 的比例。
    """
    def __init__(self, top_ratio=0.1):
        super().__init__()
        self.top_ratio = top_ratio

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        k = int(preds.shape[0] * self.top_ratio)
        if k == 0: return torch.zeros(preds.shape[1:], device=preds.device)

        _, top_indices = torch.topk(preds, k, dim=0)
        top_targets = torch.gather(targets, 0, top_indices)
        
        hit_rate = (top_targets > 0).float().mean(dim=0)
        return hit_rate.mean(dim=0) if hit_rate.dim() > 1 else hit_rate
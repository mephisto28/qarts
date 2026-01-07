import torch
import torch.nn as nn
from .registry import register_metric


@register_metric('coverage')
class RiskCoverage(nn.Module):
    def forward(self, preds, targets):
        covered = targets < torch.abs(preds)
        return covered.float().mean(axis=0)


@register_metric('interval_width')
class IntervalWidth(nn.Module):
    def forward(self, preds, targets):
        return preds.abs().mean(axis=0)
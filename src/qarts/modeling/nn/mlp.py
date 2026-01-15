import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from .registery import register_model

__all__ = ['ResidualMLP', 'DualMLP']


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_hidden * 2)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return F.silu(x1) * x2


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
        use_post_ln=False,
        use_swiglu=False,
        stochastic_depth_prob=0.0,
    ):
        super().__init__()
        self.use_post_ln = use_post_ln
        self.stochastic_depth_prob = stochastic_depth_prob

        # 只注册一个 LayerNorm，避免 DDP 检查失败
        self.norm = nn.LayerNorm(dim)

        # 激活函数结构
        if use_swiglu:
            self.activation = SwiGLU(dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, dim)
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        if self.use_post_ln:
            out = x
        else:
            out = self.norm(x)

        if hasattr(self, "activation"):
            out = self.activation(out)
            out = self.dropout(out)
            out = self.proj_out(out)
        else:
            out = self.fc1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.fc2(out)

        if self.training and self.stochastic_depth_prob > 0.0:
            b = out.shape[0]
            mask = torch.rand((b, 1)) > self.stochastic_depth_prob
            out = out * mask.to(device=out.device).float()

        if self.use_post_ln:
            return self.norm(residual + out)
        else:
            return residual + out


def get_time_embedding(time_indices, d_model=8, max_t=240):
    """
    time_indices: 形状为 (batch_size,) 的张量，存储分钟序号 (1-240)
    d_model: 输出的编码维度
    """
    assert d_model % 2 == 0
    
    device = time_indices.device
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(500.0) / d_model)).to(device)
    
    t = time_indices.unsqueeze(1).float()
    pe = torch.zeros(time_indices.size(0), d_model).to(device)
    pe[:, 0::2] = torch.sin(t * div_term)
    pe[:, 1::2] = torch.cos(t * div_term)
    return pe


@register_model('mlp')
class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        dropout=0.0,
        use_post_ln=False,
        use_swiglu=False,
        stochastic_depth_prob=0.0,
        use_final_norm=False,
        use_pre_bn=False,
        use_pre_ln=False,
        use_positional_encoding=False,
        pe_dim=64
    ):
        super().__init__()
        self.use_pre_bn = use_pre_bn
        self.use_pre_ln = use_pre_ln
        self.use_positional_encoding = use_positional_encoding
        if use_pre_bn:
            self.in_norm = nn.BatchNorm1d(input_dim, affine=False, momentum=0.01)
        elif use_pre_ln:
            self.in_norm = nn.LayerNorm(input_dim, bias=False)
        self.in_projection = nn.Linear(input_dim, hidden_dim)
        if self.use_positional_encoding:
            self.pe_dim = pe_dim
            self.pos_proj = nn.Linear(pe_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_post_ln=use_post_ln,
                use_swiglu=use_swiglu,
                stochastic_depth_prob=stochastic_depth_prob,
            )
            for _ in range(num_blocks)
        ])
        self.out_projection = nn.Linear(hidden_dim, output_dim)
        if use_final_norm:
            self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x, pos_idx=None):
        if self.use_pre_bn or self.use_pre_ln:
            x = self.in_norm(x)
        x = self.in_projection(x)

        if self.use_positional_encoding:
            pos_emb = get_time_embedding(pos_idx, d_model=self.pe_dim)
            x = x + self.pos_proj(pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.out_projection(x)
        if hasattr(self, "final_norm"):
            x = self.final_norm(x)
        return x


@register_model('dual_mlp')
class DualMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dims,
        num_blocks,
        dropout=0.0,
        use_post_ln=False,
        use_swiglu=False,
        stochastic_depth_prob=0.0,
        use_final_norm=False,
        use_pre_bn=False,
        use_pre_ln=False,
        use_positional_encoding=False,
    ):
        super().__init__()
        self.use_pre_bn = use_pre_bn
        self.use_pre_ln = use_pre_ln
        if use_pre_bn:
            self.in_norm = nn.BatchNorm1d(input_dim, affine=False, momentum=0.01)
        elif use_pre_ln:
            self.in_norm = nn.LayerNorm(input_dim, bias=False)
        if use_positional_encoding:
            self.pos_proj = nn.Linear(240, hidden_dim)
        self.in_projection = nn.Linear(input_dim, hidden_dim)

        self.blocks1 = nn.ModuleList([
            ResidualMLPBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_post_ln=use_post_ln,
                use_swiglu=use_swiglu,
                stochastic_depth_prob=stochastic_depth_prob,
            )
            for _ in range(num_blocks)
        ])
        self.blocks2 = nn.ModuleList([
            ResidualMLPBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_post_ln=use_post_ln,
                use_swiglu=use_swiglu,
                stochastic_depth_prob=stochastic_depth_prob,
            )
            for _ in range(num_blocks)
        ])
        self.out1 = nn.Linear(hidden_dim, output_dims[0])
        self.out2 = nn.Linear(hidden_dim, output_dims[1])
        if use_final_norm:
            self.final_norm1 = nn.LayerNorm(output_dims[0])
            self.final_norm2 = nn.LayerNorm(output_dims[1])

    def forward(self, x):
        if self.use_pre_bn or self.use_pre_ln:
            x = self.in_norm(x)
        x = x0 = self.in_projection(x)
        for block in self.blocks1:
            x = block(x)
        y1 = self.out1(x)
        if hasattr(self, "final_norm1"):
            y1 = self.final_norm1(y1)
        
        x = x0
        for block in self.blocks2:
            x = block(x)
        y2 = self.out2(x)
        if hasattr(self, "final_norm2"):
            y2 = self.final_norm2(y2)
        return torch.cat([y1, y2], dim=-1)
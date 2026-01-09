import torch
import torch.nn as nn
import torch.nn.functional as F
from .registery import register_model


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
    ):
        super().__init__()
        self.use_pre_bn = use_pre_bn
        self.use_pre_ln = use_pre_ln
        if use_pre_bn:
            self.in_norm = nn.BatchNorm1d(input_dim, affine=False, momentum=0.01)
        elif use_pre_ln:
            self.in_norm = nn.LayerNorm(input_dim, bias=False)
        self.in_projection = nn.Linear(input_dim, hidden_dim)

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

    def forward(self, x):
        if self.use_pre_bn or self.use_pre_ln:
            x = self.in_norm(x)
        x = self.in_projection(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_projection(x)
        if hasattr(self, "final_norm"):
            x = self.final_norm(x)
        return x
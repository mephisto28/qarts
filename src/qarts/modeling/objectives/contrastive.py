import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .registry import register_loss_fn


class MemoryEfficientContrastiveLossForSeq(nn.Module):
    def __init__(self, chunk_size=32, epsilon=1e-8):
        super().__init__()
        self.chunk_size = chunk_size
        self.epsilon = epsilon

    @staticmethod
    def _contrastive_chunk_op_loss_only(
        preds_chunk, 
        preds_full, 
        targets_chunk, 
        targets_full, 
        masks_chunk, 
        masks_full, 
        start_idx
    ):
        """
        仅计算 Loss 的 Checkpoint 函数。
        只返回 Loss Sum，不再返回 Mask Sum (在外层计算)。
        """
        # 1. 确保 targets/masks 不带梯度，节省显存并防止图泄露
        targets_chunk = targets_chunk.detach()
        targets_full = targets_full.detach()
        masks_chunk = masks_chunk.detach()
        masks_full = masks_full.detach()

        C, L = preds_chunk.shape
        B = preds_full.shape[0]
        device = preds_chunk.device

        # 2. 生成上三角掩码 (Logical Upper Triangular Mask)
        row_indices = torch.arange(start_idx, start_idx + C, device=device).unsqueeze(1)
        col_indices = torch.arange(B, device=device).unsqueeze(0)
        valid_pair_mask = col_indices > row_indices # (C, B)

        # 快速剪枝：如果没有有效对，返回带梯度的 0
        if not valid_pair_mask.any():
            return preds_chunk.sum() * 0.0 

        # 3. Mask 计算 (C, B, L)
        # 注意：这里只计算参与 Loss 的部分
        mask_or = masks_chunk[:, None, :] & masks_full[None, :, :]
        mask_final = mask_or & valid_pair_mask.unsqueeze(-1)
        
        if not mask_final.any():
            return preds_chunk.sum() * 0.0

        # 4. 核心计算 (显存峰值处)
        # preds: (C, 1, L) - (1, B, L) -> (C, B, L)
        pred_diff = preds_chunk[:, None, :] - preds_full[None, :, :]
        
        target_diff = targets_chunk[:, None, :] - targets_full[None, :, :]
        target_gt = (target_diff > 0.001).float()

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_diff, target_gt, reduction='none'
        )
        
        # 5. Masked Sum
        masked_loss = bce_loss * mask_final
        
        # 返回标量，Checkpoint 会保存 inputs 用于在此处重新运行 forward 以计算 preds 的梯度
        return masked_loss.sum()

    def forward(self, preds, targets, masks):
        B, L, N = preds.shape
        device = preds.device
        total_loss = 0.0
        
        # 确保输入连续，避免 View 导致的显存碎片
        preds = preds.contiguous()
        targets = targets.contiguous()
        masks = masks.contiguous()

        for i in range(N):
            pred_i = preds[:, :, i]     # (B, L)
            target_i = targets[:, :, i] # (B, L)
            mask_i = masks[:, :, i]     # (B, L)

            loss_accum = 0.0
            mask_accum = 0.0
            
            # 外层循环计算 Mask Sum (不需要梯度，不需要 Checkpoint)
            # 为了防止 mask_accum 计算占用显存，我们也分块计算它，但不需要 autograd
            with torch.no_grad():
                # 预先计算 mask sum，因为这部分完全是逻辑运算，不涉及 float 梯度
                # 如果显存极其紧张，这里也需要像下面一样分块 loop
                pass 

            # --- Chunking 循环 ---
            for start_idx in range(0, B, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, B)
                
                # Slicing
                pred_chunk = pred_i[start_idx:end_idx]
                target_chunk = target_i[start_idx:end_idx]
                mask_chunk = mask_i[start_idx:end_idx]
                
                # A. 计算 Mask Sum (无梯度，纯计算)
                # 这部分代码显存占用极小 (Bool/Byte)，不需要 Checkpoint
                with torch.no_grad():
                    C = end_idx - start_idx
                    row_indices = torch.arange(start_idx, end_idx, device=device).unsqueeze(1)
                    col_indices = torch.arange(B, device=device).unsqueeze(0)
                    valid_pair_mask = col_indices > row_indices # (C, B)
                    
                    mask_or = mask_chunk[:, None, :] & mask_i[None, :, :] # (C, B, L)
                    mask_final = mask_or & valid_pair_mask.unsqueeze(-1)
                    chunk_mask_sum = mask_final.float().sum()
                    
                    mask_accum += chunk_mask_sum.item() # 取出 Python float，断开 Tensor 图
                    
                    # 及时删除临时变量
                    del mask_or, mask_final, valid_pair_mask

                # B. 计算 Loss (Gradient Checkpointing)
                if chunk_mask_sum > 0:
                    chunk_loss = checkpoint.checkpoint(
                        self._contrastive_chunk_op_loss_only,
                        pred_chunk,     # requires_grad
                        pred_i,         # requires_grad
                        target_chunk,   # constant
                        target_i,       # constant
                        mask_chunk,     # constant
                        mask_i,         # constant
                        start_idx,      # constant
                        use_reentrant=False # 关键配置
                    )
                    loss_accum = loss_accum + chunk_loss
                
            # 计算当前特征维度的平均 Loss
            if mask_accum > 0:
                total_loss = total_loss + (loss_accum / (mask_accum + self.epsilon))
        
        return total_loss / N


@register_loss_fn('contrastive')
class MemoryEfficientPairwiseLoss(nn.Module):
    def __init__(self, chunk_size=64, margin=0.001):
        super().__init__()
        self.chunk_size = chunk_size
        self.margin = margin

    @staticmethod
    def _chunk_op(p_chunk, p_full, t_chunk, t_full, m_chunk, m_full, start_idx):
        """
        Input Shapes: 
            Chunk: (S, C), Full: (B, C)
        Returns:
            masked_loss_sum (Scalar Tensor with grad)
            valid_count (Scalar Tensor detached)
        """
        # 1. 广播计算差异 (S, B, C)
        p_diff = p_chunk[:, None, :] - p_full[None, :, :]
        t_diff = t_chunk[:, None, :] - t_full[None, :, :]
        
        # 2. 构建上三角掩码 (只保留 j > i)
        # S: chunk_size, B: batch_size
        S, B, _ = p_diff.shape
        curr_rows = torch.arange(start_idx, start_idx + S, device=p_chunk.device)[:, None]
        all_cols = torch.arange(B, device=p_chunk.device)[None, :]
        triu_mask = (all_cols > curr_rows).unsqueeze(-1) # (S, B, 1)

        # 3. 合并 Input Mask 和 Triu Mask
        # (S, 1, C) & (1, B, C) & (S, B, 1) -> (S, B, C)
        mask_final = m_chunk[:, None, :] & m_full[None, :] & triu_mask
        
        # 快速剪枝
        if not mask_final.any():
            return p_chunk.sum() * 0.0, torch.tensor(0.0, device=p_chunk.device)

        # 4. 计算 Loss
        target_gt = (t_diff > 0.001).float() # 使用 self.margin (需传入或硬编码)
        loss = F.binary_cross_entropy_with_logits(p_diff, target_gt, reduction='none')
        
        # 5. Masking
        masked_loss = loss * mask_final
        
        # 返回 Loss Sum (带梯度) 和 Count (无梯度)
        return masked_loss.sum(), mask_final.float().sum().detach()

    def forward(self, preds, targets, masks = None):
        """
        preds, targets, masks: (B, C)
        """
        B, C = preds.shape
        total_loss = 0.0
        total_count = 0.0
        
        # 显式连续化，防止碎片
        if masks is None:
            masks = torch.ones_like(preds, dtype=torch.bool)
        preds, targets, masks = preds.contiguous(), targets.contiguous(), masks.contiguous()

        for start_idx in range(0, B, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, B)
            
            # Slicing (View operation, no copy)
            p_chunk = preds[start_idx:end_idx]
            t_chunk = targets[start_idx:end_idx]
            m_chunk = masks[start_idx:end_idx]

            # Gradient Checkpointing
            # 这里的 margin 0.001 硬编码在静态方法里了，若需动态可作为参数传入
            chunk_loss, chunk_count = checkpoint.checkpoint(
                self._chunk_op,
                p_chunk, preds,      # inputs with grad
                t_chunk, targets,    # no grad
                m_chunk, masks,      # no grad
                start_idx,           # scalar
                use_reentrant=False  # 关键: 防止图构建问题
            )
            
            total_loss = total_loss + chunk_loss
            total_count = total_count + chunk_count.item()

        return total_loss / (total_count + 1e-8)
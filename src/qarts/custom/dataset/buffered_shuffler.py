import time

import torch
from loguru import logger
import torch.utils.data as data


class BufferedShuffleIterator:
    def __init__(self, loader, buffer_size_samples, batch_size):
        """
        Args:
            loader: 基础 DataLoader，batch_size 必须为 None (即由 Dataset 直接返回 chunk)
            buffer_size_samples: Buffer 的容量 (样本数，例如 100 * batch_size)
            batch_size: 训练时的 mini-batch 大小
        """
        self.loader = loader
        self.loader_iter = iter(loader)
        self.buffer_capacity = buffer_size_samples
        self.batch_size = batch_size

        self.feature_names = None
        self.target_names = None
        # 内部状态
        self.buffer_features = None
        self.buffer_targets = None
        self.current_size = 0  # 当前 buffer 里有多少有效数据
        self.ptr = 0           # 并不是循环指针，而是指示初始化填充进度的
        
        # 用于缓存从 loader 读出来但还没塞进 buffer 的剩余数据
        self.pending_features = None
        self.pending_targets = None
        self.pending_cursor = 0
        
        # 耗尽标志
        self.loader_exhausted = False


    def _fetch_next_chunk(self):
        """尝试从 DataLoader 获取下一个 4096 块"""
        try:
            # 获取数据 (4096, ...)
            batch = next(self.loader_iter)
            f, t = batch['features'], batch['targets']
            if self.feature_names is None:
                self.feature_names = batch['feature_names']
                self.target_names = batch['target_names']
            
            return f, t
        except StopIteration:
            self.loader_exhausted = True
            return None, None

    def _init_buffer(self):
        """
        优化版初始化：预分配内存，避免 torch.cat 和 clone 带来的双倍内存峰值。
        """
        logger.info(f">>> Initializing Buffer... (Pre-allocation Strategy) with capacity {self.buffer_capacity}")
        
        filled_so_far = 0
        
        while filled_so_far < self.buffer_capacity:
            # 1. 获取下一个 chunk
            f, t = self._fetch_next_chunk()

            # 如果数据集读完了，Buffer 没填满也得停
            if f is None:
                break
            
            chunk_len = f.shape[0]
            
            # 2. 如果是第一次读到数据，根据数据形状预分配 Buffer 内存
            #    只分配这一次，避免后续扩容
            if self.buffer_features is None:
                # 这里的 shape 是 (buffer_capacity, 240, 101)
                # 使用 pin_memory=False (如果显存够大可手动 pin，但在 Dataset 里由 DataLoader pin 更安全)
                self.buffer_features = torch.empty(
                    (self.buffer_capacity, *f.shape[1:]), 
                    dtype=f.dtype
                )
                self.buffer_targets = torch.empty(
                    (self.buffer_capacity, *t.shape[1:]), 
                    dtype=t.dtype
                )
            
            # 3. 计算填充位置
            #    我们要把当前 chunk 填入 buffer [start : end]
            needed = self.buffer_capacity - filled_so_far
            # 实际能拿多少：取决于 chunk 够不够，或者 buffer 剩多少空位
            take_len = min(needed, chunk_len)
            
            start = filled_so_far
            end = start + take_len
            
            # 4. 原地填入 (In-place copy)，不产生额外大张量
            self.buffer_features[start:end] = f[:take_len]
            self.buffer_targets[start:end] = t[:take_len]
            
            filled_so_far += take_len
            logger.info(f">>> Filled {filled_so_far} / {self.buffer_capacity} in buffer")
            
            # 5. 处理溢出部分 (Pending)
            #    如果当前 chunk 比剩余空位大，多出来的部分存入 pending
            if chunk_len > take_len:
                # 这一小部分只能 clone 保存了，因为它属于下一次循环的数据
                self.pending_features = f[take_len:].clone()
                self.pending_targets = t[take_len:].clone()
                self.pending_cursor = 0
            
            # 关键：手动删除引用，确保 Python 尽快回收 chunk 内存
            del f, t

        # 更新 buffer 有效大小
        self.current_size = filled_so_far
        
        if self.current_size == 0:
            raise ValueError("Dataset is empty, failed to initialize buffer.")
            
        logger.info(f">>> Buffer Initialized. Current Size: {self.current_size} / {self.buffer_capacity}")

    def __iter__(self):
        return self

    def __next__(self):
        # 1. 如果 Buffer 还没初始化，先初始化
        if self.buffer_features is None:
            self._init_buffer()

        # 2. 如果 Buffer 数据不足一个 batch 且无法补充，停止迭代 (Epoch End)
        #    这里简化逻辑：如果 buffer 没空就能取，但通常我们希望保证 batch_size 完整
        if self.current_size < self.batch_size and self.loader_exhausted and self.pending_features is None:
            raise StopIteration

        # ===============================================
        # 核心逻辑：采样 + 替换
        # ===============================================
        
        # A. 生成随机索引 (从 buffer 中挑 batch_size 个)
        #    如果 buffer 不满 (尾部)，只取有效区域
        valid_range = self.current_size
        fetch_size = self.batch_size
        
        # 边界处理：如果最后剩的数据不够一个 batch
        if valid_range < fetch_size:
            fetch_size = valid_range
        
        indices = torch.randint(0, valid_range, (fetch_size,))
        
        # B. 取出数据 (Deep Copy，因为我们要覆盖 Buffer 位置)
        batch_f = self.buffer_features[indices].clone()
        batch_t = self.buffer_targets[indices].clone()
        
        # C. 补充新数据 (Refill)
        #    我们要找 fetch_size 个新样本填入刚刚被取走的 indices 位置
        
        # 检查 pending 区域是否有足够数据
        needed = fetch_size
        filled_count = 0
        
        while filled_count < needed:
            # 1. 尝试从 pending 获取
            if self.pending_features is not None:
                available = self.pending_features.shape[0] - self.pending_cursor
                to_take = min(needed - filled_count, available)
                
                # 源数据切片
                src_slice_f = self.pending_features[self.pending_cursor : self.pending_cursor + to_take]
                src_slice_t = self.pending_targets[self.pending_cursor : self.pending_cursor + to_take]
                
                # 填入 Buffer 的对应位置 (indices 中前 to_take 个位置)
                target_indices = indices[filled_count : filled_count + to_take]
                
                self.buffer_features[target_indices] = src_slice_f
                self.buffer_targets[target_indices] = src_slice_t
                
                self.pending_cursor += to_take
                filled_count += to_take
                
                # 如果 pending 用完了，清空
                if self.pending_cursor >= self.pending_features.shape[0]:
                    self.pending_features = None
                    self.pending_targets = None
            
            # 2. 如果 pending 空了，且还没填满，从 Loader 拉取新 Chunk
            if filled_count < needed:
                if self.loader_exhausted:
                    # 没有新数据了，无法填补。
                    # 此时 Buffer 里的这些位置实际上变成了“废弃/重复”数据，或者我们缩小 valid_size
                    # 简单策略：将 Buffer 尾部的数据移到这些空位，然后缩小 current_size
                    remaining_holes = needed - filled_count
                    holes_indices = indices[filled_count:]
                    
                    # 将 Buffer 尾部的数据搬运到空洞处 (类似 vector remove 的 swap 操作)
                    # 这里的逻辑稍微复杂，为保持高性能，简单的做法是：
                    # 直接缩小 current_size，下次不采样这些区域即可
                    # 但由于我们是随机采样，这就意味着这次取出的数据位置变成了“无效区”。
                    # 最简单的 Swap Remove:
                    # 把 buffer 末尾的 valid_data 填入 holes_indices
                    for h_idx in holes_indices:
                        if self.current_size - 1 != h_idx: # 避免自己交换自己
                            self.buffer_features[h_idx] = self.buffer_features[self.current_size - 1]
                            self.buffer_targets[h_idx] = self.buffer_targets[self.current_size - 1]
                        self.current_size -= 1
                    
                    # 填补循环结束
                    break
                else:
                    # 加载新 Chunk
                    new_f, new_t = self._fetch_next_chunk()
                    if new_f is None:
                        continue # 实际上会进入上面的 exhausted 分支
                    
                    self.pending_features = new_f
                    self.pending_targets = new_t
                    self.pending_cursor = 0
        
        return {
            'features': batch_f,
            'targets': batch_t,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }


class CrossBatchDataLoader:
    def __init__(self, dataset, buffer_batch_count=50, batch_size=32, num_workers=4, collate_fn=None):
        """
        buffer_batch_count: Buffer 中容纳多少个 mini-batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size_samples = getattr(self.dataset, 'chunk_size', 4096) * buffer_batch_count
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def __iter__(self):
        # 1. 创建基础 Loader：负责多进程 IO，只吐出 4096 大块
        #    batch_size=None 表示禁用自动 batching，直接返回 Dataset 的返回值
        base_loader = data.DataLoader(
            self.dataset,
            batch_size=1, 
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=1 if self.num_workers > 0 else None,
            collate_fn=self.collate_fn
        )
        
        # 2. 返回我们的自定义迭代器
        return BufferedShuffleIterator(base_loader, self.buffer_size_samples, self.batch_size)

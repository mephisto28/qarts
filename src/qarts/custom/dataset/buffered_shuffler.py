import time

import torch
from loguru import logger
import torch.utils.data as data


class BufferedShuffleIterator:
    def __init__(self, loader, buffer_size_samples, batch_size):
        """
        通用型 Buffer Shuffle Iterator
        自动识别并混洗 Batch 中所有类型为 torch.Tensor 的字段。
        非 Tensor 字段（如 names）会被视为静态元数据，取第一批的值在每次迭代中返回。
        """
        self.loader = loader
        self.loader_iter = iter(loader)
        self.buffer_capacity = buffer_size_samples
        self.batch_size = batch_size

        # 核心数据结构：字典
        # self.buffers[key] = Large Tensor
        self.buffers = {}       
        
        # 缓存数据：字典
        # self.pending_data[key] = Remaining Tensor
        self.pending_data = {} 
        self.pending_cursor = 0
        
        # 静态元数据 (非 Tensor 字段，如 feature_names)
        self.static_meta = {}
        self.meta_captured = False

        # 状态控制
        self.current_size = 0
        self.loader_exhausted = False

    def _fetch_next_chunk(self):
        """
        从 DataLoader 获取下一个块，并自动分离 Tensor 和 非Tensor 数据
        """
        try:
            # 假设 batch 是一个 dict
            batch = next(self.loader_iter)
            
            tensor_chunk = {}
            
            # 首次运行时，捕捉元数据
            if not self.meta_captured:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        tensor_chunk[k] = v
                    else:
                        self.static_meta[k] = v
                self.meta_captured = True
            else:
                # 后续只需提取 Tensor
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        tensor_chunk[k] = v
            
            return tensor_chunk
        
        except StopIteration:
            self.loader_exhausted = True
            return None

    def _init_buffer(self):
        """
        通用初始化：根据第一次读到的数据结构，为每个 Tensor 字段创建 Buffer
        """
        logger.info(f">>> Initializing Buffer... (Generic Mode) Capacity: {self.buffer_capacity}")
        
        filled_so_far = 0
        
        while filled_so_far < self.buffer_capacity:
            # 1. 获取下一个 chunk (Dict[str, Tensor])
            chunk_data = self._fetch_next_chunk() # return dict or None

            if chunk_data is None:
                break
            
            # 获取任意一个 tensor 来计算长度 (假设所有 tensor 第一维相同)
            first_key = next(iter(chunk_data))
            chunk_len = chunk_data[first_key].shape[0]
            
            # 2. 首次分配 Buffer 内存
            if not self.buffers:
                for k, v in chunk_data.items():
                    # 动态创建对应的 Buffer，shape 为 (capacity, *original_shape[1:])
                    self.buffers[k] = torch.empty(
                        (self.buffer_capacity, *v.shape[1:]), 
                        dtype=v.dtype
                    )
            
            # 3. 计算填充位置
            needed = self.buffer_capacity - filled_so_far
            take_len = min(needed, chunk_len)
            
            start = filled_so_far
            end = start + take_len
            
            # 4. 遍历所有字段进行填充
            for k, tensor_data in chunk_data.items():
                self.buffers[k][start:end] = tensor_data[:take_len]
                
                # 5. 处理溢出部分 (Pending)
                if chunk_len > take_len:
                    if k not in self.pending_data:
                        self.pending_data[k] = None # 占位
                    # Clone 剩余部分
                    self.pending_data[k] = tensor_data[take_len:].clone()
            
            if chunk_len > take_len:
                self.pending_cursor = 0

            filled_so_far += take_len
            logger.info(f">>> Filled {filled_so_far} / {self.buffer_capacity}")
            
            # 内存回收
            del chunk_data

        self.current_size = filled_so_far
        
        if self.current_size == 0:
            raise ValueError("Dataset is empty, failed to initialize buffer.")
            
        logger.info(f">>> Buffer Initialized. Keys: {list(self.buffers.keys())}")

    def __iter__(self):
        return self

    def __next__(self):
        # 1. 初始化
        if not self.buffers:
            self._init_buffer()

        # 2. 终止条件检查
        # 只要 pending 里没数据了，且 loader 也没了，且 buffer 不够一个 batch，就停止
        has_pending = bool(self.pending_data)
        if self.current_size < self.batch_size and self.loader_exhausted and not has_pending:
            raise StopIteration

        # ===============================================
        # A. 采样 (indices 对所有字段通用)
        # ===============================================
        valid_range = self.current_size
        fetch_size = self.batch_size
        
        if valid_range < fetch_size:
            fetch_size = valid_range
        
        # 生成随机索引
        indices = torch.randint(0, valid_range, (fetch_size,))
        
        # ===============================================
        # B. 取出数据 (构造返回字典)
        # ===============================================
        result_batch = {}
        
        # 1. 放入 Tensor 数据
        for k, buf in self.buffers.items():
            result_batch[k] = buf[indices].clone()
            
        # 2. 放入静态元数据 (如 names)
        result_batch.update(self.static_meta)
        
        # ===============================================
        # C. 补充新数据 (Refill) - 同步对所有 Buffer 操作
        # ===============================================
        needed = fetch_size
        filled_count = 0
        
        while filled_count < needed:
            # --- 场景 1: 从 Pending 填充 ---
            if self.pending_data:
                # 获取任意一个 pending tensor 计算可用长度
                first_pending_key = next(iter(self.pending_data))
                available = self.pending_data[first_pending_key].shape[0] - self.pending_cursor
                to_take = min(needed - filled_count, available)
                
                # 计算 buffer 中需要填入的目标索引位置
                target_indices = indices[filled_count : filled_count + to_take]
                
                # 对所有字段执行搬运
                for k in self.buffers:
                    src = self.pending_data[k][self.pending_cursor : self.pending_cursor + to_take]
                    self.buffers[k][target_indices] = src
                
                self.pending_cursor += to_take
                filled_count += to_take
                
                # 如果 pending 耗尽，清空字典
                if self.pending_cursor >= self.pending_data[first_pending_key].shape[0]:
                    self.pending_data.clear() # 清空整个字典
            
            # --- 场景 2: 从 Loader 获取新 Chunk ---
            if filled_count < needed:
                if self.loader_exhausted:
                    # --- 场景 3: 彻底没数据了 (Swap Remove) ---
                    # 将 Buffer 尾部的数据填入被取走后留下的“空洞”
                    remaining_holes_count = needed - filled_count
                    holes_indices = indices[filled_count:]
                    
                    # 倒序遍历空洞（虽然 swap 顺序不敏感，但逻辑上清晰）
                    for h_idx in holes_indices:
                        tail_idx = self.current_size - 1
                        
                        # 如果空洞本身不在尾部，才需要搬运
                        if tail_idx != h_idx:
                            for k in self.buffers:
                                self.buffers[k][h_idx] = self.buffers[k][tail_idx]
                        
                        # 缩小有效范围
                        self.current_size -= 1
                    
                    # 既然没数据填了，强行结束填充循环
                    break
                else:
                    # 加载新数据到 Pending
                    new_chunk = self._fetch_next_chunk()
                    if new_chunk is None:
                        continue # loop back to handle exhausted
                    
                    self.pending_data = new_chunk # 整个 dict 赋值
                    self.pending_cursor = 0
        
        return result_batch


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
        self.loader = None

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
        self.loader = base_loader
        
        # 2. 返回我们的自定义迭代器
        return BufferedShuffleIterator(base_loader, self.buffer_size_samples, self.batch_size)

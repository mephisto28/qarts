import os
import json
import random

import torch
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from qarts.modeling.nn import ResidualMLP
from qarts.modeling.objectives.contrastive import MemoryEfficientPairwiseLoss
from qarts.custom.dataset import get_dataset, get_collate_fn


class Trainer:

    def __init__(self, config: dict | str):
        if isinstance(config, str):
            config = json.load(open(config))
        self.config = config
        logger.info(f'Config: {self.config}')
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[config['train'].get('dtype', 'fp32')]
        self.device = config['train'].get('device', 'cuda')

        self.prepare_model()
        self.prepare_train()
        self.prepare_dataset()

    def prepare_dataset(self):
        self.dataset_config = self.config['dataset']
        self.dataset_type = self.dataset_config['type']
        self.train_dataset = get_dataset(self.dataset_type)(self.dataset_config, is_training=True)
        self.valid_dataset = get_dataset(self.dataset_type)(self.dataset_config, is_training=False)

    def prepare_model(self):
        self.model_config = self.config['model']
        self.model_type = self.model_config['type']
        if self.model_type == 'mlp':
            self.model = ResidualMLP(**self.model_config['params'])
        else:
            raise ValueError(f'Model type {self.model_type} not supported')
        self.model.to(dtype=self.dtype, device=self.device)
        return self.model

    def prepare_train(self):
        self.train_config = self.config['train']
        self.optimizer = self.get_optimizer(self.train_config)
        self.lr_scheduler = self.get_lr_scheduler(self.train_config)
        return self.optimizer, self.lr_scheduler

    def get_optimizer(self, train_config: dict):
        optimizer_type = train_config['optimizer']
        lr = train_config['lr']
        weight_decay = train_config['weight_decay']
        if optimizer_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, **train_config['optimizer_params'])
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, **train_config['optimizer_params'])
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, **train_config['optimizer_params'])
        else:
            raise ValueError(f'Optimizer type {optimizer_type} not supported')
    
    def get_lr_scheduler(self, train_config: dict):
        lr_scheduler_type = train_config['lr_scheduler']
        if lr_scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **train_config['lr_scheduler_params'])
        else:
            raise ValueError(f'LR scheduler type {lr_scheduler_type} not supported')

    def get_loss_fn(self, train_config: dict):
        loss_fn_type = train_config['loss_fn']
        if loss_fn_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_fn_type == 'contrastive':
            return MemoryEfficientPairwiseLoss()
        else:
            raise ValueError(f'Loss function type {loss_fn_type} not supported')

    def train(self):
        train_config = self.train_config
        collate_fn = get_collate_fn(self.dataset_config['collate_fn'], **self.dataset_config['collate_fn_params'])
        dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=train_config.get('num_workers', 4), collate_fn=collate_fn)
        loss_fn = self.get_loss_fn(train_config)

        num_epochs = train_config['num_epochs']
        log_interval = train_config.get('log_interval', 100)

        step = 0
        for epoch in range(num_epochs):
            self.model.train()
            for batch in dataloader:
                X, y, t_idx = batch
                X = X.to(dtype=self.dtype, device=self.device)
                y = y.to(dtype=self.dtype, device=self.device)

                self.optimizer.zero_grad()
                preds = self.model(X)
                loss = loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                if step % log_interval == 0:
                    logger.info(f"Step {step:05d}(E{epoch}), LR: {self.optimizer.param_groups[0]['lr']:.5f}, Loss: {loss.item():.6f}")

                step += 1
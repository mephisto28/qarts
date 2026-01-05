import json
import torch
from torch.utils.data import DataLoader

from qarts.modeling.nn import ResidualMLP
from qarts.custom.dataset import get_dataset


class Trainer:

    def __init__(self, config: dict | str, device: str = 'cuda'):
        if isinstance(config, str):
            config = json.load(open(config))
        self.config = config
        self.dtype = config['train'].get('dtype', 'float32')
        self.device = device

        self.prepare_dataset()
        self.prepare_model()
        self.prepare_train()

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
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
        else:
            raise ValueError(f'Optimizer type {optimizer_type} not supported')
    
    def get_lr_scheduler(self, train_config: dict):
        lr_scheduler_type = train_config['lr_scheduler']
        if lr_scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **train_config['optimizer_params'])
        else:
            raise ValueError(f'LR scheduler type {lr_scheduler_type} not supported')

    def get_loss_fn(self, train_config: dict):
        loss_fn_type = train_config['loss_fn']
        if loss_fn_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_fn_type == 'contrastive':
            return 
        else:
            raise ValueError(f'Loss function type {loss_fn_type} not supported')

    def train(self):
        train_config = self.train_config
        dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=train_config.get('num_workers', 4))
        loss_fn = self.get_loss_fn(train_config)


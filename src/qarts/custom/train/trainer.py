import os
import json
import glob

import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from qarts.modeling.nn import ResidualMLP
from qarts.custom.dataset import get_dataset, get_collate_fn
from qarts.modeling.objectives import get_loss_fn, HybridLoss, HybridEvaluator


class Trainer:

    def __init__(self, config: dict | str, name: str):
        d = os.path.dirname
        project_dir = d(d(d(d(d(os.path.abspath(__file__))))))
        if isinstance(config, str):
            config = json.load(open(config))
        self.config = config
        self.name = name
        self.output_dir = os.path.join(project_dir, 'experiments', 'output', name)
        logger.info(f'Model {name} Config: {self.config}\nResults will be saved to {self.output_dir}')

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
        loss_fn_type = train_config['objectives']
        if isinstance(loss_fn_type, list):
            loss_fn = HybridLoss(schemas=loss_fn_type)
            return loss_fn
        else:
            return get_loss_fn(loss_fn_type)

    def get_evaluator(self, train_config: dict):
        loss_fn_type = train_config['objectives']
        evaluator = HybridEvaluator(schemas=loss_fn_type)
        return evaluator

    def load_model(self, epoch: int = -1):
        prefix = os.path.join(self.output_dir, 'ckpt')
        os.makedirs(prefix, exist_ok=True)
        model_paths = sorted(glob.glob(os.path.join(prefix, 'model_e*.pth')))
        if len(model_paths) == 0:
            logger.warning(f'No model found at {prefix}')
            return

        model_path = model_paths[epoch]
        opt_path = model_path.replace('model_e', 'optimizer_e')
        if os.path.exists(model_path) and os.path.exists(opt_path):
            self.model.load_state_dict(torch.load(model_path))
            self.optimizer.load_state_dict(torch.load(opt_path)['optimizer'])
            self.lr_scheduler.load_state_dict(torch.load(opt_path)['lr_scheduler'])
        else:
            raise FileNotFoundError(f'Model or optimizer not found at {model_path} / {opt_path}')
        logger.info(f'Model loaded from {model_path}')


    def save_model(self, epoch: int):
        prefix = os.path.join(self.output_dir, 'ckpt')
        model_path = f'model_e{epoch:02d}.pth'
        opt_path = f'optimizer_e{epoch:02d}.pth'
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(prefix, model_path))
        opt_state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        torch.save(opt_state_dict, os.path.join(prefix, opt_path))
        logger.info(f'Model saved to {prefix}: {model_path} / {opt_path}')

    def train(self):
        train_config = self.train_config
        collate_fn = get_collate_fn(self.dataset_config['collate_fn'], **self.dataset_config['collate_fn_params'])
        dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=train_config.get('num_workers', 4), collate_fn=collate_fn)
        loss_fn = self.get_loss_fn(train_config)
        evaluator = self.get_evaluator(train_config)
        num_epochs = train_config['num_epochs']
        log_interval = train_config.get('log_interval', 10)
        eval_interval = train_config.get('eval_interval', 10)

        step = 0
        self.load_model()
        wandb.init(project='qarts', name=self.name)
        for epoch in range(num_epochs):
            self.save_model(epoch)
            self.model.train()
            
            for batch in dataloader:
                X = batch['features'].to(dtype=self.dtype, device=self.device)
                y = batch['targets'].to(dtype=self.dtype, device=self.device)

                self.optimizer.zero_grad()
                preds = self.model(X)
                if isinstance(loss_fn, HybridLoss):
                    loss_fn.set_input_specs(input_columns=batch['target_names'], target_columns=batch['target_names'])
                    evaluator.set_input_specs(input_columns=batch['target_names'], target_columns=batch['target_names'])
                loss, loss_info = loss_fn(preds, y)
                eval_results = evaluator(preds, y)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                if step % log_interval == 0:
                    detailed_loss_info = ', '.join([f'{n}: {v:.4f}' for n, v in loss_info.items()])
                    logger.info(f"Step {step:05d}(E{epoch}), LR: {self.optimizer.param_groups[0]['lr']:.5f}, Loss: {loss.item():.6f}, {detailed_loss_info}")
                wandb.log({n: v for n, v in eval_results.items()})
                if step % eval_interval == 0:
                    detailed_eval_results = ', '.join([f'{n}: {v:.4f}' for n, v in eval_results.items()])
                    logger.info(f"Step {step:05d}(E{epoch}), Eval: {detailed_eval_results}")


                step += 1
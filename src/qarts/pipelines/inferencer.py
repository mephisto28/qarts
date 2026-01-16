import os
import json
import glob
import typing as T

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

from qarts.core.panel import PanelBlockDense
from qarts.modeling.nn import get_model
from qarts.pipelines import Processor, GlobalContext
from qarts.pipelines.factor_process import get_factor_process_pipeline


class ModelInferenceProcessor(Processor):
    _name: str = 'inference'

    def __init__(
        self, 
        config: dict | str, 
        epoch: int = -1, 
        output_fields: T.Optional[list[str]] = None, 
        pred_indices: T.Optional[list[int]] = None,
        device: str = 'cuda', 
        dtype: torch.dtype = torch.float32,
        trange: tuple[int, int] = (200, 238) # near close 
    ):
        if isinstance(config, str):
            model_name = os.path.basename(config).split('.')[0]
            config = json.load(open(config))
            model_config = config['model']
            factor_group_name = config.get('dataset', {}).get('factor_group', 'default')
        else:
            if 'name' not in config:
                raise ValueError('Model name is required')
            model_name = config['name']
            factor_group_name = config.get('dataset', {}).get('factor_group', 'default')
            model_config = config if 'model' not in config else config['model']
        
        if output_fields is None:
            if 'output_fields' in model_config:
                output_fields = model_config['output_fields']
            else:
                objectives = config.get('train', {}).get('objectives', [])
                alpha_objective = [o for o in objectives if o['name'] == 'alpha']
                if len(alpha_objective) > 0:
                    output_fields = alpha_objective[0]['target_fields']
                    pred_indices = alpha_objective[0]['pred_indices']
        
        self.config = model_config
        self.device = device
        self.dtype = dtype
        self.epoch = epoch
        self.model_name = model_name
        self.factor_group_name = factor_group_name
        self.pred_fields = output_fields
        self.pred_indices = pred_indices
        self.batch_size = 256
        self.trange = trange
        self.model: nn.Module = self.create_model()
        logger.info(f'Creating {model_name} output: {output_fields} Epoch: {epoch}')

        self.load_model(epoch)

    @property
    def input_fields(self) -> list[str]:
        return [f'factors_{self.factor_group_name}']

    @property
    def name(self) -> str:
        return f'inference_{self.model_name}'

    def load_model(self, epoch: int = -1):
        d = os.path.dirname
        proj_dir = d(d(d(d(os.path.abspath(__file__)))))
        out_dir = os.path.join(proj_dir, 'experiments', 'output', )
        ckpt_dir = os.path.join(out_dir, self.model_name, 'ckpt')
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, 'model_e*.pth')))
        if len(ckpt_paths) < 2:
            raise FileNotFoundError(f'No model found at {ckpt_dir}, existing checkpoints: {ckpt_paths}')
        ckpt_path = sorted(ckpt_paths)[epoch]
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict)
        logger.info(f'Model loaded from {ckpt_path}')
        return self.model

    def create_model(self) -> nn.Module:
        config = self.config
        model_name = config['type']
        Model = get_model(model_name)
        model = Model(**config['params'])
        return model.to(device=self.device, dtype=self.dtype)

    def process(self, context: GlobalContext) -> T.Any:
        factors_block = context.get(self.input_fields[0]) # (F, N, T)
        X = factors_block.data.transpose(1, 2, 0) # (N, T, F)
        X = X[factors_block.is_valid_instruments]
        X = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            preds = []
            B, T, F = X.shape
            for t in range(T):
                x = X[:, t, :]
                pos_idx = (t - self.trange[0]) / (self.trange[1] - self.trange[0]) * 240
                pos_idx = torch.full((B,), fill_value=int(pos_idx), dtype=torch.int64, device=self.device)
                preds.append(self.model(x, pos_idx))
            preds = torch.stack(preds, dim=1)
            preds = preds[..., self.pred_indices]

            # num_batches = int(np.ceil(X.shape[0] / self.batch_size))
            # for i in range(num_batches):
            #     start_idx = i * self.batch_size
            #     end_idx = min(start_idx + self.batch_size, X.shape[0])
            #     preds.append(self.model(X[start_idx:end_idx]))
            # preds = torch.cat(preds, dim=0)
        preds = preds.cpu().numpy()
        return PanelBlockDense(
            instruments=factors_block.instruments[factors_block.is_valid_instruments],
            timestamps=factors_block.timestamps,
            data=preds.transpose(2, 0, 1),
            fields=self.pred_fields or [f'pred_{i}' for i in range(preds.shape[1])],
            frequency=factors_block.frequency,
            is_valid_instruments=np.ones(len(preds), dtype=bool)
        )


def get_inference_pipeline(
    config: dict | str,
    epoch: int = -1,
    output_fields: T.Optional[list[str]] = None,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
):
    processor = ModelInferenceProcessor(config, epoch, output_fields, device, dtype)
    pipeline = get_factor_process_pipeline(processor.factor_group_name)
    pipeline.register_processor(processor)
    return pipeline
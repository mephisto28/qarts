import torch.nn as nn
from loguru import logger

from .registry import Schema, get_loss_fn, register_loss_fn


@register_loss_fn('hybrid')
class HybridLoss(nn.Module):
    def __init__(
        self, 
        schemas: list[Schema] | list[dict], 
        input_columns: list[str] = None,
        target_columns: list[str] = None,
    ):
        super().__init__()
        if isinstance(schemas[0], dict):
            schemas = [Schema(**schema) for schema in schemas]
        self.schemas = schemas
        self.loss_fns = {
            schema.name: {
                loss['type']: get_loss_fn(loss['type'])(**loss['params'])
                for loss in schema.loss
            }
            for schema in self.schemas
        }
        self.set_input_specs(input_columns, target_columns)

    def set_input_specs(self, input_columns: list[str], target_columns: list[str]):
        if input_columns is None or target_columns is None:
            return
        if getattr(self, 'input_indices', None) is None:
            self.input_columns = input_columns
            self.output_columns = target_columns
            self.input_indices = {schema.name: [input_columns.index(f) for f in schema.fields] for schema in self.schemas}
            self.output_indices = {schema.name: [target_columns.index(f) for f in schema.fields] for schema in self.schemas}
            logger.info(f"Input specifications set: {self.input_columns}, {self.output_columns}")

    def forward(self, preds, targets):
        if self.input_indices is None:
            raise ValueError("Input specifications not set, call set_input_specs for hybrid loss to work")
            
        total_loss = 0.0
        loss_details = {}
        
        for item in self.schemas:
            name = item.name
            loss_fns = self.loss_fns[name]
            input_indices = self.input_indices[name]
            output_indices = self.output_indices[name]
            for loss_name, loss_fn in loss_fns.items():   
                p_sub = preds[..., input_indices]
                t_sub = targets[..., output_indices]
                sub_loss = loss_fn(p_sub, t_sub)
                
                total_loss += sub_loss * item.weight
                final_name = f'{name}/{loss_name}'
                loss_details[final_name] = sub_loss.item()

        return total_loss, loss_details
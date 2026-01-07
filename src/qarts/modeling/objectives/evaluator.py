import torch
import torch.nn as nn
from loguru import logger
from .registry import Schema, register_metric, get_metric_fn


@register_metric('hybrid')
class HybridEvaluator(nn.Module):
    def __init__(
        self, 
        schemas: list[dict], # Expecting dicts or Schema objects
        input_columns: list[str] = None,
        target_columns: list[str] = None,
    ):
        super().__init__()
        if isinstance(schemas[0], dict):
            schemas = [Schema(**schema) for schema in schemas]
        self.schemas = schemas 
        self.metrics = {}
        
        # Initialize metrics per schema
        for schema in self.schemas:
            s_name = schema.name if hasattr(schema, 'name') else schema['name']
            s_metrics = schema.metrics if hasattr(schema, 'metrics') else schema.get('metrics', [])
            
            self.metrics[s_name] = {
                metric['type']: get_metric_fn(metric['type'])(**metric.get('params', {}))
                for metric in s_metrics
            }
            
        self.set_input_specs(input_columns, target_columns)

    def set_input_specs(self, input_columns: list[str], target_columns: list[str]):
        if input_columns is None or target_columns is None:
            return
        
        if getattr(self, 'input_indices', None) is None:
            self.input_columns = input_columns
            self.output_columns = target_columns
            self.input_indices = {}
            self.output_indices = {}
            
            for schema in self.schemas:
                # Handle both object and dict access for Schema
                name = schema.name if hasattr(schema, 'name') else schema['name']
                fields = schema.fields if hasattr(schema, 'fields') else schema['fields']
                
                self.input_indices[name] = [input_columns.index(f) for f in fields]
                self.output_indices[name] = [target_columns.index(f) for f in fields]
                
            logger.info(f"Evaluator specs set. Schemas: {list(self.metrics.keys())}")

    @torch.no_grad()
    def forward(self, preds, targets):
        if self.input_indices is None:
            raise ValueError("Input specs not set for HybridEvaluator")
            
        eval_results = {}
        
        for schema in self.schemas:
            name = schema.name if hasattr(schema, 'name') else schema['name']
            metric_fns = self.metrics[name]
            
            # Slice tensors
            input_idx = self.input_indices[name]
            output_idx = self.output_indices[name]
            
            p_sub = preds[..., input_idx]
            t_sub = targets[..., output_idx]
            
            # Compute each metric
            for metric_name, metric_fn in metric_fns.items():
                val = metric_fn(p_sub, t_sub)
                
                key = f'{name}/{metric_name}'
                
                # Handle metrics that return dicts (like stratified returns)
                if isinstance(val, dict):
                    for k, v in val.items():
                        eval_results[f'{key}/{k}'] = v.item() if torch.is_tensor(v) else v
                else:
                    for i, f in enumerate(schema.fields):
                        eval_results[f'{key}/{f}'] = val[i].item() if torch.is_tensor(val) else val[i]

        return eval_results
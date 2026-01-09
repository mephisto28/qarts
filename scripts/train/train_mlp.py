import os
import json
import random

import fire
import torch
import numpy as np
from loguru import logger

from qarts.custom.train.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f'Using seed: {seed}')


@fire.Fire
def main(exp_name: str):
    set_seed(42)
    d = os.path.dirname
    project_dir = d(d(d(os.path.abspath(__file__))))
    config_path = os.path.join(project_dir, 'experiments', 'config', f'{exp_name}.json')
    config = json.load(open(config_path))
    trainer = Trainer(config=config, name=exp_name)
    trainer.train()

if __name__ == "__main__":
    main()

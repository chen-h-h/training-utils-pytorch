import math
from typing import Any, Dict, List, Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

def WarmupCosineLR(optimizer: Optimizer, max_epochs: int, warmup_epoch: int, 
                   lr_factor_min: float) -> _LRScheduler:
    
    def lr_lambda(cur_epoch):
        lr_factor = 0
        if cur_epoch < warmup_epoch:
            lr_factor = (cur_epoch + 1) / (warmup_epoch + 1)
        else:
            lr_factor = lr_factor_min + (1 - lr_factor_min) * 0.5 * \
            (1. + math.cos(math.pi * (cur_epoch - warmup_epoch) / (max_epochs - warmup_epoch)))
        
        return lr_factor
    
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    return scheduler
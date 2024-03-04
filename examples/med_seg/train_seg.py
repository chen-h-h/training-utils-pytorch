import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from tup import Trainer, TrainingArgs, ConfigArgParser, EvalHook, HookBase, \
                set_random_seed, save_config, setup_logger, str2func, init_distributed
from medseg import test_hook

logger = logging.getLogger(__name__)


def main():
    # 1. Create an argument parser supporting loading YAML configuration file and command line parameter
    parser = ConfigArgParser(description="Training-Utils-Pytorch Example: Chest X-ray Segmentaion")
    config = parser.parse_args()

    # 2. Basic setup
    rank, local_rank, world_size = init_distributed()
    is_distributed = world_size > 1

    setup_logger(output_dir=config.training_args.output_dir, rank=rank)
    save_config(config, os.path.join(config.training_args.output_dir, "runtime_config.yaml"), rank=rank)
    set_random_seed(None if config.seed < 0 else config.seed + rank, config.deterministic)
    device = torch.device(config.device)

    # 3. Create model, data_loader, loss_fn, optimizer, lr_scheduler, training_args
    model = str2func(config.model.name)(**config.model.args if config.model.args else {}).to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    train_loader, test_loader = str2func(config.data.name)(
        is_distributed=is_distributed,
        **config.data.args if config.data.args else {})

    loss_fn = str2func(config.loss_fn.name)(**config.loss_fn.args if config.loss_fn.args else {})

    optimizer = str2func(config.optimizer.name)(
        params=model.parameters(), 
        **config.optimizer.args if config.optimizer.args else {})
    
    lr_scheduler = str2func(config.lr_scheduler.name)(
        optimizer=optimizer, 
        max_epochs=config.training_args.max_epochs, 
        **config.lr_scheduler.args if config.lr_scheduler.args else {})
    
    training_args = TrainingArgs(**config.training_args)

    # 4. Create Trainer
    trainer = Trainer(model, train_loader, loss_fn, optimizer, lr_scheduler, training_args)
    n_classes = config.model.args.output_dim
    trainer.register_hooks([test_hook(model, test_loader, n_classes)] if rank == 0 else [])
    trainer.train(auto_resume=True)

if __name__=='__main__':
    main()
    
"""PyTorch MNIST example.

The code is modified from: https://github.com/pytorch/examples/blob/main/mnist/main.py
It supports both single-gpu and multi-gpu training.
"""
import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from tup import Trainer, TrainingArgs, ConfigArgParser, EvalHook, HookBase, \
                set_random_seed, save_config, setup_logger, str2func, init_distributed
from inference_hook import InferenceHook

logger = logging.getLogger(__name__)

def test(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            img, target = batch['input'].to(device), batch['target'].to(device)
            output = model(img)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))

def main():
    # 1. Create an argument parser supporting loading YAML configuration file and command line parameter
    parser = ConfigArgParser(description="Training-Utils-Pytorch Example")
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
    
    train_loader, test_loader = str2func(config.data.name)(
        is_distributed=is_distributed,
        **config.data.args if config.data.args else {})

    loss_fn = str2func(config.loss_fn.name)(**config.loss_fn.args if config.loss_fn.args else {})

    optimizer = str2func(config.optimizer.name)(
        params=model.parameters(), 
        **config.optimizer.args if config.optimizer.args else {})
    
    lr_scheduler = str2func(config.lr_scheduler.name)(
        optimizer=optimizer, 
        **config.lr_scheduler.args if config.lr_scheduler.args else {})
    
    training_args = TrainingArgs(**config.training_args)

    # 4. Create Trainer
    trainer = Trainer(model, train_loader, loss_fn, optimizer, lr_scheduler, training_args)
    trainer.register_hooks([
        EvalHook(1, lambda: test(model, test_loader)),
        InferenceHook(test_loader.dataset)
    ] if rank == 0 else [])
    trainer.train(auto_resume=False)

if __name__=='__main__':
    main()
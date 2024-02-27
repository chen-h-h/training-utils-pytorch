import logging
import os
import os.path as osp
import time
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .distributed import gather, get_rank, get_world_size, is_main_process
from .history_buffer import HistoryBuffer
from .hooks import CheckpointHook, DistributedHook, HookBase, LoggerHook
from .logger import setup_logger
from .misc import collect_env, symlink

__all__ = ["TrainingArgs", "Trainer"]

logger = logging.getLogger(__name__)

@dataclass
class TrainingArgs:
    """
    The data class stores all the training args.

    Args:
        max_epochs (int): Total training epochs.
        output_dir (str): The output directory to save checkpoints and logs.
            Defaults to "output_dir".
        checkpoint_period (int): The period (epoch-based) to save checkpoint. Defaults to 10.
        log_period (int): The period (iter-based) to log. Defaults to 50.
        max_num_checkpoints (int): The maximum number of checkpoints to save.
            If None, save all checkpoints. Defaults to None.
        clip_grad_norm (float): Max norm of the gradients. If <= 0, will not clip gradients.
            Defaults to 0.
        enable_amp (bool): Enable the Automatic Mixed Precision (AMP) training. Defaults to False.

        iters_one_epoch (int): The number of iteration in one epoch,which equal `len(data_loader)`.
        inner_iter (int): The current inner iteration in one epoch when traiing, 
            which is `[0, iters_one_epoch - 1]`.
        cur_epoch (int): The current epoch when training, which is `[0, max_epochs - 1]`.
        start_epoch: The start epoch, which is `[0, max_epochs - 1]`. It's usually zero 
            unless you're in resume a training.
    """
    # invariant parameters in training
    max_epochs: int = 1000
    output_dir: str = "output_dir"
    checkpoint_period: int = 10
    log_period: int = 50
    max_num_checkpoints: int = None
    clip_grad_norm: float = 0.0
    enable_amp: bool = False

    # variable parameters in training
    iters_one_epoch: int = 0            # len(data_loader)
    inner_iter: int = 0                 # [0, iters_one_epoch - 1]
    cur_epoch: int = 0                  # [0, max_epochs - 1]
    start_epoch: int = 0                # [0, max_epochs - 1]

    @property
    def max_iters(self) -> int:
        """The total training iterations."""
        return self.max_epochs * self.iters_one_epoch

    @property
    def cur_iter(self) -> int:
        """The current iteration ranged in [0, max_iters - 1]."""
        return self.cur_epoch * self.iters_one_epoch + self.inner_iter

    @property
    def start_iter(self) -> int:
        """The iteration to start from. The minimum possible value is 0."""
        return self.start_epoch * self.iters_one_epoch

    @property
    def ckpt_dir(self) -> str:
        """The directory to save checkpoints. Overwrite this method to change the path."""
        return osp.join(self.output_dir, "checkpoints")

    @property
    def tb_log_dir(self) -> str:
        """The directory to save tensorboard files. Overwrite this method to change the path."""
        return osp.join(self.output_dir, "tb_logs")
    
class Trainer:
    """An epoch-based trainer.

    The class implements a simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source epoch-based optimization.
    It assumes that every step, you:

    1. Load a batch from the data_loader.
    2. Compute the loss with the batch.
    3. Compute the gradients with the above loss.
    4. Update the model with the optimizer.

    All other tasks during training (e.g., lr updating, checkpointing, logging, evaluation)
    are maintained by hooks, which can be registered by :meth:`register_hooks`.

    If you want to do anything fancier than this, subclass this class
    and implement your own :meth:`train_one_iter`.

    Args:
        model (torch.nn.Module)
        data_loader (torch.utils.data.DataLoader): Training data loader.
        loss_fn (torch.nn.Module)
        optimizer (torch.optim.Optimizer)
        lr_scheduler (optim.lr_scheduler._LRScheduler)
        args (TrainingArgs): Refer to :class:`TrainingArgs`

    Example::

        # create your model / data_loader / loss_fn / optimizer / lr_scheduler /args
        # before using the trainer
        model = ...
        data_loader = ...
        loss_fn = ...
        optimizer = ...
        lr_scheduler = ...
        # train 100 epochs
        args = TrainingArgs(max_epochs=100, ...)
        trainer = Trainer(model, data_loader, loss_fn, optimizer, lr_scheduler, args)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        args: TrainingArgs,
    ):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args

        self.metric_storage = MetricStorage()
        self.args.iters_one_epoch = len(data_loader)
        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._device = next(model.parameters()).device

        self._default_setup()

    @property
    def model_or_module(self) -> nn.Module:
        """The model not wrapped by :class:`DistributedDataParallel`."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    @property
    def hook_info(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ + f" (priority {h.priority})" for h in self._hooks]


    def _default_setup(self) -> None:
        # setup the root logger of the `tup` library to show
        # the log messages generated from this library
        setup_logger("tup", output_dir=self.args.output_dir, rank=get_rank())

        logger.info("Environment info:\n" + collect_env())

        default_hooks = [DistributedHook()]
        if is_main_process():
            default_hooks.extend([
                CheckpointHook(self.args.checkpoint_period, self.args.max_num_checkpoints),
                LoggerHook(self.args.log_period, tb_log_dir=self.args.tb_log_dir)
            ])
        self.register_hooks(default_hooks)
        logger.info(f"Registered default hooks: {self.hook_info}")

        self._grad_scaler = GradScaler(enabled=self.args.enable_amp)
        if self.args.enable_amp:
            logger.info("Automatic Mixed Precision (AMP) training is on.")

        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        split_line = "-" * 50
        logger.info(f"\n{split_line}\n"
                    f"Output directory: {self.args.output_dir}\n"
                    f"Checkpoint directory: {self.args.ckpt_dir}\n"
                    f"Tensorboard directory: {self.args.tb_log_dir}\n"
                    f"{split_line}")

    def register_hooks(self, hooks: List[HookBase]) -> None:
        """Register hooks to the trainer.

        Args:
            hooks (list[HookBase]): List of hooks to be registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: HookBase) -> None:
        """Register a hook to the trainer.

        For hooks with the same priority, they are executed in the order they are registered.

        Args:
            hook (HookBase): The hook to be registered.
        """
        assert isinstance(hook, HookBase)
        assert hook.priority >= 1 and hook.priority <= 10
        # To avoid circular reference, hooks and trainer cannot own each other. This normally
        # does not matter, but will cause memory leak if the involved objects contain __del__.
        # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        hook.trainer = weakref.proxy(self)
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if hook.priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def train_one_iter(self) -> None:
        """Train one iteration.

        Subclass of :class:`tup.trainer.Trainer` can implement your own :meth:`train_one_iter`
        to do something fancier.
        """
        iter_start_time = time.perf_counter()

        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            batch = next(self._data_iter)
        input, target = batch['input'].to(self._device), batch['target'].to(self._device)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        # If self._enable_amp=False, autocast and GradScaler’s calls become no-ops.
        # This allows switching between default precision and mixed precision
        # without if-else statements.
        with autocast(enabled=self.args.enable_amp):
            pred = self.model(input)
            loss_dict = self.loss_fn(pred, target)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        ##########################
        # 3. Calculate gradients #
        ##########################
        self.optimizer.zero_grad()
        self._grad_scaler.scale(losses).backward()
        if self.args.clip_grad_norm > 0:
            self._grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

        ##############################
        # 4. Update model parameters #
        ##############################
        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()

        self._log_iter_metrics(loss_dict, data_time, time.perf_counter() - iter_start_time)

    def train_one_epoch(self) -> None:
        # evaluation hook changes the model to `eval` mode after finishing epoch
        self.model.train()
        for self.args.inner_iter in range(self.args.iters_one_epoch):
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
        self.lr_scheduler.step()

    def train(self, resume_from_checkpoint: Optional[str] = None, auto_resume: bool = True) -> None:
        """Start training.

        If ``resume_from_checkpoint`` is specified, resume from the given checkpoint.
        Otherwise, auto resume from the latest checkpoint.

        Args:
            resume_from_checkpoint (str): Path to the checkpoint. Defaults to None.
            auto_resume (bool): Defaults to True.
        """
        if resume_from_checkpoint is not None:
            self.load_checkpoint(path=resume_from_checkpoint)
        elif resume_from_checkpoint is None and auto_resume:
            self.load_checkpoint(auto_resume=auto_resume)

        logger.info(f"Start training from epoch {self.args.start_epoch}")

        self._call_hooks("before_train")
        for self.args.cur_epoch in range(self.args.start_epoch, self.args.max_epochs):
            self._call_hooks("before_epoch")
            self.train_one_epoch()
            self._call_hooks("after_epoch")
        self._call_hooks("after_train")

    def save_checkpoint(self, file_name: str) -> None:
        """Save training state: ``epoch``, ``num_gpus``, ``model``, ``optimizer``, ``lr_scheduler``,
        ``metric_storage``, ``hooks`` (optional), ``grad_scaler`` (optional).

        Args:
            filename (str): The checkpoint will be saved as ``ckpt_dir/filename``.
        """
        data = {
            "epoch": self.args.cur_epoch,
            "num_gpus": get_world_size(),
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage,
        }
        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states
        if self.args.enable_amp:
            data["grad_scaler"] = self._grad_scaler.state_dict()

        file_path = osp.join(self.args.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag the latest checkpoint
        dst_file = osp.join(self.args.ckpt_dir, "latest.pth")
        symlink(file_name, dst_file)

    def load_checkpoint(self, path: Optional[str] = None, auto_resume: bool = False):
        """Load the given checkpoint or resume from the latest checkpoint.

        Args:
            path (str): Path to the checkpoint to load.
            auto_resume (bool): If True, automatically resume from the latest checkpoint.
        """
        if path is None and auto_resume:
            latest_ckpt = osp.join(self.args.ckpt_dir, "latest.pth")
            if not os.path.exists(latest_ckpt):
                logger.warning(f"Not found {latest_ckpt} to auto resume from.")
            else:
                logger.info(f"Found {latest_ckpt} to auto resume from.")
                path = latest_ckpt
        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")
        else:
            logger.info("Skip loading checkpoint.")
            return

        # check if the number of GPUs is consistent with the checkpoint
        num_gpus = get_world_size()
        ckpt_num_gpus = checkpoint["num_gpus"]
        assert num_gpus == ckpt_num_gpus, (
            f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
            f"but currently only have {num_gpus} GPUs.")

        # 1. load epoch
        self.args.start_epoch = checkpoint["epoch"] + 1

        # 2. load model
        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning("Encounter missing keys when loading model weights:\n"
                           f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning("Encounter unexpected keys when loading model weights:\n"
                           f"{incompatible.unexpected_keys}")

        # 3. load metric_storage
        self.metric_storage = checkpoint["metric_storage"]

        # 4. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 5. load lr_scheduler
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 6. load grad scaler
        consistent_amp = not (self.args.enable_amp ^ ("grad_scaler" in checkpoint))
        assert consistent_amp, "Found inconsistent AMP training setting when loading checkpoint."
        if self.args.enable_amp:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        # 7. load hooks
        hook_states = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}")

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break

    def log(self, *args, **kwargs) -> None:
        """Update the metrics stored in :obj:`self.trainer.metric_storage`."""
        self.metric_storage.update(*args, **kwargs)

    def _log_iter_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float,
                          iter_time: float) -> None:
        """
        Args:
            loss_dict (dict): Dict of scalar losses.
            data_time (float): Time taken by the dataloader iteration.
            iter_time (float): Time taken by one complete iteration.
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict.update(data_time=data_time, iter_time=iter_time)
        # gather metrics among all workers for logging
        all_metrics_dict = gather(metrics_dict)

        if is_main_process():
            self.log(self.args.cur_iter, lr=self.optimizer.param_groups[0]["lr"], smooth=False)

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            self.log(self.args.cur_iter, data_time=data_time)

            # same as data_time
            iter_time = np.max([x.pop("iter_time") for x in all_metrics_dict])
            self.log(self.args.cur_iter, iter_time=iter_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(losses_reduced):
                raise FloatingPointError(f"Loss became infinite or NaN at epoch={self.args.cur_epoch}! "
                                         f"loss_dict={metrics_dict}.")

            self.log(self.args.cur_iter, total_loss=losses_reduced)
            if len(metrics_dict) > 1:
                self.log(self.args.cur_iter, **metrics_dict)

class MetricStorage(dict):
    """The class stores the values of multiple metrics (some of them may be noisy, e.g., loss,
    batch time) in training process, and provides access to the smoothed values for better logging.

    The class is designed for automatic tensorboard logging. User should specify the ``smooth``
    when calling :meth:`update`, so that we can determine which metrics should be
    smoothed when performing tensorboard logging.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss will be smoothed, but lr will not
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}
        >>> # like dict, can be indexed by string
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: Dict[str, HistoryBuffer] = self
        self._smooth: Dict[str, bool] = {}
        self._latest_iter: Dict[str, int] = {}

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (int): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (bool): If True, return the smoothed values of these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same ``smooth`` in different calls to :meth:`update`.
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
                self._history[key] = HistoryBuffer(window_size=self._window_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key]
                self._latest_iter[key] = iter
            else:
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """Return the smoothed values or the latest values of multiple metrics.
        The specific behavior depends on the ``smooth`` when updating metrics.

        Returns:
            dict[str -> (int, float)]:
                Mapping from metric name to its (the latest iteration, the avg / the latest value)
                pair.
        """
        return {
            key: (self._latest_iter[key], his_buf.avg if self._smooth[key] else his_buf.latest)
            for key, his_buf in self._history.items()
        }

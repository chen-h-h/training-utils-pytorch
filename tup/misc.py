import logging
import os
import random
import sys
import importlib
import inspect
from collections import defaultdict
from typing import Optional, Callable

import numpy as np
import torch
from tabulate import tabulate

__all__ = ["collect_env", "set_random_seed", "symlink", "str2func"]

logger = logging.getLogger(__name__)


def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The value of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
        - TorchVision (optional): TorchVision version.
        - OpenCV (optional): OpenCV version.

    Returns:
        str: A string describing the running environment.
    """
    env_info = []
    env_info.append(("sys.platform", sys.platform))
    env_info.append(("Python", sys.version.replace("\n", "")))
    env_info.append(("Numpy", np.__version__))

    cuda_available = torch.cuda.is_available()
    env_info.append(("CUDA available", cuda_available))

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info.append(("GPU " + ",".join(device_ids), name))

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision
        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass

    try:
        import cv2
        env_info.append(("OpenCV", cv2.__version__))
    except ModuleNotFoundError:
        pass

    return tabulate(env_info)


def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        logger.info(f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}")
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down your training considerably.")


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    # Note: When use relitive path, src is the relative path with respect to dst, 
    # and dst is the relative path with respect to the current path
    os.symlink(src, dst, **kwargs)


# ALLOWED_MODULES = ['torch.optim', 'torch.optim', 'numpy', 'tup.lr_scheduler']  # List of trusted module paths
def str2func(func_str: str) -> Callable:
    """
    Convert a string representing a function (possibly including module path) to 
    the corresponding function object.
    
    Args:
        func_str (str): String representing the function (possibly including module path).
    Returns:
        function: The function object.
    """
    # Get the stack information of the caller
    caller_frame = inspect.stack()[1]
    # Get the global variables dictionary of the caller's scope
    caller_globals = caller_frame[0].f_globals 
    # Split the function string into module path and function name
    parts = func_str.split('.')
    
    if len(parts) == 1:
        # If no module path is provided, assume it's a built-in function or from the current module
        function_name = parts[0]

        # Iterate through the global variables to find the module
        for var_name, var_value in caller_globals.items():
            if hasattr(var_value, "__name__"):
                # If the variable is a module, try to get the function object
                try:
                    function = getattr(var_value, func_str)
                    return function
                except AttributeError:
                    pass
        
        raise ValueError("Function {} not found in the caller's global variables.".format(func_str))

    else:
        module_name = '.'.join(parts[:-1])
        function_name = parts[-1] 

        # if module_name not in ALLOWED_MODULES:
        #     raise ValueError("Module {} is not allowed.".format(module_name))
        
        # Dynamically import module
        module = importlib.import_module(module_name)
        # Get function object
        function = getattr(module, function_name)
        
        return function
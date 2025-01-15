import argparse
import logging
from omegaconf import OmegaConf, DictConfig
from typing import List

logger = logging.getLogger(__name__)

class ConfigArgParser():
    """Argument parser that supports loading a multilevel YAML configuration file.

    By working with omegaconf, we can load a multilevel configuration, and refer and modify 
    configuration value using dot operator (I like that).
    """

    def __init__(self, *args, **kwargs):
        """Same as :meth:`argparse.ArgumentParser`."""
        self.config_parser = argparse.ArgumentParser(*args, **kwargs)
        self.config_parser.add_argument("-c", "--config", default=None, metavar="FILE", required=True, 
                                        help="Where to load YAML configuration.")
        self.config_parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                                        help="""Modify config options at the end of the command.
                                            Such as "path.key1=value1 path.key2=value2".""".strip())

    def add_argument(self, *args, **kwargs):
        """Same as :meth:`ArgumentParser.add_argument`."""
        arg = self.config_parser.add_argument(*args, **kwargs)
        return arg

    def parse_args(self, *args, **kwargs) -> DictConfig:
        """The `args` is same as :meth:`ArgumentParser.parse_args`."""
        _args = self.config_parser.parse_args(*args, **kwargs)
        
        config_cmd = _args.opts
        config_file = OmegaConf.load(_args.config)
        
        config = self.merge_config(config_file, config_cmd)
        
        # merge other parameters that added by `add_argument()` 
        for key, value in vars(_args).items():
            if key not in ["opts", "config"]:
                OmegaConf.update(config, key, value, merge=True)
        
        return config
    
    def merge_config(self, config_file: DictConfig, config_cmd: List[str]) -> DictConfig:
        """
        Merge the config_cmd into config_file. Priority: config_cmd > config_file.
        Reference https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/lazy.py#L318

        Args:
            config_file: an omegaconf contents of cfg
            config_cmd: list of strings in the format of "a=b" to override configs.
        """
        if config_cmd==None:
            pass

        for o in config_cmd:
            key, value = o.split("=")
            try:
                value = eval(value, {})
            except NameError:
                pass
            except SyntaxError:
                pass
            self._safe_update(config_file, key, value)

        return config_file
    
    def _safe_update(self, cfg, key, value):
        # TODO: Do more safe checking on options from the command line
        v = OmegaConf.select(cfg, key, default=None)
        if v is None:
            logging.warning(
                f"\nTrying to update key **{key}**, but it is not "
                f"one of the config options.\n")
        OmegaConf.update(cfg, key, value, merge=True)

def save_config(configs: DictConfig, filepath: str, rank: int = 0) -> None:
    """If in master process, save ``config`` to a YAML file. Otherwise, do nothing.

    Args:
        configs (omegaconf.DictConfig): The configs to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert isinstance(configs, DictConfig)
    assert filepath.endswith(".yaml")
    if rank != 0:
        return
    OmegaConf.save(configs,filepath)
    logger.info(f"Configs is saved to {filepath}.")

import argparse
import logging
from omegaconf import OmegaConf, DictConfig

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
        self.option_names = []

    def add_argument(self, *args, **kwargs):
        """Same as :meth:`ArgumentParser.add_argument`."""
        arg = self.config_parser.add_argument(*args, **kwargs)
        self.option_names.append(arg.dest)
        return arg

    def parse_args(self) -> DictConfig:
        """The `args` is same as :meth:`ArgumentParser.parse_args`."""
        config_cmd = self.config_parser.parse_args()
        config_file = OmegaConf.load(config_cmd.config)
        
        config = self.merge_config(config_file, config_cmd)

        return config
    
    def merge_config(self, config_file: DictConfig, config_cmd: argparse.Namespace) -> DictConfig:
        """Merge the config_cmd into config_file. Priority: config_cmd > config_cmd"""
        # TODO: complete the function
        return config_file

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

"""project_tag"""

""" 
Other global variables
"""
from typing import List, Optional

import dataclasses
import os
from argparse import Namespace
from dataclasses import dataclass, field, make_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from dotenv import load_dotenv
from eztils import abspath, datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from rich import print
import torch

load_dotenv()


def setup_experiment():
    """
    Sets up the experiment by creating a run directory and a log directory, and creating a symlink from the repo directory to the run directory.
    """
    print("Setting up experiment...")

    """SETUP CONFIG"""
    parser = HfArgumentParser(Config)
    parser.add_argument("-c", "--config", type=str)

    conf: Config
    extras: Namespace
    conf, extras = parser.parse_args_into_dataclasses()

    if extras.config is not None:  # parse config file
        (original_conf,) = parser.parse_json_file(extras.config)
        for field_ in dataclasses.fields(original_conf):
            val = getattr(original_conf, field_.name)
            if isinstance(val, list):
                setattr(
                    field_, "default_factory", lambda x=val: x
                )
                setattr(original_conf, field_.name, field_)
        parser = HfArgumentParser(update_dataclass_defaults(Config, original_conf))
        parser.add_argument("-c", "--config", type=str)
        conf, extras = parser.parse_args_into_dataclasses()

    return conf


@dataclass
class Config:
    block_size: int = 1024
    recent_context: int = 20
    add_prompt: int = True
    n_layer: int = 3
    n_head: int = 1
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    seed: int = 42
    wandb: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""
    model_api: str = ""
    model_name: str = ""
    device: str = ""
    temperatures: List[float] = field(default_factory=lambda: [0.01, 1.0])


def main():
    conf = setup_experiment()
    # conf_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[bold green]Welcome to Transcendence LLM Domain")
    
    import os
    print("Current working directory:", os.getcwd())

    from transcendence_llm.llm_temp_sampling import run_llm
    
    run_llm(conf)


if __name__ == "__main__":
    main()

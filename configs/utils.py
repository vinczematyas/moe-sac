import torch
import yaml
from dataclasses import dataclass

@dataclass
class DummyClass:
    def update(self, new_args):
        for key, value in new_args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

def args_to_cfg(args, cfg):
    nested_keys = []
    for key, value in args.items():
        if type(value) == dict:
            setattr(cfg, key, DummyClass())
            args_to_cfg(value, getattr(cfg, key))
        else:
            setattr(cfg, key, value)

def init_cfg(path):
    cfg = DummyClass()
    with open(path, 'r') as f:
        args = yaml.safe_load(f.read())
        args_to_cfg(args, cfg)
    return cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from types import SimpleNamespace
import yaml


class NestedDictToNamespace:
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return NestedDictToNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, NestedDictToNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else: # this is the only addition
                setattr(self, key, val)

    def to_dict(self):
        result = {}
        for key, val in self.__dict__.items():
            if isinstance(val, NestedDictToNamespace):
                result[key] = val.to_dict()
            elif isinstance(val, list):
                result[key] = [v.to_dict() if isinstance(v, NestedDictToNamespace) else v for v in val]
            else:
                result[key] = val
        return result

    def to_yaml(self, file_path):
        with open(file_path, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)


def init_cfg(cfg_file):
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return NestedDictToNamespace(**cfg)

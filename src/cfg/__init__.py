import os.path as osp
from src.utils import yaml_load

THIS_DIR = osp.dirname(osp.realpath(__file__))
DEFAULT_CFG = yaml_load(osp.join(THIS_DIR, 'defaults.yaml'))


def override_cfg(cfg: dict, overrites: dict):
    for k, v in overrites.items():
        if k not in cfg:
            raise Exception(f"Cofiguration key {k} not recognized")
        if v is not None:
            cfg[k] = v
    return cfg

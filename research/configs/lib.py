# lib.py
import functools as ft
import numpy as np

from .config import Config

# ----------------------------
# Small, reasonable building blocks
# ----------------------------

DATA_DIM = 32
OUT_DIM = DATA_DIM  # match dataset's target size
DEPTH_LAYERS = 3  # number of Linear layers (here: in->hid1, hid1->hid2, hid2->out)

def mlp_2h(width: int):
    from model import MLP
    dims = [DATA_DIM, width, width, OUT_DIM]
    return Config(
        obj=MLP,
        params={"dims": dims, "bias": False},
    )

def mup_parametrization(n_layers: int = DEPTH_LAYERS):
    from parametrization import abc_parametrization
    al = [-0.5] + [0.0] * (n_layers - 2) + [0.5]
    bl = [0.5] + [0.5] * (n_layers - 2) + [0.5]
    cl = [0.0] + [0.0] * (n_layers - 2) + [0.0]
    return Config(obj=abc_parametrization, params={"al": al, "bl": bl, "cl": cl})

def sgd(lr: float):
    from torch.optim import SGD
    return Config(obj=SGD, params={"lr": lr})

def adamw(lr: float):
    from torch.optim import AdamW
    return Config(obj=AdamW, params={"lr": lr})

def const_lr_scheduler():
    from parametrization import constant_lr_scheduler
    return Config(obj=constant_lr_scheduler, params={})

def _free_params(dims):
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))

def toy_data(width: int):
    from data import SyntheticNormalDataset
    dims = [DATA_DIM, width, width, OUT_DIM]
    n_params = _free_params(dims)
    return Config(
        obj=SyntheticNormalDataset,
        params={
            "dataset_size": n_params,
            "batch_size": n_params,
            "width": DATA_DIM,   # <-- FIX: input feature dimension must match model's first dim
            "resample": False,
            "signal_strength": 0.0,
            "label_init": "random",
        },
    )

def training_small(n_steps: int = 1000, seed: int = 0, log_freq: int = 1):
    from train import train
    return Config(obj=train, params={"seed": seed, "n_train_steps": n_steps, "log_freq": log_freq})

# ----------------------------
# Single experiment grid: width × LR
# ----------------------------
def width_lr_grid(
    widths=(32, 64, 128, 256),
    lrs=(1e-1, 3e-2, 1e-2),
    optimizer="sgd",  # "sgd" or "adamw"
):
    """
    Yields (run_id, run_name, param_args) for main().
    param_args = (training_config, model_config, optimizer_config, lr_scheduler_config,
                  parametrization_config, data_config)
    """
    for wi, w in enumerate(widths):
        for li, lr in enumerate(lrs):
            run_id = wi * len(lrs) + li

            def model_cfg(w=w):
                return mlp_2h(w)

            def data_cfg(w=w):
                return toy_data(w)

            def opt_cfg(lr=lr, which=optimizer):
                return sgd(lr) if which == "sgd" else adamw(lr)

            def param_cfg(n_layers=DEPTH_LAYERS):
                return mup_parametrization(n_layers)

            def lr_sched_cfg():
                return const_lr_scheduler()

            def training_cfg():
                return training_small(n_steps=1000, seed=0, log_freq=1)

            run_name = f"w{w}_lr{lr:.3g}_{optimizer}"
            param_args = (training_cfg, model_cfg, opt_cfg, lr_sched_cfg, param_cfg, data_cfg)
            yield run_id, run_name, param_args

# configs/lib.py
from .config import Config

# ----------------------------
# Small, reasonable building blocks
# ----------------------------

# Dataset-dependent constants
SYNTH_IN_DIM = 32                 # SyntheticNormalDataset feature dim
CIFAR_IN_DIM = 32 * 32 * 3        # 3072 flattened CIFAR10
CIFAR_NUM_CLASSES = 10

DEPTH_LAYERS = 3  # in -> hid1 -> hid2 -> out


def mlp_2h_with_dims(in_dim: int, width: int, out_dim: int):
    from model import MLP
    dims = [in_dim, width, width, out_dim]
    return Config(obj=MLP, params={"dims": dims, "bias": False})


def mup_parametrization(n_layers: int = DEPTH_LAYERS):
    from parametrization import abc_parametrization
    # simple muP-like exponents
    al = [-0.5] + [0.0] * (n_layers - 2) + [0.5]
    bl = [0.5]  + [0.5] * (n_layers - 2) + [0.5]
    cl = [0.0]  + [0.0] * (n_layers - 2) + [0.0]
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


def training_small(n_steps: int = 1000, seed: int = 0, log_freq: int = 1):
    from train import train
    return Config(obj=train, params={"seed": seed, "n_train_steps": n_steps, "log_freq": log_freq})


def metrics_alignment_and_rL():
    # .build() returns the spec list consumed by train.py
    return Config(obj=lambda spec: spec, params={"spec": ["alignment", "rL"]})


# ----------------------------
# Data configs
# ----------------------------

def _free_params(dims):
    # weights only (no biases)
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))


def synthetic_data_for_dims(dims, *, resample=False, signal_strength=0.0, label_init="random"):
    """
    Synthetic regression where targets match input dim when label_init='random'.
    """
    from data import SyntheticNormalDataset
    in_dim = dims[0]
    n_params = _free_params(dims)
    return Config(
        obj=SyntheticNormalDataset,
        params={
            "dataset_size": n_params,
            "batch_size": n_params,   # adjust down if too big
            "width": in_dim,          # input feature dimension
            "resample": resample,
            "signal_strength": signal_strength,
            "label_init": label_init, # 'random' => y ∈ R^{in_dim} (matches out_dim=in_dim)
        },
    )


def cifar10_data(batch_size=256, **noise_kwargs):
    """
    CIFAR10 classification (in_dim=3072, out_dim=10).
    noise_kwargs: signal_fn, signal_strength, signal_range, signal_period, total_steps
    """
    from data import CIFAR10Dataset
    return Config(obj=CIFAR10Dataset, params={"batch_size": batch_size, **noise_kwargs})


# ----------------------------
# Single experiment grid: width × LR, with a single selected dataset
# ----------------------------
def width_lr_grid(
    widths=(32, 64, 128),
    lrs=(1e0, 6e-1, 3e-1),
    optimizer="sgd",                 # "sgd" or "adamw"
    dataset="synth",                 # "synth" or "cifar"
):
    """
    Yields (run_id, run_name, param_args) for main().

    param_args = (
        training_config, model_config, optimizer_config, lr_scheduler_config,
        parametrization_config, data_config, metrics_config
    )

    dataset controls input/output dims:
      - "synth": in_dim=SYNTH_IN_DIM, out_dim=in_dim (regression)
      - "cifar": in_dim=3072, out_dim=10 (classification)
    """
    # Resolve dataset mapping once
    if dataset == "synth":
        ds_tag = "synth"
        base_in_dim = SYNTH_IN_DIM
        base_out_dim = base_in_dim

        def data_cfg_factory(dims):
            return synthetic_data_for_dims(dims, resample=False, signal_strength=0.0, label_init="random")

    elif dataset == "cifar":
        ds_tag = "cifar"
        base_in_dim = CIFAR_IN_DIM
        base_out_dim = CIFAR_NUM_CLASSES

        def data_cfg_factory(_dims):
            return cifar10_data(batch_size=256, signal_fn="const", signal_strength=1.0, signal_period=1000, total_steps=1000)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    run_id = 0
    for w in widths:
        for lr in lrs:
            dims = [base_in_dim, w, w, base_out_dim]

            # Model & Data (bound to dims)
            def model_cfg(dims=dims):
                return mlp_2h_with_dims(dims[0], dims[1], dims[3])

            def data_cfg(dims=dims):
                return data_cfg_factory(dims)

            # Optimizer bound to LR
            def opt_cfg(lr=lr, which=optimizer):
                return sgd(lr) if which == "sgd" else adamw(lr)

            # Parametrization sized to depth
            def param_cfg(n_layers=DEPTH_LAYERS):
                return mup_parametrization(n_layers)

            # LR scheduler
            def lr_sched_cfg():
                return const_lr_scheduler()

            # Training & Metrics
            def training_cfg():
                return training_small(n_steps=1000, seed=0, log_freq=1)

            def metrics_cfg():
                return metrics_alignment_and_rL()

            run_name = f"{ds_tag}_w{w}_lr{lr:.3g}_{optimizer}"
            param_args = (training_cfg, model_cfg, opt_cfg, lr_sched_cfg, param_cfg, data_cfg, metrics_cfg)

            yield run_id, run_name, param_args
            run_id += 1


# Convenience aliases (useful with your launcher)
def width_lr_grid_synth(**kwargs):
    return width_lr_grid(dataset="synth", **kwargs)

def width_lr_grid_cifar(**kwargs):
    return width_lr_grid(dataset="cifar", **kwargs)

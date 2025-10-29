# configs/lib.py
from .config import Config

# ----------------------------
# Small, reasonable building blocks
# ----------------------------

# Dataset-dependent constants
SYNTH_IN_DIM = 32                 # SyntheticNormalDataset feature dim
CIFAR_IN_DIM = 32 * 32 * 3        # 3072 flattened CIFAR10
CIFAR_NUM_CLASSES = 10
N_STEPS = 10000

DEPTH_LAYERS = 3  # in -> hid1 -> hid2 -> out


def mlp_with_dims(dims: list[int]):
    from model import MLP
    return Config(obj=MLP, params={"dims": dims, "bias": False})


def cifar_vit(dim, n_layers):
    from model import ViT
    return Config(
        obj=ViT,
        params={
            "image_size": 32,
            "patch_size": 4,
            "dim": dim,
            "hidden_dim": 4 * dim,
            "n_layers": n_layers,
            "n_classes": 10
        }
    )


def mup_parametrization(opt, alignment, n_layers):
    from parametrization import abc_parametrization

    al = [-0.5] +  [0.0] * (n_layers - 2) +  [0.5]
    bl =  [0.5] +  [0.5] * (n_layers - 2) +  [0.5]

    if alignment == 'full':
        if opt == 'sgd':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]
        elif opt == 'adam':
            cl =  [0.5] +  [1.0] * (n_layers - 2) +  [0.5]
        elif opt == 'ada':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.0]
    elif alignment == 'no':
        if opt == 'sgd':
            cl =  [0.0] + [-0.5] * (n_layers - 2) +  [0.0]
        elif opt == 'adam':
            cl =  [0.5] +  [0.5] * (n_layers - 2) +  [0.0]
        elif opt == 'ada':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]

    return Config(
        obj=abc_parametrization,
        params={
            "al": al,
            "bl": bl,
            "cl": cl,
        }
    )

def ntk_parametrization(opt, alignment, n_layers):
    from parametrization import abc_parametrization

    al =  [0.0] +  [0.5] * (n_layers - 2) +  [0.5]
    bl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]

    if alignment == 'full':
        if opt == 'sgd':
            cl = [-0.5] + [-0.5] * (n_layers - 2) +  [0.0]
        elif opt == 'adam':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.5]
        elif opt == 'ada':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.5]
    elif alignment == 'no':
        if opt == 'sgd':
            cl = [-0.5] + [-1.0] * (n_layers - 2) + [-0.5]
        elif opt == 'adam':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]
        elif opt == 'ada':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]

    return Config(
        obj=abc_parametrization,
        params={
            "al": al,
            "bl": bl,
            "cl": cl,
        }
    )

def mfp_parametrization(opt, alignment, n_layers):
    from parametrization import abc_parametrization

    al =  [0.0] +  [0.5] * (n_layers - 2) +  [1.0]
    bl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]

    if alignment == 'full':
        if opt == 'sgd':
            cl = [-1.0] + [-1.0] * (n_layers - 2) + [-1.0]
        elif opt == 'adam':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.0]
        elif opt == 'ada':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.0]
    elif alignment == 'no':
        if opt == 'sgd':
            cl = [-1.0] + [-1.5] * (n_layers - 2) + [-1.0]
        elif opt == 'adam':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.5]
        elif opt == 'ada':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]

    return Config(
        obj=abc_parametrization,
        params={
            "al": al,
            "bl": bl,
            "cl": cl,
        }
    )

def standard_parametrization(opt, alignment, n_layers):
    from parametrization import abc_parametrization

    al =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]
    bl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.5]

    if alignment == 'full':
        if opt == 'sgd':
            cl = [-0.5] +  [0.5] * (n_layers - 2) +  [1.0]
        elif opt == 'adam':
            cl =  [0.0] +  [1.0] * (n_layers - 2) +  [1.0]
        elif opt == 'ada':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.5]
    elif alignment == 'no':
        if opt == 'sgd':
            cl = [-0.5] +  [0.0] * (n_layers - 2) +  [0.5]
        elif opt == 'adam':
            cl =  [0.0] +  [0.5] * (n_layers - 2) +  [0.5]
        elif opt == 'ada':
            cl =  [0.0] +  [0.0] * (n_layers - 2) +  [0.0]

    return Config(
        obj=abc_parametrization,
        params={
            "al": al,
            "bl": bl,
            "cl": cl,
        }
    )


def sgd(lr: float):
    from torch.optim import SGD
    return Config(obj=SGD, params={"lr": lr})


def adamw(lr: float):
    from torch.optim import AdamW
    return Config(obj=AdamW, params={"lr": lr})


def const_lr_scheduler(*args, **kwargs):
    from parametrization import constant_lr_scheduler
    return Config(obj=constant_lr_scheduler, params={})


def max_lr_scheduler(n, al, bl, lr_prefactor, feature_learning=False):
    from parametrization import maximal_lr_scheduler
    return Config(
        obj=maximal_lr_scheduler,
        params={"n": n, "al": al, "bl": bl, "lr_prefactor": lr_prefactor, "feature_learning": feature_learning}
    )


def training_small(n_steps: int = N_STEPS, seed: int = 0, log_freq: int = 1):
    from train import train
    return Config(obj=train, params={"seed": seed, "n_train_steps": n_steps, "log_freq": log_freq})


def metrics_alignment_and_rL(resample_w0=False):
    return Config(obj=lambda spec, resample_w0: (spec, resample_w0), params={"spec": ["alignment", "rL"], "resample_w0": resample_w0})

def metrics_none():
    return Config(obj=lambda spec: spec, params={"spec": []})


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

def cifar10_nclass_data(n_classes: int, batch_size=256, **noise_kwargs):
    """
    CIFAR10 classification restricted to the first n_classes in sorted order.
    noise_kwargs: signal_fn, signal_strength, signal_range, signal_period, total_steps
    """
    from data import CIFAR10FewClassDataset
    return Config(obj=CIFAR10FewClassDataset, params={"batch_size": batch_size, "n_classes": n_classes, **noise_kwargs})

# ----------------------------
# Single experiment grid: width × LR, with a single selected dataset
# ----------------------------
def depth_width_lr_grid(
    depths=(3, 4),
    widths=(32, 64, 128),
    lrs=(6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 8e-2, 6e-2),
    optimizer="sgd",                 # "sgd" or "adamw"
    dataset="synth",                 # "synth" or "cifar"
    lr_scheduler=const_lr_scheduler  # "const_lr_scheduler" or "max_lr_scheduler"
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
            return cifar10_data(batch_size=256, signal_fn="const", signal_strength=1.0, signal_period=N_STEPS, total_steps=N_STEPS)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    run_id = 0
    for d in depths:
        for w in widths:
            for lr in lrs:
                # Model & Data (bound to dims)
                def model_cfg(dim=w, n_layers=d):
                    return cifar_vit(dim, n_layers)

                def data_cfg(dims=None):
                    if dims is None:
                        dims = [base_in_dim] + (d - 1) * [w] + [base_out_dim]
                    return data_cfg_factory(dims)

                # Optimizer bound to LR
                def opt_cfg(lr=lr, which=optimizer):
                    return sgd(lr) if which == "sgd" else adamw(lr)

                # Parametrization sized to depth
                def param_cfg(n_layers=3):
                    # return mup_parametrization("sgd", alignment="full", n_layers=n_layers)
                    return mup_parametrization(optimizer, alignment="full", n_layers=n_layers)

                # LR scheduler
                def lr_sched_cfg():
                    param = param_cfg()
                    al = param['al']
                    bl = param['bl']
                    return lr_scheduler(n=w, al=al, bl=bl, lr_prefactor=lr)

                # Training & Metrics
                def training_cfg():
                    return training_small(n_steps=N_STEPS, seed=0, log_freq=1)

                def metrics_cfg():
                    # return metrics_none()
                    return metrics_alignment_and_rL(resample_w0=True)

                run_name = f"{ds_tag}_{lr_scheduler.__name__}_d{d}_w{w}_lr{lr:.3g}_{optimizer}"
                param_args = (training_cfg, model_cfg, opt_cfg, lr_sched_cfg, param_cfg, data_cfg, metrics_cfg)

                yield run_id, run_name, param_args
                run_id += 1


# Convenience aliases (useful with your launcher)
def depth_width_lr_grid_synth(**kwargs):
    return depth_width_lr_grid(dataset="synth", **kwargs)

def depth_width_lr_grid_cifar(**kwargs):
    return depth_width_lr_grid(dataset="cifar", **kwargs)

def cifar_single(**kwargs):
    return depth_width_lr_grid(widths=(512,), lrs=(2e-1,), optimizer="sgd", dataset="cifar")

def cifar_baseline_grid(**kwargs):
    return depth_width_lr_grid_cifar(
        depths=(3, 4, 5),
        widths=(128, 256, 512),
        lrs=(6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 8e-2, 6e-2),
        optimizer="adam",
        lr_scheduler=const_lr_scheduler
    )

def cifar_vit_baseline_grid(**kwargs):
    return depth_width_lr_grid_cifar(
        depths=(4, 6, 8),
        widths=(128, 256, 512),
        lrs=(1e-1, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 8e-3, 6e-3),
        optimizer="adam",
        lr_scheduler=const_lr_scheduler
    )

def cifar_test_vit(**kwargs):
    return depth_width_lr_grid_cifar(
        depths=(8,),
        widths=(512,),
        lrs=(5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 8e-3, 6e-3, 4e-3, 1e-3, 6e-4, 4e-4, 1e-4),
        optimizer="adam",
        lr_scheduler=const_lr_scheduler
    )

def cifar_maxlr_grid(**kwargs):
    return depth_width_lr_grid_cifar(
        depths=(3, 4, 5),
        widths=(128, 256, 512),
        lrs=(6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 8e-2, 6e-2),
        optimizer="adam",
        lr_scheduler=max_lr_scheduler
    )

def cifar_maxlr_resamplew0_keepz0_grid(**kwargs):
    return depth_width_lr_grid_cifar(
        depths=(3, 4, 5),
        widths=(128, 256, 512),
        lrs=(6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1, 8e-2, 6e-2),
        optimizer="adam",
        lr_scheduler=max_lr_scheduler
    )

def cifar_nclass_sweep(n_classes_list=(2, 5, 8, 10), width=512, lr=2e-1, optimizer="sgd"):
    """
    Sweep over number of CIFAR classes with a fixed width/lr.
    Yields (run_id, run_name, param_args) for main().
    """
    base_in_dim = CIFAR_IN_DIM

    run_id = 0
    for n_cls in n_classes_list:
        out_dim = n_cls
        dims = [base_in_dim, width, width, out_dim]

        def model_cfg(dims=dims):
            return mlp_with_dims(dims)

        def data_cfg(n_cls=n_cls):
            return cifar10_nclass_data(
                n_classes=n_cls,
                batch_size=256,
                signal_fn="const",
                signal_strength=1.0,
                signal_period=N_STEPS,
                total_steps=N_STEPS,
            )

        def opt_cfg(lr=lr, which=optimizer):
            return sgd(lr) if which == "sgd" else adamw(lr)

        def param_cfg(n_layers=DEPTH_LAYERS):
            return mup_parametrization("sgd", alignment="full", n_layers=n_layers)

        def lr_sched_cfg():
            return const_lr_scheduler()

        def training_cfg():
            # Keep logging every 10 steps by default
            return training_small(n_steps=N_STEPS, seed=0, log_freq=1)

        def metrics_cfg():
            return metrics_alignment_and_rL()

        run_name = f"cifar_ncls{n_cls}_w{width}_lr{lr:.3g}_{optimizer}"
        param_args = (training_cfg, model_cfg, opt_cfg, lr_sched_cfg, param_cfg, data_cfg, metrics_cfg)
        yield run_id, run_name, param_args
        run_id += 1
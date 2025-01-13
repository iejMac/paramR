import functools as ft

import numpy as np

from .config import Config


# Jascha setup (https://arxiv.org/abs/2402.06184):
N_FRAC_1 = 16
def mlp1h():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [N_FRAC_1, N_FRAC_1, 1],
            "bias": False,
        },
    )
N_FREE_PARAMS_1 = N_FRAC_1 * N_FRAC_1 + N_FRAC_1 * 1

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

def jascha_data():
    from data import SyntheticNormalDataset
    return Config(
        obj=SyntheticNormalDataset,
        params={
            "dataset_size": N_FREE_PARAMS_1,
            "batch_size": N_FREE_PARAMS_1,
            "width": N_FRAC_1,
            "resample": False,
            "signal_strength": 0.0,
        }
    )

def jascha_grid():
    resolution = 4
    c1_grid = np.linspace(-1.5, -0.5, num=resolution).tolist()
    c2_grid = np.linspace(-1.5, -0.5, num=resolution).tolist()

    for c1_id, c1 in enumerate(c1_grid):
        for c2_id, c2 in enumerate(c2_grid):
            exp_id = c1_id * len(c2_grid) + c2_id
            def mean_field_parametrization_with_diff_cl():
                mf_cfg = mfp_parametrization('sgd', 'full', 2)
                mf_cfg['cl'] = [c1, c2]
                return mf_cfg

            param_args = (training_frac, mlp1h, sgd_frac, mean_field_parametrization_with_diff_cl, jascha_data)

            run_name = f"mfp_c1_{c1:.14f}_c2_{c2:.14f}"
            yield exp_id, run_name, param_args



# New setup:
DATA_DIM=8
N_FRAC_2 = 256
def mlp2h():
    from model import MLP
    return Config(
        obj=MLP,
        params={
            "dims": [DATA_DIM, N_FRAC_2, N_FRAC_2, 1],
            "bias": False,
        },
    )
N_FREE_PARAMS_2 = (DATA_DIM * N_FRAC_2) + (N_FRAC_2 * N_FRAC_2) + (N_FRAC_2 * 1)

def ab_data():
    from data import SyntheticNormalDataset
    return Config(
        obj=SyntheticNormalDataset,
        params={
            "dataset_size": N_FREE_PARAMS_2,
            "batch_size": N_FREE_PARAMS_2,
            "width": DATA_DIM,
            "resample": False,
            "signal_fn": 'const',
            "signal_strength": 1.0,
            "signal_period": 1000
        }
    )

def ab_grid(param_cfg, model_cfg, opt_cfg, l, resolution=5, ab_range=0.2):
    cfg = param_cfg()
    a_grid = np.linspace(cfg['al'][l - 1] - ab_range, cfg['al'][l - 1] + ab_range, num=resolution).tolist()
    b_grid = np.linspace(cfg['bl'][l - 1] - ab_range, cfg['bl'][l - 1] + ab_range, num=resolution).tolist()

    for a_id, a in enumerate(a_grid):
        for b_id, b in enumerate(b_grid):
            exp_id = a_id * len(b_grid) + b_id

            def diff_ab():
                cfg = param_cfg()
                cfg['al'][l - 1] = a
                cfg['bl'][l - 1] = b
                return cfg

            param_args = (training_frac, model_cfg, opt_cfg, diff_ab, ab_data)

            run_name = f"grid_a{l}_{a:.2f}_b{l}_{b:.2f}"
            yield exp_id, run_name, param_args

def ab_eps_grid(param_cfg, model_cfg, opt_cfg, l, resolution=5, t_resolution=11, ab_range=0.2):
    cfg = param_cfg()
    a_grid = np.linspace(cfg['al'][l - 1] - ab_range, cfg['al'][l - 1] + ab_range, num=resolution).tolist()
    b_grid = np.linspace(cfg['bl'][l - 1] - ab_range, cfg['bl'][l - 1] + ab_range, num=resolution).tolist()
    eps_grid = np.linspace(0.0, 1.0, num=t_resolution).tolist()

    for a_id, a in enumerate(a_grid):
        for b_id, b in enumerate(b_grid):
            for eps_id, eps in enumerate(eps_grid):
                exp_id = (a_id * len(b_grid) + b_id) * len(eps_grid) + eps_id

                def diff_ab(a=a, b=b):
                    cfg = param_cfg()
                    cfg['al'][l - 1] = a
                    cfg['bl'][l - 1] = b
                    return cfg

                def ab_data_w_signal(eps=eps):
                    data_cfg = ab_data()
                    data_cfg["signal_strength"] = eps
                    return data_cfg

                param_args = (training_frac, model_cfg, opt_cfg, diff_ab, ab_data_w_signal)
                run_name = f"grid_a{l}_{a:.2f}_b{l}_{b:.2f}_eps_{eps:.2f}"
                yield exp_id, run_name, param_args

def ab_lr_grid(param_cfg, model_cfg, opt_cfg, l, resolution=5, ab_range=0.2):
    cfg = param_cfg()
    a_grid = np.linspace(cfg['al'][l - 1] - ab_range, cfg['al'][l - 1] + ab_range, num=resolution).tolist()
    b_grid = np.linspace(cfg['bl'][l - 1] - ab_range, cfg['bl'][l - 1] + ab_range, num=resolution).tolist()
    lr_grid = [5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2]

    for a_id, a in enumerate(a_grid):
        for b_id, b in enumerate(b_grid):
            for lr_id, lr in enumerate(lr_grid):
                exp_id = (a_id * len(b_grid) + b_id) * len(lr_grid) + lr_id

                def diff_ab(a=a, b=b):
                    cfg = param_cfg()
                    cfg['al'][l - 1] = a
                    cfg['bl'][l - 1] = b
                    return cfg

                def opt_w_lr(lr=lr):
                    opt_config = opt_cfg()
                    opt_config['lr'] = lr
                    return opt_config

                param_args = (training_frac, model_cfg, opt_w_lr, diff_ab, ab_data)
                run_name = f"grid_a{l}_{a:.2f}_b{l}_{b:.2f}_lr_{lr:.6f}"
                yield exp_id, run_name, param_args



# Grid definitions:
def mup_a3b3_grid():
    return ab_grid(
        param_cfg=ft.partial(mup_parametrization, "adam", "full", 3),
        model_cfg=mlp2h,
        opt_cfg=adamw_frac,
        l=3,
        resolution=5,
        ab_range=0.2
    )

def mup_a3b3_eps_grid():
    return ab_eps_grid(
        param_cfg=ft.partial(mup_parametrization, "adam", "full", 3),
        model_cfg=mlp2h,
        opt_cfg=adamw_frac,
        l=3,
        resolution=5,
        t_resolution=11,
        ab_range=0.2
    )

def mup_a3b3_loss_grid():
    return ab_lr_grid(
        param_cfg=ft.partial(mup_parametrization, "adam", "full", 3),
        model_cfg=mlp2h,
        opt_cfg=adamw_frac,
        l=3,
        resolution=5,
        ab_range=0.2
    )



# Common:
def sgd_frac():
    from torch.optim import SGD
    return Config(
        obj=SGD,
        params={
            "lr": 1e-1,
        },
    )

def adamw_frac():
    from torch.optim import AdamW
    return Config(
        obj=AdamW,
        params={
            "lr": 5e-3,
        },
    )

def ada_frac():
    from torch.optim import Adafactor
    return Config(
        obj=Adafactor,
        params={
            "lr": 5e-3,
        },
    )

def training_frac():
    from fractal import train
    N_STEPS = 1000
    return Config(
        obj=train,
        params={
            "seed": 0,
            "n_train_steps": N_STEPS,
            "log_freq": 1,
        },
    )

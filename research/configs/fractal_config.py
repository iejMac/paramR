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

def parametrization(type, opt='adam', alignment='full', n_layers=3):
    from parametrization import abc_parametrization

    if type == 'mup':
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
    elif type == 'ntk':
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
    elif type == 'mfp':
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
    elif type == 'standard':
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
                mf_cfg = parametrization('mfp', 'sgd', n_layers=2)
                mf_cfg['cl'] = [c1, c2]
                return mf_cfg

            param_args = (training_frac, mlp1h, sgd_frac, mean_field_parametrization_with_diff_cl, jascha_data)

            run_name = f"mfp_c1_{c1:.14f}_c2_{c2:.14f}"
            yield exp_id, run_name, param_args



# New setup:
DATA_DIM=2
N_FRAC_2 = 64
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
            # "signal_strength": 0.5,
            "signal_strength": 0.2,
        }
    )

def ab_grid(*, param='mup', opt='adam', alignment='full', n_layers=3, l=3, resolution=5, ab_range=0.2):
    cfg = parametrization(param, opt, alignment, n_layers)
    a_grid = np.linspace(cfg['al'][l - 1] - ab_range, cfg['al'][l - 1] + ab_range, num=resolution).tolist()
    b_grid = np.linspace(cfg['bl'][l - 1] - ab_range, cfg['bl'][l - 1] + ab_range, num=resolution).tolist()

    for a_id, a in enumerate(a_grid):
        for b_id, b in enumerate(b_grid):
            exp_id = a_id * len(b_grid) + b_id

            def diff_ab():
                cfg = parametrization(param, opt, alignment, n_layers)
                cfg['al'][l - 1] = a
                cfg['bl'][l - 1] = b
                return cfg

            param_args = (training_frac, mlp2h, opt_frac(opt), diff_ab, ab_data)

            run_name = f"{param}_{opt}_{alignment}_a{l}_{a:.2f}_b{l}_{b:.2f}"
            yield exp_id, run_name, param_args

def ab_eps_grid(*, param='mup', opt='adam', alignment='full', n_layers=3, l=3, resolution=5, t_resolution=11, ab_range=0.2):
    cfg = parametrization(param, opt, alignment, n_layers)
    a_grid = np.linspace(cfg['al'][l - 1] - ab_range, cfg['al'][l - 1] + ab_range, num=resolution).tolist()
    b_grid = np.linspace(cfg['bl'][l - 1] - ab_range, cfg['bl'][l - 1] + ab_range, num=resolution).tolist()
    eps_grid = np.linspace(0.0, 1.0, num=t_resolution).tolist()

    for a_id, a in enumerate(a_grid):
        for b_id, b in enumerate(b_grid):
            for eps_id, eps in enumerate(eps_grid):
                exp_id = (a_id * len(b_grid) + b_id) * len(eps_grid) + eps_id

                def diff_ab(a=a, b=b):
                    cfg = parametrization(param, opt, alignment, n_layers)
                    cfg['al'][l - 1] = a
                    cfg['bl'][l - 1] = b
                    return cfg

                def ab_data_w_signal(eps=eps):
                    data_cfg = ab_data()
                    data_cfg["signal_strength"] = eps
                    return data_cfg

                param_args = (training_frac, mlp2h, opt_frac(opt), diff_ab, ab_data_w_signal)
                run_name = f"{param}_{opt}_{alignment}_a{l}_{a:.2f}_b{l}_{b:.2f}_eps_{eps:.2f}"
                yield exp_id, run_name, param_args

def ab_lr_grid(*, param='mup', opt='adam', alignment='full', n_layers=3, l=3, resolution=5, ab_range=0.2):
    cfg = parametrization(param, opt, alignment, n_layers)
    a_grid = np.linspace(cfg['al'][l - 1] - ab_range, cfg['al'][l - 1] + ab_range, num=resolution).tolist()
    b_grid = np.linspace(cfg['bl'][l - 1] - ab_range, cfg['bl'][l - 1] + ab_range, num=resolution).tolist()
    lr_grid = [5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2]

    for a_id, a in enumerate(a_grid):
        for b_id, b in enumerate(b_grid):
            for lr_id, lr in enumerate(lr_grid):
                exp_id = (a_id * len(b_grid) + b_id) * len(lr_grid) + lr_id

                def diff_ab(a=a, b=b):
                    cfg = parametrization(param, opt, alignment, n_layers)
                    cfg['al'][l - 1] = a
                    cfg['bl'][l - 1] = b
                    return cfg

                def opt_w_lr(lr=lr):
                    opt_config = opt_frac(opt)
                    opt_config['lr'] = lr
                    return opt_config

                param_args = (training_frac, mlp2h, opt_w_lr, diff_ab, ab_data)
                run_name = f"{param}_{opt}_{alignment}_a{l}_{a:.2f}_b{l}_{b:.2f}_lr_{lr:.6f}"
                yield exp_id, run_name, param_args

def mup_adam_a3b3_grid():
    return Config(
        obj=ab_grid,
        params={
            "param": 'mup',
            "opt": 'adam',
            "alignment": 'full',
            "n_layers": 3,
            "l": 3,
            "resolution": 5,
            "ab_range": 0.2,
        },
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

def opt_frac(type):
    if type == 'sgd':
        return sgd_frac()
    elif type == 'adam':
        return adamw_frac()
    elif type == 'ada':
        return ada_frac()

def training_frac():
    from fractal import train
    N_STEPS = 200
    return Config(
        obj=train,
        params={
            "seed": 0,
            "n_train_steps": N_STEPS,
            "log_freq": 1,
        },
    )

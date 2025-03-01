import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from parametrization import maximal_lr_scheduler, constant_lr_scheduler

torch.set_default_dtype(torch.float64)

class BinaryLogger:
    def __init__(self, log_dir, n_steps, metrics):
        self.log_dir = log_dir
        self.n_steps = n_steps

        self.metrics = {}
        for m_name, m_shape in metrics.items():
            m_full_shape = (n_steps,) + m_shape
            self.metrics[m_name] = np.zeros(m_full_shape, dtype=np.float64)
   
    def log(self, metric):
        step = metric.pop("step")
        for m_name, m in metric.items():
            self.metrics[m_name][step] = m

    def save(self):
        for m_name, m_dat in self.metrics.items():
            np.save(os.path.join(self.log_dir, f"{m_name}.npy"), m_dat)

def train(
        model_config, optimizer_config, parametrization_config, lr_scheduler_config, data_config,
        n_train_steps,
        log_freq,
        seed=0,
        run_dir="./runs", data_dir="./data",
    ):
    torch.manual_seed(seed)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    model = model_config().build()
    model = model.to(device)

    opt_cfg = optimizer_config()

    width = model_config()["dims"][1]  # Feature dimension (width of hidden layer)
    params = parametrization_config().build(mlp=model, n=width, lr_prefactor=opt_cfg['lr'], std_prefactor=1.0)

    opt = opt_cfg.build(params=params)

    param_cfg = parametrization_config()
    lr_scheduler = lr_scheduler_config().build(optimizer=opt)
    alignment_warmup = 100  # how many steps before doing alignment-based LR schedules

    train_loader = data_config().build(device=device)

    metrics = {
        "losses": (1,),
        "rLs": (1,),
    }

    fractal_mode = False
    if not fractal_mode:
        metrics.update({
            "Als": (model.n_layers, 4),  # A_cum, A_alpha, A_omega, A_u
            "lrs": (model.n_layers,)
        })

    logger = BinaryLogger(run_dir, n_steps=n_train_steps, metrics=metrics)

    if not fractal_mode:
        log_sample_size = 32
        trace = {}
        def trace_layer(name):
            def hook(model, input, output):
                trace[name] = {
                    "input": input[0][:log_sample_size].detach(),
                    "output": output[:log_sample_size].detach(),
                    # "weight": model.weight.data.detach(),
                    "weight": model.lin.weight.data.detach(),
                }
            return hook
        for l_id, l in enumerate(model.layers):
            l.register_forward_hook(trace_layer(f'lin_{l_id}'))

        # Compute initial features
        measurement_X, _ = next(iter(train_loader))
        with torch.no_grad():
            model(measurement_X)
        zL_init = trace[f'lin_{model.n_layers - 1}']["input"].clone()
        trace_init = copy.deepcopy(trace)
    elif 'rLs' in metrics:
        # faster for rLs
        trace = {}

        def get_activation(name):
            def hook(model, input, output):
                trace[name] = {'input': input[0].detach()}  # input[0] is the input to the last layer
            return hook

        # Register hook on the last layer to capture its input (features)
        model.layers[-1].register_forward_hook(get_activation(f'lin_{model.n_layers - 1}'))

        # Set up a fixed measurement batch for logging
        measurement_X, _ = next(iter(train_loader))
        # Compute initial features
        with torch.no_grad():
            model(measurement_X)
            zL_init = trace[f'lin_{model.n_layers - 1}']['input']

    s = 0
    diverged = False
    for X, y in train_loader:
        if s >= n_train_steps or diverged:
            break
        opt.zero_grad()
        y_hat = model(X)

        if train_loader.type == "classification":
            loss = F.cross_entropy(y_hat, y)
        elif train_loader.type == "regression":
            loss = F.mse_loss(y_hat, y)
        loss_item = loss.item()

        # Check for divergence
        if not np.isfinite(loss_item):  # Detect inf/nan
            diverged = True  # Set flag to stop further computation
            for m_name in logger.metrics:
                logger.metrics[m_name][s:] = np.inf
            print("Exiting early due to divergence...")
            break

        metric = {
            "step": s,
        }

        if "losses" in metrics:
            metric["losses"] = loss_item

        if s % log_freq == 0:
            with torch.no_grad():
                if "rLs" in metrics:
                    # compute feature_learning metric rL
                    zL_current = trace[f'lin_{model.n_layers - 1}']["input"]
                    delta_zL = zL_current - zL_init
                    norm_delta_zL = delta_zL.norm().item()
                    if norm_delta_zL > 0:
                        rL = np.log(norm_delta_zL) / np.log(width)
                    else:
                        rL = -np.inf if s > 0 else 0  # (first step its exactly 0)
                    metric["rLs"] = rL

                # compute alignment metric
                if "Als" in metrics:
                    Al = []
                    for l_id in range(model.n_layers):
                        rms_norm = lambda x: torch.sqrt(torch.mean(x ** 2))
                        z, w, o = trace[f"lin_{l_id}"]["input"], trace[f"lin_{l_id}"]["weight"], trace[f"lin_{l_id}"]["output"]
                        z_n, w_n, o_n = rms_norm(z), rms_norm(w), rms_norm(o)

                        z_init, w_init = trace_init[f"lin_{l_id}"]["input"], trace_init[f"lin_{l_id}"]["weight"]
                        z_init_n, w_init_n = rms_norm(z_init), rms_norm(w_init)
                        dw, dz = w - w_init, z - z_init
                        dw_n, dz_n = rms_norm(dw), rms_norm(dz)

                        # I. cumulative alignment
                        # NOTE: this is supposed to be log_{fan_in}, for now its always always fan_in = width
                        A_cum = (torch.log(o_n) - torch.log(z_n * w_n)) / torch.log(torch.tensor(width))
                        if dw_n + dz_n != 0.0:
                            # II. alpha alignment
                            # o = z @ dw.T
                            o = z_init @ dw.T
                            o_n = rms_norm(o)
                            # NOTE: mistake in the paper implies commented out line when in fact it should be uncommented
                            # A_alpha = (torch.log(o_n) - torch.log(z_n * dw_n)) / torch.log(torch.tensor(width))
                            A_alpha = (torch.log(o_n) - torch.log(z_init_n * dw_n)) / torch.log(torch.tensor(width))
                            if l_id > 0:  # by definition z = x therefore dz = 0
                                # III. omega alignment
                                # o = dz @ w.T
                                o = dz @ w_init.T
                                o_n = rms_norm(o)
                                # NOTE: mistake in the paper implies commented out line when in fact it should be uncommented
                                # A_omega = (torch.log(o_n) - torch.log(dz_n * w_n)) / torch.log(torch.tensor(width))
                                A_omega = (torch.log(o_n) - torch.log(dz_n * w_init_n)) / torch.log(torch.tensor(width))
                                # IV. u alignment
                                o = dz @ dw.T
                                o_n = rms_norm(o)
                                A_u = (torch.log(o_n) - torch.log(dz_n * dw_n)) / torch.log(torch.tensor(width))
                            else:
                                temp = -torch.inf if s > 0 else torch.tensor(0)
                                A_omega, A_u = temp, temp
                        else:
                            temp = -torch.inf if s > 0 else torch.tensor(0)
                            A_alpha, A_omega, A_u = temp, temp, temp

                        Al.append(torch.tensor([A_cum, A_alpha, A_omega, A_u]))
                    Al = torch.stack(Al).detach().cpu().numpy()

                    # TODO: a temporary fix to avoid inf/nan values in the alignment metric
                    Al[np.isneginf(Al)] = 0.
                    Al[np.isinf(Al)] = 1.
                    Al[np.isnan(Al)] = 0.

                    metric["Als"] = Al

            if "Als" in metrics:
                alpha_l = Al[:, 1].tolist()
                omega_l = Al[:, 2].tolist()
                u_l = Al[:, 3].tolist()
            else:
                alpha_l, omega_l, u_l = None, None, None

            if not fractal_mode: 
                lrs = [0.0] * len(alpha_l)  # TEMP: placeholder
                if s > alignment_warmup:
                    lrs = lr_scheduler(alpha_l=alpha_l, u_l=u_l, omega_l=omega_l)
            else:
                lrs = None

            if "lrs" in metrics:
                metric["lrs"] = lrs

            logger.log(metric)

        loss.backward()

        opt.step()
        s += 1

    # Save the final arrays to binary format
    logger.save()

def main(run_name, exp_name, training_config, model_config, optimizer_config, lr_scheduler_config, parametrization_config, data_config):
    run_dir = os.path.join("./runs", exp_name, run_name)

    os.makedirs(run_dir, exist_ok=True)
    configs = {
        "training": training_config,
        "model": model_config,
        "optimizer": optimizer_config,
        "lr_scheduler": lr_scheduler_config,
        "parametrization": parametrization_config,
        "data": data_config,
    }
    for config_name, config in configs.items():
        config_path = os.path.join(run_dir, f"{config_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config().get_params(), f, indent=4)

    training_config().build(
        model_config=model_config, 
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        parametrization_config=parametrization_config,
        data_config=data_config,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    worker_id = int(os.environ.get("WORKER_ID", 0))
    n_workers = int(os.environ.get("N_WORKERS", 1))

    import configs.fractal_config
    exp_name = sys.argv[1]
    grid = getattr(configs.fractal_config, exp_name)

    for run_id, run_name, param_args in grid():
        if run_id % n_workers == worker_id:
            t0 = time.time()
            main(run_name, exp_name, *param_args)
            tf = time.time()
            print(f"Experiment {run_name} (id={run_id}) completed in {tf - t0:.2f} seconds by worker {worker_id}.")
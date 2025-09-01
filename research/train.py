# train.py
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from logger import BinaryLogger
from metrics.tracing import Tracer
from metrics.lib import build_metric_set, schema_from_metrics, compute_all

torch.set_default_dtype(torch.float64)


def train(
    model_config, optimizer_config, parametrization_config, lr_scheduler_config, data_config,
    n_train_steps,
    log_freq=1,
    seed=0,
    run_dir="./runs",
    metrics_config=None,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build model/optimizer/params
    model = model_config().build().to(device)
    opt_cfg = optimizer_config()

    width = model_config()["dims"][1]  # fan-in width for parametrization
    params = parametrization_config().build(mlp=model, n=width, lr_prefactor=opt_cfg['lr'], std_prefactor=1.0)
    opt = opt_cfg.build(params=params)

    lr_scheduler = lr_scheduler_config().build(optimizer=opt)
    alignment_warmup = 100

    # --- Data
    train_loader = data_config().build(device=device)

    # --- Tracer
    tracer = Tracer(model, sample_size=32)
    measurement_X, _ = next(iter(train_loader))
    tracer.capture_initial(measurement_X)

    # --- Metrics
    metrics_spec = metrics_config().build() if metrics_config is not None else ["alignment", "rL"]
    metric_set = build_metric_set(metrics_spec)

    # --- Logger schema
    current0 = tracer.capture(step=0, measurement_X=measurement_X)
    window0 = tracer.window(current0)
    base_schema = {"losses": (1,), "lrs": (window0.n_layers,)}
    extra_schema = schema_from_metrics(metric_set, window0) if metric_set else {}
    schema = {**base_schema, **extra_schema}
    logger = BinaryLogger(run_dir, n_steps=n_train_steps, metrics=schema)

    # --- Train loop
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
        loss_item = float(loss.item())

        if not np.isfinite(loss_item):
            diverged = True
            for m_name in logger.metrics:
                logger.metrics[m_name][s:] = np.inf
            print("Exiting early due to divergence...")
            break

        metric_row = {"step": s, "losses": loss_item}

        # Compute metrics BEFORE backward/step
        if s % log_freq == 0:
            with torch.no_grad():
                current = tracer.capture(step=s, measurement_X=measurement_X)
                window = tracer.window(current)

                if metric_set:
                    mvals = compute_all(metric_set, window)
                    metric_row.update(mvals)
                else:
                    mvals = {}

                # LR scheduling using alignment if present
                lrs = [0.0] * window.n_layers
                if "Als" in mvals:
                    Al = mvals["Als"]  # [L, 4]
                    alpha_l = Al[:, 1].tolist()
                    omega_l = Al[:, 2].tolist()
                    u_l     = Al[:, 3].tolist()
                else:
                    alpha_l = omega_l = u_l = None

                if s > alignment_warmup:
                    try:
                        lrs = lr_scheduler(alpha_l=alpha_l, u_l=u_l, omega_l=omega_l)
                    except TypeError:
                        lrs = [0.0] * window.n_layers

                metric_row["lrs"] = lrs

            logger.log(metric_row)

        # Backprop + step
        loss.backward()
        tracer.collect_weight_grads_after_backward()  # grad_weight captured for next measurement
        opt.step()
        tracer.on_optimizer_step()                    # update_weight captured for next measurement

        s += 1

    logger.save()


def main(run_name, exp_name, training_config, model_config, optimizer_config, lr_scheduler_config, parametrization_config, data_config, metrics_config=None):
    run_dir = os.path.join("./runs", exp_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save configs (metrics saved like the rest)
    configs = {
        "training": training_config,
        "model": model_config,
        "optimizer": optimizer_config,
        "lr_scheduler": lr_scheduler_config,
        "parametrization": parametrization_config,
        "data": data_config,
    }
    if metrics_config is not None:
        configs["metrics"] = metrics_config

    for config_name, config in configs.items():
        config_path = os.path.join(run_dir, f"{config_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config().get_params(), f, indent=4)

    # Kick off training via the training config
    training_config().build(
        model_config=model_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        parametrization_config=parametrization_config,
        data_config=data_config,
        metrics_config=metrics_config,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    worker_id = int(os.environ.get("WORKER_ID", 0))
    n_workers = int(os.environ.get("N_WORKERS", 1))

    import configs.lib
    exp_name = sys.argv[1]
    grid = getattr(configs.lib, exp_name)

    for run_id, run_name, param_args in grid():
        if run_id % n_workers == worker_id:
            t0 = time.time()
            main(run_name, exp_name, *param_args)
            tf = time.time()
            print(f"Experiment {run_name} (id={run_id}) completed in {tf - t0:.2f} seconds by worker {worker_id}.")
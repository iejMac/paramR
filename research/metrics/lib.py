# metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Any

import numpy as np
import torch

from metrics.tracing import TraceWindow

MetricOutput = Dict[str, Any]
Schema = Dict[str, Tuple[int, ...]]

# ---------- registry ----------

_REGISTRY: Dict[str, "Metric"] = {}

def register(metric_cls):
    _REGISTRY[metric_cls.name] = metric_cls  # type: ignore
    return metric_cls

def build_metric_set(names_and_kwargs: List[Tuple[str, Dict[str, Any]]]) -> List["Metric"]:
    out: List[Metric] = []
    for name, kwargs in names_and_kwargs:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown metric '{name}'. Available: {list(_REGISTRY)}")
        out.append(_REGISTRY[name](**kwargs))
    return out

def schema_from_metrics(metrics: List["Metric"], window: TraceWindow) -> Schema:
    schema: Schema = {}
    for m in metrics:
        schema.update(m.schema(window))
    return schema

def compute_all(metrics: List["Metric"], window: TraceWindow) -> MetricOutput:
    out: MetricOutput = {}
    for m in metrics:
        out.update(m.compute(window))
    return out

# ---------- base class ----------

class Metric:
    name: str = "base"
    def __init__(self, **kwargs): pass
    def schema(self, window: TraceWindow) -> Schema:
        raise NotImplementedError
    @torch.no_grad()
    def compute(self, window: TraceWindow) -> MetricOutput:
        raise NotImplementedError

# ---------- helpers ----------

def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    # robust RMS (avoid underflow for tiny tensors)
    return torch.sqrt(torch.mean(x.double() ** 2) + 1e-32)

def _cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").contiguous().numpy()

# ---------- rL (feature learning) ----------

@register
class FeatureRL(Metric):
    """
    rL metric on the *input* to the last layer (features).
    rL = log(||z_L - z_L_init||) / log(width), with sane behavior for zero norm.
    """
    name = "rL"
    key = "rLs"

    def schema(self, window: TraceWindow) -> Schema:
        return {self.key: (1,)}

    @torch.no_grad()
    def compute(self, window: TraceWindow) -> MetricOutput:
        last = window.layer_names[-1]
        z_init = window.init.layers[last].input
        z_cur  = window.current.layers[last].input
        assert z_init is not None and z_cur is not None, "Missing last-layer inputs for rL"

        delta = (z_cur - z_init).double()
        norm_delta = float(torch.norm(delta))
        if norm_delta > 0.0:
            rL = np.log(norm_delta) / np.log(window.width_fan_in)
        else:
            # by convention: exactly 0 at step 0, else -inf
            rL = 0.0 if window.current.step == 0 else -np.inf
        return {self.key: rL}

# ---------- Alignment (A_cum, A_alpha, A_omega, A_u) ----------

@register
class Alignment(Metric):
    """
    Alignment metrics per layer, following your implementation.
    Produces array [n_layers, 4]: [A_cum, A_alpha, A_omega, A_u].
    """
    name = "alignment"
    key = "Als"

    def schema(self, window: TraceWindow) -> Schema:
        return {self.key: (window.n_layers, 4)}

    @torch.no_grad()
    def compute(self, window: TraceWindow) -> MetricOutput:
        width = window.width_fan_in
        L = window.n_layers
        res = torch.zeros((L, 4), dtype=torch.float64)

        for idx, lname in enumerate(window.layer_names):
            cur = window.current.layers[lname]
            init = window.init.layers[lname]

            z, w, o = cur.input, cur.weight, cur.output
            z0, w0  = init.input, init.weight
            assert z is not None and w is not None and o is not None
            assert z0 is not None and w0 is not None

            z_n  = _rms_norm(z)
            w_n  = _rms_norm(w)
            o_n  = _rms_norm(o)
            z0_n = _rms_norm(z0)
            w0_n = _rms_norm(w0)

            dz   = (z - z0)
            dw   = (w - w0)
            dz_n = _rms_norm(dz)
            dw_n = _rms_norm(dw)

            # I. cumulative alignment
            A_cum = (torch.log(o_n) - torch.log(z_n * w_n)) / torch.log(torch.tensor(width, dtype=torch.float64))

            # Initialize others
            A_alpha = torch.tensor(0.0, dtype=torch.float64)
            A_omega = torch.tensor(0.0, dtype=torch.float64)
            A_u     = torch.tensor(0.0, dtype=torch.float64)

            if float(dw_n + dz_n) != 0.0:
                # II. alpha alignment: o = z0 @ dw^T
                o_alpha = z0 @ dw.T
                o_alpha_n = _rms_norm(o_alpha)
                A_alpha = (torch.log(o_alpha_n) - torch.log(z0_n * dw_n)) / torch.log(torch.tensor(width, dtype=torch.float64))

                if idx > 0:  # by definition z=x at l=0 => dz=0
                    # III. omega alignment: o = dz @ w0^T
                    o_omega = dz @ w0.T
                    o_omega_n = _rms_norm(o_omega)
                    A_omega = (torch.log(o_omega_n) - torch.log(dz_n * w0_n)) / torch.log(torch.tensor(width, dtype=torch.float64))

                    # IV. u alignment: o = dz @ dw^T
                    o_u = dz @ dw.T
                    o_u_n = _rms_norm(o_u)
                    A_u = (torch.log(o_u_n) - torch.log(dz_n * dw_n)) / torch.log(torch.tensor(width, dtype=torch.float64))
                else:
                    A_omega = torch.tensor(-float("inf") if window.current.step > 0 else 0.0, dtype=torch.float64)
                    A_u     = A_omega.clone()
            else:
                tmp = torch.tensor(-float("inf") if window.current.step > 0 else 0.0, dtype=torch.float64)
                A_alpha, A_omega, A_u = tmp, tmp.clone(), tmp.clone()

            res[idx, 0] = A_cum
            res[idx, 1] = A_alpha
            res[idx, 2] = A_omega
            res[idx, 3] = A_u

        # sanitize (match your temporary fix)
        A = _cpu(res)
        A[np.isneginf(A)] = 0.0
        A[np.isinf(A)]    = 1.0
        A[np.isnan(A)]    = 0.0
        return {self.key: A}
# metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union


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

SpecItem = Union[str, Tuple[str, Dict[str, Any]], Dict[str, Any]]

def build_metric_set(spec: List[SpecItem], resample_w0: bool = False) -> List["Metric"]:
    """
    Flexible spec:
      ["alignment", "rL"]
      [("alignment", {"foo": 1}), ("rL", {})]
      [{"name": "alignment"}, {"name": "rL", "kwargs": {"foo": 1}}]

    resample_w0: If True, pass to Alignment metric to enable w0 resampling
    """
    items: List[Tuple[str, Dict[str, Any]]] = []
    for it in spec:
        if isinstance(it, str):
            items.append((it, {}))
        elif isinstance(it, tuple) and len(it) == 2:
            name, kwargs = it
            items.append((name, dict(kwargs or {})))
        elif isinstance(it, dict):
            name = it.get("name")
            if not name:
                raise ValueError(f"Metric spec dict missing 'name': {it}")
            items.append((name, dict(it.get("kwargs", {}) or {})))
        else:
            raise ValueError(f"Unrecognized metric spec entry: {it!r}")

    out: List["Metric"] = []
    for name, kwargs in items:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown metric '{name}'. Available: {list(_REGISTRY)}")

        # Pass resample_w0 to Alignment metric
        if name == "alignment" and "resample_w0" not in kwargs:
            kwargs["resample_w0"] = resample_w0

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

def _rms_norm(x: torch.Tensor, dim: int=None) -> torch.Tensor:
    # robust RMS (avoid underflow for tiny tensors)
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.mean(x ** 2) + 1e-32)
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + 1e-32)

def _l2_norm(x: torch.Tensor, dim: int=None) -> torch.Tensor:
    # robust RMS (avoid underflow for tiny tensors)
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.sum(x ** 2) + 1e-32)
    return torch.sqrt(torch.sum(x ** 2, dim=dim) + 1e-32)

def _spectral_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x, 2)


def _cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").contiguous().numpy()


def _resample_w0(shape: Tuple[int, ...], al: float, bl: float, std_prefactor: float = 2**0.5) -> torch.Tensor:
    """
    Resample initial weights using abc parametrization initialization scheme.
    shape: (out_features, in_features) for Linear weights
    al, bl: parametrization exponents
    std_prefactor: initialization std multiplier (default sqrt(2) for ReLU)
    """
    n = shape[1]  # fan-in (width)
    var_l = n ** (-2 * bl)
    std = std_prefactor * (var_l ** 0.5)
    return torch.randn(shape, dtype=torch.float64) * std


def _compute_z0_from_w0(z_prev: torch.Tensor, w0_new: torch.Tensor, al: float, apply_relu: bool = True) -> torch.Tensor:
    """
    Compute z0 for a layer given the previous layer's z0 and a resampled w0.
    z_prev: input to this layer (batch_size, in_features)
    w0_new: resampled initial weight (out_features, in_features)
    al: parametrization exponent for layer multiplier
    apply_relu: whether to apply ReLU activation (False for last layer)
    """
    n = w0_new.shape[1]  # fan-in
    layer_mult = n ** (-al)
    z0 = z_prev @ w0_new.T * layer_mult
    if apply_relu:
        z0 = torch.relu(z0)
    return z0

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

    def __init__(self, resample_w0: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.resample_w0 = resample_w0

    def schema(self, window: TraceWindow) -> Schema:
        return {self.key: (window.n_layers, 4)}

    @torch.no_grad()
    def compute(self, window: TraceWindow) -> MetricOutput:
        width = window.width_fan_in
        L = window.n_layers
        res = torch.zeros((L, 4), dtype=torch.float64)

        # Prepare w0 and z0 for all layers
        w0_dict = {}
        z0_dict = {}

        if self.resample_w0 and window.init_params is not None:
            al_list = window.init_params.get('al', [])
            bl_list = window.init_params.get('bl', [])

            # First pass: resample all w0s
            for idx, lname in enumerate(window.layer_names):
                init = window.init.layers[lname]
                w0_stored = init.weight

                if idx < len(al_list) and idx < len(bl_list):
                    al, bl = al_list[idx], bl_list[idx]
                    # Resample w0 with same dtype as stored weights
                    w0_dict[lname] = _resample_w0(w0_stored.shape, al=al, bl=bl).to(w0_stored.dtype)
                else:
                    w0_dict[lname] = w0_stored

            # Second pass: compute z0s via forward pass
            for idx, lname in enumerate(window.layer_names):
                init = window.init.layers[lname]
                z0_input = init.input  # Input to this layer at initialization

                if idx == 0:
                    # First layer: use the original input (doesn't change)
                    z0_dict[lname] = z0_input
                else:
                    # Compute z0 for this layer from previous layer's output
                    prev_lname = window.layer_names[idx - 1]
                    z0_prev = z0_dict[prev_lname]  # Input to this layer (output of prev)
                    w0_prev = w0_dict[prev_lname]
                    al_prev = al_list[idx - 1] if idx - 1 < len(al_list) else 0.0

                    # Compute output of previous layer with resampled w0_prev
                    apply_relu = (idx < L)  # All layers except last have ReLU
                    z0_dict[lname] = _compute_z0_from_w0(z0_prev, w0_prev, al_prev, apply_relu=apply_relu)
        else:
            # Use stored values
            for lname in window.layer_names:
                init = window.init.layers[lname]
                w0_dict[lname] = init.weight
                z0_dict[lname] = init.input

        # Main computation loop
        for idx, lname in enumerate(window.layer_names):
            cur = window.current.layers[lname]
            z, w, o = cur.input, cur.weight, cur.output
            w0 = w0_dict[lname]
            z0 = z0_dict[lname]

            assert z is not None and w is not None and o is not None
            assert z0 is not None and w0 is not None

            # z_n  = _rms_norm(z, dim=-1)
            z_n  = _l2_norm(z, dim=-1)

            # w_n  = _rms_norm(w)
            w_n  = _spectral_norm(w)

            # o_n  = _rms_norm(o, dim=-1)
            o_n  = _l2_norm(o, dim=-1)

            # z0_n = _rms_norm(z0, dim=-1)
            z0_n = _l2_norm(z0, dim=-1)

            # w0_n = _rms_norm(w0)
            w0_n = _spectral_norm(w0)

            dz   = (z - z0)
            dw   = (w - w0)

            # dz_n = _rms_norm(dz, dim=-1)
            dz_n = _l2_norm(dz, dim=-1)

            # dw_n = _rms_norm(dw)
            dw_n = _spectral_norm(dw)

            # I. cumulative alignment
            # A_cum = (torch.log(o_n) - torch.log(z_n * w_n)) / torch.log(torch.tensor(width, dtype=torch.float64))
            A_cum = torch.mean(o_n / (z_n * w_n))

            # Initialize others
            A_alpha = torch.tensor(0.0, dtype=torch.float64)
            A_omega = torch.tensor(0.0, dtype=torch.float64)
            A_u     = torch.tensor(0.0, dtype=torch.float64)

            if float((dw_n + dz_n).sum()) != 0.0:
                # II. alpha alignment: o = z0 @ dw^T
                o_alpha = z0 @ dw.T
                # o_alpha_n = _rms_norm(o_alpha, dim=-1)
                o_alpha_n = _l2_norm(o_alpha, dim=-1)

                # A_alpha = (torch.log(o_alpha_n) - torch.log(z0_n * dw_n)) / torch.log(torch.tensor(width, dtype=torch.float64))
                A_alpha = torch.mean(o_alpha_n / (z0_n * dw_n))

                if idx > 0:  # by definition z=x at l=0 => dz=0
                    # III. omega alignment: o = dz @ w0^T
                    o_omega = dz @ w0.T
                    # o_omega_n = _rms_norm(o_omega, dim=-1)
                    o_omega_n = _l2_norm(o_omega, dim=-1)

                    # A_omega = (torch.log(o_omega_n) - torch.log(dz_n * w0_n)) / torch.log(torch.tensor(width, dtype=torch.float64))
                    A_omega = torch.mean(o_omega_n / (dz_n * w0_n))

                    # IV. u alignment: o = dz @ dw^T
                    o_u = dz @ dw.T
                    # o_u_n = _rms_norm(o_u, dim=-1)
                    o_u_n = _l2_norm(o_u, dim=-1)

                    # A_u = (torch.log(o_u_n) - torch.log(dz_n * dw_n)) / torch.log(torch.tensor(width, dtype=torch.float64))
                    A_u = torch.mean(o_u_n / (dz_n * dw_n))
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
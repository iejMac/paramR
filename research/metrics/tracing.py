# tracing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import contextlib
import torch
import torch.nn as nn

Tensor = torch.Tensor

@dataclass
class LayerSnap:
    input: Optional[Tensor] = None
    output: Optional[Tensor] = None
    weight: Optional[Tensor] = None
    grad_weight: Optional[Tensor] = None
    update_weight: Optional[Tensor] = None

@dataclass
class StepTrace:
    step: int
    layers: Dict[str, LayerSnap]

@dataclass
class TraceWindow:
    init: StepTrace
    current: StepTrace
    n_layers: int
    width_fan_in: int
    layer_names: List[str]
    # Initialization parameters (a, b from abc parametrization)
    init_params: Optional[Dict[str, List[float]]] = None

class Tracer:
    """
    Minimal tracer for nn.Linear layers.
    Captures: input, output, weight (on measurement forwards) +
              grad_weight (after backward) +
              update_weight (after optimizer.step()).
    """
    def __init__(self, model: nn.Module, sample_size: int = 32, init_params: Optional[Dict[str, List[float]]] = None):
        self.model = model
        self.sample_size = sample_size
        self.init_params = init_params  # Store al, bl for resampling

        # collect linear modules in definition order
        self.layer_names: List[str] = []
        self.modules: List[nn.Linear] = []

        def traverse_model(m: nn.Module, prefix: str):
            for name, layer in m.named_children():
                if isinstance(layer, nn.Linear):  # TODO: add embedding and layer norm?
                    self.layer_names.append(f"{prefix}.{name}")
                    self.modules.append(layer)
                else:
                    traverse_model(layer, f"{prefix}.{name}")

        traverse_model(self.model, prefix="model")

        if not self.modules:
            raise RuntimeError("Tracer: no layers found.")

        self._armed = False
        self._snap: Dict[str, LayerSnap] = {}

        # state for grads and updates
        self._last_grad_cpu: Dict[str, Tensor] = {}
        self._last_update_cpu: Dict[str, Tensor] = {}
        self._prev_weight_gpu: Dict[str, Tensor] = {}

        for name, mod in zip(self.layer_names, self.modules):
            self._last_update_cpu[name] = torch.zeros_like(mod.weight, device="cpu")
            self._prev_weight_gpu[name] = mod.weight.data.detach().clone()

        # single forward hook per linear
        self._handles = [mod.register_forward_hook(self._make_hook(name))
                         for name, mod in zip(self.layer_names, self.modules)]

        self.initial: Optional[StepTrace] = None

    def _make_hook(self, name: str):
        def hook(module: nn.Linear, inputs, output):
            if not self._armed:
                return
            x = inputs[0]
            if self.sample_size is not None:
                x = x[: self.sample_size]
                output = output[: self.sample_size]
            self._snap[name] = LayerSnap(
                input=x.detach().to("cpu").contiguous(),
                output=output.detach().to("cpu").contiguous(),
                weight=module.weight.detach().to("cpu").contiguous(),
                grad_weight=self._last_grad_cpu.get(name),
                update_weight=self._last_update_cpu.get(name),
            )
        return hook

    @contextlib.contextmanager
    def trace(self):
        self._armed, self._snap = True, {}
        try:
            yield
        finally:
            self._armed = False

    @torch.no_grad()
    def capture_initial(self, measurement_X: Tensor) -> StepTrace:
        with self.trace():
            _ = self.model(measurement_X)
        self.initial = StepTrace(step=0, layers=self._snap)
        return self.initial

    @torch.no_grad()
    def capture(self, step: int, measurement_X: Tensor) -> StepTrace:
        with self.trace():
            _ = self.model(measurement_X)
        return StepTrace(step=step, layers=self._snap)

    def collect_weight_grads_after_backward(self):
        # Call right after loss.backward()
        for name, mod in zip(self.layer_names, self.modules):
            g = mod.weight.grad
            if g is not None:
                self._last_grad_cpu[name] = g.detach().to("cpu").contiguous()

    def on_optimizer_step(self):
        # Call right after optimizer.step()
        for name, mod in zip(self.layer_names, self.modules):
            w = mod.weight.data
            prev = self._prev_weight_gpu[name]
            self._last_update_cpu[name] = (w - prev).detach().to("cpu").contiguous()
            self._prev_weight_gpu[name] = w.detach().clone()

    def window(self, current: StepTrace) -> TraceWindow:
        if self.initial is None:
            raise RuntimeError("Tracer.initial not set; call capture_initial() first.")
        first = next(iter(current.layers.values()))
        width = int(first.weight.shape[1])  # fan-in
        return TraceWindow(
            init=self.initial, current=current,
            n_layers=len(self.layer_names),
            width_fan_in=width,
            layer_names=self.layer_names,
            init_params=self.init_params,
        )
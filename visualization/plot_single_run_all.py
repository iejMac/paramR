#!/usr/bin/env python3
"""
Line chart visualizer for a single run directory.

Expected files (any subset is fine):
  - losses.npy           # shape [T] or [T,1]
  - rLs.npy              # shape [T] or [T,1]
  - lrs.npy              # shape [T, L] (layerwise learning rates)
  - Als.npy              # shape [T, L, 4] (A_cum, A_alpha, A_omega, A_u)

Also reads:
  - model_config.json (optional, for dims/n_layers in titles)

Outputs (saved into the run dir):
  - losses.png (with last-K mean horizontal line)
  - rLs.png
  - lrs_lines.png
  - A_cum_lines.png, A_alpha_lines.png, A_omega_lines.png, A_u_lines.png
  - all_plots.pdf (optional, one page per chart)

Usage:
  python visualization/plot_single_run_all_lines.py \
      --run ./runs/cifar_single/cifar_w64_lr0.2_sgd \
      --k 10 \
      --pdf
"""
import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --------- IO helpers ---------

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _load_npy(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        return np.asarray(np.load(path))
    except Exception:
        return None

def _squeeze_1d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.reshape(-1)
    return arr

# --------- shape normalization ---------

def _infer_steps_hint(*arrays) -> Optional[int]:
    """Heuristic to guess T (steps)."""
    cands = []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a)
        if a.ndim == 1:
            cands.append(a.shape[0])
        elif a.ndim == 2:
            cands.extend(list(a.shape))
        elif a.ndim == 3:
            # probably [T,L,4] or [L,T,4]; either of first two axes is T
            cands.extend(list(a.shape[:2]))
    return int(max(cands)) if cands else None

def _time_major_1d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return _squeeze_1d(arr).astype(float)

def _time_major_2d(arr: Optional[np.ndarray], steps_hint: Optional[int]) -> Optional[np.ndarray]:
    """Return [T, L]. If arr is [L,T], transpose; else assume axis 0 is time."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim != 2:
        return None
    if steps_hint is not None:
        if a.shape[0] == steps_hint:
            return a
        if a.shape[1] == steps_hint:
            return a.T
    # fallback: prefer axis 0 as time
    return a

def _time_major_3d_Als(arr: Optional[np.ndarray], steps_hint: Optional[int]) -> Optional[np.ndarray]:
    """Normalize Als to [T, L, 4] from any permutation containing a 4-axis."""
    if arr is None:
        return None
    A = np.asarray(arr, dtype=float)
    if A.ndim != 3:
        return None
    # find the components axis (=4)
    try:
        comp_axis = list(A.shape).index(4)
    except ValueError:
        return None
    if comp_axis != 2:
        A = np.moveaxis(A, comp_axis, 2)  # now A[..., 4]
    # choose T vs L among first two axes
    T, L, _ = A.shape
    if steps_hint is not None and L == steps_hint and T != steps_hint:
        A = np.transpose(A, (1, 0, 2))
    return A  # [T, L, 4]

# --------- plotting ---------

def _savefig(fig, out_path: str, pdf: Optional[PdfPages]):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")

def _plot_series(y: np.ndarray, title: str, ylabel: str, out_png: str, hline_lastk: Optional[float]=None, pdf: Optional[PdfPages]=None):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    x = np.arange(len(y))
    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline_lastk is not None and np.isfinite(hline_lastk):
        ax.axhline(hline_lastk, linestyle="--", linewidth=1.0)
        ax.text(0.99, 0.02, f"last-k mean: {hline_lastk:.6g}", transform=ax.transAxes,
                ha="right", va="bottom")
    _savefig(fig, out_png, pdf)

def _plot_lines_per_layer(M: np.ndarray, title: str, ylabel: str, out_png: str, max_legend_layers: int = 12, pdf: Optional[PdfPages]=None):
    """M: [T, L] -> one line per layer on a single axes."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    T, L = M.shape
    x = np.arange(T)
    for l in range(L):
        ax.plot(x, M[:, l], linewidth=1.2, label=f"layer {l}" if l < max_legend_layers else None)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if L <= max_legend_layers:
        ax.legend(frameon=False, ncol=2)
    _savefig(fig, out_png, pdf)

# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to run directory")
    ap.add_argument("--k", type=int, default=10, help="Average of last K steps for loss hline")
    ap.add_argument("--pdf", action="store_true", help="Also save combined PDF (all_plots.pdf)")
    ap.add_argument("--legend-cap", type=int, default=12, help="Max layers shown in legends")
    args = ap.parse_args()

    run_dir = args.run
    os.makedirs(run_dir, exist_ok=True)

    # Load arrays
    losses = _load_npy(os.path.join(run_dir, "losses.npy"))
    rLs    = _load_npy(os.path.join(run_dir, "rLs.npy"))
    lrs    = _load_npy(os.path.join(run_dir, "lrs.npy"))
    Als    = _load_npy(os.path.join(run_dir, "Als.npy"))

    # Steps hint + normalization
    steps_hint = _infer_steps_hint(losses, rLs, lrs, Als)
    losses_tm = _time_major_1d(losses)
    rLs_tm    = _time_major_1d(rLs)
    lrs_tm    = _time_major_2d(lrs, steps_hint)         # [T, L]
    Als_tm    = _time_major_3d_Als(Als, steps_hint)     # [T, L, 4]

    # Meta (optional)
    model_cfg = _load_json(os.path.join(run_dir, "model_config.json")) or {}
    dims = model_cfg.get("dims", None)
    n_layers = (len(dims) - 1) if isinstance(dims, list) else (lrs_tm.shape[1] if lrs_tm is not None else None)
    title_suffix = f" (n_layers={n_layers})" if n_layers is not None else ""

    pdf = PdfPages(os.path.join(run_dir, "all_plots.pdf")) if args.pdf else None

    # ---- losses (with last-k mean line)
    if losses_tm is not None:
        k = max(1, min(args.k, len(losses_tm)))
        final_mean = float(np.mean(losses_tm[-k:]))
        _plot_series(
            losses_tm,
            title=f"Loss over steps{title_suffix}",
            ylabel="loss",
            out_png=os.path.join(run_dir, "losses.png"),
            hline_lastk=final_mean,
            pdf=pdf,
        )
        print(f"Final loss (mean of last {k}): {final_mean:.6g}")

    # ---- rL (single line)
    if rLs_tm is not None:
        _plot_series(
            rLs_tm,
            title=f"rL over steps{title_suffix}",
            ylabel="rL",
            out_png=os.path.join(run_dir, "rLs.png"),
            pdf=pdf,
        )

    # ---- learning rates (per-layer lines)
    if lrs_tm is not None:
        _plot_lines_per_layer(
            lrs_tm,
            title=f"Learning rates per layer{title_suffix}",
            ylabel="lr",
            out_png=os.path.join(run_dir, "lrs_lines.png"),
            max_legend_layers=args.legend_cap,
            pdf=pdf,
        )

    # ---- alignment metrics (each component as its own per-layer line chart)
    if Als_tm is not None and Als_tm.shape[-1] == 4:
        names = ["A_cum", "A_alpha", "A_omega", "A_u"]
        for i, name in enumerate(names):
            M = Als_tm[:, :, i]  # [T, L]
            _plot_lines_per_layer(
                M,
                title=f"{name} per layer{title_suffix}",
                ylabel=name,
                out_png=os.path.join(run_dir, f"{name}_lines.png"),
                max_legend_layers=args.legend_cap,
                pdf=pdf,
            )

    if pdf is not None:
        pdf.close()
        print(f"Saved combined PDF: {os.path.join(run_dir, 'all_plots.pdf')}")

    print("Done.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Plot: LR (x) vs Final Loss (avg of last K steps) with one curve per width.
- Walks runs/<exp_name>/*, loads losses, width, and lr from each run dir.
- Groups by width; for each width plots LR vs final loss.
- Colors are mapped proportional to width and a colorbar is shown.

Usage:
  python scripts/plot_width_lr_loss.py --runs ./runs --exp width_lr_grid --k 10 --out ./runs/width_lr_grid/width_lr_final_loss.png
"""

import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------- Helpers ----------

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _squeeze_1d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and (arr.shape[1] == 1 or arr.shape[0] == 1):
        return arr.reshape(-1)
    return arr

def _load_losses(run_dir: str) -> Optional[np.ndarray]:
    """
    Try a few common layouts:
      - run_dir/losses.npy
      - run_dir/metrics.npz (key 'losses' or similar)
      - any .npy with 'loss' in the name
    Returns 1D float array or None.
    """
    # 1) losses.npy
    p = os.path.join(run_dir, "losses.npy")
    if os.path.exists(p):
        try:
            arr = np.load(p)
            return _squeeze_1d(arr.astype(float))
        except Exception:
            pass

    # 2) metrics.npz
    p = os.path.join(run_dir, "metrics.npz")
    if os.path.exists(p):
        try:
            data = np.load(p)
            for key in ["losses", "loss", "loss_array"]:
                if key in data:
                    return _squeeze_1d(np.array(data[key], dtype=float))
            # fallback: first key that contains 'loss'
            for key in data.files:
                if "loss" in key.lower():
                    return _squeeze_1d(np.array(data[key], dtype=float))
        except Exception:
            pass

    # 3) any .npy with 'loss' in the filename
    try:
        for fname in os.listdir(run_dir):
            if fname.endswith(".npy") and "loss" in fname.lower():
                arr = np.load(os.path.join(run_dir, fname))
                return _squeeze_1d(arr.astype(float))
    except Exception:
        pass

    return None

_RUN_NAME_RE = re.compile(r"w(?P<w>\d+)_lr(?P<lr>[-+eE0-9\.]+)")

def _parse_width_from_runname(run_name: str) -> Optional[int]:
    m = _RUN_NAME_RE.search(run_name)
    if not m:
        return None
    try:
        return int(m.group("w"))
    except Exception:
        return None

def _parse_lr_from_runname(run_name: str) -> Optional[float]:
    m = _RUN_NAME_RE.search(run_name)
    if not m:
        return None
    try:
        return float(m.group("lr"))
    except Exception:
        return None

def _get_width(run_dir: str, run_name: str) -> Optional[int]:
    # Prefer model_config.json
    cfg = _load_json(os.path.join(run_dir, "model_config.json"))
    if cfg and isinstance(cfg.get("dims"), list) and len(cfg["dims"]) >= 2:
        try:
            return int(cfg["dims"][1])
        except Exception:
            pass
    # Fallback: parse from run_name
    return _parse_width_from_runname(run_name)

def _get_lr(run_dir: str, run_name: str) -> Optional[float]:
    # Prefer optimizer_config.json
    cfg = _load_json(os.path.join(run_dir, "optimizer_config.json"))
    if cfg and "lr" in cfg:
        try:
            return float(cfg["lr"])
        except Exception:
            pass
    # Fallback: parse from run_name
    return _parse_lr_from_runname(run_name)

def _final_k_mean(losses: np.ndarray, k: int) -> Optional[float]:
    if losses is None or losses.size == 0:
        return None
    finite = np.isfinite(losses)
    if not finite.any():
        return None
    vals = losses[finite]
    k = min(k, len(vals))
    if k <= 0:
        return None
    return float(np.mean(vals[-k:]))

# ---------- Main logic ----------

def collect_runs(exp_dir: str, k: int) -> Dict[int, List[Tuple[float, float, str]]]:
    """
    Returns {width: [(lr, final_loss, run_name), ...], ...}
    """
    grouped: Dict[int, List[Tuple[float, float, str]]] = {}
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    for run_name in sorted(os.listdir(exp_dir)):
        run_dir = os.path.join(exp_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        losses = _load_losses(run_dir)
        if losses is None:
            # No losses found; skip quietly
            continue

        width = _get_width(run_dir, run_name)
        lr = _get_lr(run_dir, run_name)
        if width is None or lr is None:
            # Missing metadata; skip
            continue

        final = _final_k_mean(losses, k=k)
        if final is None or not np.isfinite(final):
            continue

        grouped.setdefault(width, []).append((lr, final, run_name))

    # Sort within each width by LR
    for w, items in grouped.items():
        items.sort(key=lambda t: t[0])  # sort by lr
    return grouped

def plot_width_lr_curves(grouped, exp_name: str, k: int, out_path: Optional[str]):
    if not grouped:
        print("No data to plot.")
        return

    widths = sorted(grouped.keys())

    # Normalize widths → colors
    w_min, w_max = min(widths), max(widths)
    norm = plt.Normalize(vmin=w_min, vmax=w_max)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required when mappable isn't tied to an image

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)

    for w in widths:
        series = grouped[w]
        lrs = [lr for lr, _, _ in series]
        finals = [fl for _, fl, _ in series]
        color = cmap(norm(w))
        ax.plot(lrs, finals, marker="o", linewidth=1.5, markersize=4, color=color, label=f"W={w}")

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel(f"Final loss (mean of last {k} steps)")
    ax.set_title(f"{exp_name}: LR vs Final Loss")

    # Colorbar for width, attached to this Axes
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Width")

    if len(widths) <= 10:
        ax.legend(loc="best", frameon=False)
    else:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), frameon=False)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="./runs", help="Path to runs root directory")
    ap.add_argument("--exp", type=str, required=True, help="Experiment name (subdir under --runs)")
    ap.add_argument("--k", type=int, default=10, help="Average over last K steps")
    ap.add_argument("--out", type=str, default="", help="Output path for PNG (if omitted, will show instead)")
    args = ap.parse_args()

    exp_dir = os.path.join(args.runs, args.exp)
    grouped = collect_runs(exp_dir, k=args.k)
    out_path = args.out if args.out else None
    plot_width_lr_curves(grouped, exp_name=args.exp, k=args.k, out_path=out_path)

if __name__ == "__main__":
    main()
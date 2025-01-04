#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

###############################################################################
#                              CONFIGURATION
###############################################################################
BASE_PATH = "/home/maciej/code/paramR/runs"
run_dirs = [
    os.path.join(BASE_PATH, d) 
    for d in os.listdir(BASE_PATH) 
    if os.path.isdir(os.path.join(BASE_PATH, d))
]

# Keys in your JSON for param1/param2. E.g. "al.2" = a3, "bl.2" = b3
param1_name = "al.2"  
param2_name = "bl.2"

output_filename = "overlayed_losses_best_lr_from_all_runs.png"

###############################################################################
#                        HELPER FUNCTIONS
###############################################################################
def extract_param_values(subdir_path: str, param_key1: str, param_key2: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Load 'parametrization_config.json' to extract the specified (param1, param2).
    Returns (v1, v2) or (None, None) if missing or invalid.
    """
    config_path = os.path.join(subdir_path, "parametrization_config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return (None, None)
    
    def get_nested_value(cfg, dotted_key: str):
        keys = dotted_key.split('.')
        val = cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, list):
                try:
                    idx = int(k)
                    val = val[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return val

    v1 = get_nested_value(config, param_key1)
    v2 = get_nested_value(config, param_key2)
    if v1 is None or v2 is None:
        return (None, None)
    return (float(v1), float(v2))

def extract_lr(subdir_path: str) -> Optional[float]:
    """
    Load 'optimizer_config.json' to extract 'lr' (learning rate).
    Returns None if not found or invalid.
    """
    opt_config_path = os.path.join(subdir_path, "optimizer_config.json")
    try:
        with open(opt_config_path, "r") as f:
            cfg = json.load(f)
        return float(cfg["lr"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None

def load_losses(subdir_path: str) -> Optional[np.ndarray]:
    """
    Load 'losses.npy' if present and valid. Return None otherwise.
    """
    losses_path = os.path.join(subdir_path, "losses.npy")
    try:
        return np.load(losses_path)
    except (FileNotFoundError, OSError, ValueError):
        return None

# ADDED: A new function to load steps
def load_steps(subdir_path: str) -> Optional[np.ndarray]:
    """
    Load 'steps.npy' if present. Return None otherwise.
    """
    steps_path = os.path.join(subdir_path, "steps.npy")
    if not os.path.exists(steps_path):
        return None
    try:
        return np.load(steps_path)
    except (OSError, ValueError):
        return None

def score_losses(losses: np.ndarray, window: int = 20) -> float:
    """
    Return a 'score' for deciding which losses are "best."
    Defaults to the average of the last `window` steps.
    Lower score = better.
    """
    if len(losses) < window:
        return float(np.mean(losses))  # fallback for short arrays
    return float(np.mean(losses[-window:]))

###############################################################################
#               PER-RUN FUNCTION: GATHER BEST LR FOR EACH (p1, p2)
###############################################################################
def compute_best_data_for_run_dir(
    run_dir: str, 
    param1: str, 
    param2: str
) -> Dict[Tuple[float, float], Tuple[float, np.ndarray, np.ndarray]]:
    """
    Go through all subdirectories in `run_dir`.
    For each subdir with valid (p1_val, p2_val) and LR, load losses and steps.
    Among multiple LRs for the same (p1_val, p2_val), pick the best based on score.

    Returns a dict: 
       { (p1_val, p2_val): (best_lr, best_losses_array, best_steps_array) }
    """
    if not os.path.isdir(run_dir):
        return {}

    # Collect all (lr, losses, steps) for each (p1_val, p2_val) in this run
    temp_dict = {}  # (p1_val, p2_val) -> list of (lr, losses, steps)
    subdirs = [
        d for d in os.listdir(run_dir) 
        if os.path.isdir(os.path.join(run_dir, d))
    ]
    for subdir in subdirs:
        subdir_path = os.path.join(run_dir, subdir)
        p1_val, p2_val = extract_param_values(subdir_path, param1, param2)
        if p1_val is None or p2_val is None:
            continue
        
        lr = extract_lr(subdir_path)
        if lr is None:
            continue
        
        losses = load_losses(subdir_path)
        if losses is None:
            continue

        # ADDED: Load steps (or fallback)
        steps = load_steps(subdir_path)
        if steps is None:
            steps = np.arange(len(losses))  # fallback if no steps.npy found

        # If there's a mismatch in length, adjust or skip
        if len(steps) != len(losses):
            min_len = min(len(steps), len(losses))
            steps = steps[:min_len]
            losses = losses[:min_len]

        key = (p1_val, p2_val)
        if key not in temp_dict:
            temp_dict[key] = []
        temp_dict[key].append((lr, losses, steps))

    # Now pick the best LR for each (p1_val, p2_val)
    best_dict = {}
    for key, list_of_lr_losses in temp_dict.items():
        best_lr = None
        best_losses = None
        best_steps = None
        best_score = float("inf")

        for (lr, losses_array, steps_array) in list_of_lr_losses:
            sc = score_losses(losses_array, window=20)
            if sc < best_score:
                best_score = sc
                best_lr = lr
                best_losses = losses_array
                best_steps = steps_array
        
        if best_lr is not None and best_losses is not None and best_steps is not None:
            best_dict[key] = (best_lr, best_losses, best_steps)

    return best_dict

###############################################################################
#   COMBINE DATA FROM ALL RUN DIRS: (p1, p2) -> List[ (run_label, best_lr, best_losses) ]
###############################################################################
def collect_data_across_runs(
    run_dirs: List[str],
    param1: str,
    param2: str
) -> Dict[Tuple[float, float], List[Tuple[str, float, np.ndarray, np.ndarray]]]:
    """
    For each run_dir in `run_dirs`, produce its best_dict, i.e.:
        { (p1_val, p2_val): (best_lr, best_losses, best_steps) }

    Then we combine them so that each (p1_val, p2_val) maps to 
    a list of (run_label, best_lr, best_losses, best_steps) from all runs that have data.
    """
    # Combined structure
    combined_dict = {}  # (p1_val, p2_val) -> list of (run_label, best_lr, best_losses, best_steps)

    for run_dir in run_dirs:
        run_label = os.path.basename(run_dir) or run_dir  # fallback if empty
        best_dict = compute_best_data_for_run_dir(run_dir, param1, param2)

        for key, (lr, losses, steps) in best_dict.items():
            if key not in combined_dict:
                combined_dict[key] = []
            combined_dict[key].append((run_label, lr, losses, steps))

    return combined_dict

###############################################################################
#                          PLOTTING (OVERLAYED)
###############################################################################
def plot_overlayed_losses_with_best_lr(
    combined_dict: Dict[Tuple[float, float], List[Tuple[str, float, np.ndarray, np.ndarray]]],
    param1: str,
    param2: str,
    output_filename: str
):
    """
    Creates a 2D grid of subplots (rows = unique param1, cols = unique param2).
    In each cell, we overlay one line per run_dir—BUT each line is the "best" LR 
    that was selected for that run_dir + (p1, p2) combo.

    We color-code by run_dir (a single color for each run_dir) and create 
    a single figure-level legend mapping run_dir -> color.

    We also expand the subplot title to list each run's best LR. 
    """
    if not combined_dict:
        print("No valid data found to plot.")
        return

    # Collect sorted param1 and param2 values
    param1_values = sorted({p1 for (p1, _) in combined_dict.keys()})
    param2_values = sorted({p2 for (_, p2) in combined_dict.keys()})

    n_rows = len(param1_values)
    n_cols = len(param2_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=False, sharey=False)
    
    # Handle 1D cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([ax for ax in axes]).reshape(n_rows, 1)

    # Initialize
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)

    # We’ll pick a color for each run_label so it's consistent across subplots.
    run_label_to_color = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    next_color_idx = 0

    # For each cell, we might have multiple runs, each with a best LR
    # We'll store a mini-title line summarizing each run's LR
    for (p1_val, p2_val), runs_list in combined_dict.items():
        row = param1_values.index(p1_val)
        col = param2_values.index(p2_val)
        ax = axes[row, col]

        # Clear placeholder
        ax.clear()

        # We'll accumulate lines of text to add to the subplot's title
        best_lr_lines = []

        # Plot each run's best curve
        for (run_label, best_lr, best_losses, best_steps) in runs_list:
            # Assign color if not assigned yet
            if run_label not in run_label_to_color:
                run_label_to_color[run_label] = color_cycle[next_color_idx % len(color_cycle)]
                next_color_idx += 1

            c = run_label_to_color[run_label]
            # CHANGED: Plot with best_steps on the x-axis
            ax.plot(best_steps, best_losses, color=c)
            # Keep track of the LR in the title
            best_lr_lines.append(f"{run_label}: LR={best_lr:.2e}")

        # Title: param1 & param2 on first line, then run info
        title_lines = [
            f"{param1}={p1_val:.2f}, {param2}={p2_val:.2f}",
            *best_lr_lines  # each run on its own line
        ]
        ax.set_title("\n".join(title_lines), fontsize=8)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    # Build a single legend: each run_label -> color
    legend_handles = []
    for run_label, c in run_label_to_color.items():
        line = plt.Line2D([0], [0], color=c, label=run_label)
        legend_handles.append(line)

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        fontsize=8
    )

    fig.tight_layout()
    print(f"Saving figure to {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

###############################################################################
#                                 MAIN
###############################################################################
def main():
    # 1) Collect data from all run directories, picking the best LR
    combined_dict = collect_data_across_runs(run_dirs, param1_name, param2_name)

    # 2) Plot them in a 2D grid, overlaying different runs in each cell
    plot_overlayed_losses_with_best_lr(combined_dict, param1_name, param2_name, output_filename)

if __name__ == "__main__":
    main()
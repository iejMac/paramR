import os
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import List, Dict, Tuple

# Enable LaTeX-like rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "dejavusans",
})

# Styling constants
COMPONENT_STYLES = {
    'Cumulative': '-',      # Solid line for Cumulative
    'Alpha': '--',          # Dashed line for Alpha
    'Omega': ':',           # Dotted line for Omega
    'U': '-.'               # Dash-dot line for U
}

COMPONENT_LABELS = {
    'Cumulative': r'A',
    'Alpha': r'$\alpha$',
    'Omega': r'$\omega$',
    'U': r'$\mathcal{u}$'
}

LAYER_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

# Figure settings to ensure even dimensions
FIG_WIDTH_INCHES = 13
FIG_HEIGHT_INCHES = 12
DPI = 300

# Calculate pixel dimensions
pixel_width = int(FIG_WIDTH_INCHES * DPI)
pixel_height = int(FIG_HEIGHT_INCHES * DPI)

# Ensure even dimensions
pixel_width += pixel_width % 2
pixel_height += pixel_height % 2

# Recalculate figsize based on even pixel dimensions
FIG_WIDTH_INCHES = pixel_width / DPI
FIG_HEIGHT_INCHES = pixel_height / DPI

def load_params_from_json(subdir_path: str, grid_parameters: List[str]) -> Tuple[float, float]:
    """Load specified parameters from parametrization config."""
    config_path = os.path.join(subdir_path, 'parametrization_config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Missing parametrization_config.json in {subdir_path}")
        return (None, None)
    except json.JSONDecodeError:
        print(f"JSON parsing error in {config_path}")
        return (None, None)
    
    # Extract the specified parameters from config
    parameter_values = []
    for param in grid_parameters:
        # Handle nested keys if necessary
        keys = param.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                try:
                    idx = int(key)
                    value = value[idx]
                except (ValueError, IndexError):
                    print(f"Invalid key '{key}' for list in parameter '{param}' (dir: {subdir_path})")
                    value = None
                    break
            else:
                value = None
                break
        parameter_values.append(value)
    return tuple(parameter_values)

def load_data_config(subdir_path: str) -> Dict:
    """Load the data configuration."""
    config_path = os.path.join(subdir_path, 'data_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def create_single_plot(directories: List[str], output_path: str, signal_strength: float, grid_parameters: List[str]):
    """Create a single plot for a group of directories."""
    # Extract parameter values
    param_pairs = []
    for dir_path in directories:
        params = load_params_from_json(dir_path, grid_parameters)
        if None in params:
            print(f"Skipping {dir_path}: Missing parameter(s) {grid_parameters}")
            continue
        param_pairs.append(params)

    # Get unique sorted values
    param1_values = sorted(set(pair[0] for pair in param_pairs if pair[0] is not None))
    param2_values = sorted(set(pair[1] for pair in param_pairs if pair[1] is not None))

    # Grid sizes
    N = len(param1_values)
    M = len(param2_values)

    if N == 0 or M == 0:
        print("No valid parameter values found.")
        return

    # Create position mapping
    # (This is optional in your code, but let's keep it for clarity.)
    pos_map = {(p1, p2): (i, j) for i, p1 in enumerate(param1_values) for j, p2 in enumerate(param2_values)}

    # Create figure with even dimensions
    fig, axes = plt.subplots(N, M, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
    fig.suptitle(f'Alignment Metrics (signal_strength={signal_strength:.4f})', fontsize=16, y=0.95)

    # Ensure axes is a 2D array
    axes = np.atleast_2d(axes)
    if axes.shape != (N, M):
        axes = axes.reshape(N, M)

    used_layer_indices = set()

    # Initialize all subplots with "No data" placeholders
    for i in range(N):
        for j in range(M):
            ax = axes[i, j]
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_xlabel(grid_parameters[0])
            ax.set_ylabel(grid_parameters[1])
            ax.grid(alpha=0.3)

    # Plot for each directory
    for dir_path in directories:
        # Load parameters
        params = load_params_from_json(dir_path, grid_parameters)
        if None in params:
            continue
        try:
            row = param1_values.index(params[0])
            col = param2_values.index(params[1])
            ax = axes[row, col]
            # Clear the "No data" placeholder if we can plot something
            ax.clear()
        except ValueError:
            print(f"Skipping {dir_path}: Parameter values not found in param_values")
            continue

        als_path = os.path.join(dir_path, 'Als.npy')
        losses_path = os.path.join(dir_path, 'losses.npy')

        # Attempt to load alignment_metrics
        try:
            alignment_metrics = np.load(als_path)
        except FileNotFoundError:
            print(f"Missing Als.npy in {dir_path}. Subplot will remain empty.")
            ax.text(0.5, 0.5, 'Missing Als.npy', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            continue
        except Exception as e:
            print(f"Error loading {als_path}: {str(e)}. Subplot will remain empty.")
            ax.text(0.5, 0.5, 'Error loading Als.npy', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            continue

        # Attempt to load losses
        try:
            losses = np.load(losses_path)
        except FileNotFoundError:
            print(f"Missing losses.npy in {dir_path}. Subplot will remain empty.")
            ax.text(0.5, 0.5, 'Missing losses.npy', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            continue
        except Exception as e:
            print(f"Error loading {losses_path}: {str(e)}. Subplot will remain empty.")
            ax.text(0.5, 0.5, 'Error loading losses.npy', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            continue

        # Now we have both alignment_metrics and losses
        avg_last_20_losses = np.mean(losses[-20:])
        n_steps, n_layers, n_components = alignment_metrics.shape

        # Create plots
        for layer in range(n_layers):
            used_layer_indices.add(layer)
            for comp_idx, comp_name in enumerate(COMPONENT_STYLES.keys()):
                line_style = COMPONENT_STYLES[comp_name]
                ax.plot(
                    alignment_metrics[:, layer, comp_idx],
                    line_style,
                    color=LAYER_COLORS[layer % len(LAYER_COLORS)],
                    alpha=0.7 if comp_idx > 0 else 1.0
                )

        # Configure subplot
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_title(
            f'{grid_parameters[0]}={params[0]:.2f}, {grid_parameters[1]}={params[1]:.2f}\n'
            f'Loss(Last 20)={avg_last_20_losses:.4f}',
            fontsize=8
        )
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.grid(True, alpha=0.3)

    # Create legends
    used_layer_legend_elements = [
        Line2D([0], [0], color=LAYER_COLORS[i % len(LAYER_COLORS)], label=f'Layer {i}')
        for i in sorted(used_layer_indices)
    ]

    component_legend_elements = [
        Line2D([0], [0], color='black', linestyle=COMPONENT_STYLES[comp_name], label=COMPONENT_LABELS[comp_name])
        for comp_name in COMPONENT_STYLES.keys()
    ]

    # Add legends
    fig.legend(
        handles=used_layer_legend_elements + component_legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(used_layer_legend_elements) + len(component_legend_elements),
        title="Layers and Components",
        fontsize=8,
        frameon=True
    )

    # Save plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust rect to make space for legend
    plt.savefig(output_path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def group_directories_by_signal_strength(root_dir: str) -> List[Tuple[float, List[str]]]:
    """Group directories by signal strength and return sorted groups."""
    strength_groups = defaultdict(list)
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        try:
            data_config = load_data_config(full_path)
            signal_strength = float(data_config['signal_strength'])
            strength_groups[signal_strength].append(full_path)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Skipping {subdir}: {str(e)}")
            continue
    
    return sorted(strength_groups.items(), key=lambda x: x[0])

def save_frames(root_dir: str = '/app/maciej/junk/fractal/runs', grid_parameters: List[str] = ['cl.0', 'cl.1']):
    """Save frames grouped by signal strength."""
    print("Starting frame generation process...")
    
    # Create output directory
    output_dir = os.path.join('figures', 'alignment_evolution')
    os.makedirs(output_dir, exist_ok=True)
    
    # Group directories
    sorted_groups = group_directories_by_signal_strength(root_dir)
    
    if not sorted_groups:
        raise ValueError("No valid directories found with signal strength data")
    
    # Generate frames
    for frame_idx, (signal_strength, directories) in enumerate(sorted_groups):
        print(f"Processing signal strength {signal_strength:.4f} ({frame_idx + 1}/{len(sorted_groups)})")
        
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        create_single_plot(directories, frame_path, signal_strength, grid_parameters)
    
    # Create ffmpeg shell script
    create_ffmpeg_script(output_dir, len(sorted_groups))
    
    print(f"Frames saved to {output_dir}")
    print(f"Total frames generated: {len(sorted_groups)}")

def create_ffmpeg_script(output_dir: str, num_frames: int):
    """Create a shell script with the ffmpeg command to generate a video."""
    script_path = os.path.join(output_dir, 'create_video.sh')
    video_output = 'alignment_evolution.mp4'
    
    # FFmpeg command with scaling filter to ensure even dimensions
    ffmpeg_command = (
        f"#!/bin/bash\n\n"
        f"ffmpeg -framerate 1 -i figures/alignment_evolution/frame_%04d.png -vf \"scale=ceil(iw/2)*2:ceil(ih/2)*2\" "
        f"-c:v libx264 -r 30 -pix_fmt yuv420p {video_output}\n"
    )
    
    with open(script_path, 'w') as script_file:
        script_file.write(ffmpeg_command)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"FFmpeg script created at: {script_path}")
    print(f"To generate the video, navigate to {output_dir} and run ./create_video.sh")

if __name__ == "__main__":
    # Set this path as needed
    # ROOT_DIR = '/home/maciej/code/salmon/workloads/cifar-10/runs/runs_no_sched'
    ROOT_DIR = '/home/maciej/code/salmon/workloads/cifar-10/runs/runs_max_sched'
    GRID_PARAMETERS = ['al.2', 'bl.2']  # Modify this list to specify different grid parameters
    save_frames(ROOT_DIR, GRID_PARAMETERS)
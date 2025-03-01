#!/usr/bin/env python3

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any, Union, NamedTuple
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorsys


@dataclass
class MetricConfig:
    """Configuration for a specific metric type"""
    name: str  # Internal name (e.g., 'Als')
    display_name: str  # Display name (e.g., 'Alignment')
    component_names: Optional[List[str]] = None  # Names for multi-dimensional metrics
    component_labels: Optional[Dict[str, str]] = None  # LaTeX or special formatting for components

# Define configurations for known metrics
METRIC_CONFIGS = {
    # Alignment metrics
    'Als': MetricConfig(
        name='Als',
        display_name='Alignment',
        component_names=['Cumulative', 'Alpha', 'Omega', 'U'],
        component_labels={
            'Cumulative': 'A',
            'Alpha': r'$\alpha$',
            'Omega': r'$\omega$',
            'U': r'$\mathcal{u}$'
        }
    ),
    # Loss metrics
    'losses': MetricConfig(
        name='losses',
        display_name='Loss'
    ),
    # Learning rate metrics
    'lrs': MetricConfig(
        name='lrs',
        display_name='Learning Rate'
    )
}

def get_metric_config(metric_name: str) -> MetricConfig:
    """Get metric configuration, falling back to basic config if not found"""
    if metric_name in METRIC_CONFIGS:
        return METRIC_CONFIGS[metric_name]
    return MetricConfig(name=metric_name, display_name=metric_name)


@dataclass
class MetricData:
    """Normalized representation of metric data"""
    data: np.ndarray  # Always 3D: [steps, layers, metrics]
    n_steps: int
    n_layers: int = 1
    n_metrics: int = 1
    metric_names: Optional[List[str]] = None
    
    @staticmethod
    def from_array(arr: np.ndarray, metric_names: Optional[List[str]] = None) -> 'MetricData':
        """Convert any 1D-3D array to standard 3D format"""
        # Add missing dimensions
        if arr.ndim == 1:
            arr = arr[:, np.newaxis, np.newaxis]  # [steps] -> [steps, 1, 1]
        elif arr.ndim == 2:
            if metric_names and len(metric_names) > 1:
                # If multiple metric names provided, treat second dim as metrics
                arr = arr[:, np.newaxis, :]  # [steps, metrics] -> [steps, 1, metrics]
            else:
                arr = arr[:, :, np.newaxis]  # [steps, layers] -> [steps, layers, 1]
            
        n_steps, n_layers, n_metrics = arr.shape
        
        if metric_names and len(metric_names) != n_metrics:
            raise ValueError(f"Number of metric names ({len(metric_names)}) doesn't match data shape ({n_metrics})")
            
        return MetricData(
            data=arr,
            n_steps=n_steps,
            n_layers=n_layers,
            n_metrics=n_metrics,
            metric_names=metric_names
        )

    def smooth(self, alpha: float = 0.1) -> 'MetricData':
        """
        Apply exponential moving average smoothing to the data.
        
        Args:
            alpha (float): Smoothing factor between 0 and 1.
                Lower values mean more smoothing.
        
        Returns:
            MetricData: New MetricData object with smoothed data
        """
        smoothed_data = np.zeros_like(self.data)
        
        # Apply EMA smoothing to each layer and metric
        for l in range(self.n_layers):
            for m in range(self.n_metrics):
                data = self.data[:, l, m]
                smoothed = data.copy()
                for i in range(1, len(data)):
                    smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
                smoothed_data[:, l, m] = smoothed
        
        return MetricData(
            data=smoothed_data,
            n_steps=self.n_steps,
            n_layers=self.n_layers,
            n_metrics=self.n_metrics,
            metric_names=self.metric_names
        )


class GroupKey:
    """Structured key for grouping runs with metadata"""
    def __init__(self, features: Dict[str, Any]):
        self.features = features
        # Convert to frozenset for hashing
        self._hashable = frozenset(features.items())
    
    def __hash__(self) -> int:
        return hash(self._hashable)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupKey):
            return NotImplemented
        return self._hashable == other._hashable
    
    def __str__(self) -> str:
        """Format the group key for display"""
        return ", ".join(f"{k}={v}" for k, v in self.features.items())

class AggregationResult(NamedTuple):
    """Result of aggregating runs with metadata"""
    run: 'RunData'
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        """Format the aggregation result for display"""
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in self.metadata.items())

@dataclass
class RunData:
    """Container for data from a single training run"""
    path: str
    model_config: Dict
    optimizer_config: Dict
    lr_scheduler_config: Dict
    parametrization_config: Dict
    data_config: Dict
    training_config: Dict
    metrics: Dict[str, MetricData]  # Normalized metrics
    
    @classmethod
    def from_directory(cls, path: str) -> 'RunData':
        """Load run data from directory"""
        # Load configs
        configs = {}
        for name in ['model', 'optimizer', 'lr_scheduler', 'parametrization', 'data', 'training']:
            config_path = os.path.join(path, f"{name}_config.json")
            try:
                with open(config_path, 'r') as f:
                    configs[f"{name}_config"] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                configs[f"{name}_config"] = {}
        
        # Load metrics
        metrics = {}
        for file in os.listdir(path):
            if file.endswith('.npy'):
                metric_name = file[:-4]
                try:
                    data = np.load(os.path.join(path, file))
                    metrics[metric_name] = MetricData.from_array(data)
                except:
                    continue
        
        return cls(
            path=path,
            metrics=metrics,
            **configs
        )

class RunCollector:
    """Collects and filters training runs"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.runs: List[RunData] = []
        self._collect_runs()
    
    def _collect_runs(self):
        """Recursively collect all runs from base_dir"""
        for root, _, files in os.walk(self.base_dir):
            if all(f"{x}_config.json" in files for x in ['model', 'optimizer', 'parametrization', 'data', 'training']):
                try:
                    run_data = RunData.from_directory(root)
                    self.runs.append(run_data)
                except Exception as e:
                    print(f"Error loading run from {root}: {str(e)}")
                    continue

    def filter(self, condition: Callable[[RunData], bool]) -> 'RunCollector':
        """Filter runs based on a condition"""
        new_collector = RunCollector(self.base_dir)
        new_collector.runs = [run for run in self.runs if condition(run)]
        return new_collector
    
    def group_by(self, key_fn: Callable[[RunData], Dict[str, Any]]) -> Dict[GroupKey, List[RunData]]:
        """Group runs by a key function that returns a dict of features"""
        groups = defaultdict(list)
        for run in self.runs:
            features = key_fn(run)
            groups[GroupKey(features=features)].append(run)
        return dict(groups)

class MetricAggregator:
    """Aggregates metrics across multiple runs"""
    
    # @staticmethod
    # def best_by_final_loss(runs: List[RunData], window: int = 10) -> AggregationResult:
    #     """Select run with best (lowest) average loss over last window steps"""
    #     def score(run: RunData) -> float:
    #         if 'losses' not in run.metrics:
    #             return float('inf')
    #         losses = run.metrics['losses'].data[:, 0, 0]  # Get 1D array
    #         return float(np.mean(losses[-window:]))
        
    #     best_run = min(runs, key=score)
    #     best_score = score(best_run)

    #     return AggregationResult(
    #         run=best_run,
    #         metadata={
    #             'final_loss': best_score,
    #             'window_size': window
    #         }
    #     )

    @staticmethod
    def best_by_final_loss(runs: List[RunData], window: int = 10) -> AggregationResult:
        """Select run with best (lowest) average loss over last window steps"""
        def score(run: RunData) -> float:
            if 'losses' not in run.metrics:
                return float('inf')
            losses = run.metrics['losses'].data[:, 0, 0]  # Get 1D array
            return float(np.mean(losses[-window:]))
        
        best_run = min(runs, key=score)
        best_score = score(best_run)

        return AggregationResult(
            run=best_run,
            metadata={}
        )



class StyleManager:
    """Manages colors, patterns, and other visual elements"""
    
    def __init__(self):
        # Base colors for different runs
        self.base_colors = [
            '#e41a1c',  # Red
            '#377eb8',  # Blue
            '#4daf4a',  # Green
            '#984ea3',  # Purple
            '#ff7f00',  # Orange
        ]
        
        # Line styles for different metrics
        self.metric_styles = ['-', '--', ':', '-.']
        
        # Alpha values for different layers
        self.min_alpha = 0.3
        self.max_alpha = 1.0
    
    def get_run_color(self, run_idx: int, layer_idx: int = 0, n_layers: int = 1) -> Tuple[float, float, float, float]:
        """Get color for a run, adjusted for layer if needed"""
        base_color = self.base_colors[run_idx % len(self.base_colors)]
        
        # Convert hex to RGB
        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
        hsv = colorsys.rgb_to_hsv(*rgb)
        
        # Adjust value (brightness) based on layer
        if n_layers > 1:
            v_range = 0.5  # How much to vary the value
            v_min = max(0, hsv[2] - v_range/2)
            v_max = min(1, hsv[2] + v_range/2)
            value = v_min + (v_max - v_min) * (layer_idx / (n_layers - 1))
            
            # Convert back to RGB
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], value)
        
        # Add alpha
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (layer_idx / max(1, n_layers - 1))
        return (*rgb, alpha)
    
    def get_metric_style(self, metric_idx: int) -> str:
        """Get line style for a metric"""
        return self.metric_styles[metric_idx % len(self.metric_styles)]
    
    def get_layer_colors(self, n_layers: int) -> List[Tuple[float, float, float, float]]:
        """Get grayscale colors for layer legend"""
        colors = []
        for i in range(n_layers):
            # Create grayscale colors from dark to light
            value = self.min_alpha + (self.max_alpha - self.min_alpha) * (i / max(1, n_layers - 1))
            colors.append((value, value, value, 1.0))
        return colors



class GridVisualizer:
    """Enhanced grid visualization system with separated legends"""
    
    def __init__(self):
        self.fig_size_per_subplot = (4, 3)  # width, height in inches per subplot
        self.style_manager = StyleManager()
        self.legend_spacing = 0.08  # Vertical spacing between legends
        
    def create_grid(self, groups: Dict[GroupKey, List[AggregationResult]]) -> Tuple[plt.Figure, np.ndarray, List[str], List[Any], List[Any]]:
        """Create a grid based on group features"""
        features = list(next(iter(groups.keys())).features.keys())
        grid_features = features[:2]
        
        x_feature, y_feature = grid_features
        x_values = sorted({group.features[x_feature] for group in groups.keys()})
        y_values = sorted({group.features[y_feature] for group in groups.keys()})
        
        fig_width = len(x_values) * self.fig_size_per_subplot[0]
        fig_height = len(y_values) * self.fig_size_per_subplot[1]
        
        # Add extra height for multiple legends
        fig_height += 1.5  # Space for legends
        
        fig, axes = plt.subplots(len(y_values), len(x_values), 
                               figsize=(fig_width, fig_height),
                               squeeze=False)
        
        for ax in axes.flat:
            ax.grid(alpha=0.3)
            ax.set_xlabel('Steps')
        
        return fig, axes, grid_features, x_values, y_values
    
    def create_separated_legends(self, 
                               fig: plt.Figure, 
                               run_elements: List[Line2D],
                               layer_elements: List[Line2D],
                               metric_elements: List[Line2D]):
        """Create separate legends arranged horizontally at the bottom"""
        n_legends = sum([bool(run_elements), bool(layer_elements), bool(metric_elements)])
        if n_legends == 0:
            return

        # Calculate positions for each legend
        legend_width = 1.0 / max(n_legends, 1)  # Divide available width
        current_x = 0.0  # Start from left
        bottom_y = 0.02  # Fixed bottom position

        def add_legend(elements: List[Line2D], title: str, x_pos: float) -> None:
            if not elements:
                return
            # Make columns more compact
            ncol = min(3, len(elements))
            legend = fig.legend(handles=elements, 
                              title=title,
                              loc='lower center',
                              bbox_to_anchor=(x_pos + legend_width/2, bottom_y),
                              ncol=ncol,
                              fontsize=7,  # Smaller font
                              title_fontsize=8,  # Smaller title
                              columnspacing=1.0,  # Reduce space between columns
                              handlelength=1.5)  # Shorter lines in legend
            fig.add_artist(legend)

        # Add legends side by side
        legends_to_add = [(run_elements, "Runs"), 
                         (layer_elements, "Layers"), 
                         (metric_elements, "Metrics")]
        
        for elements, title in legends_to_add:
            if elements:
                add_legend(elements, title, current_x)
                current_x += legend_width

        # Adjust layout to account for legends
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Fixed space at bottom for legends

    def plot_grid(self,
                 named_groups: Dict[str, Dict[GroupKey, AggregationResult]],
                 metric_name: str,
                 title: Optional[str] = None,
                 subsample_config: Dict[str, List[int]] = None,
                 smoothing_alpha: Optional[float] = None) -> plt.Figure:
        """Plot a metric with named groups and separated legends"""
        # Get metric configuration
        metric_config = get_metric_config(metric_name)
        # Use metric display name if no title provided
        if title is None:
            title = metric_config.display_name
            
        # Get all unique group keys
        all_keys = set()
        for groups in named_groups.values():
            all_keys.update(groups.keys())
            
        # Create grid based on first group's structure
        sample_key = next(iter(all_keys))
        grid_features = list(sample_key.features.keys())[:2]
        x_feature, y_feature = grid_features
        
        x_values = sorted({key.features[x_feature] for key in all_keys})
        y_values = sorted({key.features[y_feature] for key in all_keys})
        
        fig, axes = plt.subplots(len(y_values), len(x_values),
                                figsize=(len(x_values) * self.fig_size_per_subplot[0],
                                        len(y_values) * self.fig_size_per_subplot[1] + 1.5),
                                squeeze=False)
        
        fig.suptitle(title, y=0.98)
        
        # Initialize legend elements
        run_elements = []
        layer_elements = []
        metric_elements = []
        
        for group_key in all_keys:
            i = y_values.index(group_key.features[y_feature])
            j = x_values.index(group_key.features[x_feature])
            ax = axes[i, j]
            ax.grid(alpha=0.3)
            
            # Plot each named group's data
            for run_idx, (run_name, groups) in enumerate(named_groups.items()):
                if group_key not in groups:
                    continue
                    
                agg_result = groups[group_key]
                if metric_name not in agg_result.run.metrics:
                    continue
                
                metric_data = agg_result.run.metrics[metric_name]
                # Inside the plotting loop where metric_data is used:
                if smoothing_alpha is not None:
                    metric_data = metric_data.smooth(alpha=smoothing_alpha)

                
                # Add run to legend if first time seeing it
                if run_idx >= len(run_elements):
                    # base_color = self.style_manager.get_run_color(run_idx, 0, 1)
                    # run_elements.append(
                    #     Line2D([0], [0], color=base_color, label=run_name)
                    # )

                    base_color = self.style_manager.get_run_color(run_idx, 0, 1)
                    # Get lr from the run's optimizer config

                    # TODO: make this configurable
                    lr = agg_result.run.optimizer_config['params']['lr']
                    lr_str = f" (lr={lr})" if lr is not None else ""
                    run_elements.append(
                        Line2D([0], [0], color=base_color, label=f"{run_name}{lr_str}")
                    )
                
                # Add layer legend elements if needed and not already added
                if metric_data.n_layers > 1 and not layer_elements:
                    for layer_idx in range(metric_data.n_layers):
                        if subsample_config is not None and "layers" in subsample_config:
                            if layer_idx not in subsample_config["layers"]:
                                continue
                        color = self.style_manager.get_layer_colors(metric_data.n_layers)[layer_idx]
                        layer_elements.append(
                            Line2D([0], [0], color=color, label=f"Layer {layer_idx}")
                        )
                
                # Add metric legend elements if needed and not already added
                if metric_data.n_metrics > 1 and not metric_elements:
                    config = get_metric_config(metric_name)
                    for metric_idx in range(metric_data.n_metrics):
                        if subsample_config is not None and "metrics" in subsample_config:
                            if metric_idx not in subsample_config["metrics"]:
                                continue
                        style = self.style_manager.get_metric_style(metric_idx)
                        # Use component names/labels from config if available
                        if config.component_names and metric_idx < len(config.component_names):
                            component_name = config.component_names[metric_idx]
                            label = (config.component_labels.get(component_name, component_name) 
                                   if config.component_labels 
                                   else component_name)
                        else:
                            label = (metric_data.metric_names[metric_idx] 
                                   if metric_data.metric_names 
                                   else f"Metric {metric_idx}")
                        metric_elements.append(
                            Line2D([0], [0], color='black', linestyle=style, label=label)
                        )
                
                # Plot the actual data
                for layer_idx in range(metric_data.n_layers):
                    if subsample_config is not None and "layers" in subsample_config:
                        if layer_idx not in subsample_config["layers"]:
                            continue
                    for metric_idx in range(metric_data.n_metrics):
                        if subsample_config is not None and "metrics" in subsample_config:
                            if metric_idx not in subsample_config["metrics"]:
                                continue
                        color = self.style_manager.get_run_color(
                            run_idx, layer_idx, metric_data.n_layers
                        )
                        style = self.style_manager.get_metric_style(metric_idx)
                        data = metric_data.data[:, layer_idx, metric_idx]
                        ax.plot(data, style, color=color)
            
            # Subplot title
            title_lines = [str(group_key)]
            if hasattr(agg_result, 'metadata'):
                metadata_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in agg_result.metadata.items())
                title_lines.append(metadata_str)
            ax.set_title('\n'.join(title_lines), fontsize=8)
        
        # Create separated legends
        self.create_separated_legends(fig, run_elements, layer_elements, metric_elements)
            
        return fig



def main():
    # Collect runs from both experiments
    prefix = "sgd_relu_align_warmup_"

    exps = {}

    # sgd_exp_dir = "/home/maciej/code/paramR/runs/mup_sgd_lw_cifar_grid_lr_schedule_ablation"
    # sgd_exp_collector = RunCollector(sgd_exp_dir)
    # sgd_exp_collector = sgd_exp_collector.filter(lambda run: (run.model_config["params"]["dims"][1] >= 1024) & (len(run.model_config["params"]["dims"]) >= 7 + 1))

    # sgd_constant_collector = sgd_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'constant_lr_scheduler')
    # sgd_max_collector = sgd_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'maximal_lr_scheduler')
    # exps["sgd_const"] = sgd_constant_collector
    # exps["sgd_max"] = sgd_max_collector

    # adamw_exp_dir = "/home/maciej/code/paramR/runs/mup_adamw_lw_cifar_grid_lr_schedule_ablation"
    # adamw_exp_collector = RunCollector(adamw_exp_dir)
    # adamw_exp_collector = adamw_exp_collector.filter(lambda run: (run.model_config["params"]["dims"][1] >= 1024) & (len(run.model_config["params"]["dims"]) >= 7 + 1))

    # adamw_constant_collector = adamw_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'constant_lr_scheduler')
    # adamw_max_collector = adamw_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'maximal_lr_scheduler')
    # exps["adamw_const"] = adamw_constant_collector
    # exps["adamw_max"] = adamw_max_collector

    # adamw_relu_exp_dir = "/home/maciej/code/paramR/runs/mup_adamw_lw_cifar_grid_lr_schedule_ablation_relu"
    # adamw_relu_exp_collector = RunCollector(adamw_relu_exp_dir)
    # # adamw_relu_exp_collector = adamw_relu_exp_collector.filter(lambda run: (run.model_config["params"]["dims"][1] >= 1024) & (len(run.model_config["params"]["dims"]) >= 7 + 1))

    # adamw_relu_constant_collector = adamw_relu_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'constant_lr_scheduler')
    # adamw_relu_max_collector = adamw_relu_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'maximal_lr_scheduler')
    # exps["adamw_relu_const"] = adamw_relu_constant_collector
    # exps["adamw_relu_max"] = adamw_relu_max_collector

    # adamw_relu_align_warmup_exp_dir = "/home/maciej/code/paramR/runs/mup_adamw_lw_cifar_grid_lr_schedule_ablation_relu_w_alignment_warmup"
    # adamw_relu_align_warmup_exp_collector = RunCollector(adamw_relu_align_warmup_exp_dir)
    # # adamw_relu_align_warmup_exp_collector = adamw_relu_align_warmup_exp_collector.filter(lambda run: (run.model_config["params"]["dims"][1] >= 1024) & (len(run.model_config["params"]["dims"]) >= 7 + 1))

    # adamw_relu_align_warmup_constant_collector = adamw_relu_align_warmup_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'constant_lr_scheduler')
    # adamw_relu_align_warmup_max_collector = adamw_relu_align_warmup_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'maximal_lr_scheduler')
    # exps["adamw_relu_align_warmup_const"] = adamw_relu_align_warmup_constant_collector
    # exps["adamw_relu_align_warmup_max"] = adamw_relu_align_warmup_max_collector

    sgd_relu_align_warmup_exp_dir = "/home/maciej/code/paramR/runs/mup_sgd_lw_cifar_grid_lr_schedule_ablation_relu_w_alignment_warmup"
    sgd_relu_align_warmup_exp_collector = RunCollector(sgd_relu_align_warmup_exp_dir)
    # sgd_relu_align_warmup_exp_collector = sgd_relu_align_warmup_exp_collector.filter(lambda run: (run.model_config["params"]["dims"][1] >= 1024) & (len(run.model_config["params"]["dims"]) >= 7 + 1))

    sgd_relu_align_warmup_constant_collector = sgd_relu_align_warmup_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'constant_lr_scheduler')
    sgd_relu_align_warmup_max_collector = sgd_relu_align_warmup_exp_collector.filter(lambda run: run.lr_scheduler_config['obj'] == 'maximal_lr_scheduler')
    exps["sgd_relu_align_warmup_const"] = sgd_relu_align_warmup_constant_collector
    exps["sgd_relu_align_warmup_max"] = sgd_relu_align_warmup_max_collector

    # Load and group runs
    def group_by_architecture(run: RunData) -> Dict[str, Any]:
        dims = run.model_config["params"]["dims"]
        return {
            "depth": len(dims) - 1,
            "width": dims[1]
        }

    for exp_name, exp in exps.items():
        exp_grouped = exp.group_by(group_by_architecture)
        exp_agg = {
            key: MetricAggregator.best_by_final_loss(runs)
            for key, runs in exp_grouped.items()
        }
        exps[exp_name] = exp_agg

    combined_groups = exps

    # Create visualization
    viz = GridVisualizer()
    
    # Plot losses (automatically handles multiple runs)
    fig1 = viz.plot_grid(
        combined_groups, 
        "losses", 
        "Loss Comparison",
        smoothing_alpha=0.1,
    )
    fig1.savefig(f"{prefix}losses.png", bbox_inches='tight', dpi=300)

    # Plot losses (automatically handles multiple runs)
    fig1 = viz.plot_grid(
        combined_groups, 
        "rLs", 
        "Feature Learning Residuals",
    )
    fig1.savefig(f"{prefix}rLs.png", bbox_inches='tight', dpi=300)

    # Plot learning rates (automatically handles layers)
    fig2 = viz.plot_grid(
        combined_groups, 
        "lrs", 
        "Learning Rate Schedules",
        subsample_config={"layers": [0, 3, 6, 8]},
    )
    for ax in fig2.axes:
        ax.set_yscale('log')
        ax.set_ylim(0, 1e1)
    fig2.savefig(f"{prefix}learning_rates.png", bbox_inches='tight', dpi=300)
    
    # Plot alignment (automatically handles layers and metrics)
    for al_idx in range(4):
        fig3 = viz.plot_grid(
            combined_groups, 
            "Als", 
            "Alignment Analysis", 
            subsample_config={"metrics": [al_idx], "layers": [0, 3, 6, 8]},
            smoothing_alpha=0.1,
        )
        for ax in fig3.axes:
            ax.set_ylim(0.4, 1.0)
        al_name = METRIC_CONFIGS['Als'].component_names[al_idx]
        fig3.savefig(f"{prefix}alignment_{al_name}.png", bbox_inches='tight', dpi=300)
    
    plt.close('all')


if __name__ == "__main__":
    main()
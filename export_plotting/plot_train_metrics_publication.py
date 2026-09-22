# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
from collections import defaultdict
import warnings
import re
warnings.filterwarnings('ignore')
# ===== Third-Party Imports =====
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# ===== Own Modules =====
from settings import setting


class TensorBoardPlotter:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the TensorBoard plotter.
    # Args:
    #   logdir (Path | str): Path to TensorBoard logs directory
    #   prob_dir (Path | str | None): Path to directory containing probability .npz files
    #   output_folder (Path | str | None): Folder for output plots (defaults to settings['pth_output'])
    #   output_format (str): 'png', 'tiff', 'svg', 'pdf'
    #   raster_dpi (int): DPI for raster formats
    #   mode (str): 'auto', 'crossval', or 'single'
    #   roc_curve_epoch (str | int): Which epoch to use for ROC curves
    #   pr_curve_epoch (str | int): Which epoch to use for PR curves
    #   checkpoint_map (dict | None): Map of run names to epoch numbers (from Confidence Analyzer)
    #   train_line_style (str): Line style for training curves
    #   val_line_style (str): Line style for validation curves
    #   standard_line_style (str): Line style for standard (unweighted) accuracy curves
    #   line_width (float): Width of plot lines
    #   show_markers (bool): Show markers at data points
    #   marker_size (int): Size of markers
    #   marker_frequency (int): Show every Nth marker
    #   smoothing (float): Smoothing factor for curves
    #   train_color (str): Color for training curves
    #   val_color (str): Color for validation curves
    #   standard_train_color (str): Color for standard training curves
    #   standard_val_color (str): Color for standard validation curves
    #   lr_color (str): Color for learning rate curve
    #   f1_weighted_color (str): Color for weighted F1 curve
    #   f1_macro_color (str): Color for macro F1 curve
    #   auc_color (str): Color for AUC curve
    #   ap_color (str): Color for Average Precision curve
    #   composite_score_color (str): Color for composite score curve
    #   balanced_accuracy_color (str): Color for balanced accuracy curve
    #   class_std_color (str): Color for class accuracy std dev curve
    #   min_class_acc_color (str): Color for minimum class accuracy curve
    #   gpu_memory_color (str): Color for GPU memory curve
    #   per_class_colormap (str): Colormap for variable-line plots
    #   class_weights_colormap (str): Colormap for class weights bar chart
    #   class_counts_colormap (str): Colormap for class counts bar chart
    #   class_weights_colors (dict | None): Per-class colors for weights bar chart
    #   class_counts_colors (dict | None): Per-class colors for counts bar chart
    #   bar_edge_width (float): Border thickness for bars
    #   bar_label_fontsize (int): Font size for bar labels
    #   bar_label_offset (float): Offset for bar labels as fraction of max value
    #   use_fixed_axes_height (bool): Consistent axes height across plots
    #   fixed_axes_height (float): Height of axes in inches
    #   font_family (str): Font family
    #   master_font_size (int | None): Uniform font size for all text
    #   axis_label_size (int): Font size for axis labels
    #   title_font_size (int): Font size for plot titles
    #   tick_label_size (int): Font size for tick labels
    #   legend_font_size (int): Font size for legend text
    #   loss_legend_loc (str): Legend position for loss curves
    #   loss_legend_outside (bool): Place loss legend outside
    #   accuracy_legend_loc (str): Legend position for accuracy curves
    #   accuracy_legend_outside (bool): Place accuracy legend outside
    #   f1_legend_loc (str): Legend position for F1 curves
    #   f1_legend_outside (bool): Place F1 legend outside
    #   per_class_legend_loc (str): Legend position for per-class accuracy
    #   per_class_legend_outside (bool): Place per-class legend outside
    #   roc_legend_loc (str): Legend position for ROC curves
    #   roc_legend_outside (bool): Place ROC legend outside
    #   pr_legend_loc (str): Legend position for PR curves
    #   pr_legend_outside (bool): Place PR legend outside
    #   class_std_legend_loc (str): Legend position for class std curve
    #   class_std_legend_outside (bool): Place class std legend outside
    #   min_class_acc_legend_loc (str): Legend position for min class accuracy
    #   min_class_acc_legend_outside (bool): Place min class accuracy legend outside
    #   composite_legend_loc (str): Legend position for composite score
    #   composite_legend_outside (bool): Place composite score legend outside
    #   balanced_acc_legend_loc (str): Legend position for balanced accuracy
    #   balanced_acc_legend_outside (bool): Place balanced accuracy legend outside
    #   gpu_memory_legend_loc (str): Legend position for GPU memory
    #   gpu_memory_legend_outside (bool): Place GPU memory legend outside
    #   lr_legend_loc (str): Legend position for learning rate
    #   lr_legend_outside (bool): Place learning rate legend outside
    #   auc_legend_loc (str): Legend position for AUC
    #   auc_legend_outside (bool): Place AUC legend outside
    #   ap_legend_loc (str): Legend position for Average Precision
    #   ap_legend_outside (bool): Place Average Precision legend outside
    #   legend_outside_bbox (tuple): Bbox for outside legend
    #   show_legend (bool): Show/hide legend
    #   show_grid (bool): Show background grid
    #   grid_alpha (float): Grid line transparency
    #   grid_linestyle (str): Grid line style
    #   grid_color (str): Grid line color
    #   x_axis_min (float): Minimum value for x-axis
    #   y_axis_min (float): Minimum value for y-axis
    #   min_class_acc_threshold (float): Threshold for minimum class accuracy
    #   plot_loss (bool): Generate loss plots
    #   plot_accuracy (bool): Generate accuracy plots
    #   plot_f1 (bool): Generate F1 plots
    #   plot_lr (bool): Generate learning rate plots
    #   plot_auc (bool): Generate AUC plots
    #   plot_ap (bool): Generate Average Precision plots
    #   plot_per_class_accuracy (bool): Generate per-class accuracy plots
    #   plot_class_weights (bool): Generate class weights bar charts
    #   plot_class_counts (bool): Generate class counts bar charts
    #   plot_composite_score (bool): Generate composite score plots
    #   plot_balanced_accuracy (bool): Generate balanced accuracy plots
    #   plot_class_std (bool): Generate class accuracy std dev plots
    #   plot_min_class_acc (bool): Generate minimum class accuracy plots
    #   plot_gpu_memory (bool): Generate GPU memory usage plots
    #   plot_roc_curves (bool): Generate ROC curve plots
    #   plot_pr_curves (bool): Generate PR curve plots
    def __init__(
        self,
        logdir=None,
        prob_dir=None,
        output_folder=None,
        output_format='tiff',
        raster_dpi=600,
        mode='auto',
        roc_curve_epoch='composite_score',
        pr_curve_epoch='composite_score',
        checkpoint_map=None,
        train_line_style='-',
        val_line_style='-',
        standard_line_style='--',
        line_width=2.0,
        show_markers=False,
        marker_size=4,
        marker_frequency=1,
        smoothing=0.95,
        train_color='#2E86AB',
        val_color='#A23B72',
        standard_train_color='#2E86AB',
        standard_val_color='#A23B72',
        lr_color='#2E86AB',
        f1_weighted_color='#2E86AB',
        f1_macro_color='#F39C12',
        auc_color='#27AE60',
        ap_color='#8E44AD',
        composite_score_color='#E74C3C',
        balanced_accuracy_color='#1ABC9C',
        class_std_color='#E67E22',
        min_class_acc_color='#1ABC9C',
        gpu_memory_color='#95A5A6',
        per_class_colormap='tab10',
        class_weights_colormap='viridis',
        class_counts_colormap='plasma',
        class_weights_colors=None,
        class_counts_colors=None,
        bar_edge_width=2.0,
        bar_label_fontsize=10,
        bar_label_offset=0.05,
        use_fixed_axes_height=True,
        fixed_axes_height=5.0,
        font_family='Arial',
        master_font_size=None,
        axis_label_size=12,
        title_font_size=14,
        tick_label_size=10,
        legend_font_size=10,
        loss_legend_loc='upper right',
        loss_legend_outside=False,
        accuracy_legend_loc='lower right',
        accuracy_legend_outside=False,
        f1_legend_loc='lower right',
        f1_legend_outside=False,
        per_class_legend_loc='best',
        per_class_legend_outside=False,
        roc_legend_loc='lower right',
        roc_legend_outside=False,
        pr_legend_loc='lower left',
        pr_legend_outside=False,
        class_std_legend_loc='upper right',
        class_std_legend_outside=False,
        min_class_acc_legend_loc='lower right',
        min_class_acc_legend_outside=False,
        composite_legend_loc='lower right',
        composite_legend_outside=False,
        balanced_acc_legend_loc='lower right',
        balanced_acc_legend_outside=False,
        gpu_memory_legend_loc='best',
        gpu_memory_legend_outside=False,
        lr_legend_loc='best',
        lr_legend_outside=False,
        auc_legend_loc='lower right',
        auc_legend_outside=False,
        ap_legend_loc='lower right',
        ap_legend_outside=False,
        legend_outside_bbox=(1.05, 0.5),
        show_legend=True,
        show_grid=True,
        grid_alpha=0.3,
        grid_linestyle='--',
        grid_color='gray',
        x_axis_min=0,
        y_axis_min=0,
        min_class_acc_threshold=0.60,
        plot_loss=True,
        plot_accuracy=True,
        plot_f1=True,
        plot_lr=True,
        plot_auc=True,
        plot_ap=True,
        plot_per_class_accuracy=True,
        plot_class_weights=True,
        plot_class_counts=True,
        plot_composite_score=True,
        plot_balanced_accuracy=True,
        plot_class_std=True,
        plot_min_class_acc=True,
        plot_gpu_memory=True,
        plot_roc_curves=True,
        plot_pr_curves=True,
    ) -> None:

        # Cache best epoch
        self._best_epoch_cache: dict = {}

        # Input/Output settings
        if logdir is None:
            raise ValueError("logdir must be specified")
        self.logdir = Path(logdir)

        # Probability directory - auto-determine if not specified
        if prob_dir is None:
            if (self.logdir / "probabilities").exists():
                self.prob_dir = self.logdir / "probabilities"
            else:
                self.prob_dir = self.logdir.parent / "probabilities" if self.logdir.parent != self.logdir else self.logdir
        else:
            self.prob_dir = Path(prob_dir)

        if output_folder is None:
            output_folder = setting.get('pth_output', 'output')
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_format = output_format.lower()
        self.raster_dpi = raster_dpi
        self.use_fixed_axes_height = use_fixed_axes_height
        self.fixed_axes_height = fixed_axes_height

        # Mode settings
        self.mode = mode
        self.run_type = None

        # Plot flags
        self.plot_loss = plot_loss
        self.plot_accuracy = plot_accuracy
        self.plot_f1 = plot_f1
        self.plot_lr = plot_lr
        self.plot_auc = plot_auc
        self.plot_ap = plot_ap
        self.plot_per_class_accuracy = plot_per_class_accuracy
        self.plot_class_weights = plot_class_weights
        self.plot_class_counts = plot_class_counts
        self.plot_composite_score = plot_composite_score
        self.plot_balanced_accuracy = plot_balanced_accuracy
        self.plot_class_std = plot_class_std
        self.plot_min_class_acc = plot_min_class_acc
        self.plot_gpu_memory = plot_gpu_memory
        self.plot_roc_curves = plot_roc_curves
        self.plot_pr_curves = plot_pr_curves

        # ROC/PR options
        self.roc_curve_epoch = roc_curve_epoch
        self.pr_curve_epoch = pr_curve_epoch

        # Checkpoint map
        self.checkpoint_map = checkpoint_map or {}

        # Line styling
        self.train_line_style = train_line_style
        self.val_line_style = val_line_style
        self.standard_line_style = standard_line_style
        self.line_width = line_width
        self.show_markers = show_markers
        self.marker_size = marker_size
        self.marker_frequency = marker_frequency
        self.smoothing = smoothing

        # Colors for fixed-line plots
        self.train_color = train_color
        self.val_color = val_color
        self.standard_train_color = standard_train_color
        self.standard_val_color = standard_val_color
        self.lr_color = lr_color
        self.f1_weighted_color = f1_weighted_color
        self.f1_macro_color = f1_macro_color
        self.auc_color = auc_color
        self.ap_color = ap_color
        self.composite_score_color = composite_score_color
        self.balanced_accuracy_color = balanced_accuracy_color
        self.class_std_color = class_std_color
        self.min_class_acc_color = min_class_acc_color
        self.gpu_memory_color = gpu_memory_color

        # Colormap for variable-line plots
        self.per_class_colormap = per_class_colormap

        # Bar chart settings
        self.class_weights_colormap = class_weights_colormap
        self.class_counts_colormap = class_counts_colormap
        self.class_weights_colors = class_weights_colors
        self.class_counts_colors = class_counts_colors
        self.bar_edge_width = bar_edge_width
        self.bar_label_fontsize = bar_label_fontsize
        self.bar_label_offset = bar_label_offset

        # Font settings
        self.font_family = font_family
        self.master_font_size = master_font_size

        if master_font_size is not None:
            self.axis_label_size = master_font_size
            self.title_font_size = master_font_size
            self.tick_label_size = master_font_size
            self.legend_font_size = master_font_size
        else:
            self.axis_label_size = axis_label_size
            self.title_font_size = title_font_size
            self.tick_label_size = tick_label_size
            self.legend_font_size = legend_font_size

        # Legend position settings
        self.loss_legend_loc = loss_legend_loc
        self.loss_legend_outside = loss_legend_outside
        self.accuracy_legend_loc = accuracy_legend_loc
        self.accuracy_legend_outside = accuracy_legend_outside
        self.f1_legend_loc = f1_legend_loc
        self.f1_legend_outside = f1_legend_outside
        self.per_class_legend_loc = per_class_legend_loc
        self.per_class_legend_outside = per_class_legend_outside
        self.roc_legend_loc = roc_legend_loc
        self.roc_legend_outside = roc_legend_outside
        self.pr_legend_loc = pr_legend_loc
        self.pr_legend_outside = pr_legend_outside
        self.class_std_legend_loc = class_std_legend_loc
        self.class_std_legend_outside = class_std_legend_outside
        self.min_class_acc_legend_loc = min_class_acc_legend_loc
        self.min_class_acc_legend_outside = min_class_acc_legend_outside
        self.composite_legend_loc = composite_legend_loc
        self.composite_legend_outside = composite_legend_outside
        self.balanced_acc_legend_loc = balanced_acc_legend_loc
        self.balanced_acc_legend_outside = balanced_acc_legend_outside
        self.gpu_memory_legend_loc = gpu_memory_legend_loc
        self.gpu_memory_legend_outside = gpu_memory_legend_outside
        self.lr_legend_loc = lr_legend_loc
        self.lr_legend_outside = lr_legend_outside
        self.auc_legend_loc = auc_legend_loc
        self.auc_legend_outside = auc_legend_outside
        self.ap_legend_loc = ap_legend_loc
        self.ap_legend_outside = ap_legend_outside
        self.legend_outside_bbox = legend_outside_bbox

        # Other options
        self.show_legend = show_legend
        self.show_grid = show_grid
        self.grid_alpha = grid_alpha
        self.grid_linestyle = grid_linestyle
        self.grid_color = grid_color
        self.min_class_acc_threshold = min_class_acc_threshold

        # Axis limits
        self.x_axis_min = x_axis_min
        self.y_axis_min = y_axis_min

        # Data storage
        self.runs_data = {}
        self.per_class_data = {}
        self.class_weight_data = {}
        self.class_count_data = {}
        self.class_names = None
        self.roc_data = {}
        self.pr_data = {}

        self._setup_plot_style()
        self._print_configuration()

    #############################################################################################################
    # METHODS

    # Set publication-ready matplotlib defaults.
    def _setup_plot_style(self) -> None:
        light_gray = '#C0C0C0'

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [self.font_family, 'Arial', 'Helvetica'],
            'font.size': self.tick_label_size,
            'axes.labelsize': self.axis_label_size,
            'axes.titlesize': self.title_font_size,
            'axes.titleweight': 'bold',
            'xtick.labelsize': self.tick_label_size,
            'ytick.labelsize': self.tick_label_size,
            'figure.dpi': self.raster_dpi,
            'savefig.dpi': self.raster_dpi,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.color': light_gray,
            'grid.linestyle': self.grid_linestyle,
            'grid.alpha': 1.0,
        })
        self.grid_color = light_gray
        self.grid_alpha = 1.0

    # Print current configuration.
    def _print_configuration(self) -> None:
        print("=" * 60)
        print("COMPLETE TENSORBOARD PUBLICATION PLOTTER - MULTI-RUN VERSION")
        print("NOTE: All epochs in plots start at 1 (shifted from internal 0-index)")
        print("NOTE: All axes start at 0 (x_min=0, y_min=0)")
        if self.use_fixed_axes_height:
            print(f"NOTE: Fixed axes height = {self.fixed_axes_height} inches (all plots have identical axes dimensions)")
        if self.checkpoint_map:
            print(f"Using pre-selected checkpoints from Confidence Analyzer: {len(self.checkpoint_map)} run(s)")
        print("=" * 60)
        print(f"Log directory:       {self.logdir}")
        print(f"Probabilities dir:   {self.prob_dir}")
        print(f"Output folder:       {self.output_folder}")
        print(f"Output format:       {self.output_format.upper()}")
        if self.output_format in ['png', 'tiff', 'tif']:
            print(f"Resolution:          {self.raster_dpi} DPI")
        if self.use_fixed_axes_height:
            print(f"Plot axes height:    {self.fixed_axes_height} inches")
        if self.master_font_size is not None:
            print(f"Font size:           {self.master_font_size} points (uniform)")
        print(f"Smoothing factor:    {self.smoothing}" + (" (no smoothing)" if self.smoothing == 0 else ""))
        print(f"Markers:             {'Yes' if self.show_markers else 'No'}")
        print(f"Bar label font size: {self.bar_label_fontsize}")
        print(f"Axis limits:         x_min={self.x_axis_min}, y_min={self.y_axis_min}")
        print(f"Grid color:          {self.grid_color}, Grid linestyle: {self.grid_linestyle}, Grid alpha: {self.grid_alpha}")
        print("=" * 60)

    # Clean run name for plot title (empty for single-training timestamp runs).
    def _clean_run_name_for_title(self, run_name: str) -> str:
        if self.run_type == 'single' and self._is_timestamp_run(run_name):
            return ""
        return run_name

    # Add legend to a plot.
    def _add_legend(self, ax, loc: str = 'best', outside: bool = False, outside_bbox: tuple = (1.05, 0.5)):
        if not self.show_legend:
            return None

        if outside:
            legend = ax.legend(
                fontsize=self.legend_font_size,
                frameon=True,
                framealpha=1.0,
                edgecolor=self.grid_color,
                loc='center left',
                bbox_to_anchor=outside_bbox,
            )
        else:
            legend = ax.legend(
                loc=loc,
                fontsize=self.legend_font_size,
                frameon=True,
                framealpha=1.0,
                edgecolor=self.grid_color,
            )

        if legend:
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_linestyle(self.grid_linestyle)

        return legend

    # Create figure and axes with consistent sizing.
    def _create_figure_and_ax(self, has_legend: bool = False, has_colorbar: bool = False,
                              has_bbox_legend: bool = False, legend_outside: bool = False):
        if self.use_fixed_axes_height:
            axes_height = self.fixed_axes_height

            top_margin = 0.6
            bottom_margin = 0.8

            fig_height = axes_height + top_margin + bottom_margin

            if legend_outside:
                fig_width = 12.5
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.78, bottom=0.12, top=0.92)
            elif has_bbox_legend:
                fig_width = 11.0
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.85, bottom=0.12, top=0.92)
            elif has_colorbar:
                fig_width = 9.0
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)
            else:
                fig_width = 8.0
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

            return fig, ax
        else:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            return fig, ax

    # Calculate balanced accuracy from per-class accuracies (fallback).
    def _calculate_balanced_accuracy_from_per_class(self, run_name: str, epoch_1indexed: int) -> float | None:
        if run_name not in self.per_class_data:
            return None

        per_class_data = self.per_class_data[run_name]
        if not per_class_data:
            return None

        class_accs = []
        for class_name, df in per_class_data.items():
            epoch_row = df[df['step'] == epoch_1indexed]
            if not epoch_row.empty:
                class_accs.append(epoch_row['value'].values[0])

        if not class_accs:
            return None

        return np.mean(class_accs)

    # Get the best epoch for a given metric.
    # Results are cached per (run_name, metric_name) so that subsequent calls
    # (e.g. from plot-title generation and probability loading) do not repeat
    # the search or the console output.
    def _get_best_epoch_by_metric(self, run_name: str, metric_name: str = 'composite_score') -> int | None:
        cache_key = (run_name, metric_name)
        if cache_key in self._best_epoch_cache:
            return self._best_epoch_cache[cache_key]

        if run_name not in self.runs_data:
            self._best_epoch_cache[cache_key] = None
            return None

        run_data = self.runs_data[run_name]

        if metric_name == 'composite_score':
            if 'composite_score' not in run_data:
                self._best_epoch_cache[cache_key] = None
                return None
            scores = run_data['composite_score']['value']
            epochs = run_data['composite_score']['step']
            if len(scores) == 0:
                self._best_epoch_cache[cache_key] = None
                return None
            valid = ~np.isnan(scores)
            if not np.any(valid):
                self._best_epoch_cache[cache_key] = None
                return None
            best_idx = np.argmax(scores[valid])
            best_epoch = int(epochs[valid][best_idx])
            best_score = scores[valid][best_idx]
            print(f"      Best epoch by {metric_name}: {best_epoch} (score={best_score:.4f})")
            self._best_epoch_cache[cache_key] = best_epoch
            return best_epoch

        elif metric_name == 'balanced_accuracy':
            if 'balanced_accuracy' in run_data:
                scores = run_data['balanced_accuracy']['value']
                epochs = run_data['balanced_accuracy']['step']
                if len(scores) > 0:
                    valid = ~np.isnan(scores)
                    if np.any(valid):
                        best_idx = np.argmax(scores[valid])
                        best_epoch = int(epochs[valid][best_idx])
                        best_score = scores[valid][best_idx]
                        print(f"      Best epoch by {metric_name}: {best_epoch} (score={best_score:.4f})")
                        self._best_epoch_cache[cache_key] = best_epoch
                        return best_epoch

            if run_name not in self.per_class_data:
                self._best_epoch_cache[cache_key] = None
                return None

            per_class_data = self.per_class_data[run_name]
            if not per_class_data:
                self._best_epoch_cache[cache_key] = None
                return None

            first_class = next(iter(per_class_data.keys()))
            epochs = per_class_data[first_class]['step'].values

            best_epoch = None
            best_score = -float('inf')

            for epoch in epochs:
                bal_acc = self._calculate_balanced_accuracy_from_per_class(run_name, epoch)
                if bal_acc is not None and bal_acc > best_score:
                    best_score = bal_acc
                    best_epoch = int(epoch)

            if best_epoch is not None:
                print(f"      Best epoch by {metric_name} (calculated): {best_epoch} (score={best_score:.4f})")
                self._best_epoch_cache[cache_key] = best_epoch
                return best_epoch

            self._best_epoch_cache[cache_key] = None
            return None

        else:
            self._best_epoch_cache[cache_key] = None
            return None

    # Shift 0-indexed steps to 1-indexed.
    def _get_shifted_steps(self, steps_0indexed: list) -> list:
        return [s + 1 for s in steps_0indexed]

    # Check if run name matches cross-validation pattern (ds01, ds02, ...).
    def _is_cross_validation_run(self, run_name: str) -> bool:
        return bool(re.match(r'^ds\d{2}$', run_name))

    # Check if run name matches timestamp pattern (YYYYMMDD-HHMMSS).
    def _is_timestamp_run(self, run_name: str) -> bool:
        return bool(re.match(r'^\d{8}-\d{6}$', run_name))

    # Auto-detect run type.
    def _detect_run_type(self, runs: list) -> str:
        if self.mode == 'crossval':
            print("✓ Mode forced: CROSS-VALIDATION")
            return 'crossval'
        elif self.mode == 'single':
            print("✓ Mode forced: SINGLE TRAINING")
            return 'single'

        crossval_count = 0
        timestamp_count = 0

        for run_path, run_name, _ in runs:
            if self._is_cross_validation_run(run_name):
                crossval_count += 1
            elif self._is_timestamp_run(run_name):
                timestamp_count += 1

        if crossval_count > 0 and timestamp_count == 0:
            print(f"✓ Auto-detected: CROSS-VALIDATION mode ({crossval_count} dsXX folders)")
            return 'crossval'
        elif timestamp_count > 0 and crossval_count == 0:
            print(f"✓ Auto-detected: SINGLE TRAINING mode ({timestamp_count} timestamp folders)")
            return 'single'
        elif crossval_count > 0 and timestamp_count > 0:
            print(f"⚠ Mixed run types detected! Using CROSS-VALIDATION mode as default.")
            return 'crossval'
        else:
            if list(self.logdir.glob("events.out.tfevents.*")):
                print(f"✓ Auto-detected: SINGLE TRAINING mode (direct event files)")
                return 'single'
            print(f"⚠ Could not determine run type. Using CROSS-VALIDATION mode as default.")
            return 'crossval'

    # Get the probability subdirectory based on run type (supports new and old layouts).
    def _get_probability_subdir(self, run_name: str):
        if not self.prob_dir:
            return None

        if self.run_type == 'crossval':
            # Run names are zero-padded (ds01..ds20), but on-disk folders are
            # not (dataset_1..dataset_20). Normalize by stripping the prefix
            # and re-parsing as int, so both sides agree.
            suffix = run_name.replace("ds", "")
            try:
                dataset_num = int(suffix)
                dataset_name = f"dataset_{dataset_num}"
            except ValueError:
                dataset_name = f"dataset_{suffix}"

            prob_subdir = self.prob_dir / dataset_name / "logs" / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            prob_subdir = self.logdir / run_name / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            prob_subdir = self.logdir / dataset_name / "logs" / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            return None
        else:
            if self.prob_dir.name == "probabilities":
                return self.prob_dir
            prob_subdir = self.logdir / "probabilities"
            if prob_subdir.exists():
                return prob_subdir
            if run_name and self.logdir.name != run_name:
                prob_subdir = self.logdir / run_name / "probabilities"
            else:
                prob_subdir = self.logdir / "probabilities"
            return prob_subdir

    # Find runs in the log directory (supports both old and new structures).
    def find_runs(self) -> list:
        all_event_files = list(self.logdir.rglob("events.out.tfevents.*"))

        if not all_event_files:
            print(f"❌ No event files found in {self.logdir}")
            return []

        folders = defaultdict(list)

        for ef in all_event_files:
            parent = ef.parent

            if parent.name == "logs" and parent.parent.name.startswith("dataset_"):
                folders[parent.parent].append(ef)
            elif parent.name == "logs" and parent.parent.name.startswith("ds"):
                folders[parent.parent].append(ef)
            elif parent.name.startswith("ds") and parent.parent.name == "logs":
                folders[parent].append(ef)
            elif parent.name == "logs" and parent.parent.name not in ["cross_validation", "acv_results"]:
                folders[parent.parent].append(ef)
            else:
                folders[parent].append(ef)

        runs = []
        for folder_path, files in folders.items():
            # Zero-pad dataset_N -> dsNN (matches folder naming and paper)
            if folder_path.name.startswith("dataset_"):
                num = folder_path.name.replace("dataset_", "")
                try:
                    run_name = f"ds{int(num):02d}"
                except ValueError:
                    run_name = folder_path.name.replace("dataset_", "ds")
            elif folder_path.name.startswith("ds") and folder_path.parent.name == "logs":
                # Zero-pad dsN -> dsNN if not already padded
                suffix = folder_path.name[2:]
                if suffix.isdigit():
                    run_name = f"ds{int(suffix):02d}"
                else:
                    run_name = folder_path.name
            else:
                run_name = folder_path.name
            runs.append((folder_path, run_name, files))

        if runs:
            self.run_type = self._detect_run_type(runs)

        return runs

    # Apply exponential moving average smoothing to values.
    def apply_smoothing(self, values: list, alpha: float = 0.95) -> list:
        if alpha <= 0 or len(values) == 0:
            return values

        smoothed = np.zeros_like(values, dtype=float)
        last = values[0]
        for i, val in enumerate(values):
            smoothed_val = alpha * last + (1 - alpha) * val
            smoothed[i] = smoothed_val
            last = smoothed_val
        return smoothed

    # Check if a color is white (for edge detection in bar charts).
    def _is_white_color(self, color) -> bool:
        if isinstance(color, str):
            color_lower = color.lower()
            if color_lower in ['white', '#ffffff', '#fff', 'w']:
                return True
            if color_lower in ['whitesmoke', 'snow', 'ivory', 'floralwhite', 'oldlace']:
                return True
            if color.startswith('#'):
                if len(color) == 7:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    if r > 240 and g > 240 and b > 240:
                        return True
            return False
        return False

    # Get colors for bar chart classes.
    def _get_bar_colors(self, class_names: list, color_dict: dict | None, colormap_name: str) -> list:
        if color_dict is not None:
            return [color_dict.get(cls, '#808080') for cls in class_names]
        else:
            cmap = plt.cm.get_cmap(colormap_name)
            return [cmap(i / len(class_names)) for i in range(len(class_names))]

    # Add markers to a plot.
    def _add_markers(self, ax, x, y, color=None) -> None:
        if not self.show_markers:
            return

        marker_indices = slice(None, None, self.marker_frequency)
        x_markers = np.array(x)[marker_indices]
        y_markers = np.array(y)[marker_indices]

        ax.plot(x_markers, y_markers, 'o', markersize=self.marker_size,
                color=color, linestyle='none', label='_nolegend_')

    # Extract a scalar series from TensorBoard events.
    def _extract_metric_from_events(self, events, metric_name: str, target_value: str | None = None):
        steps = []
        values = []

        for event in events:
            if isinstance(event.value, dict):
                if target_value and target_value in event.value:
                    steps.append(event.step)
                    values.append(event.value[target_value])
                elif metric_name in event.value:
                    steps.append(event.step)
                    values.append(event.value[metric_name])
            else:
                steps.append(event.step)
                values.append(event.value)

        if not steps:
            return None, None

        step_dict = defaultdict(list)
        for step, val in zip(steps, values):
            step_dict[step].append(val)

        unique_steps = sorted(step_dict.keys())
        unique_values = [np.mean(step_dict[step]) for step in unique_steps]

        return unique_steps, unique_values

    # Extract all scalar data from a run's event files.
    def extract_run_data(self, run_path: Path, event_files: list):
        all_metrics = defaultdict(list)

        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(
                    str(event_file),
                    size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE,
                )
                ea.Reload()

                folder_name = event_file.parent.name

                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)

                    if folder_name == 'Metrics_F1_Macro' or folder_name.startswith('Metrics_F1_Macro'):
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        if steps:
                            all_metrics[('f1_macro', 'f1_macro_data')].append((steps, values))
                        continue

                    if folder_name == 'Metrics_F1_Weighted' or folder_name.startswith('Metrics_F1_Weighted'):
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        if steps:
                            all_metrics[('f1_weighted', 'f1_weighted_data')].append((steps, values))
                        continue

                    if tag == 'Metrics/F1_Macro' or tag == 'F1_Macro' or tag == 'f1_macro':
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        if steps:
                            all_metrics[('f1_macro', folder_name)].append((steps, values))
                        continue

                    if tag == 'Metrics/F1_Weighted' or tag == 'F1_Weighted' or tag == 'f1_weighted':
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        if steps:
                            all_metrics[('f1_weighted', folder_name)].append((steps, values))
                        continue

                    if tag == 'Metrics/F1':
                        steps, values = self._extract_metric_from_events(events, 'Macro', 'Macro')
                        if steps:
                            all_metrics[('f1_macro', folder_name)].append((steps, values))

                        steps, values = self._extract_metric_from_events(events, 'Weighted', 'Weighted')
                        if steps:
                            all_metrics[('f1_weighted', folder_name)].append((steps, values))
                        continue

                    if tag == 'Metrics/F1':
                        continue

                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    if steps:
                        all_metrics[(tag, folder_name)].append((steps, values))

            except Exception:
                continue

        if not all_metrics:
            return None, None, None, None

        run_data = {}
        per_class_data = {}
        class_weight_data = {}
        class_count_data = {}

        for (tag, folder_name), metric_lists in all_metrics.items():
            all_steps = []
            all_values = []
            for steps, values in metric_lists:
                all_steps.extend(steps)
                all_values.extend(values)

            if not all_steps:
                continue

            step_dict = defaultdict(list)
            for step, val in zip(all_steps, all_values):
                step_dict[step].append(val)

            steps_0indexed = sorted(step_dict.keys())
            values = [np.mean(step_dict[step]) for step in steps_0indexed]
            steps_1indexed = self._get_shifted_steps(steps_0indexed)

            tag_lower = tag.lower()

            if tag == 'Metrics/F1':
                continue

            if 'loss' in tag_lower:
                if 'train' in tag_lower:
                    metric_name = 'loss_train'
                elif 'val' in tag_lower or 'valid' in tag_lower:
                    metric_name = 'loss_val'
                else:
                    continue

            elif 'accuracy' in tag_lower or 'acc' in tag_lower:
                if 'train' in tag_lower:
                    if 'standard' in tag_lower:
                        metric_name = 'acc_train_standard'
                    else:
                        metric_name = 'acc_train'
                elif 'val' in tag_lower or 'valid' in tag_lower:
                    if 'standard' in tag_lower:
                        metric_name = 'acc_val_standard'
                    else:
                        metric_name = 'acc_val'
                elif 'class' in tag_lower:
                    if 'std' in tag_lower or 'stddev' in tag_lower:
                        metric_name = 'class_std'
                    elif 'min' in tag_lower:
                        metric_name = 'min_class_acc'
                    else:
                        parts = tag.split('/')
                        class_name = parts[-1] if parts else tag
                        metric_name = f'per_class_{class_name}'
                else:
                    continue

            elif 'balanced' in tag_lower and 'accuracy' in tag_lower:
                metric_name = 'balanced_accuracy'

            elif 'f1' in tag_lower:
                if 'weighted' in tag_lower:
                    metric_name = 'f1_weighted'
                elif 'macro' in tag_lower:
                    metric_name = 'f1_macro'
                else:
                    continue

            elif 'auc' in tag_lower:
                if 'class' in tag_lower:
                    parts = tag.split('/')
                    class_name = parts[-1] if parts else 'unknown'
                    metric_name = f'auc_class_{class_name}'
                else:
                    metric_name = 'auc'

            elif 'ap' in tag_lower or 'average_precision' in tag_lower:
                if 'class' in tag_lower:
                    parts = tag.split('/')
                    class_name = parts[-1] if parts else 'unknown'
                    metric_name = f'ap_class_{class_name}'
                else:
                    metric_name = 'ap'

            elif 'lr' in tag_lower or 'learning_rate' in tag_lower:
                metric_name = 'lr'

            elif 'composite' in tag_lower:
                metric_name = 'composite_score'

            elif 'gpu_memory' in tag_lower:
                metric_name = 'gpu_memory'

            elif 'class_count' in tag_lower and 'data' in tag_lower:
                parts = tag.split('/')
                class_name = parts[-1] if parts else 'unknown'
                class_count_data[class_name] = {
                    'step': steps_1indexed,
                    'step_0indexed': steps_0indexed,
                    'value': values,
                }
                continue

            elif 'class_weight' in tag_lower and 'data' in tag_lower:
                parts = tag.split('/')
                class_name = parts[-1] if parts else 'unknown'
                class_weight_data[class_name] = {
                    'step': steps_1indexed,
                    'step_0indexed': steps_0indexed,
                    'value': values,
                }
                continue

            else:
                continue

            if metric_name in ['lr', 'class_std', 'min_class_acc', 'composite_score', 'balanced_accuracy']:
                smoothed = values
            else:
                smoothed = self.apply_smoothing(values, self.smoothing)

            df = pd.DataFrame({
                'step': steps_1indexed,
                'step_0indexed': steps_0indexed,
                'value': values,
                'smoothed': smoothed,
            })

            if metric_name.startswith('per_class_'):
                class_name = metric_name.replace('per_class_', '')
                per_class_data[class_name] = df
            elif metric_name.startswith('auc_class_'):
                class_name = metric_name.replace('auc_class_', '')
                if 'per_class_auc' not in run_data:
                    run_data['per_class_auc'] = {}
                run_data['per_class_auc'][class_name] = df
            elif metric_name.startswith('ap_class_'):
                class_name = metric_name.replace('ap_class_', '')
                if 'per_class_ap' not in run_data:
                    run_data['per_class_ap'] = {}
                run_data['per_class_ap'][class_name] = df
            else:
                if metric_name not in run_data or len(df) > len(run_data[metric_name]):
                    run_data[metric_name] = df

        # Stage-one F1 merge from subfolder event files
        for (key, fname), metric_list in all_metrics.items():
            if key == 'f1_macro' and fname == 'f1_macro_data':
                all_steps = []
                all_values = []
                for steps, values in metric_list:
                    all_steps.extend(steps)
                    all_values.extend(values)
                if all_steps:
                    step_dict = defaultdict(list)
                    for step, val in zip(all_steps, all_values):
                        step_dict[step].append(val)
                    steps_0indexed = sorted(step_dict.keys())
                    values = [np.mean(step_dict[step]) for step in steps_0indexed]
                    steps_1indexed = self._get_shifted_steps(steps_0indexed)
                    run_data['f1_macro'] = pd.DataFrame({
                        'step': steps_1indexed,
                        'step_0indexed': steps_0indexed,
                        'value': values,
                        'smoothed': values,
                    })
                    print(f"    Extracted f1_macro with {len(steps_1indexed)} points")

            elif key == 'f1_weighted' and fname == 'f1_weighted_data':
                all_steps = []
                all_values = []
                for steps, values in metric_list:
                    all_steps.extend(steps)
                    all_values.extend(values)
                if all_steps:
                    step_dict = defaultdict(list)
                    for step, val in zip(all_steps, all_values):
                        step_dict[step].append(val)
                    steps_0indexed = sorted(step_dict.keys())
                    values = [np.mean(step_dict[step]) for step in steps_0indexed]
                    steps_1indexed = self._get_shifted_steps(steps_0indexed)
                    run_data['f1_weighted'] = pd.DataFrame({
                        'step': steps_1indexed,
                        'step_0indexed': steps_0indexed,
                        'value': values,
                        'smoothed': values,
                    })
                    print(f"    Extracted f1_weighted with {len(steps_1indexed)} points")

        # Also handle F1 variants that came in under other tag names
        for (key, fname), metric_list in all_metrics.items():
            if (key == 'f1_macro' and fname == 'f1_macro_data') or \
               (key == 'f1_weighted' and fname == 'f1_weighted_data'):
                continue

            if key == 'f1_macro':
                all_steps = []
                all_values = []
                for steps, values in metric_list:
                    all_steps.extend(steps)
                    all_values.extend(values)
                if all_steps:
                    step_dict = defaultdict(list)
                    for step, val in zip(all_steps, all_values):
                        step_dict[step].append(val)
                    steps_0indexed = sorted(step_dict.keys())
                    values = [np.mean(step_dict[step]) for step in steps_0indexed]
                    steps_1indexed = self._get_shifted_steps(steps_0indexed)
                    if 'f1_macro' not in run_data:
                        run_data['f1_macro'] = pd.DataFrame({
                            'step': steps_1indexed,
                            'step_0indexed': steps_0indexed,
                            'value': values,
                            'smoothed': values,
                        })
                        print(f"    Extracted f1_macro with {len(steps_1indexed)} points")

            elif key == 'f1_weighted':
                all_steps = []
                all_values = []
                for steps, values in metric_list:
                    all_steps.extend(steps)
                    all_values.extend(values)
                if all_steps:
                    step_dict = defaultdict(list)
                    for step, val in zip(all_steps, all_values):
                        step_dict[step].append(val)
                    steps_0indexed = sorted(step_dict.keys())
                    values = [np.mean(step_dict[step]) for step in steps_0indexed]
                    steps_1indexed = self._get_shifted_steps(steps_0indexed)
                    if 'f1_weighted' not in run_data:
                        run_data['f1_weighted'] = pd.DataFrame({
                            'step': steps_1indexed,
                            'step_0indexed': steps_0indexed,
                            'value': values,
                            'smoothed': values,
                        })
                        print(f"    Extracted f1_weighted with {len(steps_1indexed)} points")

        return run_data, per_class_data, class_weight_data, class_count_data

    # Load probability data for a run.
    def _load_probability_data(self, run_name: str, epoch_choice):
        npz_files = []

        prob_subdir = self._get_probability_subdir(run_name)

        if prob_subdir and prob_subdir.exists():
            npz_files = sorted(prob_subdir.glob("probabilities_epoch_*.npz"))

        if not npz_files and self.prob_dir.exists():
            npz_files = sorted(self.prob_dir.glob(f"*{run_name}*_probabilities_epoch_*.npz"))

        if not npz_files and self.prob_dir.exists():
            npz_files = sorted(self.prob_dir.glob("probabilities_epoch_*.npz"))

        if not npz_files:
            return None, None, None

        epoch_files = []
        for f in npz_files:
            try:
                match = re.search(r'epoch_(\d+)', f.stem)
                if match:
                    internal_epoch = int(match.group(1))
                    epoch_files.append((internal_epoch, f))
            except Exception:
                continue

        if not epoch_files:
            return None, None, None

        epoch_files.sort(key=lambda x: x[0])

        selected_internal_epoch = None
        selected_file = None

        map_key = run_name
        if self.run_type == 'single' and self._is_timestamp_run(run_name):
            if run_name in self.checkpoint_map:
                map_key = run_name
            elif 'single_run' in self.checkpoint_map:
                map_key = 'single_run'

        if self.checkpoint_map and map_key in self.checkpoint_map:
            target_epoch_1indexed = self.checkpoint_map[map_key]
            target_internal_epoch = target_epoch_1indexed - 1
            print(f"      Using pre-selected epoch {target_epoch_1indexed} from checkpoint map")
            for internal_epoch, f in epoch_files:
                if internal_epoch == target_internal_epoch:
                    selected_internal_epoch, selected_file = internal_epoch, f
                    break
            if selected_internal_epoch is None:
                print(f"      WARNING: Pre-selected epoch {target_epoch_1indexed} not found")

        if selected_internal_epoch is None:
            if epoch_choice == 'last':
                selected_internal_epoch, selected_file = epoch_files[-1]
            elif epoch_choice == 'composite_score':
                best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
                if best_epoch is not None:
                    best_internal_epoch = best_epoch - 1
                    for internal_epoch, f in epoch_files:
                        if internal_epoch == best_internal_epoch:
                            selected_internal_epoch, selected_file = internal_epoch, f
                            break
                if selected_internal_epoch is None:
                    selected_internal_epoch, selected_file = epoch_files[-1]
            elif epoch_choice == 'balanced_accuracy':
                best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
                if best_epoch is not None:
                    best_internal_epoch = best_epoch - 1
                    for internal_epoch, f in epoch_files:
                        if internal_epoch == best_internal_epoch:
                            selected_internal_epoch, selected_file = internal_epoch, f
                            break
                if selected_internal_epoch is None:
                    selected_internal_epoch, selected_file = epoch_files[-1]
            else:
                try:
                    target_epoch_1indexed = int(epoch_choice)
                    target_internal_epoch = target_epoch_1indexed - 1
                    for internal_epoch, f in epoch_files:
                        if internal_epoch == target_internal_epoch:
                            selected_internal_epoch, selected_file = internal_epoch, f
                            break
                    if selected_internal_epoch is None:
                        selected_internal_epoch, selected_file = epoch_files[-1]
                except Exception:
                    selected_internal_epoch, selected_file = epoch_files[-1]

        selected_epoch_1indexed = selected_internal_epoch + 1
        print(f"      Using epoch {selected_epoch_1indexed} from {selected_file.name}")
        data = np.load(selected_file)

        probabilities = data['probabilities']
        labels = data['labels']

        if 'classes' in data:
            class_names = data['classes'].tolist()
        else:
            class_names = [f'Class_{i}' for i in range(probabilities.shape[1])]

        return probabilities, labels, class_names

    # Process all runs: extract scalar data, merge F1 subfolders, load probability data.
    def process_runs(self) -> bool:

        self._best_epoch_cache = {}
        runs = self.find_runs()

        if not runs:
            print("\n❌ No training runs found.")
            return False

        print("\n📊 Extracting scalar data from TensorBoard...")

        f1_macro_run = None
        f1_weighted_run = None
        main_runs = []

        for run_path, run_name, event_files in runs:
            if run_name == 'Metrics_F1_Macro' or run_name.startswith('Metrics_F1_Macro'):
                f1_macro_run = (run_path, run_name, event_files)
                print(f"  Found F1 Macro subfolder: {run_name}")
            elif run_name == 'Metrics_F1_Weighted' or run_name.startswith('Metrics_F1_Weighted'):
                f1_weighted_run = (run_path, run_name, event_files)
                print(f"  Found F1 Weighted subfolder: {run_name}")
            else:
                main_runs.append((run_path, run_name, event_files))

        for i, (run_path, run_name, event_files) in enumerate(main_runs, 1):
            print(f"  {i}/{len(main_runs)}: {run_name} ({len(event_files)} event files)")
            run_data, per_class_data, class_weight_data, class_count_data = self.extract_run_data(run_path, event_files)

            if run_data and len(run_data) > 0:
                self.runs_data[run_name] = run_data
                print(f"    ✓ Metrics: {list(run_data.keys())}")

            if per_class_data and len(per_class_data) > 0:
                self.per_class_data[run_name] = per_class_data
                print(f"    ✓ Per-class: {list(per_class_data.keys())}")

            if class_weight_data and len(class_weight_data) > 0:
                self.class_weight_data[run_name] = class_weight_data
                print(f"    ✓ Class weights: {list(class_weight_data.keys())}")

            if class_count_data and len(class_count_data) > 0:
                self.class_count_data[run_name] = class_count_data
                print(f"    ✓ Class counts: {list(class_count_data.keys())}")

        # Stage one: process F1 subfolders and merge into currently loaded runs
        if f1_macro_run or f1_weighted_run:
            print("\n📊 Extracting F1 data from subfolders...")

            if f1_macro_run:
                run_path, run_name, event_files = f1_macro_run
                print(f"  Processing {run_name}...")

                for event_file in event_files:
                    try:
                        ea = event_accumulator.EventAccumulator(
                            str(event_file),
                            size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE,
                        )
                        ea.Reload()

                        for tag in ea.Tags()['scalars']:
                            events = ea.Scalars(tag)
                            steps = [e.step for e in events]
                            values = [e.value for e in events]

                            if steps:
                                step_dict = defaultdict(list)
                                for step, val in zip(steps, values):
                                    step_dict[step].append(val)
                                steps_0indexed = sorted(step_dict.keys())
                                values = [np.mean(step_dict[step]) for step in steps_0indexed]
                                steps_1indexed = self._get_shifted_steps(steps_0indexed)

                                for main_run_name in self.runs_data.keys():
                                    self.runs_data[main_run_name]['f1_macro'] = pd.DataFrame({
                                        'step': steps_1indexed,
                                        'step_0indexed': steps_0indexed,
                                        'value': values,
                                        'smoothed': values,
                                    })
                                    print(f"    Merged f1_macro into {main_run_name}")
                    except Exception as e:
                        print(f"    Error processing {run_name}: {e}")

            if f1_weighted_run:
                run_path, run_name, event_files = f1_weighted_run
                print(f"  Processing {run_name}...")

                for event_file in event_files:
                    try:
                        ea = event_accumulator.EventAccumulator(
                            str(event_file),
                            size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE,
                        )
                        ea.Reload()

                        for tag in ea.Tags()['scalars']:
                            events = ea.Scalars(tag)
                            steps = [e.step for e in events]
                            values = [e.value for e in events]

                            if steps:
                                step_dict = defaultdict(list)
                                for step, val in zip(steps, values):
                                    step_dict[step].append(val)
                                steps_0indexed = sorted(step_dict.keys())
                                values = [np.mean(step_dict[step]) for step in steps_0indexed]
                                steps_1indexed = self._get_shifted_steps(steps_0indexed)

                                for main_run_name in self.runs_data.keys():
                                    self.runs_data[main_run_name]['f1_weighted'] = pd.DataFrame({
                                        'step': steps_1indexed,
                                        'step_0indexed': steps_0indexed,
                                        'value': values,
                                        'smoothed': values,
                                    })
                                    print(f"    Merged f1_weighted into {main_run_name}")
                    except Exception as e:
                        print(f"    Error processing {run_name}: {e}")

        # Remove F1 subfolder placeholder runs, if any made it into runs_data
        for run_name in list(self.runs_data.keys()):
            if run_name == 'Metrics_F1_Macro' or run_name.startswith('Metrics_F1_Macro'):
                del self.runs_data[run_name]
            elif run_name == 'Metrics_F1_Weighted' or run_name.startswith('Metrics_F1_Weighted'):
                del self.runs_data[run_name]

        # Stage two: merge F1 data into cross-validation runs
        print("\n📊 Merging F1 data into cross-validation runs...")

        f1_macro_data = None
        f1_weighted_data = None
        f1_macro_run_name = None
        f1_weighted_run_name = None

        for run_name, run_data in self.runs_data.items():
            if run_name == 'Metrics_F1_Macro' or run_name.startswith('Metrics_F1_Macro'):
                f1_macro_data = run_data
                f1_macro_run_name = run_name
            elif run_name == 'Metrics_F1_Weighted' or run_name.startswith('Metrics_F1_Weighted'):
                f1_weighted_data = run_data
                f1_weighted_run_name = run_name

        if f1_macro_data is not None or f1_weighted_data is not None:
            for run_name in list(self.runs_data.keys()):
                if run_name.startswith('ds') and len(run_name) == 4:
                    if run_name not in self.runs_data:
                        continue

                    if f1_macro_data is not None:
                        if 'f1_macro' in f1_macro_data:
                            self.runs_data[run_name]['f1_macro'] = f1_macro_data['f1_macro']
                            print(f"    Merged f1_macro into {run_name}")

                    if f1_weighted_data is not None:
                        if 'f1_weighted' in f1_weighted_data:
                            self.runs_data[run_name]['f1_weighted'] = f1_weighted_data['f1_weighted']
                            print(f"    Merged f1_weighted into {run_name}")

            if f1_macro_run_name and f1_macro_run_name in self.runs_data:
                del self.runs_data[f1_macro_run_name]
            if f1_weighted_run_name and f1_weighted_run_name in self.runs_data:
                del self.runs_data[f1_weighted_run_name]

            print(f"✓ Merged F1 data into {len([r for r in self.runs_data.keys() if r.startswith('ds')])} cross-validation runs")
        else:
            print("  No F1 data found to merge.")

        print(f"\n✓ Loaded {len(self.runs_data)} run(s)")

        # Load probability data for ROC/PR curves
        if self.plot_roc_curves or self.plot_pr_curves:
            print("\n📊 Loading probability data for ROC/PR curves...")
            for run_name in self.runs_data.keys():
                print(f"  {run_name}:")
                probs_roc, labels_roc, class_names = self._load_probability_data(run_name, self.roc_curve_epoch)
                if probs_roc is not None:
                    self.roc_data[run_name] = (probs_roc, labels_roc, class_names)
                else:
                    self.roc_data[run_name] = None

                if self.roc_curve_epoch == self.pr_curve_epoch:
                    self.pr_data[run_name] = self.roc_data[run_name]
                else:
                    probs_pr, labels_pr, class_names = self._load_probability_data(run_name, self.pr_curve_epoch)
                    self.pr_data[run_name] = (probs_pr, labels_pr, class_names) if probs_pr is not None else None

        return True

    #############################################################################################################
    # PLOTTING METHODS

    # Plot loss curves for a run.
    def plot_loss_curves_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.loss_legend_outside)

        if 'loss_train' in metrics:
            x = metrics['loss_train']['step']
            y = metrics['loss_train']['smoothed']
            ax.plot(x, y,
                    linestyle=self.train_line_style, linewidth=self.line_width,
                    color=self.train_color, label='Training')
            self._add_markers(ax, x, y, color=self.train_color)

        if 'loss_val' in metrics:
            x = metrics['loss_val']['step']
            y = metrics['loss_val']['smoothed']
            ax.plot(x, y,
                    linestyle=self.val_line_style, linewidth=self.line_width,
                    color=self.val_color, label='Validation')
            self._add_markers(ax, x, y, color=self.val_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Loss Curves - {clean_name}')
        else:
            ax.set_title(f'Loss Curves')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.loss_legend_loc, outside=self.loss_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_loss_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot accuracy curves for a run.
    def plot_accuracy_curves_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.accuracy_legend_outside)

        if 'acc_train' in metrics:
            x = metrics['acc_train']['step']
            y = metrics['acc_train']['smoothed']
            ax.plot(x, y,
                    linestyle=self.train_line_style, linewidth=self.line_width,
                    color=self.train_color, label='Training (weighted)')
            self._add_markers(ax, x, y, color=self.train_color)

        if 'acc_train_standard' in metrics:
            x = metrics['acc_train_standard']['step']
            y = metrics['acc_train_standard']['smoothed']
            ax.plot(x, y,
                    linestyle=self.standard_line_style, linewidth=self.line_width,
                    color=self.standard_train_color, alpha=0.7, label='Training (standard)')
            self._add_markers(ax, x, y, color=self.standard_train_color)

        if 'acc_val' in metrics:
            x = metrics['acc_val']['step']
            y = metrics['acc_val']['smoothed']
            ax.plot(x, y,
                    linestyle=self.val_line_style, linewidth=self.line_width,
                    color=self.val_color, label='Validation (weighted)')
            self._add_markers(ax, x, y, color=self.val_color)

        if 'acc_val_standard' in metrics:
            x = metrics['acc_val_standard']['step']
            y = metrics['acc_val_standard']['smoothed']
            ax.plot(x, y,
                    linestyle=self.standard_line_style, linewidth=self.line_width,
                    color=self.standard_val_color, alpha=0.7, label='Validation (standard)')
            self._add_markers(ax, x, y, color=self.standard_val_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Accuracy Curves - {clean_name}')
        else:
            ax.set_title(f'Accuracy Curves')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.accuracy_legend_loc, outside=self.accuracy_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_accuracy_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot F1 scores for a run.
    def plot_f1_scores_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.f1_legend_outside)

        def get_f1_data(data):
            if data is None:
                return None, None
            if isinstance(data, pd.DataFrame):
                if 'step' in data.columns and 'smoothed' in data.columns:
                    return data['step'].values, data['smoothed'].values
            elif isinstance(data, dict):
                if 'step' in data and 'smoothed' in data:
                    return data['step'], data['smoothed']
            return None, None

        f1_weighted_found = False
        f1_macro_found = False

        if 'f1_weighted' in metrics:
            x, y = get_f1_data(metrics['f1_weighted'])
            if x is not None and len(x) > 0:
                ax.plot(x, y,
                        linestyle='-', linewidth=self.line_width,
                        color=self.f1_weighted_color, label='Weighted F1')
                self._add_markers(ax, x, y, color=self.f1_weighted_color)
                f1_weighted_found = True

        if 'f1_macro' in metrics:
            x, y = get_f1_data(metrics['f1_macro'])
            if x is not None and len(x) > 0:
                ax.plot(x, y,
                        linestyle='--', linewidth=self.line_width,
                        color=self.f1_macro_color, label='Macro F1')
                self._add_markers(ax, x, y, color=self.f1_macro_color)
                f1_macro_found = True

        if not f1_weighted_found and not f1_macro_found:
            ax.text(0.5, 0.5, 'No F1 data available',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=self.title_font_size)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'F1 Scores - {clean_name}')
        else:
            ax.set_title(f'F1 Scores')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend and (f1_weighted_found or f1_macro_found):
            self._add_legend(ax, loc=self.f1_legend_loc, outside=self.f1_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_f1_scores.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot learning rate for a run.
    def plot_learning_rate_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.lr_legend_outside)

        if 'lr' not in metrics:
            return None

        x = metrics['lr']['step']
        y = metrics['lr']['value']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.lr_color, label='Learning Rate')
        self._add_markers(ax, x, y, color=self.lr_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Learning Rate Schedule - {clean_name}')
        else:
            ax.set_title(f'Learning Rate Schedule')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.lr_legend_loc, outside=self.lr_legend_outside, outside_bbox=self.legend_outside_bbox)

        filename = f"{run_name}_learning_rate.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot AUC for a run.
    def plot_auc_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.auc_legend_outside)

        if 'auc' not in metrics:
            return None

        x = metrics['auc']['step']
        y = metrics['auc']['smoothed']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.auc_color, label='AUC')
        self._add_markers(ax, x, y, color=self.auc_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'AUC over Training - {clean_name}')
        else:
            ax.set_title(f'AUC over Training')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.auc_legend_loc, outside=self.auc_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_auc_over_epochs.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot Average Precision for a run.
    def plot_ap_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.ap_legend_outside)

        if 'ap' not in metrics:
            return None

        x = metrics['ap']['step']
        y = metrics['ap']['smoothed']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.ap_color, label='Average Precision')
        self._add_markers(ax, x, y, color=self.ap_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Precision')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Average Precision over Training - {clean_name}')
        else:
            ax.set_title(f'Average Precision over Training')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.ap_legend_loc, outside=self.ap_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_ap_over_epochs.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot per-class accuracy for a run.
    def plot_per_class_accuracy_for_run(self, run_name: str, per_class_data: dict):
        if not per_class_data:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.per_class_legend_outside)

        cmap = plt.get_cmap(self.per_class_colormap)
        class_idx = 0

        for class_name, df in per_class_data.items():
            color = cmap(class_idx % cmap.N)
            class_idx += 1
            x = df['step']
            y = df['smoothed']
            ax.plot(x, y,
                    linestyle='-', linewidth=self.line_width,
                    color=color, label=class_name)
            self._add_markers(ax, x, y, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Per-Class Accuracy - {clean_name}')
        else:
            ax.set_title(f'Per-Class Accuracy')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)

        if self.show_legend:
            self._add_legend(ax, loc=self.per_class_legend_loc, outside=self.per_class_legend_outside, outside_bbox=self.legend_outside_bbox)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        filename = f"{run_name}_per_class_accuracy.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot composite score for a run.
    def plot_composite_score_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.composite_legend_outside)

        if 'composite_score' not in metrics:
            return None

        x = metrics['composite_score']['step']
        y = metrics['composite_score']['value']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.composite_score_color, label='Composite Score')
        self._add_markers(ax, x, y, color=self.composite_score_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Composite Score')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Composite Score - {clean_name}')
        else:
            ax.set_title(f'Composite Score')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.composite_legend_loc, outside=self.composite_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_composite_score.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot balanced accuracy for a run.
    def plot_balanced_accuracy_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.balanced_acc_legend_outside)

        if 'balanced_accuracy' in metrics:
            x = metrics['balanced_accuracy']['step']
            y = metrics['balanced_accuracy']['value']
        else:
            if run_name not in self.per_class_data:
                return None

            per_class_data = self.per_class_data[run_name]
            if not per_class_data:
                return None

            first_class = next(iter(per_class_data.keys()))
            epochs = per_class_data[first_class]['step'].values

            bal_acc_values = []
            for epoch in epochs:
                bal_acc = self._calculate_balanced_accuracy_from_per_class(run_name, epoch)
                if bal_acc is not None:
                    bal_acc_values.append(bal_acc)
                else:
                    bal_acc_values.append(np.nan)

            x = epochs
            y = np.array(bal_acc_values)

        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.balanced_accuracy_color, label='Balanced Accuracy')
        self._add_markers(ax, x, y, color=self.balanced_accuracy_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Balanced Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Balanced Accuracy over Training - {clean_name}')
        else:
            ax.set_title(f'Balanced Accuracy over Training')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.balanced_acc_legend_loc, outside=self.balanced_acc_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_balanced_accuracy.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot class accuracy standard deviation for a run.
    def plot_class_std_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.class_std_legend_outside)

        if 'class_std' not in metrics:
            return None

        x = metrics['class_std']['step']
        y = metrics['class_std']['value']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.class_std_color, label='Class Accuracy Std Dev')
        self._add_markers(ax, x, y, color=self.class_std_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Standard Deviation')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Class Accuracy Variation - {clean_name}')
        else:
            ax.set_title(f'Class Accuracy Variation')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.class_std_legend_loc, outside=self.class_std_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_class_accuracy_std.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot minimum class accuracy for a run.
    def plot_min_class_acc_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.min_class_acc_legend_outside)

        if 'min_class_acc' not in metrics:
            return None

        x = metrics['min_class_acc']['step']
        y = metrics['min_class_acc']['value']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.min_class_acc_color, label='Minimum Class Accuracy')
        self._add_markers(ax, x, y, color=self.min_class_acc_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Minimum Class Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Worst-Performing Class Accuracy - {clean_name}')
        else:
            ax.set_title(f'Worst-Performing Class Accuracy')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)

        if self.min_class_acc_threshold is not None:
            ax.axhline(y=self.min_class_acc_threshold, color='red',
                       linestyle='--', alpha=0.5, label=f'Threshold ({self.min_class_acc_threshold:.0%})')

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.min_class_acc_legend_loc, outside=self.min_class_acc_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_min_class_accuracy.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot GPU memory usage for a run.
    def plot_gpu_memory_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.gpu_memory_legend_outside)

        if 'gpu_memory' not in metrics:
            return None

        x = metrics['gpu_memory']['step']
        y = metrics['gpu_memory']['value']
        ax.plot(x, y,
                linestyle='-', linewidth=self.line_width,
                color=self.gpu_memory_color, label='GPU Memory Usage')
        self._add_markers(ax, x, y, color=self.gpu_memory_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Memory Usage Ratio')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'GPU Memory Usage - {clean_name}')
        else:
            ax.set_title(f'GPU Memory Usage')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.gpu_memory_legend_loc, outside=self.gpu_memory_legend_outside, outside_bbox=self.legend_outside_bbox)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_gpu_memory.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot per-class AUC for a run.
    def plot_per_class_auc_for_run(self, run_name: str, metrics: dict):
        if 'per_class_auc' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_bbox_legend=True)

        cmap = plt.get_cmap(self.per_class_colormap)
        class_idx = 0

        for class_name, df in metrics['per_class_auc'].items():
            color = cmap(class_idx % cmap.N)
            class_idx += 1
            x = df['step']
            y = df['smoothed']
            ax.plot(x, y,
                    linestyle='-', linewidth=self.line_width,
                    color=color, label=class_name)
            self._add_markers(ax, x, y, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Per-Class AUC - {clean_name}')
        else:
            ax.set_title(f'Per-Class AUC')
        ax.set_xlim(left=self.x_axis_min)
        ax.set_ylim(bottom=self.y_axis_min, top=1.05)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            legend = ax.legend(fontsize=self.legend_font_size - 1, frameon=True,
                               framealpha=1.0, edgecolor=self.grid_color,
                               loc='center left', bbox_to_anchor=(1, 0.5))
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_linestyle(self.grid_linestyle)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        filename = f"{run_name}_per_class_auc.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Generate ROC curve for a run.
    def generate_roc_curve_for_run(self, run_name: str, probabilities, labels, class_names, epoch_info: str):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.roc_legend_outside)

        cmap = plt.get_cmap(self.per_class_colormap)
        colors = [cmap(i % cmap.N) for i in range(len(class_names))]

        roc_auc_values = {}

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]

            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            roc_auc_val = auc(fpr, tpr)
            roc_auc_values[class_name] = roc_auc_val

            ax.plot(fpr, tpr, color=color, lw=self.line_width,
                    label=f'{class_name} (AUC = {roc_auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random (AUC=0.5)')

        ax.set_xlim([self.x_axis_min, 1.0])
        ax.set_ylim([self.y_axis_min, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=self.axis_label_size)
        ax.set_ylabel('True Positive Rate', fontsize=self.axis_label_size)

        if self.run_type == 'crossval':
            title_display = run_name
        else:
            title_display = "Single Run"

        if "Best Epoch" in epoch_info:
            match = re.search(r'Best Epoch (\d+)', epoch_info)
            if match:
                epoch_num = match.group(1)
                ax.set_title(f'ROC Curves - {title_display} (Best Epoch {epoch_num})',
                             fontsize=self.title_font_size, fontweight='bold')
            else:
                ax.set_title(f'ROC Curves - {title_display}',
                             fontsize=self.title_font_size, fontweight='bold')
        else:
            ax.set_title(f'ROC Curves - {title_display} ({epoch_info})',
                         fontsize=self.title_font_size, fontweight='bold')

        if self.show_legend:
            self._add_legend(ax, loc=self.roc_legend_loc, outside=self.roc_legend_outside, outside_bbox=self.legend_outside_bbox)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)

        plt.tight_layout()

        filename = f"{run_name}_roc_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)

        auc_df = pd.DataFrame([roc_auc_values]).T
        auc_df.columns = ['AUC']
        auc_df.index.name = 'Class'
        auc_df.to_csv(self.output_folder / f"{run_name}_roc_auc_values.csv")

        return filename

    # Generate PR curve for a run.
    def generate_pr_curve_for_run(self, run_name: str, probabilities, labels, class_names, epoch_info: str):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.pr_legend_outside)

        cmap = plt.get_cmap(self.per_class_colormap)
        colors = [cmap(i % cmap.N) for i in range(len(class_names))]

        ap_values = {}

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]

            precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
            ap_val = average_precision_score(binary_labels, class_probs)
            ap_values[class_name] = ap_val

            ax.plot(recall, precision, color=color, lw=self.line_width,
                    label=f'{class_name} (AP = {ap_val:.3f})')

        ax.set_xlim([self.x_axis_min, 1.0])
        ax.set_ylim([self.y_axis_min, 1.05])
        ax.set_xlabel('Recall', fontsize=self.axis_label_size)
        ax.set_ylabel('Precision', fontsize=self.axis_label_size)

        if self.run_type == 'crossval':
            title_display = run_name
        else:
            title_display = "Single Run"

        if "Best Epoch" in epoch_info:
            match = re.search(r'Best Epoch (\d+)', epoch_info)
            if match:
                epoch_num = match.group(1)
                ax.set_title(f'Precision-Recall Curves - {title_display} (Best Epoch {epoch_num})',
                             fontsize=self.title_font_size, fontweight='bold')
            else:
                ax.set_title(f'Precision-Recall Curves - {title_display}',
                             fontsize=self.title_font_size, fontweight='bold')
        else:
            ax.set_title(f'Precision-Recall Curves - {title_display} ({epoch_info})',
                         fontsize=self.title_font_size, fontweight='bold')

        if self.show_legend:
            self._add_legend(ax, loc=self.pr_legend_loc, outside=self.pr_legend_outside, outside_bbox=self.legend_outside_bbox)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)

        plt.tight_layout()

        filename = f"{run_name}_pr_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)

        ap_df = pd.DataFrame([ap_values]).T
        ap_df.columns = ['Average Precision']
        ap_df.index.name = 'Class'
        ap_df.to_csv(self.output_folder / f"{run_name}_pr_ap_values.csv")

        return filename

    # Plot class weights bar chart for a run.
    def plot_class_weights_for_run(self, run_name: str, class_weight_data: dict):
        if not class_weight_data:
            return None

        fig, ax = self._create_figure_and_ax()

        class_names = sorted(class_weight_data.keys())
        weights = [class_weight_data[cls]['value'][-1] for cls in class_names]
        weights_pct = np.array(weights) * 100

        fill_colors = self._get_bar_colors(class_names, self.class_weights_colors, self.class_weights_colormap)

        max_val = max(weights_pct)
        label_offset = max_val * self.bar_label_offset

        for i, (class_name, val, fill_color) in enumerate(zip(class_names, weights_pct, fill_colors)):
            if self._is_white_color(fill_color):
                edge_color = '#000000'
                edge_width = self.bar_edge_width
            else:
                edge_color = fill_color
                edge_width = 0

            bar = ax.bar(class_name, val,
                         color=fill_color,
                         edgecolor=edge_color,
                         linewidth=edge_width)

            ax.text(bar[0].get_x() + bar[0].get_width() / 2, val + label_offset,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=self.bar_label_fontsize, color='black')

        ax.set_xlabel('Class')
        ax.set_ylabel('Loss Weight (%)')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Class Weights for Loss Function - {clean_name}')
        else:
            ax.set_title(f'Class Weights for Loss Function')
        ax.set_ylim(bottom=self.y_axis_min, top=max_val * 1.10)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color, axis='y')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        filename = f"{run_name}_class_weights.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot class counts bar chart for a run.
    def plot_class_counts_for_run(self, run_name: str, class_count_data: dict):
        if not class_count_data:
            return None

        fig, ax = self._create_figure_and_ax()

        class_names = sorted(class_count_data.keys())
        counts = [class_count_data[cls]['value'][0] for cls in class_names]

        fill_colors = self._get_bar_colors(class_names, self.class_counts_colors, self.class_counts_colormap)

        max_count = max(counts)
        label_offset = max_count * self.bar_label_offset

        bars = []
        for i, (class_name, count, fill_color) in enumerate(zip(class_names, counts, fill_colors)):
            if self._is_white_color(fill_color):
                edge_color = '#000000'
                edge_width = self.bar_edge_width
            else:
                edge_color = fill_color
                edge_width = 0

            bar = ax.bar(class_name, count,
                         color=fill_color,
                         edgecolor=edge_color,
                         linewidth=edge_width)
            bars.append(bar[0])

        total = sum(counts)
        for bar, count, fill_color in zip(bars, counts, fill_colors):
            pct = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, count + label_offset,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=self.bar_label_fontsize, color='black')

        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        clean_name = self._clean_run_name_for_title(run_name)
        if clean_name:
            ax.set_title(f'Class Distribution in Training Set - {clean_name}')
        else:
            ax.set_title(f'Class Distribution in Training Set')
        ax.set_ylim(bottom=self.y_axis_min, top=max_count * 1.10)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color, axis='y')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        filename = f"{run_name}_class_counts.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Save figure with appropriate settings and print size.
    def _save_figure(self, fig, output_path: Path) -> None:
        save_kwargs = {'dpi': self.raster_dpi, 'bbox_inches': 'tight'}

        if self.output_format in ['png', 'jpg', 'jpeg']:
            fig.savefig(output_path, **save_kwargs)
        elif self.output_format in ['tiff', 'tif']:
            fig.savefig(output_path, format='tiff', **save_kwargs)
        else:
            fig.savefig(output_path, format=self.output_format, **save_kwargs)

        file_size_kb = output_path.stat().st_size / 1024
        file_size_mb = file_size_kb / 1024

        if file_size_mb >= 1:
            print(f"    ✓ {output_path.name} ({file_size_mb:.1f} MB)")
        else:
            print(f"    ✓ {output_path.name} ({file_size_kb:.0f} KB)")

        plt.close(fig)

    #############################################################################################################
    # MAIN GENERATION METHOD

    # Generate all plots for all runs.
    def generate_all_plots(self) -> list:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS FOR EACH RUN")
        print("NOTE: All plots show epochs starting from 1")
        print("NOTE: All axes start at 0 (x_min=0, y_min=0)")
        if self.use_fixed_axes_height:
            print(f"NOTE: All plots have identical axes height: {self.fixed_axes_height} inches")
        print("=" * 60)

        all_generated = []

        for run_name in self.runs_data.keys():
            print(f"\n📁 Processing run: {run_name}")
            metrics = self.runs_data[run_name]
            generated = []

            if self.plot_loss:
                if self.plot_loss_curves_for_run(run_name, metrics):
                    generated.append("loss_curves")

            if self.plot_accuracy:
                if self.plot_accuracy_curves_for_run(run_name, metrics):
                    generated.append("accuracy_curves")

            if self.plot_f1:
                if self.plot_f1_scores_for_run(run_name, metrics):
                    generated.append("f1_scores")

            if self.plot_lr:
                if self.plot_learning_rate_for_run(run_name, metrics):
                    generated.append("learning_rate")

            if self.plot_auc:
                if self.plot_auc_for_run(run_name, metrics):
                    generated.append("auc_over_epochs")
                if self.plot_per_class_auc_for_run(run_name, metrics):
                    generated.append("per_class_auc")

            if self.plot_ap:
                if self.plot_ap_for_run(run_name, metrics):
                    generated.append("ap_over_epochs")

            if self.plot_per_class_accuracy and run_name in self.per_class_data:
                if self.plot_per_class_accuracy_for_run(run_name, self.per_class_data[run_name]):
                    generated.append("per_class_accuracy")

            if self.plot_class_weights and run_name in self.class_weight_data:
                if self.plot_class_weights_for_run(run_name, self.class_weight_data[run_name]):
                    generated.append("class_weights")

            if self.plot_class_counts and run_name in self.class_count_data:
                if self.plot_class_counts_for_run(run_name, self.class_count_data[run_name]):
                    generated.append("class_counts")

            if self.plot_composite_score:
                if self.plot_composite_score_for_run(run_name, metrics):
                    generated.append("composite_score")

            if self.plot_balanced_accuracy:
                if self.plot_balanced_accuracy_for_run(run_name, metrics):
                    generated.append("balanced_accuracy")

            if self.plot_class_std:
                if self.plot_class_std_for_run(run_name, metrics):
                    generated.append("class_accuracy_std")

            if self.plot_min_class_acc:
                if self.plot_min_class_acc_for_run(run_name, metrics):
                    generated.append("min_class_accuracy")

            if self.plot_gpu_memory:
                if self.plot_gpu_memory_for_run(run_name, metrics):
                    generated.append("gpu_memory")

            # ROC curves
            if self.plot_roc_curves and run_name in self.roc_data and self.roc_data[run_name] is not None:
                probabilities, labels, class_names = self.roc_data[run_name]

                if self.checkpoint_map:
                    map_key = run_name
                    if self.run_type == 'single' and self._is_timestamp_run(run_name):
                        if run_name in self.checkpoint_map:
                            map_key = run_name
                        elif 'single_run' in self.checkpoint_map:
                            map_key = 'single_run'

                    if map_key in self.checkpoint_map:
                        epoch_num = self.checkpoint_map[map_key]
                        epoch_info = f"Best Epoch {epoch_num} (by Conf. Analyzer)"
                    elif self.roc_curve_epoch == 'composite_score':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Composite Score)"
                        else:
                            epoch_info = "Best Epoch (by Composite Score)"
                    elif self.roc_curve_epoch == 'balanced_accuracy':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Balanced Accuracy)"
                        else:
                            epoch_info = "Best Epoch (by Balanced Accuracy)"
                    elif self.roc_curve_epoch == 'last':
                        if run_name in self.runs_data and 'acc_val' in self.runs_data[run_name]:
                            epochs = self.runs_data[run_name]['acc_val']['step']
                            if len(epochs) > 0:
                                last_epoch = int(epochs[-1])
                                epoch_info = f"Last Epoch {last_epoch}"
                            else:
                                epoch_info = "Last Epoch"
                        else:
                            epoch_info = "Last Epoch"
                    else:
                        epoch_info = f"Epoch {self.roc_curve_epoch}"
                else:
                    if self.roc_curve_epoch == 'composite_score':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Composite Score)"
                        else:
                            epoch_info = "Best Epoch (by Composite Score)"
                    elif self.roc_curve_epoch == 'balanced_accuracy':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Balanced Accuracy)"
                        else:
                            epoch_info = "Best Epoch (by Balanced Accuracy)"
                    elif self.roc_curve_epoch == 'last':
                        if run_name in self.runs_data and 'acc_val' in self.runs_data[run_name]:
                            epochs = self.runs_data[run_name]['acc_val']['step']
                            if len(epochs) > 0:
                                last_epoch = int(epochs[-1])
                                epoch_info = f"Last Epoch {last_epoch}"
                            else:
                                epoch_info = "Last Epoch"
                        else:
                            epoch_info = "Last Epoch"
                    else:
                        epoch_info = f"Epoch {self.roc_curve_epoch}"

                if self.generate_roc_curve_for_run(run_name, probabilities, labels, class_names, epoch_info):
                    generated.append("roc_curves")

            # PR curves
            if self.plot_pr_curves and run_name in self.pr_data and self.pr_data[run_name] is not None:
                probabilities, labels, class_names = self.pr_data[run_name]

                if self.checkpoint_map:
                    map_key = run_name
                    if self.run_type == 'single' and self._is_timestamp_run(run_name):
                        if run_name in self.checkpoint_map:
                            map_key = run_name
                        elif 'single_run' in self.checkpoint_map:
                            map_key = 'single_run'

                    if map_key in self.checkpoint_map:
                        epoch_num = self.checkpoint_map[map_key]
                        epoch_info = f"Best Epoch {epoch_num} (by Conf. Analyzer)"
                    elif self.pr_curve_epoch == 'composite_score':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Composite Score)"
                        else:
                            epoch_info = "Best Epoch (by Composite Score)"
                    elif self.pr_curve_epoch == 'balanced_accuracy':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Balanced Accuracy)"
                        else:
                            epoch_info = "Best Epoch (by Balanced Accuracy)"
                    elif self.pr_curve_epoch == 'last':
                        if run_name in self.runs_data and 'acc_val' in self.runs_data[run_name]:
                            epochs = self.runs_data[run_name]['acc_val']['step']
                            if len(epochs) > 0:
                                last_epoch = int(epochs[-1])
                                epoch_info = f"Last Epoch {last_epoch}"
                            else:
                                epoch_info = "Last Epoch"
                        else:
                            epoch_info = "Last Epoch"
                    else:
                        epoch_info = f"Epoch {self.pr_curve_epoch}"
                else:
                    if self.pr_curve_epoch == 'composite_score':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Composite Score)"
                        else:
                            epoch_info = "Best Epoch (by Composite Score)"
                    elif self.pr_curve_epoch == 'balanced_accuracy':
                        best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
                        if best_epoch is not None:
                            epoch_info = f"Best Epoch {best_epoch} (by Balanced Accuracy)"
                        else:
                            epoch_info = "Best Epoch (by Balanced Accuracy)"
                    elif self.pr_curve_epoch == 'last':
                        if run_name in self.runs_data and 'acc_val' in self.runs_data[run_name]:
                            epochs = self.runs_data[run_name]['acc_val']['step']
                            if len(epochs) > 0:
                                last_epoch = int(epochs[-1])
                                epoch_info = f"Last Epoch {last_epoch}"
                            else:
                                epoch_info = "Last Epoch"
                        else:
                            epoch_info = "Last Epoch"
                    else:
                        epoch_info = f"Epoch {self.pr_curve_epoch}"

                if self.generate_pr_curve_for_run(run_name, probabilities, labels, class_names, epoch_info):
                    generated.append("pr_curves")

            print(f"  ✓ Generated {len(generated)} plot(s)")
            all_generated.extend([f"{run_name}/{g}" for g in generated])

        print("\n" + "=" * 60)
        print(f"✅ Complete: {len(all_generated)} plot(s) generated for {len(self.runs_data)} run(s)")
        print(f"📁 {self.output_folder.absolute()}")
        print("=" * 60)

        return all_generated


#############################################################################################################
# HELPER FUNCTION: Extract checkpoint map from Confidence Analyzer output

# Extract a checkpoint map (run_name -> epoch, 1-indexed / user-visible)
# from a Confidence Analyzer CSV.
# Args:
#   analyzer_csv_path (str | Path): Path to the analyzer output CSV
#   run_type (str): 'auto', 'single' or 'crossval'
# Returns:
#   dict: Mapping of run_name (or 'single_run') to selected epoch
def extract_checkpoint_map_from_analyzer(analyzer_csv_path, run_type: str = 'auto') -> dict:
    analyzer_csv_path = Path(analyzer_csv_path)
    if not analyzer_csv_path.exists():
        print(f"⚠️ Analyzer CSV not found: {analyzer_csv_path}")
        return {}

    df = pd.read_csv(analyzer_csv_path)
    checkpoint_map = {}

    for _, row in df.iterrows():
        dataset_num = int(row['dataset'])
        checkpoint_name = row['checkpoint']

        match = re.search(r'_e(\d+)', checkpoint_name)
        if match:
            epoch = int(match.group(1))

            ds_match = re.search(r'_ds(\d+)', checkpoint_name)
            if ds_match:
                run_name = f"ds{dataset_num:02d}"
                checkpoint_map[run_name] = epoch
                print(f"  {run_name} -> epoch {epoch}")
            else:
                if run_type == 'single' or run_type == 'auto':
                    checkpoint_map['single_run'] = epoch
                    print(f"  single_run -> epoch {epoch}")

    return checkpoint_map


#############################################################################################################
# MAIN

if __name__ == "__main__":

    ########################
    # For CROSS-VALIDATION #
    ########################

    # TENSORBOARD_LOGDIR = r"D:\CLN7 AI - Results\ACV&CA\acv&ca_6_augment_ADAMW_val_from_test\Training10_PAPER\acv_results\logs"
    # PROBABILITY_DIR = None
    # ANALYZER_CSV = r"D:\CLN7 AI - Results\ACV&CA\acv&ca_6_augment_ADAMW_val_from_test\Training10_PAPER\ca_results_correct_80-100%_bal_acc\used_checkpoints.csv"

    #######################
    # For SINGLE TRAINING #
    #######################

    TENSORBOARD_LOGDIR = r"D:\CLN7 AI - Paper\Data\9cl_CNN_Training\Training 2\logs\20260522-082550"
    PROBABILITY_DIR = TENSORBOARD_LOGDIR + r"\probabilities"
    ANALYZER_CSV = None

    MODE = 'single'
    BEST_EPOCH = 'balanced_accuracy'

    checkpoint_map = {}
    if ANALYZER_CSV is not None:
        checkpoint_map = extract_checkpoint_map_from_analyzer(ANALYZER_CSV, run_type=MODE)

    if checkpoint_map:
        print(f"\n✓ Loaded {len(checkpoint_map)} pre-selected checkpoints from Confidence Analyzer")
    else:
        print(f"\n⚠ No pre-selected checkpoints found. Using '{BEST_EPOCH}' for epoch selection.")

    plotter = TensorBoardPlotter(
        logdir=TENSORBOARD_LOGDIR,
        prob_dir=PROBABILITY_DIR,
        output_folder=setting.get('pth_output', 'output'),
        output_format='tiff',
        raster_dpi=300,
        mode=MODE,
        plot_loss=True,
        plot_accuracy=True,
        plot_f1=True,
        plot_lr=True,
        plot_auc=False,
        plot_ap=False,
        plot_per_class_accuracy=True,
        plot_class_weights=False,
        plot_class_counts=False,
        plot_composite_score=False,
        plot_balanced_accuracy=True,
        plot_class_std=False,
        plot_min_class_acc=False,
        plot_gpu_memory=False,
        plot_roc_curves=True,
        plot_pr_curves=True,
        roc_curve_epoch=BEST_EPOCH,
        pr_curve_epoch=BEST_EPOCH,
        checkpoint_map=checkpoint_map,
        train_line_style='-',
        val_line_style='-',
        standard_line_style='--',
        line_width=2.0,
        show_markers=True,
        marker_size=5,
        marker_frequency=1,
        smoothing=0.0,
        per_class_colormap='tab10',
        class_weights_colors={'KO': '#000000', 'WT': '#FFFFFF'},
        class_counts_colors={'KO': '#000000', 'WT': '#FFFFFF'},
        class_weights_colormap='viridis',
        class_counts_colormap='plasma',
        bar_edge_width=0.5,
        bar_label_fontsize=12,
        bar_label_offset=0.05,
        use_fixed_axes_height=True,
        fixed_axes_height=5.0,
        font_family='Arial',
        master_font_size=22,
        axis_label_size=12,
        title_font_size=14,
        tick_label_size=10,
        legend_font_size=10,
        loss_legend_loc='upper right',
        loss_legend_outside=False,
        accuracy_legend_loc='lower right',
        accuracy_legend_outside=False,
        f1_legend_loc='lower right',
        f1_legend_outside=False,
        per_class_legend_loc='best',
        per_class_legend_outside=True,
        roc_legend_loc='lower right',
        roc_legend_outside=True,
        pr_legend_loc='lower left',
        pr_legend_outside=True,
        class_std_legend_loc='upper right',
        class_std_legend_outside=False,
        min_class_acc_legend_loc='lower right',
        min_class_acc_legend_outside=False,
        composite_legend_loc='lower right',
        composite_legend_outside=False,
        balanced_acc_legend_loc='lower right',
        balanced_acc_legend_outside=False,
        gpu_memory_legend_loc='best',
        gpu_memory_legend_outside=False,
        lr_legend_loc='best',
        lr_legend_outside=False,
        auc_legend_loc='lower right',
        auc_legend_outside=False,
        ap_legend_loc='lower right',
        ap_legend_outside=False,
        legend_outside_bbox=(1.05, 0.5),
        show_legend=True,
        show_grid=True,
        grid_alpha=0.3,
        grid_linestyle='--',
        grid_color='gray',
        x_axis_min=0,
        y_axis_min=0,
        min_class_acc_threshold=0.65,
    )

    if plotter.process_runs():
        plotter.generate_all_plots()
    else:
        print("\n❌ Failed to process runs.")
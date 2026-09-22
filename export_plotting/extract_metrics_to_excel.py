# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
from collections import defaultdict
import re
# ===== Third-Party Imports =====
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# ===== Own Modules =====
from settings import setting


# Default colour palette for ROC / PR chart series
CHART_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


class TensorBoardExporter:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the exporter with paths and output settings.
    # If roc_epoch / pr_epoch are not given, the selector is derived from
    # chckpt_selection_method in settings at the start of export_with_charts.
    # This keeps the ROC/PR curves in sync with whichever metric was used
    # to pick the best checkpoint during training.
    # Args:
    #   logdir (str | Path | None): TensorBoard log directory
    #   output_file (str | Path | None): Destination .xlsx path
    #   prob_dir (str | Path | None): Root directory containing probability .npz files
    #   roc_epoch (str | int | None): Explicit override for ROC epoch selector
    #   pr_epoch (str | int | None): Explicit override for PR epoch selector
    #   mode (str | None): 'auto', 'crossval' or 'single'
    def __init__(
        self,
        logdir: str | Path | None = None,
        output_file: str | Path | None = None,
        prob_dir: str | Path | None = None,
        roc_epoch: str | int | None = None,
        pr_epoch: str | int | None = None,
        mode: str | None = None,
    ) -> None:

        if mode is None:
            mode = setting.get('export_mode', 'auto')

        # Keep explicit overrides only; the effective selector is resolved
        # lazily at export time from chckpt_selection_method.
        self.roc_epoch_override = roc_epoch
        self.pr_epoch_override = pr_epoch
        self.roc_epoch: str | int | None = None
        self.pr_epoch: str | int | None = None

        if output_file is None:
            output_file = Path(setting['pth_output']) / "train_metrics.xlsx"

        # logdir must always be supplied by the caller
        self.logdir: Path = Path(logdir) if logdir else Path(".")

        # prob_dir: if not passed, default to logdir (works for both layouts,
        # since _get_probability_subdir appends the correct subfolder)
        if prob_dir is None:
            prob_dir = self.logdir

        self.output_file: Path = Path(output_file) if output_file else Path("./output.xlsx")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.prob_dir: Path | None = Path(prob_dir) if prob_dir else None
        self.mode: str = mode
        self.runs_data: dict = {}
        self.run_type: str | None = None
        self.saved_checkpoints: dict = {}
        self._prob_cache: dict = {}

        print("\nTensorBoardExporter configuration:")
        print(f"  Logdir: {self.logdir}")
        print(f"  Output: {self.output_file}")
        print(f"  Prob dir: {self.prob_dir}")
        print(f"  ROC epoch override: {self.roc_epoch_override}")
        print(f"  PR epoch override: {self.pr_epoch_override}")
        print(f"  Mode: {self.mode}")

    #############################################################################################################
    # METHODS

    # Calculate balanced accuracy from per-class accuracy columns as a fallback.
    # Args:
    #   run_df (pd.DataFrame): DataFrame containing the run metrics
    #   epoch_1indexed (int): Epoch number (1-indexed, user-visible)
    # Returns:
    #   float | None: Mean per-class accuracy, or None if not computable
    def _calculate_balanced_accuracy_from_per_class(
        self,
        run_df: pd.DataFrame,
        epoch_1indexed: int,
    ) -> float | None:

        per_class_cols = [col for col in run_df.columns if col.startswith('Accuracy_per_class_')]
        if not per_class_cols:
            return None

        epoch_row = run_df[run_df['epoch'] == epoch_1indexed]
        if epoch_row.empty:
            return None

        class_accs = []
        for col in per_class_cols:
            val = epoch_row[col].values[0]
            if not np.isnan(val):
                class_accs.append(val)

        if not class_accs:
            return None

        return float(np.mean(class_accs))

    # Find the best epoch for a given metric column.
    # Falls back to computed balanced accuracy if the column is missing.
    # Returns 1-indexed epochs (user-visible), matching df['epoch'].
    # Args:
    #   run_df (pd.DataFrame): DataFrame containing the run metrics
    #   metric_name (str): Column name to optimise
    # Returns:
    #   int | None: Best epoch (1-indexed), or None if not found
    def _get_best_epoch_by_metric(
        self,
        run_df: pd.DataFrame,
        metric_name: str = 'Composite_score',
    ) -> int | None:

        if metric_name not in run_df.columns:
            print(f"      Metric '{metric_name}' not found in DataFrame")

            if metric_name == 'Balanced_accuracy':
                best_epoch = None
                best_score = -float('inf')

                for epoch in run_df['epoch'].values:
                    bal_acc = self._calculate_balanced_accuracy_from_per_class(run_df, epoch)
                    if bal_acc is not None and bal_acc > best_score:
                        best_score = bal_acc
                        best_epoch = int(epoch)

                if best_epoch is not None:
                    print(f"      Calculated balanced accuracy from per-class metrics (fallback): "
                          f"best epoch={best_epoch}, score={best_score:.4f}")
                    return best_epoch

            return None

        scores = run_df[metric_name].values
        epochs = run_df['epoch'].values
        valid = ~np.isnan(scores)

        if not np.any(valid):
            return None

        best_idx = np.argmax(scores[valid])
        best_epoch = int(epochs[valid][best_idx])
        best_score = scores[valid][best_idx]

        print(f"      Best epoch by {metric_name}: {best_epoch} (score={best_score:.4f})")
        return best_epoch

    # Check whether a run folder name matches the cross-validation pattern (dsXX).
    # Args:
    #   run_name (str): Run folder name
    # Returns:
    #   bool: True if the name matches the cross-validation pattern
    def _is_cross_validation_run(self, run_name: str) -> bool:
        return bool(re.match(r'^ds\d{2}$', run_name))

    # Check whether a run folder name matches the single-training timestamp pattern.
    # Args:
    #   run_name (str): Run folder name
    # Returns:
    #   bool: True if the name matches the timestamp pattern
    def _is_timestamp_run(self, run_name: str) -> bool:
        return bool(re.match(r'^\d{8}-\d{6}$', run_name))

    # Detect the run type from the list of discovered runs.
    # Respects an explicitly forced mode set via settings.
    # Args:
    #   runs (list): List of (run_path, run_name, event_files) tuples
    # Returns:
    #   str: 'crossval' or 'single'
    def _detect_run_type(self, runs: list) -> str:

        if self.mode == 'crossval':
            print("✓ Mode forced: CROSS-VALIDATION")
            return 'crossval'

        if self.mode == 'single':
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

        if timestamp_count > 0 and crossval_count == 0:
            print(f"✓ Auto-detected: SINGLE TRAINING mode ({timestamp_count} timestamp folders)")
            return 'single'

        if crossval_count > 0 and timestamp_count > 0:
            print("⚠ Mixed run types detected! Using CROSS-VALIDATION mode as default.")
            print(f"   ({crossval_count} dsXX, {timestamp_count} timestamp)")
            return 'crossval'

        print("⚠ Could not determine run type. Using CROSS-VALIDATION mode as default.")
        return 'crossval'

    # Convert a run folder name into a display name for the Excel sheet.
    # Args:
    #   run_name (str): Run folder name
    # Returns:
    #   str: Display name
    def _get_display_name(self, run_name: str) -> str:

        if self.run_type == 'crossval':
            return run_name

        if self._is_timestamp_run(run_name):
            return run_name[:31]

        return "training_run"

    # Resolve the probability subdirectory for a run, supporting new and old layouts.
    # Args:
    #   run_name (str): Run folder name
    # Returns:
    #   Path | None: Probability directory, or None if unavailable
    def _get_probability_subdir(self, run_name: str) -> Path | None:

        if not self.prob_dir:
            return None

        if self.run_type == 'crossval':
            # Support the new dataset_X/logs/probabilities layout
            if run_name.startswith("ds") and run_name[2:].isdigit():
                dataset_folder = f"dataset_{int(run_name[2:])}"
                candidates = [
                    self.prob_dir / dataset_folder / "logs" / "probabilities",
                    self.logdir / dataset_folder / "logs" / "probabilities",
                    self.logdir / "logs" / dataset_folder / "probabilities",
                ]
                for c in candidates:
                    if c.exists():
                        return c

            # Old layout: dsXX/probabilities/
            prob_subdir = self.prob_dir / run_name / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            prob_subdir = self.logdir / run_name / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            # Old layout alt: logdir/logs/dsXX/probabilities/
            prob_subdir = self.logdir / "logs" / run_name / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            # Fallback so downstream glob reports a real path
            if run_name.startswith("ds") and run_name[2:].isdigit():
                dataset_folder = f"dataset_{int(run_name[2:])}"
                return self.prob_dir / dataset_folder / "logs" / "probabilities"

            return self.prob_dir / run_name / "probabilities"

        # Single training
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

    # Collect the set of epochs (1-indexed, user-visible) for which checkpoints were saved.
    # Cross-validation checkpoints live in one of two layouts:
    #   - New system: <run_path>/checkpoints/
    #   - Old system: <run_path>/<run_name>/checkpoints/  or  <logdir>/<run_name>/checkpoints/
    # Args:
    #   run_path (Path): Run directory
    #   run_name (str): Run folder name
    # Returns:
    #   set: Set of 1-indexed epochs with saved checkpoints
    def _get_saved_checkpoints(self, run_path: Path, run_name: str) -> set:

        saved_epochs: set = set()
        checkpoints_dir: Path | None = None

        if self.run_type == 'crossval':
            candidates = [
                run_path / "checkpoints",              # NEW: dataset_X/checkpoints/
                run_path / "logs" / "checkpoints",     # in case logs has its own
                run_path / run_name / "checkpoints",   # old fallback
                self.logdir / run_name / "checkpoints",
            ]
            for c in candidates:
                if c.exists() and list(c.glob("*.pt")):
                    checkpoints_dir = c
                    break
        else:
            checkpoints_dir = run_path.parent.parent / "checkpoints"
            if not checkpoints_dir.exists():
                checkpoints_dir = run_path.parent / "checkpoints"
            if not checkpoints_dir.exists():
                checkpoints_dir = run_path / "checkpoints"

        if not checkpoints_dir or not checkpoints_dir.exists():
            print(f"    No checkpoints directory found for {run_name}")
            return saved_epochs

        checkpoint_files = list(checkpoints_dir.glob("*.pt"))
        print(f"    Found {len(checkpoint_files)} checkpoint files in {checkpoints_dir}")

        # Checkpoint filenames use 1-indexed epochs (see train.py: f"e{epoch+1:02d}")
        for f in checkpoint_files:
            try:
                match = re.search(r'_e(\d+)', f.stem)
                if match:
                    epoch_from_filename = int(match.group(1))
                    saved_epochs.add(epoch_from_filename)
                else:
                    match = re.search(r'(\d+)\.pt$', f.name)
                    if match:
                        epoch_from_filename = int(match.group(1))
                        saved_epochs.add(epoch_from_filename)
            except (ValueError, IndexError):
                continue

        return saved_epochs

    # Discover all TensorBoard runs in the configured logdir.
    # Supports the new dataset_X/logs layout, the old logs/dsXX layout,
    # a single-run layout, and a generic fallback.
    # Returns:
    #   list: List of (run_path, run_name, event_files) tuples
    def find_runs(self) -> list:

        runs: list = []

        # Case 1: logdir itself contains event files (single run)
        if list(self.logdir.glob("events.out.tfevents.*")):
            runs.append((self.logdir, self.logdir.name, [self.logdir]))
            print(f"✓ Single run: {self.logdir.name}")
        else:
            # Case 2 (NEW layout): logdir contains dataset_X/ folders, each with logs/
            new_layout_runs = []

            for child in self.logdir.iterdir():
                if not child.is_dir():
                    continue

                logs_dir = child / "logs"
                if logs_dir.is_dir() and list(logs_dir.rglob("events.out.tfevents.*")):

                    if child.name.startswith("dataset_"):
                        num = child.name.replace("dataset_", "")
                        try:
                            run_name = f"ds{int(num):02d}"
                        except ValueError:
                            run_name = child.name
                    elif child.name.startswith("ds"):
                        run_name = child.name
                    else:
                        run_name = child.name

                    event_files = list(child.rglob("events.out.tfevents.*"))
                    new_layout_runs.append((child, run_name, event_files))

            if new_layout_runs:
                runs = new_layout_runs
                print(f"✓ Found {len(runs)} training run(s) (new layout: dataset_X/logs/)")
            else:
                # Case 3 (OLD layout): logdir contains a single 'logs' folder
                logs_dir = self.logdir / "logs"
                if logs_dir.is_dir():
                    for child in logs_dir.iterdir():
                        if not child.is_dir():
                            continue
                        event_files = list(child.rglob("events.out.tfevents.*"))
                        if event_files:
                            runs.append((child, child.name, event_files))
                    if runs:
                        print(f"✓ Found {len(runs)} training run(s) (old layout: logs/dsXX/)")

                # Case 4 (fallback): every direct child with event files
                if not runs:
                    for child in self.logdir.iterdir():
                        if not child.is_dir():
                            continue
                        event_files = list(child.rglob("events.out.tfevents.*"))
                        if event_files:
                            runs.append((child, child.name, event_files))
                    if runs:
                        print(f"✓ Found {len(runs)} training run(s) (fallback layout)")

        if not runs:
            print(f"❌ No runs found in {self.logdir}")
            return []

        self.run_type = self._detect_run_type(runs)
        return runs

    # Extract a scalar series from a list of TensorBoard events.
    # Handles both plain scalar events and dict-valued events.
    # Args:
    #   events (list): List of TensorBoard scalar events
    #   metric_name (str): Fallback key for dict-valued events
    #   target_value (str | None): Preferred key for dict-valued events
    # Returns:
    #   tuple: (sorted_steps, averaged_values) or (None, None)
    @staticmethod
    def _extract_metric_from_events(
        events: list,
        metric_name: str,
        target_value: str | None = None,
    ) -> tuple:

        steps, values = [], []

        for e in events:
            if isinstance(e.value, dict):
                if target_value and target_value in e.value:
                    steps.append(e.step)
                    values.append(e.value[target_value])
                elif metric_name in e.value:
                    steps.append(e.step)
                    values.append(e.value[metric_name])
            else:
                steps.append(e.step)
                values.append(e.value)

        if not steps:
            return None, None

        d = defaultdict(list)
        for s, v in zip(steps, values):
            d[s].append(v)

        steps_sorted = sorted(d.keys())
        return steps_sorted, [np.mean(d[s]) for s in steps_sorted]

    # Extract all scalar metrics for a single run into a DataFrame.
    # Adds the balanced accuracy fallback and the Checkpoint column.
    # The 'epoch' column is 1-indexed (user-visible), matching train.py's display.
    # The Checkpoint column contains 'X' for cross-validation (checkpoint saved)
    # or 'Saved' for single training, and is empty otherwise.
    # Args:
    #   run_path (Path): Run directory
    #   run_name (str): Run folder name
    # Returns:
    #   tuple: (DataFrame | None, saved_checkpoints_set)
    def extract_run_data(self, run_path: Path, run_name: str) -> tuple:

        run_path = Path(run_path)
        event_files = list(run_path.rglob("events.out.tfevents.*"))
        if not event_files:
            return None, None

        saved_checkpoints = self._get_saved_checkpoints(run_path, run_name)
        self.saved_checkpoints[run_name] = saved_checkpoints
        print(f"    Saved checkpoints (1-indexed): {sorted(saved_checkpoints)}")

        all_metrics: dict = defaultdict(list)

        for ef in event_files:
            try:
                ea = event_accumulator.EventAccumulator(
                    str(ef),
                    size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE,
                )
                ea.Reload()

                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)

                    if tag == 'Metrics/F1':
                        s, v = self._extract_metric_from_events(events, 'Macro', 'Macro')
                        if s:
                            all_metrics['F1_macro'].extend(zip(s, v))
                        s, v = self._extract_metric_from_events(events, 'Weighted', 'Weighted')
                        if s:
                            all_metrics['F1_weighted'].extend(zip(s, v))
                        continue

                    s = [e.step for e in events]
                    v = [e.value for e in events]
                    if not s:
                        continue

                    tag_lower = tag.lower()

                    if 'loss' in tag_lower:
                        mname = 'Loss_train' if 'train' in tag_lower else ('Loss_val' if 'val' in tag_lower else None)

                    elif 'accuracy' in tag_lower or 'acc' in tag_lower:
                        if 'train' in tag_lower:
                            mname = 'Accuracy_train_standard' if 'standard' in tag_lower else 'Accuracy_train_weighted'
                        elif 'val' in tag_lower:
                            mname = 'Accuracy_val_standard' if 'standard' in tag_lower else 'Accuracy_val_weighted'
                        elif 'class' in tag_lower:
                            if 'std' in tag_lower:
                                mname = 'Class_Accuracy_StdDev'
                            elif 'min' in tag_lower:
                                mname = 'Min_Class_Accuracy'
                            else:
                                mname = f'Accuracy_per_class_{tag.split("/")[-1]}'
                        else:
                            continue

                    elif 'balanced' in tag_lower and 'accuracy' in tag_lower:
                        mname = 'Balanced_accuracy'

                    elif 'f1' in tag_lower:
                        mname = 'F1_weighted' if 'weighted' in tag_lower else 'F1_macro'

                    elif 'auc' in tag_lower:
                        mname = 'AUC_overall' if 'class' not in tag_lower else f'AUC_{tag.split("/")[-1]}'

                    elif 'ap' in tag_lower or 'average_precision' in tag_lower:
                        mname = 'AP_overall' if 'class' not in tag_lower else f'AP_{tag.split("/")[-1]}'

                    elif 'lr' in tag_lower or 'learning_rate' in tag_lower:
                        mname = 'Learning_rate'

                    elif 'composite' in tag_lower:
                        mname = 'Composite_score'

                    elif 'gpu_memory' in tag_lower:
                        mname = 'GPU_memory_usage'

                    elif 'class_count' in tag_lower and 'data' in tag_lower:
                        mname = f'Class_count_{tag.split("/")[-1]}'

                    elif 'class_weight' in tag_lower and 'data' in tag_lower:
                        mname = f'Class_weight_{tag.split("/")[-1]}'

                    else:
                        continue

                    if mname:
                        all_metrics[mname].extend(zip(s, v))

            except Exception as e:
                print(f"    WARNING: Failed to read event file {ef.name}: {e}")
                continue

        if not all_metrics:
            return None, saved_checkpoints

        # TensorBoard steps are 0-indexed; convert to 1-indexed (user-visible)
        all_epochs_0indexed = sorted(set().union(*[set(dict(pairs).keys()) for pairs in all_metrics.values()]))
        all_epochs_1indexed = [ep + 1 for ep in all_epochs_0indexed]

        temp_data = {'epoch': all_epochs_1indexed}
        for mname, pairs in all_metrics.items():
            d = dict(pairs)
            temp_data[mname] = [d.get(ep_0indexed, np.nan) for ep_0indexed in all_epochs_0indexed]

        df = pd.DataFrame(temp_data)

        if 'Balanced_accuracy' not in df.columns:
            per_class_cols = [col for col in df.columns if col.startswith('Accuracy_per_class_')]

            if per_class_cols:
                bal_acc_values = []
                for idx, epoch in enumerate(all_epochs_1indexed):
                    class_accs = []
                    for col in per_class_cols:
                        val = df[col].values[idx]
                        if not np.isnan(val):
                            class_accs.append(val)
                    if class_accs:
                        bal_acc_values.append(np.mean(class_accs))
                    else:
                        bal_acc_values.append(np.nan)
                df['Balanced_accuracy'] = bal_acc_values
                print("    Added Balanced_accuracy column (calculated from per-class accuracies as fallback)")
            else:
                df['Balanced_accuracy'] = np.nan
        else:
            print("    Balanced_accuracy column loaded directly from TensorBoard")

        # Build the Checkpoint column.
        # Cross-validation: 'X' when a checkpoint was saved at this epoch.
        # Single training: 'Saved' when a checkpoint was saved at this epoch.
        # Everything else: empty string.
        checkpoint_column = []
        for ep_1indexed in all_epochs_1indexed:
            is_saved = ep_1indexed in saved_checkpoints

            if self.run_type == 'single':
                cell_value = 'Saved' if is_saved else ''
            else:
                cell_value = 'X' if is_saved else ''

            checkpoint_column.append(cell_value)

        df.insert(1, 'Checkpoint', checkpoint_column)

        cols = ['epoch', 'Checkpoint'] + [c for c in df.columns if c not in ['epoch', 'Checkpoint']]
        df = df[cols]

        return df, saved_checkpoints

    # Load probability arrays for a run at a chosen epoch.
    # Results are cached per (run_name, epoch_choice).
    # .npz filenames are 0-indexed internally; they are converted to 1-indexed
    # (user-visible) on load, so all comparisons below use user-visible epochs.
    # Args:
    #   run_df (pd.DataFrame): DataFrame with run metrics (used for metric-based selection)
    #   epoch_choice (str | int): 'last', 'composite_score', 'balanced_accuracy', or an int
    #   run_name (str | None): Run folder name
    # Returns:
    #   tuple: (probs, labels, class_names, selected_epoch_1indexed) or (None, None, None, None)
    def load_probabilities_for_run(
        self,
        run_df: pd.DataFrame,
        epoch_choice: str | int,
        run_name: str | None = None,
    ) -> tuple:

        cache_key = (run_name, str(epoch_choice))
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key]

        if not self.prob_dir or not self.prob_dir.exists():
            return None, None, None, None

        prob_subdir = self._get_probability_subdir(run_name)

        if not prob_subdir or not prob_subdir.exists():
            prob_subdir = self.prob_dir

        npz_files = sorted(prob_subdir.glob("probabilities_epoch_*.npz"))

        if not npz_files and run_name:
            npz_files = sorted(self.prob_dir.glob(f"*{run_name}*_probabilities_epoch_*.npz"))

        if not npz_files:
            alt_dir = self.logdir / run_name / "probabilities" if run_name else None
            if alt_dir and alt_dir.exists():
                npz_files = sorted(alt_dir.glob("probabilities_epoch_*.npz"))

        if not npz_files:
            print(f"      No probability files found in {prob_subdir}")
            return None, None, None, None

        # .npz filenames are 0-indexed; convert to 1-indexed (user-visible)
        # to match df['epoch'].
        epoch_files = []
        for f in npz_files:
            try:
                ep_internal = int(f.stem.split('_')[-1])
                epoch_files.append((ep_internal + 1, f))
            except (ValueError, IndexError):
                continue

        if not epoch_files:
            return None, None, None, None

        epoch_files.sort(key=lambda x: x[0])

        selected_epoch = None
        selected_file = None

        if epoch_choice == 'last':
            selected_epoch, selected_file = epoch_files[-1]
            print(f"      Using last epoch: {selected_epoch}")

        elif epoch_choice == 'composite_score':
            best_epoch = self._get_best_epoch_by_metric(run_df, 'Composite_score')
            if best_epoch is not None:
                for ep, f in epoch_files:
                    if ep == best_epoch:
                        selected_epoch, selected_file = ep, f
                        break
            if selected_epoch is None:
                print("      WARNING: Best composite_score epoch not found, using last epoch")
                selected_epoch, selected_file = epoch_files[-1]

        elif epoch_choice == 'balanced_accuracy':
            best_epoch = self._get_best_epoch_by_metric(run_df, 'Balanced_accuracy')
            if best_epoch is not None:
                for ep, f in epoch_files:
                    if ep == best_epoch:
                        selected_epoch, selected_file = ep, f
                        break
            if selected_epoch is None:
                print("      WARNING: Best balanced_accuracy epoch not found, using last epoch")
                selected_epoch, selected_file = epoch_files[-1]

        else:
            try:
                target = int(epoch_choice)
                for ep, f in epoch_files:
                    if ep == target:
                        selected_epoch, selected_file = ep, f
                        break
                else:
                    print(f"      WARNING: Specified epoch {target} not found, using last epoch")
                    selected_epoch, selected_file = epoch_files[-1]
            except (ValueError, TypeError):
                print(f"      WARNING: Invalid epoch choice '{epoch_choice}', using last epoch")
                selected_epoch, selected_file = epoch_files[-1]

        print(f"      Using epoch {selected_epoch} from {selected_file.name}")
        data = np.load(selected_file)
        probs = data['probabilities']
        labels = data['labels']
        class_names = data['classes'].tolist() if 'classes' in data else [f'Class_{i}' for i in range(probs.shape[1])]

        result = (probs, labels, class_names, selected_epoch)
        self._prob_cache[cache_key] = result
        return result

    # Compute one-vs-rest ROC curves for all classes.
    # Args:
    #   probs (np.ndarray): Predicted probabilities, shape (N, C)
    #   labels (np.ndarray): Ground-truth labels, shape (N,)
    #   class_names (list): Class names
    # Returns:
    #   list: List of (class_name, auc_value, fpr, tpr) tuples
    @staticmethod
    def compute_roc(probs: np.ndarray, labels: np.ndarray, class_names: list) -> list:

        curves = []
        for i, cls in enumerate(class_names):
            y_true = (labels == i).astype(int)
            y_score = probs[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            curves.append((cls, roc_auc, fpr, tpr))

        return curves

    # Compute one-vs-rest precision-recall curves for all classes.
    # Args:
    #   probs (np.ndarray): Predicted probabilities, shape (N, C)
    #   labels (np.ndarray): Ground-truth labels, shape (N,)
    #   class_names (list): Class names
    # Returns:
    #   list: List of (class_name, ap_value, recall, precision) tuples
    @staticmethod
    def compute_pr(probs: np.ndarray, labels: np.ndarray, class_names: list) -> list:

        curves = []
        for i, cls in enumerate(class_names):
            y_true = (labels == i).astype(int)
            y_score = probs[:, i]
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            curves.append((cls, ap, recall, precision))

        return curves

    # Export all discovered runs to an Excel workbook with scalar charts,
    # ROC curves and precision-recall curves.
    # Prints a warning if the epoch chosen for ROC/PR is not one of the
    # epochs where a checkpoint was actually saved during training,
    # which indicates a mismatch between the training-time checkpoint
    # selection method and the current export setting.
    # Returns:
    #   None
    def export_with_charts(self) -> None:

        # Resolve once for the whole export so all folds use the same selector
        self.roc_epoch = self._resolve_selector()
        self.pr_epoch = self.roc_epoch

        print("=" * 60)
        print("TENSORBOARD EXPORTER – 3-COLUMN CHART GRID")
        print("=" * 60)
        print(f"Logdir: {self.logdir}")
        print(f"Output: {self.output_file}")

        if self.prob_dir:
            print(f"Probability dir: {self.prob_dir}")

        print(f"ROC/PR epoch selector: {self.roc_epoch}")
        print(f"  (resolved from chckpt_selection_method = {setting.get('chckpt_selection_method')})")

        runs = self.find_runs()
        if not runs:
            print("No runs found.")
            return

        print("\nExtracting scalar metrics from each run...")

        for run_path, run_name, event_files in runs:
            display_name = self._get_display_name(run_name)
            print(f"  {run_name} -> sheet '{display_name}'")

            df, saved_checkpoints = self.extract_run_data(run_path, run_name)

            if df is not None and len(df) > 0:
                self.runs_data[display_name] = (df, run_name, saved_checkpoints)
                print(f"    -> {len(df)} epochs, {len(df.columns)-1} metrics")

                if saved_checkpoints:
                    print(f"    -> Saved checkpoints at epochs: {sorted(saved_checkpoints)}")

                # Warn if the epoch chosen for ROC/PR is not one of the epochs
                # where a checkpoint was saved. This indicates that the export
                # setting (chckpt_selection_method) does not match the setting
                # used during training.
                self._warn_on_checkpoint_epoch_mismatch(df, saved_checkpoints)
            else:
                print("    -> No data extracted")

        if not self.runs_data:
            print("No scalar data to export.")
            return

        print("\nWriting Excel file...")

        with pd.ExcelWriter(self.output_file, engine='xlsxwriter') as writer:
            workbook = writer.book

            for display_name, (df, original_run_name, saved_checkpoints) in self.runs_data.items():
                sheet_name = display_name[:31].replace('[', '_').replace(']', '_').replace(':', '_')
                sheet_name = sheet_name.replace('*', '_').replace('?', '_').replace('/', '_')
                print(f"\n--- Run: {original_run_name} -> sheet '{sheet_name}'")

                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, startcol=0)
                worksheet = writer.sheets[sheet_name]

                if self.run_type == 'single':
                    note = "Checkpoint column: 'Saved' = checkpoint saved at this epoch, Empty = no checkpoint"
                else:
                    note = "Checkpoint column: 'X' = checkpoint saved at this epoch"

                worksheet.write(0, len(df.columns), note, workbook.add_format({'italic': True, 'color': '#666666', 'font_size': 8}))

                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    worksheet.set_column(i, i, min(max_len, 30))

                scalar_cols = len(df.columns)
                roc_start_col = scalar_cols + 1
                pr_start_col = roc_start_col + 1        # safe default in case ROC is skipped
                roc_curves = None
                pr_curves = None
                detail_row_roc = None
                detail_row_pr = None
                roc_selected_epoch = None
                pr_selected_epoch = None

                if self.prob_dir and self.prob_dir.exists():

                    probs, labels, class_names, roc_selected_epoch = self.load_probabilities_for_run(
                        df, self.roc_epoch, original_run_name
                    )

                    if probs is not None:
                        roc_curves = self.compute_roc(probs, labels, class_names)

                        if roc_selected_epoch is not None:
                            header_text = f"ROC CURVES (RAW DATA) - Epoch {roc_selected_epoch}"
                        else:
                            header_text = "ROC CURVES (RAW DATA)"

                        worksheet.write(0, roc_start_col, header_text, workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
                        worksheet.write(1, roc_start_col, "Class", workbook.add_format({'bold': True}))
                        worksheet.write(1, roc_start_col + 1, "AUC", workbook.add_format({'bold': True}))

                        for i, (cls, auc_val, _, _) in enumerate(roc_curves):
                            worksheet.write(2 + i, roc_start_col, cls)
                            worksheet.write(2 + i, roc_start_col + 1, auc_val)

                        detail_row_roc = 2 + len(roc_curves) + 2
                        col_offset = 0
                        for cls, auc_val, fpr, tpr in roc_curves:
                            worksheet.write(detail_row_roc, roc_start_col + col_offset,
                                            f"{cls} (AUC={auc_val:.3f})", workbook.add_format({'bold': True}))
                            worksheet.write(detail_row_roc + 1, roc_start_col + col_offset, "FPR")
                            worksheet.write(detail_row_roc + 1, roc_start_col + col_offset + 1, "TPR")

                            for j, (x, y) in enumerate(zip(fpr, tpr)):
                                worksheet.write(detail_row_roc + 2 + j, roc_start_col + col_offset, x)
                                worksheet.write(detail_row_roc + 2 + j, roc_start_col + col_offset + 1, y)

                            col_offset += 3

                        roc_columns_used = col_offset
                        pr_start_col = roc_start_col + roc_columns_used + 1

                    probs_pr, labels_pr, class_names_pr, pr_selected_epoch = self.load_probabilities_for_run(
                        df, self.pr_epoch, original_run_name
                    )

                    if probs_pr is not None:
                        pr_curves = self.compute_pr(probs_pr, labels_pr, class_names_pr)

                        if pr_selected_epoch is not None:
                            header_text = f"PRECISION-RECALL CURVES (RAW DATA) - Epoch {pr_selected_epoch}"
                        else:
                            header_text = "PRECISION-RECALL CURVES (RAW DATA)"

                        worksheet.write(0, pr_start_col, header_text, workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
                        worksheet.write(1, pr_start_col, "Class", workbook.add_format({'bold': True}))
                        worksheet.write(1, pr_start_col + 1, "AP", workbook.add_format({'bold': True}))

                        for i, (cls, ap_val, _, _) in enumerate(pr_curves):
                            worksheet.write(2 + i, pr_start_col, cls)
                            worksheet.write(2 + i, pr_start_col + 1, ap_val)

                        detail_row_pr = 2 + len(pr_curves) + 2
                        col_offset = 0
                        for cls, ap_val, recall, precision in pr_curves:
                            worksheet.write(detail_row_pr, pr_start_col + col_offset,
                                            f"{cls} (AP={ap_val:.3f})", workbook.add_format({'bold': True}))
                            worksheet.write(detail_row_pr + 1, pr_start_col + col_offset, "Recall")
                            worksheet.write(detail_row_pr + 1, pr_start_col + col_offset + 1, "Precision")

                            for j, (x, y) in enumerate(zip(recall, precision)):
                                worksheet.write(detail_row_pr + 2 + j, pr_start_col + col_offset, x)
                                worksheet.write(detail_row_pr + 2 + j, pr_start_col + col_offset + 1, y)

                            col_offset += 3

                metric_cols = [c for c in df.columns if c not in ['epoch', 'Checkpoint']]
                chart_start_row = len(df) + 2
                num_columns = 3
                col_spacing = 5
                row_spacing = 16
                chart_width = 500
                chart_height = 300

                all_charts = []
                for metric in metric_cols:
                    all_charts.append(('metric', metric))
                if roc_curves is not None:
                    all_charts.append(('roc', None))
                if pr_curves is not None:
                    all_charts.append(('pr', None))

                for idx, (chart_type, metric_name) in enumerate(all_charts):
                    col_idx = idx % num_columns
                    row_idx = idx // num_columns
                    row = chart_start_row + row_idx * row_spacing
                    col = col_idx * col_spacing

                    if chart_type == 'metric':
                        metric_pos = df.columns.get_loc(metric_name)
                        num_epochs = len(df)

                        if num_epochs <= 20:
                            tick_interval = 2
                        elif num_epochs <= 40:
                            tick_interval = 5
                        elif num_epochs <= 80:
                            tick_interval = 10
                        else:
                            tick_interval = 20

                        chart = workbook.add_chart({'type': 'line'})
                        chart.add_series({
                            'name': metric_name,
                            'categories': [sheet_name, 1, 0, num_epochs, 0],
                            'values': [sheet_name, 1, metric_pos, num_epochs, metric_pos],
                            'line': {'color': 'black', 'width': 1.5},
                            'marker': {
                                'type': 'circle', 'size': 4,
                                'border': {'color': 'black'},
                                'fill': {'color': 'black'},
                            },
                        })
                        chart.set_title({'name': metric_name})
                        chart.set_x_axis({
                            'name': 'Epoch',
                            'position_axis': 'on_tick',
                            'min': 1,
                            'max': num_epochs,
                            'major_unit': tick_interval,
                            'num_format': '0',
                            'label_position': 'low',
                        })
                        chart.set_legend({'none': True})
                        chart.set_size({'width': chart_width, 'height': chart_height})
                        worksheet.insert_chart(row, col, chart)
                        print(f"    Metric chart '{metric_name}' placed at row {row}, col {col}")

                    elif chart_type == 'roc' and roc_curves is not None and detail_row_roc is not None:
                        roc_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                        col_offset = 0

                        for idx_c, (cls, auc_val, fpr, tpr) in enumerate(roc_curves):
                            data_len = len(fpr)
                            x_range = [sheet_name, detail_row_roc + 2, roc_start_col + col_offset,
                                       detail_row_roc + 2 + data_len - 1, roc_start_col + col_offset]
                            y_range = [sheet_name, detail_row_roc + 2, roc_start_col + col_offset + 1,
                                       detail_row_roc + 2 + data_len - 1, roc_start_col + col_offset + 1]

                            roc_chart.add_series({
                                'name': f"{cls} (AUC={auc_val:.3f})",
                                'categories': x_range,
                                'values': y_range,
                                'line': {'color': CHART_COLORS[idx_c % len(CHART_COLORS)], 'width': 2},
                                'marker': {'type': 'none'},
                            })
                            col_offset += 3

                        if roc_selected_epoch is not None:
                            chart_title = f'ROC Curves (Epoch {roc_selected_epoch})'
                        else:
                            chart_title = 'ROC Curves'

                        roc_chart.set_title({'name': chart_title})
                        roc_chart.set_x_axis({'name': 'False Positive Rate', 'min': 0, 'max': 1})
                        roc_chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1})
                        roc_chart.set_legend({'position': 'bottom'})
                        roc_chart.set_size({'width': chart_width, 'height': chart_height})
                        worksheet.insert_chart(row, col, roc_chart)
                        print(f"    ROC chart placed at row {row}, col {col}")

                    elif chart_type == 'pr' and pr_curves is not None and detail_row_pr is not None:
                        pr_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                        col_offset = 0

                        for idx_c, (cls, ap_val, recall, precision) in enumerate(pr_curves):
                            data_len = len(recall)
                            x_range = [sheet_name, detail_row_pr + 2, pr_start_col + col_offset,
                                       detail_row_pr + 2 + data_len - 1, pr_start_col + col_offset]
                            y_range = [sheet_name, detail_row_pr + 2, pr_start_col + col_offset + 1,
                                       detail_row_pr + 2 + data_len - 1, pr_start_col + col_offset + 1]

                            pr_chart.add_series({
                                'name': f"{cls} (AP={ap_val:.3f})",
                                'categories': x_range,
                                'values': y_range,
                                'line': {'color': CHART_COLORS[idx_c % len(CHART_COLORS)], 'width': 2},
                                'marker': {'type': 'none'},
                            })
                            col_offset += 3

                        if pr_selected_epoch is not None:
                            chart_title = f'Precision-Recall Curves (Epoch {pr_selected_epoch})'
                        else:
                            chart_title = 'Precision-Recall Curves'

                        pr_chart.set_title({'name': chart_title})
                        pr_chart.set_x_axis({'name': 'Recall', 'min': 0, 'max': 1})
                        pr_chart.set_y_axis({'name': 'Precision', 'min': 0, 'max': 1})
                        pr_chart.set_legend({'position': 'bottom'})
                        pr_chart.set_size({'width': chart_width, 'height': chart_height})
                        worksheet.insert_chart(row, col, pr_chart)
                        print(f"    PR chart placed at row {row}, col {col}")

                worksheet.freeze_panes(1, 0)

        print(f"\n✅ Export complete! File saved to: {self.output_file.absolute()}")

    # Warn if the epoch that ROC/PR will be drawn for is not one of the
    # epochs where a checkpoint was actually saved during training.
    # This indicates a mismatch between the training-time checkpoint
    # selection method and the current export setting.
    # Args:
    #   run_df (pd.DataFrame): DataFrame with the run metrics
    #   saved_checkpoints (set): Set of 1-indexed epochs with saved checkpoints
    # Returns:
    #   None
    def _warn_on_checkpoint_epoch_mismatch(
        self,
        run_df: pd.DataFrame,
        saved_checkpoints: set,
    ) -> None:

        if not saved_checkpoints:
            return

        # Map the export selector to the DataFrame column name
        if self.roc_epoch == 'balanced_accuracy':
            column_name = 'Balanced_accuracy'
            label = 'balanced accuracy'
        elif self.roc_epoch == 'composite_score':
            column_name = 'Composite_score'
            label = 'composite score'
        else:
            # 'last' or an explicit int — nothing meaningful to warn about
            return

        best_epoch = self._get_best_epoch_by_metric(run_df, column_name)
        if best_epoch is None:
            return

        if best_epoch not in saved_checkpoints:
            print(f"    ⚠️  WARNING: ROC/PR epoch {best_epoch} (best {label}) is not "
                  f"among the saved-checkpoint epochs {sorted(saved_checkpoints)}.")
            print(f"       This likely means the training-time chckpt_selection_method "
                  f"differs from the current export setting.")
            print(f"       ROC/PR curves will still be drawn for epoch {best_epoch}, "
                  f"but do not correspond to any saved checkpoint.")

    # Resolve the effective ROC/PR epoch selector.
    # Priority: explicit constructor override > chckpt_selection_method from
    # settings at the moment of export. 'both' is resolved to 'balanced_accuracy'
    # (the metric the paper reports).
    # ROC and PR always share the same selector so their curves come from the
    # same checkpoint.
    # Returns:
    #   str | int: Selector to pass to load_probabilities_for_run
    def _resolve_selector(self) -> str | int:

        # An override on either roc_epoch or pr_epoch wins for both.
        if self.roc_epoch_override is not None:
            return self.roc_epoch_override
        if self.pr_epoch_override is not None:
            return self.pr_epoch_override

        selection = setting.get('chckpt_selection_method', 'balanced_accuracy')
        if selection == 'both':
            selection = 'balanced_accuracy'

        return selection
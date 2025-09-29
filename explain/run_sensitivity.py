#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run end-to-end traffic→delay sensitivity analysis for a single path.

Workflow:
1. Load a single sample from a TFRecord file (after parsing + merging logic similar to training pipeline).
2. Select a target path (auto or user-provided).
3. Scan traffic on that path over a range while keeping all other paths fixed.
4. Run the pre-trained KAN RouteNet (delay model) and record predicted delay (loc output).
5. Plot curve and (optionally) derivative & auto annotate strongest curvature point.
6. Save artifacts (csv, png, metadata.json, sample snapshot).

Assumptions / Constraints:
- Default KAN weights: ./kan_model/137/kan_bspline/best_delay_kan_model.weights.h5
- Optional MLP comparison: ./kan_model/137/mlp_none/best_delay_model.weights.h5 (enable with --mlp_weights_dir)
- Target task is always 'delay'.
- We re-create model config consistent with routenet_tf2.py defaults unless overridden.

Outputs go to: ./explain_result (customizable via CLI).
"""

import os
import json
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Import utilities from existing training script (minimal duplication)
# We will dynamically import the routenet_tf2 module to access RouteNet & helpers.
import importlib.util

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR))
PROJECT_ROOT = ROOT_DIR  # adjust if needed

# -----------------------------
# Reuse parsing & scaling logic
# -----------------------------

def scale_fn(k, val):
    if k == 'traffic':
        return (val - 0.18) / 0.15
    if k == 'capacities':
        return val / 10.0
    return val

# Original parse function adapted (simplified for single sample extraction)
FEATURE_SPEC = {
    'traffic': tf.io.VarLenFeature(tf.float32),
    'delay': tf.io.VarLenFeature(tf.float32),
    'jitter': tf.io.VarLenFeature(tf.float32),
    'drops': tf.io.VarLenFeature(tf.float32),
    'packets': tf.io.VarLenFeature(tf.float32),
    'capacities': tf.io.VarLenFeature(tf.float32),
    'links': tf.io.VarLenFeature(tf.int64),
    'paths': tf.io.VarLenFeature(tf.int64),
    'sequences': tf.io.VarLenFeature(tf.int64),
    'n_links': tf.io.FixedLenFeature([], tf.int64),
    'n_paths': tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized):
    f = tf.io.parse_single_example(serialized, features=FEATURE_SPEC)
    for k, v in f.items():
        if isinstance(v, tf.SparseTensor):
            v = tf.sparse.to_dense(v)
        if k in ['traffic', 'capacities']:
            v = scale_fn(k, v)
        f[k] = v
    labels = {
        'delay': f.pop('delay'),
        'jitter': f.pop('jitter'),
        'drops': f.pop('drops'),
        'packets': f['packets']
    }
    return f, labels

# Merge logic (single sample, no batching). We mimic transformation_func's merging behavior.

def merge_features(single_features: Dict[str, tf.Tensor], single_labels: Dict[str, tf.Tensor]):
    merged = {
        'traffic': single_features['traffic'],
        'capacities': single_features['capacities'],
        'packets': single_features['packets'],
        'links': single_features['links'],
        'paths': single_features['paths'],
        'sequences': single_features['sequences'],
        'n_links': single_features['n_links'],
        'n_paths': single_features['n_paths'],
    }
    labels = {
        'delay': single_labels['delay'],
        'jitter': single_labels['jitter'],
        'drops': single_labels['drops'],
        'packets': single_labels['packets'],
    }
    return merged, labels

# -----------------------------
# Model loading
# -----------------------------

def load_routenet_model(weights_dir: str, is_kan: bool = True, kan_basis: str = 'bspline', mlp_weights_filename: Optional[str] = None):
    # Dynamically import routenet_tf2 (assume it is in project root / routenet/ folder)
    script_path = os.path.join(os.path.dirname(ROOT_DIR), 'routenet', 'routenet_tf2.py')
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Cannot locate routenet_tf2.py at {script_path}")
    spec = importlib.util.spec_from_file_location('routenet_tf2', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    # Recreate config (should align with training run). If there were deviations, allow CLI overrides.
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
        'kan_basis': kan_basis if is_kan else 'poly',
        'kan_grid_size': 5,
        'kan_spline_order': 3,
        'use_dropout': False,
        'dropout_rate': 0.1,
    }

    model, _ = module.create_model_and_loss_fn(config, target='delay', use_kan=is_kan, use_final_layer=True)

    if is_kan:
        weights_path = os.path.join(weights_dir, 'best_delay_kan_model.weights.h5')
    else:
        # Allow custom filename override
        fname = mlp_weights_filename or 'best_delay_model.weights.h5'
        weights_path = os.path.join(weights_dir, fname)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    # Build model by running a dummy forward pass with placeholder shapes after creating minimal dummy inputs.
    # We'll defer actual load until first forward call if needed (Keras can load after build implicitly if shape known)
    dummy_inputs = {
        'capacities': tf.zeros([1], dtype=tf.float32),
        'traffic': tf.zeros([1], dtype=tf.float32),
        'packets': tf.ones([1], dtype=tf.float32),
        'links': tf.zeros([1], dtype=tf.int64),
        'paths': tf.zeros([1], dtype=tf.int64),
        'sequences': tf.zeros([1], dtype=tf.int64),
        'n_links': tf.constant(1, dtype=tf.int64),
        'n_paths': tf.constant(1, dtype=tf.int64),
    }
    _ = model(dummy_inputs, training=False)
    model.load_weights(weights_path)
    return model

# -----------------------------
# Path selection
# -----------------------------

def choose_path(path_id: int, features: Dict[str, tf.Tensor]):
    n_paths = int(features['n_paths'].numpy())
    if path_id < 0 or path_id >= n_paths:
        raise ValueError(f"Provided path_id={path_id} out of range [0, {n_paths-1}]")
    return path_id


def auto_select_path(features: Dict[str, tf.Tensor]):
    n_paths = int(features['n_paths'].numpy())
    paths = features['paths'].numpy()
    # Compute lengths
    # lengths[path] = count occurrences in paths
    lengths = np.bincount(paths, minlength=n_paths)
    # Filter lengths >=2
    valid = np.where(lengths >= 2)[0]
    if len(valid) == 0:
        raise RuntimeError("No path with length >=2 found for analysis.")
    # Choose median length path index for stability
    candidate = valid[len(valid)//2]
    return int(candidate), lengths

# -----------------------------
# Traffic scanning
# -----------------------------

def inverse_scale_traffic(scaled: np.ndarray) -> np.ndarray:
    return scaled * 0.15 + 0.18

def scale_traffic(raw: np.ndarray) -> np.ndarray:
    return (raw - 0.18) / 0.15


def run_scan(model, features: Dict[str, tf.Tensor], path_id: int, num_steps: int, max_multiplier: float):
    traffic_scaled = features['traffic'].numpy().copy()
    traffic_raw = inverse_scale_traffic(traffic_scaled)

    baseline_raw = traffic_raw[path_id]
    max_raw = baseline_raw * max_multiplier
    values_raw = np.linspace(0.0, max_raw, num_steps)

    delays = []

    for raw_val in values_raw:
        modified_raw = traffic_raw.copy()
        modified_raw[path_id] = raw_val
        modified_scaled = scale_traffic(modified_raw)
        # Update tensor (avoid in-place mutation issues by creating new tf.constant)
        mod_features = {k: v for k, v in features.items()}
        mod_features['traffic'] = tf.convert_to_tensor(modified_scaled, dtype=tf.float32)
        preds = model(mod_features, training=False)  # shape [n_paths, 2]
        loc = preds[path_id, 0].numpy().item()
        delays.append(loc)

    return values_raw, np.array(delays), baseline_raw

# -----------------------------
# Curvature / derivative annotation
# -----------------------------

def compute_curvature_points(x: np.ndarray, y: np.ndarray):
    # Numerical first & second derivative (central differences)
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    # Strongest positive curvature index (argmax of d2y)
    idx = int(np.argmax(d2y))
    return idx, dy, d2y

# -----------------------------
# Plotting
# -----------------------------

def plot_results(x_raw, delays, baseline_raw, baseline_delay, curvature_idx, dy, d2y, output_dir, save_derivative: bool,
                 mlp_delays: Optional[np.ndarray] = None, mlp_curvature_idx: Optional[int] = None,
                 mlp_baseline_delay: Optional[float] = None):
    plt.figure(figsize=(7,5))
    plt.plot(x_raw, delays, marker='o', markersize=3, linewidth=1.2, label='KAN predicted delay')
    if mlp_delays is not None:
        plt.plot(x_raw, mlp_delays, marker='s', markersize=3, linewidth=1.2, label='MLP predicted delay', alpha=0.8)
    # Baseline marker
    plt.axvline(baseline_raw, color='gray', linestyle='--', linewidth=1, label='Baseline traffic')
    plt.scatter([baseline_raw], [baseline_delay], color='red', s=50, zorder=5, label='KAN baseline')
    if mlp_baseline_delay is not None:
        plt.scatter([baseline_raw], [mlp_baseline_delay], color='purple', s=45, zorder=5, label='MLP baseline')
    # Curvature annotation
    cx = x_raw[curvature_idx]
    cy = delays[curvature_idx]
    plt.scatter([cx], [cy], color='orange', s=60, zorder=6, label='KAN max curvature')
    plt.annotate(f'KAN max curvature\ntraffic={cx:.3f}\ndelay={cy:.3f}',
                 xy=(cx, cy), xytext=(0.55, 0.25), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='orange'), fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='orange', alpha=0.8))
    if mlp_delays is not None and mlp_curvature_idx is not None:
        mcx = x_raw[mlp_curvature_idx]
        mcy = mlp_delays[mlp_curvature_idx]
        plt.scatter([mcx], [mcy], color='green', s=55, zorder=6, label='MLP max curvature')
        plt.annotate(f'MLP max curvature\ntraffic={mcx:.3f}\ndelay={mcy:.3f}',
                     xy=(mcx, mcy), xytext=(0.05, 0.6), textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color='green'), fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.8))

    plt.title('Path Delay Sensitivity (Traffic Perturbation): KAN vs MLP' if mlp_delays is not None else 'Path Delay Sensitivity (Traffic Perturbation)')
    plt.xlabel('Traffic (raw units)')
    plt.ylabel('Predicted Delay (loc)')
    plt.grid(alpha=0.3)
    plt.legend()
    out_png = os.path.join(output_dir, 'traffic_delay_curve.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    # Derivative / curvature plot (optional)
    if save_derivative:
        fig, ax1 = plt.subplots(figsize=(7,5))
        ax1.plot(x_raw, dy, label='KAN dDelay/dTraffic', color='tab:blue')
        ax1.set_xlabel('Traffic (raw)')
        ax1.set_ylabel('First derivative', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.axvline(x_raw[curvature_idx], color='gray', linestyle='--', linewidth=1)

        ax2 = ax1.twinx()
        ax2.plot(x_raw, d2y, label='KAN Second derivative', color='tab:red', alpha=0.6)
        if mlp_delays is not None:
            mlp_dy = np.gradient(mlp_delays, x_raw)
            mlp_d2y = np.gradient(mlp_dy, x_raw)
            ax1.plot(x_raw, mlp_dy, label='MLP dDelay/dTraffic', color='tab:purple')
            ax2.plot(x_raw, mlp_d2y, label='MLP Second derivative', color='tab:green', alpha=0.6)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        ax2.set_ylabel('Second derivative', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.suptitle('Derivatives of Delay w.r.t Traffic')
        fig.tight_layout()
        out_png2 = os.path.join(output_dir, 'traffic_delay_curve_derivative.png')
        fig.savefig(out_png2, dpi=160)
        plt.close(fig)

# -----------------------------
# Main
# -----------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load sample
    raw_dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    # Skip until sample_index
    it = raw_dataset.as_numpy_iterator()
    selected = None
    for idx, serialized in enumerate(it):
        if idx == args.sample_index:
            selected = serialized
            break
    if selected is None:
        raise IndexError(f"sample_index {args.sample_index} out of range.")

    f, l = parse_example(selected)
    features, labels = merge_features(f, l)

    # Convert scalar ints to int64 tensors if not already
    for k in ['n_links', 'n_paths']:
        if not isinstance(features[k], tf.Tensor):
            features[k] = tf.convert_to_tensor(features[k], dtype=tf.int64)

    # Load models
    model = load_routenet_model(args.weights_dir, is_kan=True, kan_basis='bspline')
    mlp_model = None
    if args.mlp_weights_dir is not None:
        mlp_model = load_routenet_model(args.mlp_weights_dir, is_kan=False, mlp_weights_filename=args.mlp_weights_filename)

    # Path selection
    if args.path_id is not None:
        path_id = choose_path(args.path_id, features)
        lengths = None
    else:
        path_id, lengths = auto_select_path(features)

    # Baseline forward pass for baseline delay
    base_preds = model(features, training=False)
    baseline_delay = base_preds[path_id, 0].numpy().item()
    mlp_baseline_delay = None
    if mlp_model is not None:
        mlp_base_preds = mlp_model(features, training=False)
        mlp_baseline_delay = mlp_base_preds[path_id, 0].numpy().item()

    # Scan
    x_raw, delays, baseline_raw = run_scan(model, features, path_id, args.num_steps, args.max_multiplier)
    mlp_delays = None
    mlp_curv_idx = None
    if mlp_model is not None:
        mlp_x_raw, mlp_delays, _ = run_scan(mlp_model, features, path_id, args.num_steps, args.max_multiplier)
        # Sanity: ensure same x grid
        if not np.allclose(x_raw, mlp_x_raw):
            raise ValueError('KAN and MLP traffic grids differ, aborting.')

    # Curvature
    curvature_idx, dy, d2y = compute_curvature_points(x_raw, delays)
    if mlp_delays is not None:
        mlp_curv_idx, _, _ = compute_curvature_points(x_raw, mlp_delays)

    # Save CSV
    import csv
    csv_path = os.path.join(args.output_dir, 'traffic_delay_curve.csv')
    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['traffic_raw', 'predicted_delay_loc'])
        for xv, yv in zip(x_raw, delays):
            writer.writerow([f"{xv:.8f}", f"{yv:.8f}"])

    # Metadata
    meta = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'weights_dir': args.weights_dir,
        'weights_file': 'best_delay_kan_model.weights.h5',
        'tfrecord_path': args.tfrecord_path,
        'sample_index': args.sample_index,
        'path_id': path_id,
        'num_steps': args.num_steps,
        'max_multiplier': args.max_multiplier,
        'baseline_traffic_raw': float(baseline_raw),
        'baseline_delay_loc': float(baseline_delay),
        'curvature_point': {
            'traffic_raw': float(x_raw[curvature_idx]),
            'delay_loc': float(delays[curvature_idx]),
            'index': curvature_idx
        },
        'comparison': None
    }
    if mlp_delays is not None:
        meta['comparison'] = {
            'mlp_weights_dir': args.mlp_weights_dir,
            'mlp_weights_file': args.mlp_weights_filename or 'best_delay_model.weights.h5',
            'mlp_baseline_delay_loc': float(mlp_baseline_delay),
            'mlp_curvature_point': {
                'traffic_raw': float(x_raw[mlp_curv_idx]),
                'delay_loc': float(mlp_delays[mlp_curv_idx]),
                'index': int(mlp_curv_idx)
            }
        }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as fm:
        json.dump(meta, fm, indent=2)

    # Sample snapshot
    traffic_scaled = features['traffic'].numpy()
    traffic_raw_full = inverse_scale_traffic(traffic_scaled)
    capacities_raw = inverse_scale_traffic(features['capacities'].numpy())  # WRONG scaling intentionally avoided
    # capacities scaling: raw = scaled * 10.0 (since scaled = raw / 10.0)
    capacities_raw = features['capacities'].numpy() * 10.0

    snapshot = {
        'n_links': int(features['n_links'].numpy()),
        'n_paths': int(features['n_paths'].numpy()),
        'traffic_raw_stats': {
            'min': float(np.min(traffic_raw_full)),
            'median': float(np.median(traffic_raw_full)),
            'max': float(np.max(traffic_raw_full)),
        },
        'capacities_raw_stats': {
            'min': float(np.min(capacities_raw)),
            'median': float(np.median(capacities_raw)),
            'max': float(np.max(capacities_raw)),
        },
        'selected_path_id': path_id,
    }
    if lengths is not None:
        snapshot['selected_path_length'] = int(lengths[path_id])
    snap_path = os.path.join(args.output_dir, 'sample_snapshot.json')
    with open(snap_path, 'w') as fs:
        json.dump(snapshot, fs, indent=2)

    # Plot
    plot_results(x_raw, delays, baseline_raw, baseline_delay, curvature_idx, dy, d2y, args.output_dir, args.save_derivative,
                 mlp_delays=mlp_delays, mlp_curvature_idx=mlp_curv_idx, mlp_baseline_delay=mlp_baseline_delay)

    print(f"Saved curve PNG, CSV, metadata, snapshot to {args.output_dir}")
    print(f"Selected path_id={path_id}, baseline_traffic_raw={baseline_raw:.6f}")
    print(f"KAN baseline_delay_loc={baseline_delay:.6f}, KAN max curvature idx={curvature_idx} -> (traffic={x_raw[curvature_idx]:.6f}, delay={delays[curvature_idx]:.6f})")
    if mlp_delays is not None:
        print(f"MLP baseline_delay_loc={mlp_baseline_delay:.6f}, MLP max curvature idx={mlp_curv_idx} -> (traffic={x_raw[mlp_curv_idx]:.6f}, delay={mlp_delays[mlp_curv_idx]:.6f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic→Delay sensitivity analysis (single path)')
    parser.add_argument('--weights_dir', type=str, default='./kan_model/137/kan_bspline', help='Directory containing weights file best_delay_kan_model.weights.h5')
    parser.add_argument('--tfrecord_path', type=str, required=True, help='Path to a TFRecord file (one of training/eval files)')
    parser.add_argument('--sample_index', type=int, default=0, help='Index of sample inside TFRecord to use as baseline')
    parser.add_argument('--path_id', type=int, default=None, help='Optional explicit path id to analyze')
    parser.add_argument('--num_steps', type=int, default=60, help='Number of traffic scan points')
    parser.add_argument('--max_multiplier', type=float, default=2.0, help='Upper bound raw traffic = baseline_raw * max_multiplier')
    parser.add_argument('--output_dir', type=str, default='./explain_result', help='Directory to save outputs')
    parser.add_argument('--save_derivative', action='store_true', help='Save derivative/curvature auxiliary plot')
    # MLP comparison
    parser.add_argument('--mlp_weights_dir', type=str, default=None, help='Optional MLP weights directory for comparison')
    parser.add_argument('--mlp_weights_filename', type=str, default=None, help='Optional MLP weights filename (default: best_delay_model.weights.h5)')

    args = parser.parse_args()
    main(args)

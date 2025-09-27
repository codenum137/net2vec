#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Efficiency Analysis Script

对比 mlp_none 与 kan_none (poly) 在不同训练数据比例下 (e.g. 10%,25%,50%,100%) 的性能。
数据集：NSFNet (nsfnetbw) 与 GBN (gbnbw)

流程:
1. 为每个 (dataset, fraction) 构建一个子训练目录 (符号链接所选 TFRecord 文件)；fraction=1.0 使用原始目录。
2. 调用 routenet/routenet_tf2.py 分别训练 MLP 与 KAN (poly) 模型。
3. 使用完整评估集进行推理，计算延迟预测的相对误差： (pred - true) / true （排除 true==0）。
4. 统计 Median |Relative Error| (中位绝对相对误差) 与 Mean |Relative Error|，保存结果 CSV / JSONL。
5. 绘制样本效率曲线：X 轴 = 训练数据比例，Y 轴 = Median |Relative Error|。

注意：
- 通过抽取 TFRecord 文件数量近似模拟数据比例；假设各 shard 大致均衡。
- 如果需要更精细的样本级随机抽样，可后续改为解析 TFRecord 计数后重写采样逻辑。

运行示例：
python experiments/sample_efficiency.py \
  --base_dir . \
  --fractions 0.1,0.25,0.5,1.0 \
  --datasets nsfnetbw,gbnbw \
  --epochs 40 \
  --output_dir sample_efficiency_results

可选：加 --force 重新训练，--no-train 跳过训练只做评估（若权重已存在）。
"""
import os
import sys
import argparse
import random
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 确保可以导入项目内模块
sys.path.append(str(Path(__file__).resolve().parent.parent / 'routenet'))
from routenet_tf2 import RouteNet, create_dataset  # type: ignore

# --------------------------- 配置 & 常量 ---------------------------------
DEFAULT_CONFIG = {
    'link_state_dim': 4,
    'path_state_dim': 2,
    'T': 3,
    'readout_units': 8,
    'readout_layers': 2,
    'l2': 0.1,
    'l2_2': 0.01,
}

WEIGHT_CANDIDATES_MLP = [
    'best_delay_model.weights.h5',
    'best_model.weights.h5',
    'model.weights.h5'
]
WEIGHT_CANDIDATES_KAN = [
    'best_delay_kan_model.weights.h5',
    'best_kan_model.weights.h5',
    'best_model.weights.h5',
    'model.weights.h5'
]

# --------------------------- 数据子集构建 ---------------------------------

def build_subset(train_dir: Path, fraction: float, subset_root: Path, seed: int = 42) -> Path:
    """基于 TFRecord shard 文件按数量比例创建子集目录（使用符号链接）。
    如果 fraction>=0.999，直接返回原始 train_dir。
    """
    if fraction >= 0.999:
        return train_dir
    subset_name = f"train_frac_{int(fraction*100)}p"
    target_dir = subset_root / subset_name
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(train_dir.glob('*.tfrecords'))
    if not files:
        raise RuntimeError(f"No TFRecord files found in {train_dir}")
    k = max(1, int(round(len(files) * fraction)))
    random.Random(seed).shuffle(files)
    selected = files[:k]
    for f in selected:
        link_path = target_dir / f.name
        try:
            os.symlink(os.path.relpath(f, target_dir), link_path)
        except FileExistsError:
            pass
    return target_dir

# --------------------------- 训练与评估 ---------------------------------

def find_weight_file(model_dir: Path, use_kan: bool) -> Path:
    candidates = WEIGHT_CANDIDATES_KAN if use_kan else WEIGHT_CANDIDATES_MLP
    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No weight file found in {model_dir}")

@tf.function(experimental_relax_shapes=True)
def _forward(model, features):
    return model(features, training=False)

def evaluate_delay_model(model: tf.keras.Model, eval_dir: Path, batch_size: int =32, limit_samples: int =None) -> Dict[str, float]:
    eval_files = tf.io.gfile.glob(str(eval_dir / '*.tfrecords'))
    dataset = create_dataset(eval_files, batch_size, is_training=False)
    all_pred = []
    all_true = []
    count = 0
    for features, labels in dataset:
        preds = _forward(model, features)
        pred_delay = preds[:,0].numpy()
        true_delay = labels['delay'].numpy()
        all_pred.append(pred_delay)
        all_true.append(true_delay)
        count += len(pred_delay)
        if limit_samples and count >= limit_samples:
            break
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    mask = np.abs(true) > 1e-10
    rel = (pred[mask] - true[mask]) / true[mask]
    abs_rel = np.abs(rel)
    return {
        'median_abs_rel_error': float(np.median(abs_rel)),
        'mean_abs_rel_error': float(np.mean(abs_rel)),
        'count': int(mask.sum())
    }

# --------------------------- 主流程 ---------------------------------

def run_training(script_path: Path, train_dir: Path, eval_dir: Path, model_dir: Path, use_kan: bool, epochs: int, batch_size: int, lr: float, patience: int, force: bool=False) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_file_target = (model_dir / ('best_delay_kan_model.weights.h5' if use_kan else 'best_delay_model.weights.h5'))
    if weight_file_target.exists() and not force:
        print(f"[Skip] Weights already exist: {weight_file_target}")
        return
    cmd = [sys.executable, str(script_path),
        '--train_dir', str(train_dir),
        '--eval_dir', str(eval_dir),
        '--model_dir', str(model_dir),
        '--target', 'delay',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', str(lr),
        '--lr_schedule', 'plateau',
        '--plateau_patience', str(patience),
        '--plateau_factor', '0.5',
        '--early_stopping', '--early_stopping_patience', str(patience), '--early_stopping_min_delta', '1e-6', '--early_stopping_restore_best'
        ]
    if use_kan:
        cmd.append('--kan')
    print('[Train CMD]', ' '.join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(model_dir / 'train_stdout.log', 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {model_dir}. See train_stdout.log")


def main():
    parser = argparse.ArgumentParser(description='Sample Efficiency Analysis (MLP vs KAN)')
    parser.add_argument('--base_dir', type=str, default='.', help='Project root')
    parser.add_argument('--fractions', type=str, default='0.1,0.25,0.5,1.0', help='Comma separated fractions of training data to use')
    parser.add_argument('--datasets', type=str, default='nsfnetbw,gbnbw', help='Comma separated dataset names (folder names under data/routenet)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='sample_efficiency_results')
    parser.add_argument('--force', action='store_true', help='Force retrain even if weights exist')
    parser.add_argument('--no_train', action='store_true', help='Skip training (only evaluate existing models)')
    parser.add_argument('--seed', type=int, default=42)
    # 当前分支 routenet_tf2.py 不支持 loss_variant / no_dropout / dropout_rate 等参数
    parser.add_argument('--limit_eval_samples', type=int, default=None, help='Optional limit on evaluation samples for speed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    base = Path(args.base_dir).resolve()
    train_script = base / 'routenet' / 'routenet_tf2.py'

    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = [float(x) for x in args.fractions.split(',')]
    datasets = [x.strip() for x in args.datasets.split(',') if x.strip()]

    results = []

    for dataset in datasets:
        print(f"\n========== DATASET: {dataset} ==========")
        ds_root = base / 'data' / 'routenet' / dataset / 'tfrecords'
        train_dir_full = ds_root / 'train'
        eval_dir = ds_root / 'evaluate'
        if not train_dir_full.exists():
            raise RuntimeError(f"Train dir not found: {train_dir_full}")
        if not eval_dir.exists():
            raise RuntimeError(f"Eval dir not found: {eval_dir}")
        subset_root = ds_root  # 将子集放在同级目录下

        for frac in fractions:
            print(f"\n--- Fraction {frac:.2f} ({int(frac*100)}%) ---")
            subset_train_dir = build_subset(train_dir_full, frac, subset_root, seed=args.seed)
            frac_tag = f"{int(frac*100)}p" if frac < 0.999 else '100p'

            # 两种模型： mlp_none (MLP) 与 kan_none (KAN poly)
            for model_kind in ['mlp', 'kan']:
                use_kan = model_kind == 'kan'
                model_name = f"{model_kind}_frac_{frac_tag}"
                model_dir = out_dir / dataset / model_name
                if not args.no_train:
                    run_training(train_script, subset_train_dir, eval_dir, model_dir, use_kan, args.epochs, args.batch_size, args.learning_rate, args.patience, force=args.force)
                # 评估
                try:
                    # 构建同训练脚本的 config (简单传入默认即可；KAN 相关参数由权重结构决定)
                    config = DEFAULT_CONFIG.copy()
                    if use_kan:
                        config['kan_basis'] = 'poly'
                    # 创建模型并加载权重
                    if use_kan:
                        model = RouteNet(config, output_units=2, final_activation=None, use_kan=True)
                    else:
                        model = RouteNet(config, output_units=2, final_activation=None, use_kan=False)
                    weight_file = find_weight_file(model_dir, use_kan)
                    # 构建一次模型 (前向一个 batch) 以确保变量建立
                    sample_files = tf.io.gfile.glob(str((subset_train_dir if frac < 0.999 else train_dir_full) / '*.tfrecords'))
                    sample_dataset = create_dataset(sample_files[:1], batch_size=1, is_training=False)
                    for fts, lbs in sample_dataset.take(1):
                        _ = model(fts, training=False)
                    model.load_weights(str(weight_file))
                    metrics = evaluate_delay_model(model, eval_dir, batch_size=args.batch_size, limit_samples=args.limit_eval_samples)
                    res_entry = {
                        'dataset': dataset,
                        'fraction': frac,
                        'fraction_label': frac_tag,
                        'model': 'KAN' if use_kan else 'MLP',
                        **metrics
                    }
                    results.append(res_entry)
                    print(f"Result: {res_entry}")
                except Exception as e:
                    print(f"[Error] Evaluation failed for {model_dir}: {e}")

        # 绘图（每个 dataset 一张）
        ds_results = [r for r in results if r['dataset'] == dataset]
        if ds_results:
            fig, ax = plt.subplots(figsize=(8,5))
            for model_label in ['MLP','KAN']:
                sub = sorted([r for r in ds_results if r['model']==model_label], key=lambda x: x['fraction'])
                if not sub:
                    continue
                x = [r['fraction']*100 for r in sub]
                y = [r['median_abs_rel_error'] for r in sub]
                ax.plot(x, y, marker='o', linewidth=2, label=model_label)
                for xi, yi in zip(x,y):
                    ax.text(xi, yi, f"{yi:.3f}", fontsize=9, ha='center', va='bottom')
            ax.set_xlabel('Training Data Percentage (%)')
            ax.set_ylabel('Median |Relative Error|')
            ax.set_title(f'Sample Efficiency - {dataset}')
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plot_path = out_dir / dataset / 'sample_efficiency_curve.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=300)
            print(f"Saved plot: {plot_path}")
            plt.close(fig)

    # 保存结果表
    csv_path = out_dir / 'sample_efficiency_results.csv'
    jsonl_path = out_dir / 'sample_efficiency_results.jsonl'
    if results:
        import csv
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nAll results saved to:\n  CSV: {csv_path}\n  JSONL: {jsonl_path}")
    else:
        print("No results to save.")

    # 总结打印
    print("\n===== SUMMARY (Median |Relative Error|) =====")
    by_key = {}
    for r in results:
        key = (r['dataset'], r['model'])
        by_key.setdefault(key, []).append(r)
    for (ds, mdl), rows in by_key.items():
        rows = sorted(rows, key=lambda x: x['fraction'])
        s = ', '.join([f"{int(r['fraction']*100)}%: {r['median_abs_rel_error']:.4f}" for r in rows])
        print(f"{ds} - {mdl}: {s}")

if __name__ == '__main__':
    main()

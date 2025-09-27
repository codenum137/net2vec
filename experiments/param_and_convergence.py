#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter & Convergence Comparison Experiment
============================================

目的:
 1. 统计 MLP Readout 与 KAN Readout 的可训练参数数量（单独与全模型）。
 2. 在同等配置下训练两个模型 (target=delay)，绘制验证集损失随 epoch 的收敛曲线。

特点:
 - 额外开销极低：少量 epoch + 仅 delay 目标 + 无物理约束。
 - 直接复用 existing `routenet_tf2` 中的数据管线与损失函数。
 - 生成: 参数统计报告 (TXT/JSON)、收敛曲线 PNG、以及 CSV 历史记录。

使用示例:
python experiments/param_and_convergence.py \
  --train_dir data/routenet/nsfnetbw/train \
  --eval_dir data/routenet/nsfnetbw/eval \
  --output_dir experiment_results/param_compare \
  --epochs 15 --batch_size 32 --steps_per_epoch 50

输出文件:
  param_report.txt            参数数量 (可读格式)
  param_report.json           参数数量 (机器读取)
  convergence_history.csv     每个 epoch 的 train / val 损失
  convergence_plot.png        验证损失曲线 (MLP vs KAN)

可扩展:
 - 可增加 --kan_layers / --mlp_layers 等自定义超参数
 - 可添加 --target drops 做扩展（当前默认 delay 更具对比价值）
"""
import os
import sys
import argparse
import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 确保可以从仓库根目录导入 routenet 包
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 从主实现中导入所需组件
from routenet.routenet_tf2 import (
    create_dataset,
    RouteNet,
    heteroscedastic_loss,
)

# 固定随机种子便于复现
tf.random.set_seed(42)
np.random.seed(42)

DEFAULT_CONFIG = {
    'link_state_dim': 4,
    'path_state_dim': 2,
    'T': 3,
    'readout_units': 8,
    'readout_layers': 2,
    'l2': 0.1,
    'l2_2': 0.01,
}


def count_params(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

def count_readout_params(model: RouteNet) -> int:
    # RouteNet.readout 是 Sequential, 可以直接访问其 trainable_variables
    return int(np.sum([np.prod(v.shape) for v in model.readout.trainable_variables]))

def build_model(use_kan: bool, config: Dict) -> RouteNet:
    # target=delay => output_units=2 (loc + scale logits)
    model = RouteNet(config, output_units=2, final_activation=None, use_kan=use_kan)
    return model


def take_one_batch(dataset):
    for features, labels in dataset.take(1):
        return features, labels
    raise RuntimeError("Dataset is empty; check --train_dir path")


def run_one_epoch(model, optimizer, dataset, steps, loss_fn):
    total = 0.0
    count = 0
    for i, (features, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            preds = model(features, training=True)
            loss = loss_fn(labels, preds)
            loss += sum(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        total += float(loss)
        count += 1
        if steps and (i + 1) >= steps:
            break
    return total / max(count, 1)


def evaluate(model, dataset, steps, loss_fn):
    total = 0.0
    count = 0
    for i, (features, labels) in enumerate(dataset):
        preds = model(features, training=False)
        loss = loss_fn(labels, preds)
        loss += sum(model.losses)
        total += float(loss)
        count += 1
        if steps and (i + 1) >= steps:
            break
    return total / max(count, 1)


def train_and_record(config: Dict, train_dir: str, eval_dir: str, batch_size: int, epochs: int, steps_per_epoch: int, eval_steps: int, learning_rate: float):
    # ------------------------------------------------------------------
    # 自动发现 TFRecord 文件：允许传入上级目录，例如 data/routenet/nsfnetbw
    # 优先匹配直接目录，其次匹配常见子路径: tfrecords/train, tfrecords/evaluate, train, eval, evaluate
    # ------------------------------------------------------------------
    def discover(dir_path: str, mode: str):
        patterns = [
            '*.tfrecords',
            f'{mode}/*.tfrecords',
            f'tfrecords/{mode}/*.tfrecords',
            f'tfrecords/{"evaluate" if mode=="eval" else mode}/*.tfrecords',
            f'{"evaluate" if mode=="eval" else mode}/*.tfrecords'
        ]
        files: List[str] = []
        for p in patterns:
            glob_path = os.path.join(dir_path, p)
            found = tf.io.gfile.glob(glob_path)
            if found:
                files.extend(found)
        # 去重 & 排序
        files = sorted(list(set(files)))
        return files

    train_files = discover(train_dir, 'train')
    eval_files = discover(eval_dir, 'eval')

    if not train_files:
        raise FileNotFoundError(f"No TFRecords discovered. Tried typical patterns under train_dir={train_dir}")
    if not eval_files:
        raise FileNotFoundError(f"No TFRecords discovered. Tried typical patterns under eval_dir={eval_dir}")

    print(f"Discovered {len(train_files)} training shards, {len(eval_files)} eval shards.")

    train_ds_full = create_dataset(train_files, batch_size, is_training=True)
    eval_ds_full = create_dataset(eval_files, batch_size, is_training=False)

    histories = {}
    param_report = {}

    for label, use_kan in [("MLP", False), ("KAN", True)]:
        print(f"\n=== Building {label} model ===")
        model = build_model(use_kan=use_kan, config=config)

        # 通过跑一个 batch 构建变量 (否则还没有 weights)
        f1, l1 = take_one_batch(train_ds_full)
        _ = model(f1, training=False)

        total_params = count_params(model)
        readout_params = count_readout_params(model)

        param_report[label] = {
            'total_params': total_params,
            'readout_params': readout_params,
            'readout_fraction_percent': round(readout_params / total_params * 100, 2)
        }
        print(f"{label} Total Params: {total_params:,} | Readout: {readout_params:,} ({param_report[label]['readout_fraction_percent']}%)")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            train_loss = run_one_epoch(model, optimizer, train_ds_full, steps_per_epoch, heteroscedastic_loss)
            val_loss = evaluate(model, eval_ds_full, eval_steps, heteroscedastic_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"[{label}] Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        histories[label] = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }

    return histories, param_report


def plot_convergence(histories: Dict, output_path: str):
    epochs = range(1, len(next(iter(histories.values()))['val_loss']) + 1)
    plt.figure(figsize=(6.4, 3.2))
    for label, style, color in [("MLP", '-', '#1f77b4'), ("KAN", '--', '#ff7f0e')]:
        if label in histories:
            plt.plot(epochs, histories[label]['val_loss'], linestyle=style, color=color, linewidth=2, label=f"{label} Val Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Convergence')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved convergence plot to {output_path}")


def save_history_csv(histories: Dict, output_path: str):
    labels = list(histories.keys())
    max_epochs = max(len(histories[l]['val_loss']) for l in labels)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch']
        for l in labels:
            header += [f'{l}_train_loss', f'{l}_val_loss']
        writer.writerow(header)
        for e in range(max_epochs):
            row = [e + 1]
            for l in labels:
                train_loss = histories[l]['train_loss'][e] if e < len(histories[l]['train_loss']) else ''
                val_loss = histories[l]['val_loss'][e] if e < len(histories[l]['val_loss']) else ''
                row += [train_loss, val_loss]
            writer.writerow(row)
    print(f"Saved history CSV to {output_path}")


def save_param_report(report: Dict, txt_path: str, json_path: str):
    lines = []
    lines.append("Parameter Count Report")
    lines.append("======================")
    for label, info in report.items():
        lines.append(f"{label}:")
        lines.append(f"  Total Params        : {info['total_params']:,}")
        lines.append(f"  Readout Params      : {info['readout_params']:,}")
        lines.append(f"  Readout % of Total  : {info['readout_fraction_percent']}%")
        lines.append("")
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved param report to {txt_path} & {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Parameter & Convergence Comparison (MLP vs KAN)')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory with training TFRecords')
    parser.add_argument('--eval_dir', type=str, required=True, help='Directory with evaluation TFRecords')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--steps_per_epoch', type=int, default=50, help='Training steps per epoch cap (default: 50)')
    parser.add_argument('--eval_steps', type=int, default=20, help='Validation steps per epoch cap (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Adam learning rate (default: 1e-3)')
    parser.add_argument('--readout_units', type=int, default=8, help='Readout hidden units (default: 8)')
    parser.add_argument('--readout_layers', type=int, default=2, help='Number of readout hidden layers (default: 2)')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['readout_units'] = args.readout_units
    config['readout_layers'] = args.readout_layers

    os.makedirs(args.output_dir, exist_ok=True)

    histories, param_report = train_and_record(
        config=config,
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
    )

    # 保存参数统计
    save_param_report(param_report,
                      txt_path=os.path.join(args.output_dir, 'param_report.txt'),
                      json_path=os.path.join(args.output_dir, 'param_report.json'))

    # 保存历史 CSV
    save_history_csv(histories, os.path.join(args.output_dir, 'convergence_history.csv'))

    # 绘制收敛曲线
    plot_convergence(histories, os.path.join(args.output_dir, 'convergence_plot.png'))

    print("\nDone. You can now reference parameter counts and convergence curves in your analysis.")

if __name__ == '__main__':
    main()

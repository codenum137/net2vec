#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified RouteNet TF2 evaluation script (delay & jitter only)

Changes per requirements:
1. Removed TensorFlow version backend selection logic.
2. Removed all KAN related code/arguments.
3. Removed drop (packet loss) prediction evaluation code.
4. Removed evaluation summary printing block.
5. Kept only the core functionality: plotting relative error CDF (delay & jitter) across two test sets.
6. Added default values for nsfnet_test_dir and gbn_test_dir (pointing to typical dataset layout).

Usage example:
python experiment/plot_cdf.py \
    --delay_model_dir fixed_model/0925 \
    --output_dir evaluation_results/0925/cdf_delay \
    --batch_size 64

Optional override test dirs:
    --nsfnet_test_dir ./data/routenet/nsfnetbw/tfrecords/evaluate \
    --gbn_test_dir    ./data/routenet/gbnbw/tfrecords/evaluate
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import seaborn as sns
from routenet.routenet_tf2 import create_model_and_loss_fn, create_dataset
import math
import csv

def load_model(model_dir: str, target: str, config: dict):
    """Load delay model weights (KAN & drops removed)."""
    candidate_files = [
        os.path.join(model_dir, f"best_{target}_model.weights.h5"),
        os.path.join(model_dir, "best_model.weights.h5"),
        os.path.join(model_dir, "model.weights.h5")
    ]
    weight_path = next((p for p in candidate_files if os.path.exists(p)), None)
    if weight_path is None:
        raise FileNotFoundError(f"No model weights found under {model_dir}")
    model, _ = create_model_and_loss_fn(config, target)
    print(f"Loading delay model weights from: {weight_path}")
    return model, weight_path

def evaluate_delay_jitter_model(model, dataset, num_samples=None):
    """
    评估延迟/抖动预测模型
    
    Args:
        model: 延迟预测模型 (输出2维: [loc, scale])
        dataset: 测试数据集
        num_samples: 限制评估的样本数量
    
    Returns:
        predictions: 预测结果
        ground_truth: 真实值
        relative_errors: 相对误差
    """
    predictions = {'delay': [], 'jitter': []}
    ground_truth = {'delay': [], 'jitter': []}
    
    sample_count = 0
    
    for features, labels in tqdm(dataset, desc="Evaluating delay/jitter model"):
        # 模型预测 - 异方差输出：[loc, scale]
        pred = model(features, training=False)
        
        pred_delay = pred[:, 0].numpy()  # 延迟预测均值 (loc)
        
        # 与原版保持一致的scale计算，包含c偏移常数
        c = np.log(np.expm1(np.float32(0.098)))
        pred_scale = tf.nn.softplus(c + pred[:, 1]).numpy() + 1e-9
        
        # 根据原版的实现：jitter_prediction = scale**2
        pred_jitter = pred_scale ** 2
        
        # 收集真实值
        true_delay = labels['delay'].numpy()
        true_jitter = labels['jitter'].numpy()
        
        # 存储结果
        predictions['delay'].extend(pred_delay)
        predictions['jitter'].extend(pred_jitter)
        
        ground_truth['delay'].extend(true_delay)
        ground_truth['jitter'].extend(true_jitter)
        
        sample_count += len(pred_delay)
        if num_samples and sample_count >= num_samples:
            break
    
    # 转换为numpy数组
    for key in predictions:
        predictions[key] = np.array(predictions[key])
        ground_truth[key] = np.array(ground_truth[key])
    
    # 计算相对误差
    relative_errors = {}
    for key in predictions:
        mask = np.abs(ground_truth[key]) > 1e-10
        rel_error = np.full_like(predictions[key], np.nan)
        rel_error[mask] = (predictions[key][mask] - ground_truth[key][mask]) / ground_truth[key][mask]
        relative_errors[key] = rel_error[~np.isnan(rel_error)]
    
    return predictions, ground_truth, relative_errors

def plot_linear_focus_cdf(nsfnet_errors, gbn_errors, output_dir, model_suffix=""):
    """
    绘制线性刻度的相对误差CDF图，显示正负误差分布
    
    Args:
        nsfnet_errors: nsfnet拓扑的相对误差
        gbn_errors: gbn拓扑的相对误差  
        output_dir: 保存目录
        model_suffix: 模型后缀（如"_kan"用于区分不同模型类型）
    """
    # 设置颜色和线型
    colors = {'delay': '#1f77b4', 'jitter': '#ff7f0e', 'drops': '#2ca02c'}
    linestyles = {'nsfnet': '-', 'gbn': '--'}
    
    plt.figure(figsize=(12, 8))
    
    for metric in ['delay', 'jitter']:
        if metric in nsfnet_errors and len(nsfnet_errors[metric]) > 0:
            sorted_errors = np.sort(nsfnet_errors[metric])
            cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            plt.plot(sorted_errors, cdf_values, 
                    color=colors[metric], linestyle=linestyles['nsfnet'],
                    linewidth=2.5, label='NSFNet {}'.format(metric.upper()))
        
        if metric in gbn_errors and len(gbn_errors[metric]) > 0:
            sorted_errors = np.sort(gbn_errors[metric])
            cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            plt.plot(sorted_errors, cdf_values,
                    color=colors[metric], linestyle=linestyles['gbn'],
                    linewidth=2.5, label='GBN {}'.format(metric.upper()))
    
    # 添加理想情况的参考线
    plt.axvline(x=0, color='red', linestyle=':', linewidth=3, alpha=0.8, label='Ideal (Zero Error)')
    
    # 根据模型类型调整标题
    model_type = "KAN" if "kan" in model_suffix.lower() else "MLP"
    title = 'Relative Error CDF - {} Model - Linear Scale with Positive/Negative Errors\\n(Ideal is Vertical Red Line at 0)'.format(model_type)
    
    plt.xlabel('Relative Error (Positive: Over-prediction, Negative: Under-prediction)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right', fontsize=12)
    
    # 收集所有误差（包括正负值）来设置x轴范围
    all_signed_errors = []
    for errors_dict in [nsfnet_errors, gbn_errors]:
        for metric in ['delay', 'jitter']:
            if metric in errors_dict and len(errors_dict[metric]) > 0:
                all_signed_errors.extend(errors_dict[metric])
    
    if all_signed_errors:
        min_error = np.min(all_signed_errors)
        max_error = np.max(all_signed_errors)
        # 使用对称的范围，以0为中心
        max_abs_error = max(abs(min_error), abs(max_error))
        # 限制在合理范围内，并确保显示负值部分
        display_range = min(max_abs_error * 1.2, 1.0)
        plt.xlim(-display_range, display_range)
    else:
        plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1)
    
    # 添加统计信息，包含正负误差的信息
    detailed_stats = []
    for topo in ['nsfnet', 'gbn']:
        errors = nsfnet_errors if topo == 'nsfnet' else gbn_errors
        for metric in ['delay', 'jitter']:
            if metric in errors and len(errors[metric]) > 0:
                mean_error = np.mean(errors[metric])
                abs_errors = np.abs(errors[metric])
                median_abs_error = np.median(abs_errors)
                detailed_stats.append('{} {}: Mean={:.4f}, |Med|={:.4f}\n'.format(
                    topo.upper(), metric.upper(), mean_error, median_abs_error))
    
    detailed_stats_str = '\\n'.join(detailed_stats)
    plt.text(0.02, 0.98, detailed_stats_str, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 根据模型类型调整文件名
    filename = 'relative_error_cdf{}.png'.format(model_suffix)
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("Linear scale CDF plot with positive/negative errors saved to: {}".format(output_path))

def main():
    parser = argparse.ArgumentParser(description='RouteNet delay/jitter evaluation (simplified)')
    parser.add_argument('--delay_model_dir', type=str, required=True, help='Directory containing trained delay model weights')
    parser.add_argument('--nsfnet_test_dir', type=str, default='./data/routenet/nsfnetbw/tfrecords/evaluate', help='NSFNet test TFRecords directory')
    parser.add_argument('--gbn_test_dir', type=str, default='./data/routenet/gbnbw/tfrecords/evaluate', help='GBN test TFRecords directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save CDF plot')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None, help='Optional limit on number of samples evaluated')
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模型配置（与训练时保持一致）
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
    }
    print("Starting RouteNet delay/jitter evaluation (MLP only)...")
    print("Delay model dir: {}".format(args.delay_model_dir))
    print("NSFNet test dir: {}".format(args.nsfnet_test_dir))
    print("GBN test dir: {}".format(args.gbn_test_dir))
    
    # 加载模型
    delay_model, delay_weight_path = load_model(args.delay_model_dir, 'delay', config)
    
    # 创建数据集
    nsfnet_files = tf.io.gfile.glob(os.path.join(args.nsfnet_test_dir, '*.tfrecords'))
    gbn_files = tf.io.gfile.glob(os.path.join(args.gbn_test_dir, '*.tfrecords'))
    
    nsfnet_dataset = create_dataset(nsfnet_files, args.batch_size, is_training=False)
    gbn_dataset = create_dataset(gbn_files, args.batch_size, is_training=False)
    
    print("Found {} NSFNet test files".format(len(nsfnet_files)))
    print("Found {} GBN test files".format(len(gbn_files)))
    
    # 初始化模型权重（需要先运行一次前向传播）
    print("\nInitializing models...")
    
    # 进行一次前向传播以构建模型
    for dataset in [nsfnet_dataset.take(1)]:
        for features, labels in dataset:
            _ = delay_model(features, training=False)
            break
    
    # 加载权重
    delay_model.load_weights(delay_weight_path)
    print("Model loaded successfully!")
    
    # 评估NSFNet（同拓扑）
    print("\n" + "="*50)
    print("EVALUATING NSFNET (SAME TOPOLOGY)")
    print("="*50)
    
    # 评估delay和jitter
    nsfnet_predictions, nsfnet_ground_truth, nsfnet_delay_jitter_errors = evaluate_delay_jitter_model(
        delay_model, nsfnet_dataset, args.num_samples
    )
    nsfnet_errors = nsfnet_delay_jitter_errors
    
    # 评估GBN（跨拓扑）
    print("\n" + "="*50)
    print("EVALUATING GBN (DIFFERENT TOPOLOGY)")
    print("="*50)
    
    # 评估delay和jitter
    gbn_predictions, gbn_ground_truth, gbn_delay_jitter_errors = evaluate_delay_jitter_model(
        delay_model, gbn_dataset, args.num_samples
    )
    gbn_errors = gbn_delay_jitter_errors

    # ================= Metrics (RMSE & MAE) =================
    def compute_rmse_mae(preds: dict, gts: dict):
        results = []
        for key in ['delay', 'jitter']:
            if key in preds and len(preds[key]) > 0:
                diff = preds[key] - gts[key]
                mae = float(np.mean(np.abs(diff)))
                rmse = float(math.sqrt(np.mean(diff ** 2)))
                n = int(len(diff))
                results.append((key, n, rmse, mae))
        return results

    ns_metrics = compute_rmse_mae(nsfnet_predictions, nsfnet_ground_truth)
    gbn_metrics = compute_rmse_mae(gbn_predictions, gbn_ground_truth)

    csv_path = os.path.join(args.output_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['topology', 'metric', 'n', 'rmse', 'mae'])
        for metric, n, rmse, mae in ns_metrics:
            writer.writerow(['nsfnet', metric, n, f'{rmse:.6f}', f'{mae:.6f}'])
        for metric, n, rmse, mae in gbn_metrics:
            writer.writerow(['gbn', metric, n, f'{rmse:.6f}', f'{mae:.6f}'])
    print(
        f"Saved RMSE/MAE metrics to: {csv_path}\n"
        f"NSFNet metrics: {ns_metrics}\n"
        f"GBN metrics: {gbn_metrics}"
    )
    
    # 绘制线性刻度CDF图（仅生成linear_focus版本）
    print("\nGenerating linear focus CDF plot...")
    plot_linear_focus_cdf(nsfnet_errors, gbn_errors, args.output_dir, model_suffix="_mlp")
    
    print("\nEvaluation completed! Results saved to: {}".format(args.output_dir))

if __name__ == '__main__':
    main()

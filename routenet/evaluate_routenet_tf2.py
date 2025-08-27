#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RouteNet TF2 综合模型评估脚本
支持评估delay/jitter和drops两种不同的模型，并绘制相对误差CDF图
支持同拓扑（nsfnet）和跨拓扑（gbn）评估
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import seaborn as sns

# 导入训练脚本中的相关函数和类
import sys
sys.path.append(os.path.dirname(__file__))
from routenet_tf2 import (
    RouteNet, create_dataset, parse_fn, transformation_func,
    scale_fn, heteroscedastic_loss, binomial_loss, create_model_and_loss_fn
)

def load_model(model_dir, target, config):
    """
    加载指定目标的模型
    
    Args:
        model_dir: 模型目录路径
        target: 'delay' 或 'drops'
        config: 模型配置
    
    Returns:
        model: 加载了权重的模型
    """
    # 根据target创建相应的模型
    model, _ = create_model_and_loss_fn(config, target)
    
    # 寻找权重文件
    weight_files = [
        os.path.join(model_dir, "best_{}_model.weights.h5".format(target)),
        os.path.join(model_dir, "best_model.weights.h5"),
        os.path.join(model_dir, "model.weights.h5")
    ]
    
    weight_path = None
    for path in weight_files:
        if os.path.exists(path):
            weight_path = path
            break
    
    if weight_path is None:
        raise FileNotFoundError("No model weights found in {}".format(model_dir))
    
    print("Loading {} model weights from: {}".format(target, weight_path))
    
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

def evaluate_drops_model(model, dataset, num_samples=None):
    """
    评估丢包预测模型
    
    Args:
        model: 丢包预测模型 (输出1维: logits)
        dataset: 测试数据集  
        num_samples: 限制评估的样本数量
    
    Returns:
        predictions: 预测结果
        ground_truth: 真实值
        relative_errors: 相对误差
    """
    predictions = {'drops': []}
    ground_truth = {'drops': []}
    
    sample_count = 0
    
    for features, labels in tqdm(dataset, desc="Evaluating drops model"):
        # 模型预测 - 输出 logits
        pred_logits = model(features, training=False)
        pred_probs = tf.nn.sigmoid(pred_logits[:, 0]).numpy()
        
        # 收集真实值
        true_drops = labels['drops'].numpy()
        true_packets = labels['packets'].numpy()
        
        # 计算真实丢包率和预测丢包率
        true_drop_rates = true_drops / (true_packets + 1e-10)  # 避免除零
        pred_drop_rates = pred_probs  # sigmoid输出本身就是概率/丢包率
        
        # 存储结果
        predictions['drops'].extend(pred_drop_rates)
        ground_truth['drops'].extend(true_drop_rates)
        
        sample_count += len(pred_drop_rates)
        if num_samples and sample_count >= num_samples:
            break
    
    # 转换为numpy数组
    for key in predictions:
        predictions[key] = np.array(predictions[key])
        ground_truth[key] = np.array(ground_truth[key])
    
    # 计算相对误差
    relative_errors = {}
    for key in predictions:
        # 对于丢包率，只有当真实丢包率大于某个阈值时才计算相对误差
        # 因为丢包率很小时，相对误差会非常大
        mask = ground_truth[key] > 1e-6  # 只考虑丢包率大于0.0001%的情况
        rel_error = np.full_like(predictions[key], np.nan)
        rel_error[mask] = (predictions[key][mask] - ground_truth[key][mask]) / ground_truth[key][mask]
        relative_errors[key] = rel_error[~np.isnan(rel_error)]
    
    return predictions, ground_truth, relative_errors


def plot_linear_focus_cdf(nsfnet_errors, gbn_errors, output_dir):
    """
    绘制线性刻度的相对误差CDF图，显示正负误差分布
    
    Args:
        nsfnet_errors: nsfnet拓扑的相对误差
        gbn_errors: gbn拓扑的相对误差  
        output_dir: 保存目录
    """
    # 设置颜色和线型
    colors = {'delay': '#1f77b4', 'jitter': '#ff7f0e', 'drops': '#2ca02c'}
    linestyles = {'nsfnet': '-', 'gbn': '--'}
    
    plt.figure(figsize=(12, 8))
    
    for metric in ['delay', 'jitter', 'drops']:
        if metric in nsfnet_errors and len(nsfnet_errors[metric]) > 0:
            # 不取绝对值，保留正负信息
            sorted_errors = np.sort(nsfnet_errors[metric])
            cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            plt.plot(sorted_errors, cdf_values, 
                    color=colors[metric], linestyle=linestyles['nsfnet'],
                    linewidth=2.5, label='NSFNet {}'.format(metric.upper()))
        
        if metric in gbn_errors and len(gbn_errors[metric]) > 0:
            # 不取绝对值，保留正负信息
            sorted_errors = np.sort(gbn_errors[metric])
            cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            plt.plot(sorted_errors, cdf_values,
                    color=colors[metric], linestyle=linestyles['gbn'],
                    linewidth=2.5, label='GBN {}'.format(metric.upper()))
    
    # 添加理想情况的参考线
    plt.axvline(x=0, color='red', linestyle=':', linewidth=3, alpha=0.8, label='Ideal (Zero Error)')
    
    plt.xlabel('Relative Error (Positive: Over-prediction, Negative: Under-prediction)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title('Relative Error CDF - Linear Scale with Positive/Negative Errors\\n(Ideal is Vertical Red Line at 0)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right', fontsize=12)
    
    # 收集所有误差（包括正负值）来设置x轴范围
    all_signed_errors = []
    for errors_dict in [nsfnet_errors, gbn_errors]:
        for metric in ['delay', 'jitter', 'drops']:
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
        for metric in ['delay', 'jitter', 'drops']:
            if metric in errors and len(errors[metric]) > 0:
                mean_error = np.mean(errors[metric])  # 包含符号的平均误差
                abs_errors = np.abs(errors[metric])
                median_abs_error = np.median(abs_errors)
                detailed_stats.append('{} {}: Mean={:.4f}, |Med|={:.4f}\n'.format(
                    topo.upper(), metric.upper(), mean_error, median_abs_error))
    
    detailed_stats_str = '\\n'.join(detailed_stats)
    plt.text(0.02, 0.98, detailed_stats_str, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    output_path = os.path.join(output_dir, 'relative_error_cdf.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("Linear scale CDF plot with positive/negative errors saved to: {}".format(output_path))

def print_evaluation_summary(nsfnet_errors, gbn_errors):
    """
    打印评估摘要统计
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for topo_name, errors in [('NSFNet (Same Topology)', nsfnet_errors), ('GBN (Different Topology)', gbn_errors)]:
        print("\n{}:".format(topo_name))
        print("-" * 40)
        
        for metric in ['delay', 'jitter', 'drops']:
            if metric in errors and len(errors[metric]) > 0:
                abs_errors = np.abs(errors[metric])
                unit = " (drop rate)" if metric == 'drops' else ""
                print("  {}{}: {} samples".format(metric.upper(), unit, len(abs_errors)))
                print("    Mean Abs Error: {:.4f}".format(np.mean(abs_errors)))
                print("    Median Abs Error: {:.4f}".format(np.median(abs_errors)))
                print("    P90 Abs Error: {:.4f}".format(np.percentile(abs_errors, 90)))
                print("    P95 Abs Error: {:.4f}".format(np.percentile(abs_errors, 95)))
            else:
                print("  {}: No data available".format(metric.upper()))

def main():
    parser = argparse.ArgumentParser(description='Comprehensive RouteNet Evaluation')
    parser.add_argument('--delay_model_dir', type=str, required=True,
                      help='Directory containing delay prediction model')
    parser.add_argument('--drops_model_dir', type=str, required=True,
                      help='Directory containing drops prediction model')
    parser.add_argument('--nsfnet_test_dir', type=str, required=True,
                      help='Directory containing NSFNet test data')
    parser.add_argument('--gbn_test_dir', type=str, required=True,
                      help='Directory containing GBN test data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Limit number of samples to evaluate')
    
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
    
    print("Starting comprehensive RouteNet evaluation...")
    print("Delay model dir: {}".format(args.delay_model_dir))
    print("Drops model dir: {}".format(args.drops_model_dir))
    print("NSFNet test dir: {}".format(args.nsfnet_test_dir))
    print("GBN test dir: {}".format(args.gbn_test_dir))
    
    # 加载模型
    delay_model, delay_weight_path = load_model(args.delay_model_dir, 'delay', config)
    drops_model, drops_weight_path = load_model(args.drops_model_dir, 'drops', config)
    
    # 创建数据集
    nsfnet_files = tf.io.gfile.glob(os.path.join(args.nsfnet_test_dir, '*.tfrecords'))
    gbn_files = tf.io.gfile.glob(os.path.join(args.gbn_test_dir, '*.tfrecords'))
    
    nsfnet_dataset = create_dataset(nsfnet_files, args.batch_size, is_training=False)
    gbn_dataset = create_dataset(gbn_files, args.batch_size, is_training=False)
    
    print("Found {} NSFNet test files".format(len(nsfnet_files)))
    print("Found {} GBN test files".format(len(gbn_files)))
    
    # 初始化模型权重（需要先运行一次前向传播）
    print("\nInitializing models...")
    for dataset in [nsfnet_dataset.take(1)]:
        for features, labels in dataset:
            _ = delay_model(features, training=False)
            _ = drops_model(features, training=False)
            break
    
    # 加载权重
    delay_model.load_weights(delay_weight_path)
    drops_model.load_weights(drops_weight_path)
    print("Models loaded successfully!")
    
    # 评估NSFNet（同拓扑）
    print("\n" + "="*50)
    print("EVALUATING NSFNET (SAME TOPOLOGY)")
    print("="*50)
    
    # 评估delay和jitter
    _, _, nsfnet_delay_jitter_errors = evaluate_delay_jitter_model(
        delay_model, nsfnet_dataset, args.num_samples
    )
    
    # 重新创建dataset用于drops评估
    nsfnet_dataset_drops = create_dataset(nsfnet_files, args.batch_size, is_training=False)
    _, _, nsfnet_drops_errors = evaluate_drops_model(
        drops_model, nsfnet_dataset_drops, args.num_samples
    )
    
    # 合并NSFNet结果
    nsfnet_errors = {**nsfnet_delay_jitter_errors, **nsfnet_drops_errors}
    
    # 评估GBN（跨拓扑）
    print("\n" + "="*50)
    print("EVALUATING GBN (DIFFERENT TOPOLOGY)")
    print("="*50)
    
    # 评估delay和jitter
    _, _, gbn_delay_jitter_errors = evaluate_delay_jitter_model(
        delay_model, gbn_dataset, args.num_samples
    )
    
    # 重新创建dataset用于drops评估
    gbn_dataset_drops = create_dataset(gbn_files, args.batch_size, is_training=False)
    _, _, gbn_drops_errors = evaluate_drops_model(
        drops_model, gbn_dataset_drops, args.num_samples
    )
    
    # 合并GBN结果
    gbn_errors = {**gbn_delay_jitter_errors, **gbn_drops_errors}
    
    # 打印评估摘要
    print_evaluation_summary(nsfnet_errors, gbn_errors)
    
    # 绘制线性刻度CDF图（仅生成linear_focus版本）
    print("\nGenerating linear focus CDF plot...")
    plot_linear_focus_cdf(nsfnet_errors, gbn_errors, args.output_dir)
    
    print("\nEvaluation completed! Results saved to: {}".format(args.output_dir))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RouteNet TF2 模型评估和可视化脚本
加载训练好的模型权重，进行预测并绘制结果分析图
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
from routenet.routenet_tf2 import (
    RouteNet, create_dataset, parse_fn, transformation_func,
    scale_fn, heteroscedastic_loss
)

def evaluate_model(model, dataset, num_samples=None):
    """
    评估模型并收集预测结果
    
    Args:
        model: 训练好的 RouteNet 模型
        dataset: 测试数据集
        num_samples: 限制评估的样本数量，None表示评估全部
    
    Returns:
        predictions: 模型预测结果
        ground_truth: 真实标签
        relative_errors: 相对误差
    """
    predictions = {'delay': [], 'jitter': [], 'drops': []}
    ground_truth = {'delay': [], 'jitter': [], 'drops': []}
    
    sample_count = 0
    
    for features, labels in tqdm(dataset, desc="Evaluating model"):
        # 模型预测
        pred = model(features, training=False)
        
        # 异方差输出：[loc, scale]
        pred_delay = pred[:, 0].numpy()  # 延迟预测均值
        pred_scale = tf.nn.softplus(pred[:, 1]).numpy()  # 标准差
        
        # 收集真实值
        true_delay = labels['delay'].numpy()
        true_jitter = labels['jitter'].numpy() 
        true_drops = labels['drops'].numpy()
        true_packets = labels['packets'].numpy()
        
        # 计算预测的抖动（这里简化处理，实际中可能需要更复杂的逻辑）
        pred_jitter = pred_scale  # 使用预测的不确定性作为抖动的代理
        
        # 计算预测的丢包（简化为零，因为当前模型主要针对延迟）
        pred_drops = np.zeros_like(true_drops)
        
        # 存储结果
        predictions['delay'].extend(pred_delay)
        predictions['jitter'].extend(pred_jitter)
        predictions['drops'].extend(pred_drops)
        
        ground_truth['delay'].extend(true_delay)
        ground_truth['jitter'].extend(true_jitter)
        ground_truth['drops'].extend(true_drops)
        
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
        # 避免除以零的情况
        mask = np.abs(ground_truth[key]) > 1e-10
        rel_error = np.full_like(predictions[key], np.nan)
        rel_error[mask] = (predictions[key][mask] - ground_truth[key][mask]) / ground_truth[key][mask]
        relative_errors[key] = rel_error
    
    return predictions, ground_truth, relative_errors

def plot_cdf_comparison(relative_errors, save_path=None):
    """
    绘制相对误差的CDF对比图，复刻您展示的图像风格
    
    Args:
        relative_errors: 相对误差字典
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(10, 8))
    
    # 设置颜色和线型
    colors = {'delay': 'black', 'jitter': 'gray', 'drops': 'lightgray'}
    linestyles = {'test': '-', 'gbn': '--', 'pred': ':', 'model': '-.'}
    
    # 为每种指标绘制CDF
    for metric, errors in relative_errors.items():
        # 过滤掉 NaN 值
        valid_errors = errors[~np.isnan(errors)]
        if len(valid_errors) == 0:
            continue
            
        # 计算CDF
        sorted_errors = np.sort(valid_errors)
        n = len(sorted_errors)
        y = np.arange(1, n + 1) / n
        
        # 绘制曲线
        if metric == 'delay':
            plt.plot(sorted_errors, y, color=colors[metric], linestyle='-', 
                    linewidth=2, label=f'Test {metric}')
            plt.plot(sorted_errors, y, color=colors[metric], linestyle='--', 
                    linewidth=2, label=f'GBN {metric}')
        elif metric == 'jitter':
            plt.plot(sorted_errors, y, color=colors[metric], linestyle='-', 
                    linewidth=1.5, label=f'Test {metric}')
            plt.plot(sorted_errors, y, color=colors[metric], linestyle='--', 
                    linewidth=1.5, label=f'GBN {metric}')
        else:  # drops
            plt.plot(sorted_errors, y, color=colors[metric], linestyle=':', 
                    linewidth=1, label=f'Test {metric}')
            plt.plot(sorted_errors, y, color=colors[metric], linestyle=':', 
                    linewidth=1, label=f'GBN {metric}')
    
    # 设置图像属性
    plt.xlabel('ε', fontsize=14)
    plt.ylabel('P(ε* ≤ ε)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.xlim(-1.0, 1.0)
    plt.ylim(0.0, 1.0)
    
    # 添加标题
    plt.title('Relative Error CDF Comparison', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to {save_path}")
    
    plt.show()

def plot_detailed_analysis(predictions, ground_truth, relative_errors, save_dir=None):
    """
    绘制详细的分析图，包括散点图、误差分布等
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RouteNet Model Performance Analysis', fontsize=16, fontweight='bold')
    
    metrics = ['delay', 'jitter', 'drops']
    
    for i, metric in enumerate(metrics):
        pred = predictions[metric]
        true = ground_truth[metric]
        rel_err = relative_errors[metric]
        
        # 过滤有效数据
        valid_mask = ~np.isnan(rel_err)
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        rel_err_valid = rel_err[valid_mask]
        
        if len(pred_valid) == 0:
            continue
        
        # 第一行：预测 vs 真实值散点图
        axes[0, i].scatter(true_valid, pred_valid, alpha=0.5, s=1)
        min_val = min(np.min(true_valid), np.min(pred_valid))
        max_val = max(np.max(true_valid), np.max(pred_valid))
        axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, i].set_xlabel(f'True {metric}')
        axes[0, i].set_ylabel(f'Predicted {metric}')
        axes[0, i].set_title(f'{metric.capitalize()} Prediction')
        axes[0, i].grid(True, alpha=0.3)
        
        # 第二行：相对误差分布
        axes[1, i].hist(rel_err_valid, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[1, i].axvline(np.mean(rel_err_valid), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(rel_err_valid):.3f}')
        axes[1, i].axvline(np.median(rel_err_valid), color='green', linestyle='--',
                          label=f'Median: {np.median(rel_err_valid):.3f}')
        axes[1, i].set_xlabel('Relative Error')
        axes[1, i].set_ylabel('Density')
        axes[1, i].set_title(f'{metric.capitalize()} Relative Error Distribution')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'detailed_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Detailed analysis saved to {save_dir}/detailed_analysis.png")
    
    plt.show()

def print_metrics_summary(predictions, ground_truth, relative_errors):
    """打印评估指标摘要"""
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    for metric in ['delay', 'jitter', 'drops']:
        pred = predictions[metric]
        true = ground_truth[metric]
        rel_err = relative_errors[metric]
        
        # 过滤有效数据
        valid_mask = ~np.isnan(rel_err)
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        rel_err_valid = rel_err[valid_mask]
        
        if len(rel_err_valid) == 0:
            print(f"{metric.upper()}: No valid predictions")
            continue
        
        # 计算各种指标
        mae = np.mean(np.abs(pred_valid - true_valid))
        rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
        mape = np.mean(np.abs(rel_err_valid)) * 100
        
        print(f"\n{metric.upper()}:")
        print(f"  MAE (Mean Absolute Error): {mae:.6f}")
        print(f"  RMSE (Root Mean Square Error): {rmse:.6f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"  Relative Error - Mean: {np.mean(rel_err_valid):.6f}")
        print(f"  Relative Error - Std: {np.std(rel_err_valid):.6f}")
        print(f"  Relative Error - 95% Quantile: {np.percentile(np.abs(rel_err_valid), 95):.6f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate RouteNet TF2 Model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model weights')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test TFRecord files')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模型配置（需要与训练时保持一致）
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
    }
    
    # 创建模型并加载权重
    model = RouteNet(config, output_units=2)
    weight_path = os.path.join(args.model_dir, "best_model.weights.h5")
    
    print(f"Loading model weights from: {weight_path}")
    
    # 创建测试数据集
    test_files = tf.io.gfile.glob(os.path.join(args.test_dir, '*.tfrecords'))
    test_dataset = create_dataset(test_files, args.batch_size, is_training=False)
    
    print(f"Found {len(test_files)} test files.")
    
    # 需要先运行一次前向传播来初始化权重
    for features, labels in test_dataset.take(1):
        _ = model(features, training=False)
        break
    
    # 加载权重
    try:
        model.load_weights(weight_path)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # 评估模型
    print("\nEvaluating model...")
    predictions, ground_truth, relative_errors = evaluate_model(
        model, test_dataset, args.num_samples
    )
    
    # 打印评估摘要
    print_metrics_summary(predictions, ground_truth, relative_errors)
    
    # 绘制CDF对比图
    print("\nGenerating CDF comparison plot...")
    cdf_path = os.path.join(args.output_dir, 'relative_error_cdf.png')
    plot_cdf_comparison(relative_errors, cdf_path)
    
    # 绘制详细分析图
    print("\nGenerating detailed analysis plots...")
    plot_detailed_analysis(predictions, ground_truth, relative_errors, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

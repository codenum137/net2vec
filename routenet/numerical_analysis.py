#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RouteNet 数值性能分析脚本
计算MAE、RMSE、MAPE指标，用于精确量化模型性能
"""

import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
import pandas as pd

# 导入训练脚本中的相关函数和类
import sys
sys.path.append(os.path.dirname(__file__))
from routenet_tf2 import (
    RouteNet, create_dataset, parse_fn, transformation_func,
    scale_fn, heteroscedastic_loss, binomial_loss, create_model_and_loss_fn
)

def load_model(model_dir, target, config, use_kan=False):
    """
    加载指定目标的模型
    """
    # 根据target和use_kan参数创建相应的模型
    model, _ = create_model_and_loss_fn(config, target, use_kan=use_kan)
    
    # 寻找权重文件
    if use_kan:
        weight_files = [
            os.path.join(model_dir, "best_{}_kan_model.weights.h5".format(target)),
            os.path.join(model_dir, "best_kan_model.weights.h5"),
            os.path.join(model_dir, "best_{}_model.weights.h5".format(target)),
            os.path.join(model_dir, "best_model.weights.h5"),
        ]
    else:
        weight_files = [
            os.path.join(model_dir, "best_{}_model.weights.h5".format(target)),
            os.path.join(model_dir, "best_model.weights.h5"),
        ]
    
    weight_path = None
    for path in weight_files:
        if os.path.exists(path):
            weight_path = path
            break
    
    if weight_path is None:
        raise FileNotFoundError("No model weights found in {}".format(model_dir))
    
    model_type = "KAN" if use_kan else "MLP"
    print("Loading {} {} model weights from: {}".format(model_type, target, weight_path))
    
    return model, weight_path

def evaluate_model_metrics(model, dataset, dataset_name, num_samples=None):
    """
    评估模型并计算MAE、RMSE、MAPE指标
    
    Args:
        model: 延迟预测模型
        dataset: 测试数据集
        dataset_name: 数据集名称（用于打印）
        num_samples: 限制评估的样本数量
    
    Returns:
        results: 包含各项指标的字典
    """
    predictions_delay = []
    predictions_jitter = []
    ground_truth_delay = []
    ground_truth_jitter = []
    
    sample_count = 0
    
    print(f"\nEvaluating {dataset_name} dataset...")
    for features, labels in tqdm(dataset, desc=f"Processing {dataset_name}"):
        # 模型预测 - 异方差输出：[loc, scale]
        pred = model(features, training=False)
        
        pred_delay = pred[:, 0].numpy()  # 延迟预测均值 (loc)
        
        # 与原版保持一致的scale计算，包含c偏移常数
        c = np.log(np.expm1(np.float32(0.098)))
        pred_scale = tf.nn.softplus(c + pred[:, 1]).numpy() + 1e-9
        
        # jitter_prediction = scale**2
        pred_jitter = pred_scale ** 2
        
        # 收集真实值
        true_delay = labels['delay'].numpy()
        true_jitter = labels['jitter'].numpy()
        
        # 存储结果
        predictions_delay.extend(pred_delay)
        predictions_jitter.extend(pred_jitter)
        ground_truth_delay.extend(true_delay)
        ground_truth_jitter.extend(true_jitter)
        
        sample_count += len(pred_delay)
        if num_samples and sample_count >= num_samples:
            break
    
    # 转换为numpy数组
    pred_delay = np.array(predictions_delay)
    pred_jitter = np.array(predictions_jitter)
    true_delay = np.array(ground_truth_delay)
    true_jitter = np.array(ground_truth_jitter)
    
    print(f"{dataset_name} - Total samples: {len(pred_delay)}")
    
    # 计算各项指标
    results = {}
    
    for metric_name, pred, true in [('delay', pred_delay, true_delay), 
                                    ('jitter', pred_jitter, true_jitter)]:
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(pred - true))
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        
        # MAPE (Mean Absolute Percentage Error)
        # 避免除零，只考虑真实值大于某个阈值的样本
        mask = np.abs(true) > 1e-10
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100
        else:
            mape = float('inf')
        
        # 相对误差统计
        relative_errors = (pred[mask] - true[mask]) / true[mask] if np.sum(mask) > 0 else np.array([])
        mean_relative_error = np.mean(relative_errors) if len(relative_errors) > 0 else 0
        
        results[metric_name] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mean_relative_error': mean_relative_error,
            'samples': len(pred),
            'valid_samples_for_mape': np.sum(mask)
        }
        
        print(f"  {metric_name.upper()}:")
        print(f"    MAE:  {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    MAPE: {mape:.4f}%")
        print(f"    Mean Relative Error: {mean_relative_error:.4f}")
        print(f"    Valid samples for MAPE: {np.sum(mask)}/{len(pred)}")
    
    return results

def compare_models_performance(nsfnet_results, gbn_results, output_dir):
    """
    比较模型在不同数据集上的性能并生成报告
    """
    print("\n" + "="*80)
    print("NUMERICAL PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    # 创建性能对比表格
    metrics = ['mae', 'rmse', 'mape']
    targets = ['delay', 'jitter']
    
    # 准备数据用于表格显示
    comparison_data = []
    
    for target in targets:
        for metric in metrics:
            row = {
                'Target': target.upper(),
                'Metric': metric.upper(),
                'NSFNet (Training Topology)': f"{nsfnet_results[target][metric]:.6f}" if metric != 'mape' else f"{nsfnet_results[target][metric]:.4f}%",
                'GBN (Test Topology)': f"{gbn_results[target][metric]:.6f}" if metric != 'mape' else f"{gbn_results[target][metric]:.4f}%"
            }
            
            # 计算性能退化（从训练拓扑到测试拓扑）
            if metric == 'mape':
                degradation = gbn_results[target][metric] - nsfnet_results[target][metric]
                row['Degradation'] = f"{degradation:+.4f}%"
            else:
                degradation_ratio = (gbn_results[target][metric] - nsfnet_results[target][metric]) / nsfnet_results[target][metric] * 100
                row['Degradation'] = f"{degradation_ratio:+.2f}%"
            
            comparison_data.append(row)
    
    # 创建DataFrame并打印
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 保存到CSV文件
    csv_path = os.path.join(output_dir, 'numerical_performance_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # 计算泛化性能摘要
    print("\n" + "-"*50)
    print("GENERALIZATION PERFORMANCE SUMMARY")
    print("-"*50)
    
    for target in targets:
        print(f"\n{target.upper()} Prediction:")
        
        # MAE泛化性能
        mae_degradation = (gbn_results[target]['mae'] - nsfnet_results[target]['mae']) / nsfnet_results[target]['mae'] * 100
        print(f"  MAE Degradation: {mae_degradation:+.2f}% ({nsfnet_results[target]['mae']:.6f} → {gbn_results[target]['mae']:.6f})")
        
        # RMSE泛化性能
        rmse_degradation = (gbn_results[target]['rmse'] - nsfnet_results[target]['rmse']) / nsfnet_results[target]['rmse'] * 100
        print(f"  RMSE Degradation: {rmse_degradation:+.2f}% ({nsfnet_results[target]['rmse']:.6f} → {gbn_results[target]['rmse']:.6f})")
        
        # MAPE泛化性能
        mape_degradation = gbn_results[target]['mape'] - nsfnet_results[target]['mape']
        print(f"  MAPE Degradation: {mape_degradation:+.4f}% ({nsfnet_results[target]['mape']:.4f}% → {gbn_results[target]['mape']:.4f}%)")
        
        # 评估泛化质量
        if mae_degradation < 50 and rmse_degradation < 50:
            quality = "GOOD"
        elif mae_degradation < 100 and rmse_degradation < 100:
            quality = "MODERATE"
        else:
            quality = "POOR"
        print(f"  Generalization Quality: {quality}")

def main():
    parser = argparse.ArgumentParser(description='Numerical Performance Analysis for RouteNet')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing delay prediction model')
    parser.add_argument('--nsfnet_test_dir', type=str, required=True,
                      help='Directory containing NSFNet test data')
    parser.add_argument('--gbn_test_dir', type=str, required=True,
                      help='Directory containing GBN test data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save analysis results')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Limit number of samples to evaluate')
    parser.add_argument('--kan', action='store_true',
                      help='Evaluate KAN-based models instead of traditional MLP models')
    
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
    
    model_type = "KAN" if args.kan else "MLP"
    print(f"Starting numerical analysis for {model_type} model...")
    print(f"Model dir: {args.model_dir}")
    print(f"NSFNet test dir: {args.nsfnet_test_dir}")
    print(f"GBN test dir: {args.gbn_test_dir}")
    
    # 加载模型
    delay_model, delay_weight_path = load_model(args.model_dir, 'delay', config, use_kan=args.kan)
    
    # 创建数据集
    nsfnet_files = tf.io.gfile.glob(os.path.join(args.nsfnet_test_dir, '*.tfrecords'))
    gbn_files = tf.io.gfile.glob(os.path.join(args.gbn_test_dir, '*.tfrecords'))
    
    nsfnet_dataset = create_dataset(nsfnet_files, args.batch_size, is_training=False)
    gbn_dataset = create_dataset(gbn_files, args.batch_size, is_training=False)
    
    print(f"Found {len(nsfnet_files)} NSFNet test files")
    print(f"Found {len(gbn_files)} GBN test files")
    
    # 初始化模型权重
    print("\nInitializing model...")
    for dataset in [nsfnet_dataset.take(1)]:
        for features, labels in dataset:
            _ = delay_model(features, training=False)
            break
    
    # 加载权重
    delay_model.load_weights(delay_weight_path)
    print("Model loaded successfully!")
    
    # 评估NSFNet（训练拓扑）
    nsfnet_results = evaluate_model_metrics(
        delay_model, nsfnet_dataset, "NSFNet", args.num_samples
    )
    
    # 评估GBN（测试拓扑）
    gbn_results = evaluate_model_metrics(
        delay_model, gbn_dataset, "GBN", args.num_samples
    )
    
    # 比较性能并生成报告
    compare_models_performance(nsfnet_results, gbn_results, args.output_dir)
    
    print(f"\nNumerical analysis completed! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

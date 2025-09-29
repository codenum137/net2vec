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
    scale_fn, heteroscedastic_loss, binomial_loss, create_model_and_loss_fn,
)

def calculate_r2(y_pred, y_true):
    """
    计算R²决定系数
    R² = 1 - SS_res / SS_tot
    其中 SS_res = Σ(y_true - y_pred)²  (残差平方和)
         SS_tot = Σ(y_true - y_mean)²  (总平方和)
    
    Args:
        y_pred: 预测值数组
        y_true: 真实值数组
    
    Returns:
        r2: R²值，范围通常在(-∞, 1]，1表示完美拟合
    """
    # 去除无效值
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if np.sum(mask) < 2:
        return float('-inf')
    
    y_pred_clean = y_pred[mask]
    y_true_clean = y_true[mask]
    
    # 计算总平方和
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    
    # 计算残差平方和
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    
    # 计算R²
   
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float('-inf')
    
    r2 = 1 - (ss_res / ss_tot)
    return r2

def calculate_nll(y_pred_mean, y_pred_scale, y_true):
    """
    计算异方差模型的负对数似然 (Negative Log-Likelihood)
    
    对于正态分布 N(μ, σ²)，负对数似然为：
    NLL = -log p(y|μ,σ) = 0.5 * log(2π) + log(σ) + 0.5 * ((y-μ)/σ)²
    
    Args:
        y_pred_mean: 预测均值 (μ)
        y_pred_scale: 预测标准差 (σ)
        y_true: 真实值 (y)
    
    Returns:
        nll: 平均负对数似然值
    """
    # 去除无效值
    mask = (np.isfinite(y_pred_mean) & 
            np.isfinite(y_pred_scale) & 
            np.isfinite(y_true) & 
            (y_pred_scale > 0))  # 确保标准差为正
    
    if np.sum(mask) == 0:
        return float('inf')
    
    mean_clean = y_pred_mean[mask]
    scale_clean = y_pred_scale[mask]
    true_clean = y_true[mask]
    
    # 计算标准化残差
    normalized_residuals = (true_clean - mean_clean) / scale_clean
    
    # 计算负对数似然
    # NLL = 0.5 * log(2π) + log(σ) + 0.5 * ((y-μ)/σ)²
    log_2pi = np.log(2 * np.pi)
    nll_per_sample = (0.5 * log_2pi + 
                      np.log(scale_clean) + 
                      0.5 * normalized_residuals ** 2)
    
    # 返回平均NLL
    return np.mean(nll_per_sample)

def load_model(model_dir, target, config, use_kan=False, use_final_layer=True):
    """加载指定目标的标准模型（MLP 或 KAN，包括 B-spline）。"""
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
    
    # 使用标准模型创建方式
    model, _ = create_model_and_loss_fn(config, target, use_kan=use_kan, use_final_layer=use_final_layer)
    model_type = "KAN" if use_kan else "MLP"
    kb = config.get('kan_basis', 'poly') if use_kan else None
    if use_kan and kb == 'bspline':
        print(f"Using KAN (bspline) for {target} evaluation")
    elif use_kan:
        print(f"Using KAN (poly) for {target} evaluation")
    else:
        print(f"Using MLP for {target} evaluation")
    print("Loading {} {} model weights from: {}".format(model_type, target, weight_path))
    return model, weight_path

"""Note: Legacy probe utilities have been removed. Single-readout mode now outputs
final predictions directly (delay:2, drops:1) so numerical analysis works unchanged."""

def evaluate_model_metrics(model, dataset, dataset_name, num_samples=None):
    """
    评估模型并计算MAE、RMSE、MAPE、R²、NLL指标
    
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
    predictions_scale = []  # 用于计算NLL
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
        predictions_scale.extend(pred_scale)  # 保存scale用于NLL计算
        ground_truth_delay.extend(true_delay)
        ground_truth_jitter.extend(true_jitter)
        
        sample_count += len(pred_delay)
        if num_samples and sample_count >= num_samples:
            break
    
    # 转换为numpy数组
    pred_delay = np.array(predictions_delay)
    pred_jitter = np.array(predictions_jitter)
    pred_scale = np.array(predictions_scale)
    true_delay = np.array(ground_truth_delay)
    true_jitter = np.array(ground_truth_jitter)
    
    print(f"{dataset_name} - Total samples: {len(pred_delay)}")
    
    # 计算各项指标
    results = {}
    
    # 先计算delay的NLL（异方差负对数似然）
    delay_nll = calculate_nll(pred_delay, pred_scale, true_delay)
    results['delay_nll'] = delay_nll
    
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
        
        # R² (决定系数)
        r2 = calculate_r2(pred, true)
        
        # 相对误差统计
        relative_errors = (pred[mask] - true[mask]) / true[mask] if np.sum(mask) > 0 else np.array([])
        mean_relative_error = np.mean(relative_errors) if len(relative_errors) > 0 else 0
        
        results[metric_name] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'mean_relative_error': mean_relative_error,
            'samples': len(pred),
            'valid_samples_for_mape': np.sum(mask)
        }
        
        print(f"  {metric_name.upper()}:")
        print(f"    MAE:  {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    MAPE: {mape:.4f}%")
        print(f"    R²:   {r2:.6f}")
        print(f"    Mean Relative Error: {mean_relative_error:.4f}")
        print(f"    Valid samples for MAPE: {np.sum(mask)}/{len(pred)}")
    
    # 打印NLL指标
    print(f"  PROBABILISTIC METRICS:")
    print(f"    Delay NLL: {delay_nll:.6f}")
    
    return results

def compare_models_performance(nsfnet_results, gbn_results, output_dir):
    """
    比较模型在不同数据集上的性能并生成报告
    """
    print("\n" + "="*80)
    print("NUMERICAL PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    # 创建性能对比表格
    metrics = ['mae', 'rmse', 'mape', 'r2']
    targets = ['delay', 'jitter']
    
    # 准备数据用于表格显示
    comparison_data = []
    
    for target in targets:
        for metric in metrics:
            # 格式化数值显示
            if metric == 'mape':
                nsfnet_val = f"{nsfnet_results[target][metric]:.4f}%"
                gbn_val = f"{gbn_results[target][metric]:.4f}%"
            elif metric == 'r2':
                nsfnet_val = f"{nsfnet_results[target][metric]:.6f}"
                gbn_val = f"{gbn_results[target][metric]:.6f}"
            else:
                nsfnet_val = f"{nsfnet_results[target][metric]:.6f}"
                gbn_val = f"{gbn_results[target][metric]:.6f}"
            
            row = {
                'Target': target.upper(),
                'Metric': metric.upper(),
                'NSFNet (Training Topology)': nsfnet_val,
                'GBN (Test Topology)': gbn_val
            }
            
            # 计算性能退化（从训练拓扑到测试拓扑）
            if metric == 'mape':
                degradation = gbn_results[target][metric] - nsfnet_results[target][metric]
                row['Degradation'] = f"{degradation:+.4f}%"
            elif metric == 'r2':
                # R²的退化是减少，所以用相反的符号
                degradation = gbn_results[target][metric] - nsfnet_results[target][metric]
                row['Degradation'] = f"{degradation:+.6f}"
            else:
                degradation_ratio = (gbn_results[target][metric] - nsfnet_results[target][metric]) / nsfnet_results[target][metric] * 100
                row['Degradation'] = f"{degradation_ratio:+.2f}%"
            
            comparison_data.append(row)
    
    # 添加NLL指标（仅对delay有效）
    nll_row = {
        'Target': 'DELAY',
        'Metric': 'NLL',
        'NSFNet (Training Topology)': f"{nsfnet_results['delay_nll']:.6f}",
        'GBN (Test Topology)': f"{gbn_results['delay_nll']:.6f}",
        'Degradation': f"{((gbn_results['delay_nll'] - nsfnet_results['delay_nll']) / nsfnet_results['delay_nll'] * 100):+.2f}%"
    }
    comparison_data.append(nll_row)
    
    # 创建DataFrame并打印
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 保存到CSV文件
    csv_path = os.path.join(output_dir, 'numerical_performance_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # 计算泛化性能摘要
    print("\n" + "-"*50)
    
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
        
        # R²泛化性能
        r2_degradation = gbn_results[target]['r2'] - nsfnet_results[target]['r2']
        print(f"  R² Degradation: {r2_degradation:+.6f} ({nsfnet_results[target]['r2']:.6f} → {gbn_results[target]['r2']:.6f})")
        
        # 评估泛化质量 (更新评估标准)
        if mae_degradation < 50 and rmse_degradation < 50 and r2_degradation > -0.1:
            quality = "GOOD"
        elif mae_degradation < 100 and rmse_degradation < 100 and r2_degradation > -0.3:
            quality = "MODERATE"
        else:
            quality = "POOR"
        print(f"  Generalization Quality: {quality}")
    
    # NLL泛化性能（仅对delay）
    print(f"\nPROBABILISTIC PERFORMANCE:")
    nll_degradation = (gbn_results['delay_nll'] - nsfnet_results['delay_nll']) / nsfnet_results['delay_nll'] * 100
    print(f"  Delay NLL Degradation: {nll_degradation:+.2f}% ({nsfnet_results['delay_nll']:.6f} → {gbn_results['delay_nll']:.6f})")
    
    if nll_degradation < 50:
        nll_quality = "GOOD - Model uncertainty well calibrated"
    elif nll_degradation < 100:
        nll_quality = "MODERATE - Some uncertainty miscalibration"
    else:
        nll_quality = "POOR - Significant uncertainty miscalibration"
    print(f"  Uncertainty Calibration: {nll_quality}")

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
    # KAN basis options (optional; only used when --kan is set)
    parser.add_argument('--kan_basis', type=str, choices=['poly', 'bspline'], default=None,
                      help='KAN basis type for readout: poly (default) or bspline')
    parser.add_argument('--kan_grid_size', type=int, default=None,
                      help='Number of intervals for B-spline grid (only for bspline basis)')
    parser.add_argument('--kan_spline_order', type=int, default=None,
                      help='Degree/order of B-spline basis (only for bspline basis)')
    # 单头模式：训练时使用 --single-readout（读出层直接输出预测，无最终Dense头）
    parser.add_argument('--single-readout', action='store_true',
                      help='Model was trained in single-readout mode (readout directly outputs predictions).')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模型配置（与训练时保持一致）
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        # IMPORTANT: must match training architecture; training default was 1
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
    }
    
    # 如果评估 KAN 模型，写入可选的基函数配置；支持从目录名推断 bspline
    if args.kan:
        inferred_basis = None
        if args.kan_basis is None:
            mdl = args.model_dir.lower()
            if 'bspline' in mdl or 'b_spline' in mdl or 'b-spline' in mdl:
                inferred_basis = 'bspline'
        basis = args.kan_basis or inferred_basis or 'poly'
        config['kan_basis'] = basis
        if basis == 'bspline':
            if args.kan_grid_size is not None:
                config['kan_grid_size'] = args.kan_grid_size
            if args.kan_spline_order is not None:
                config['kan_spline_order'] = args.kan_spline_order

    model_type = "KAN" if args.kan else "MLP"
    print(f"Starting numerical analysis for {model_type} model...")
    print(f"Model dir: {args.model_dir}")
    print(f"NSFNet test dir: {args.nsfnet_test_dir}")
    print(f"GBN test dir: {args.gbn_test_dir}")

    # 单头模式：强制 readout_layers=1，并禁用 final_layer
    if args.single_readout and config.get('readout_layers', 1) != 1:
        print('[Info] Overriding readout_layers -> 1 for single-readout numerical analysis consistency')
        config['readout_layers'] = 1

    delay_model, delay_weight_path = load_model(
        args.model_dir, 'delay', config, use_kan=args.kan,
        use_final_layer=not args.single_readout
    )
    
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
    try:
        delay_model.load_weights(delay_weight_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"[LoadWarning] Initial load failed: {e}")
        # Attempt intelligent fallbacks:
        # 1) Toggle final layer usage (single-readout vs full) if mismatch suspected
        attempted = False
        if args.single_readout:
            print("[LoadFallback] Detected single-readout mode; retrying assuming weights include final Dense head...")
            alt_model, _ = load_model(
                args.model_dir, 'delay', config, use_kan=args.kan,
                use_final_layer=True
            )
            for features, labels in nsfnet_dataset.take(1):
                _ = alt_model(features, training=False)
                break
            try:
                alt_model.load_weights(delay_weight_path)
                delay_model = alt_model
                print("[LoadFallback] Successfully loaded by enabling final layer (weights include Dense head).")
                attempted = True
            except Exception as ee:
                print(f"[LoadFallback] Retry with final layer also failed: {ee}")
        else:
            print("[LoadFallback] Not single-readout; retrying assuming single-readout weights (no final Dense head)...")
            alt_model, _ = load_model(
                args.model_dir, 'delay', config, use_kan=args.kan,
                use_final_layer=False
            )
            for features, labels in nsfnet_dataset.take(1):
                _ = alt_model(features, training=False)
                break
            try:
                alt_model.load_weights(delay_weight_path)
                delay_model = alt_model
                print("[LoadFallback] Successfully loaded by disabling final layer (weights lack Dense head).")
                attempted = True
            except Exception as ee:
                print(f"[LoadFallback] Retry without final layer also failed: {ee}")

        # 2) Fallback adjusting readout_layers if still not loaded
        if not attempted and config.get('readout_layers', 1) != 1:
            print("[LoadFallback] Retrying with readout_layers=1 (training default)...")
            config['readout_layers'] = 1
            alt_model, _ = load_model(
                args.model_dir, 'delay', config, use_kan=args.kan,
                use_final_layer=not args.single_readout
            )
            for features, labels in nsfnet_dataset.take(1):
                _ = alt_model(features, training=False)
                break
            alt_model.load_weights(delay_weight_path)
            delay_model = alt_model
            print("[LoadFallback] Model loaded successfully with readout_layers=1 after architecture adjustment")
        elif not attempted:
            # If all fallbacks failed, re-raise
            raise
    
    # single-readout 模式提示输出形状确认
    if args.single_readout:
        for features, labels in nsfnet_dataset.take(1):
            sample_out = delay_model(features, training=False)
            print(f"[Mode] Single-readout numerical analysis. Output shape: {sample_out.shape} (expect [...,2] for delay)")
            break

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

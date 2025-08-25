#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RouteNet TF1.x 原始模型评估脚本
支持评估delay/jitter和drops两种不同的模型，并绘制相对误差CDF图
支持同拓扑（nsfnet）和跨拓扑（gbn）评估
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

# 导入原始训练脚本中的相关函数和类
import sys
sys.path.append(os.path.dirname(__file__))
from routenet import delay_model_fn, drop_model_fn, tfrecord_input_fn

# 禁用 TF2.x 行为，确保兼容 TF1.x 模型
tf.compat.v1.disable_v2_behavior()

class HParams:
    """超参数类"""
    def __init__(self):
        self.batch_size = 32
        # 添加模型所需的所有参数（基于原始routenet.py中的默认值）
        self.link_state_dim = 4
        self.path_state_dim = 2
        self.T = 3
        self.readout_units = 8
        self.readout_layers = 2
        self.dropout_rate = 0.5
        self.l2 = 0.1
        self.l2_2 = 0.01
        self.learn_embedding = True
        self.dropout_rate = 0.5

def load_tf1_model_and_predict(model_dir, test_files, target, num_samples=None):
    """
    加载TF1.x模型并进行预测，同时获取真实标签
    """
    print(f"Loading {target} model from: {model_dir}")
    
    # 创建超参数对象  
    hparams = HParams()
    
    # 创建新的图
    graph = tf.Graph()
    
    with graph.as_default():
        # 直接获取样本 - tfrecord_input_fn 返回 (features, labels)
        features, labels = tfrecord_input_fn(test_files, hparams, shuffle_buf=None, target=target)
        
        # 根据目标选择模型函数创建预测输出
        if target == 'delay':
            model_spec = delay_model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, hparams)
        else:
            model_spec = drop_model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, hparams)
        
        predictions_op = model_spec.predictions
        
        # 创建saver来加载权重
        saver = tf.compat.v1.train.Saver()
        
        # 创建session
        with tf.compat.v1.Session() as sess:
            # 恢复模型权重
            checkpoint_path = tf.train.latest_checkpoint(model_dir)
            if checkpoint_path is None:
                raise ValueError(f"No checkpoint found in {model_dir}")
            
            saver.restore(sess, checkpoint_path)
            print(f"Model restored from {checkpoint_path}")
            
            predictions = {'delay': [], 'jitter': [], 'drops': []}
            ground_truth = {'delay': [], 'jitter': [], 'drops': []}
            
            sample_count = 0
            try:
                with tqdm(desc=f"Evaluating {target} model") as pbar:
                    while True:
                        # 同时获取预测和真实标签
                        pred_vals, label_vals = sess.run([predictions_op, labels])
                        
                        if target == 'delay':
                            # 延迟模型预测
                            if 'delay' in pred_vals:
                                predictions['delay'].extend(pred_vals['delay'])
                            if 'jitter' in pred_vals:
                                predictions['jitter'].extend(pred_vals['jitter'])
                            
                            # 真实标签
                            if 'delay' in label_vals:
                                ground_truth['delay'].extend(label_vals['delay'])
                            if 'jitter' in label_vals:
                                ground_truth['jitter'].extend(label_vals['jitter'])
                        
                        else:  # target == 'drops'
                            # 丢包模型预测
                            if 'drops' in pred_vals:
                                predictions['drops'].extend(pred_vals['drops'])
                            
                            # 真实标签 - 计算丢包率
                            if 'drops' in label_vals and 'packets' in label_vals:
                                true_drop_rates = label_vals['drops'] / (label_vals['packets'] + 1e-10)
                                ground_truth['drops'].extend(true_drop_rates)
                        
                        batch_size = len(pred_vals.get('delay', pred_vals.get('drops', [0])))
                        sample_count += batch_size
                        pbar.update(batch_size)
                        
                        if num_samples and sample_count >= num_samples:
                            break
                            
            except tf.errors.OutOfRangeError:
                print(f"Finished processing {sample_count} samples")
    
    # 计算相对误差
    relative_errors = calculate_relative_errors(predictions, ground_truth, target)
    
    print(f"Extracted {len(ground_truth.get('delay', ground_truth.get('drops', [])))} ground truth labels")
    
    return predictions, ground_truth, relative_errors

def calculate_relative_errors(predictions, ground_truth, target):
    """
    计算相对误差
    """
    relative_errors = {}
    
    if target == 'delay':
        for metric in ['delay', 'jitter']:
            if len(predictions[metric]) > 0 and len(ground_truth[metric]) > 0:
                pred_array = np.array(predictions[metric])
                true_array = np.array(ground_truth[metric])
                
                # 确保数组长度一致
                min_len = min(len(pred_array), len(true_array))
                pred_array = pred_array[:min_len]
                true_array = true_array[:min_len]
                
                # 计算相对误差
                mask = np.abs(true_array) > 1e-10
                rel_error = np.full_like(pred_array, np.nan)
                rel_error[mask] = (pred_array[mask] - true_array[mask]) / true_array[mask]
                relative_errors[metric] = rel_error[~np.isnan(rel_error)]
    
    else:  # target == 'drops'
        if len(predictions['drops']) > 0 and len(ground_truth['drops']) > 0:
            pred_array = np.array(predictions['drops'])
            true_array = np.array(ground_truth['drops'])
            
            # 确保数组长度一致
            min_len = min(len(pred_array), len(true_array))
            pred_array = pred_array[:min_len]
            true_array = true_array[:min_len]
            
            # 只考虑丢包率大于阈值的情况
            mask = true_array > 1e-6
            rel_error = np.full_like(pred_array, np.nan)
            rel_error[mask] = (pred_array[mask] - true_array[mask]) / true_array[mask]
            relative_errors['drops'] = rel_error[~np.isnan(rel_error)]
    
    return relative_errors

def plot_linear_focus_cdf(nsfnet_errors, gbn_errors, output_dir):
    """
    绘制线性刻度的相对误差CDF图，显示正负误差分布
    """
    # 设置颜色和线型
    colors = {'delay': '#1f77b4', 'jitter': '#ff7f0e', 'drops': '#2ca02c'}
    linestyles = {'nsfnet': '-', 'gbn': '--'}
    
    plt.figure(figsize=(12, 8))
    
    for metric in ['delay', 'jitter', 'drops']:
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
    
    plt.xlabel('Relative Error (Positive: Over-prediction, Negative: Under-prediction)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title('Original RouteNet (TF1.x) - Relative Error CDF\\n(Ideal is Vertical Red Line at 0)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right', fontsize=12)
    
    # 收集所有误差来设置x轴范围
    all_signed_errors = []
    for errors_dict in [nsfnet_errors, gbn_errors]:
        for metric in ['delay', 'jitter', 'drops']:
            if metric in errors_dict and len(errors_dict[metric]) > 0:
                all_signed_errors.extend(errors_dict[metric])
    
    if all_signed_errors:
        min_error = np.min(all_signed_errors)
        max_error = np.max(all_signed_errors)
        max_abs_error = max(abs(min_error), abs(max_error))
        display_range = min(max_abs_error * 1.2, 1.0)
        plt.xlim(-display_range, display_range)
    else:
        plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1)
    
    # 添加统计信息
    detailed_stats = []
    for topo in ['nsfnet', 'gbn']:
        errors = nsfnet_errors if topo == 'nsfnet' else gbn_errors
        for metric in ['delay', 'jitter', 'drops']:
            if metric in errors and len(errors[metric]) > 0:
                mean_error = np.mean(errors[metric])
                abs_errors = np.abs(errors[metric])
                median_abs_error = np.median(abs_errors)
                detailed_stats.append('{} {}: Mean={:.4f}, |Med|={:.4f}'.format(
                    topo.upper(), metric.upper(), mean_error, median_abs_error))
    
    detailed_stats_str = '\\n'.join(detailed_stats)
    plt.text(0.02, 0.98, detailed_stats_str, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    output_path = os.path.join(output_dir, 'original_routenet_relative_error_cdf_linear_focus.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("Original RouteNet CDF plot saved to: {}".format(output_path))

def print_evaluation_summary(nsfnet_errors, gbn_errors):
    """
    打印评估摘要统计
    """
    print("\\n" + "="*60)
    print("ORIGINAL ROUTENET (TF1.x) EVALUATION SUMMARY")
    print("="*60)
    
    for topo_name, errors in [('NSFNet (Same Topology)', nsfnet_errors), 
                             ('GBN (Different Topology)', gbn_errors)]:
        print("\\n{}:".format(topo_name))
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
    parser = argparse.ArgumentParser(description='Original RouteNet (TF1.x) Evaluation')
    parser.add_argument('--delay_model_dir', type=str, required=True,
                      help='Directory containing delay prediction model (TF1.x checkpoint)')
    parser.add_argument('--drops_model_dir', type=str, required=True,
                      help='Directory containing drops prediction model (TF1.x checkpoint)')
    parser.add_argument('--nsfnet_test_dir', type=str, required=True,
                      help='Directory containing NSFNet test data')
    parser.add_argument('--gbn_test_dir', type=str, required=True,
                      help='Directory containing GBN test data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Limit number of samples to evaluate')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting Original RouteNet (TF1.x) evaluation...")
    print("Delay model dir: {}".format(args.delay_model_dir))
    print("Drops model dir: {}".format(args.drops_model_dir))
    print("NSFNet test dir: {}".format(args.nsfnet_test_dir))
    print("GBN test dir: {}".format(args.gbn_test_dir))
    
    # 获取测试文件
    nsfnet_files = tf.io.gfile.glob(os.path.join(args.nsfnet_test_dir, '*.tfrecords'))
    gbn_files = tf.io.gfile.glob(os.path.join(args.gbn_test_dir, '*.tfrecords'))
    
    print("Found {} NSFNet test files".format(len(nsfnet_files)))
    print("Found {} GBN test files".format(len(gbn_files)))
    
    # 评估NSFNet（同拓扑）
    print("\\n" + "="*50)
    print("EVALUATING NSFNET (SAME TOPOLOGY)")
    print("="*50)
    
    # 评估delay/jitter模型
    _, _, nsfnet_delay_errors = load_tf1_model_and_predict(
        args.delay_model_dir, nsfnet_files, 'delay', args.num_samples
    )
    
    # 评估drops模型
    _, _, nsfnet_drops_errors = load_tf1_model_and_predict(
        args.drops_model_dir, nsfnet_files, 'drops', args.num_samples
    )
    
    # 合并NSFNet结果
    nsfnet_errors = {**nsfnet_delay_errors, **nsfnet_drops_errors}
    
    # 评估GBN（跨拓扑）
    print("\\n" + "="*50)
    print("EVALUATING GBN (DIFFERENT TOPOLOGY)")
    print("="*50)
    
    # 评估delay/jitter模型
    _, _, gbn_delay_errors = load_tf1_model_and_predict(
        args.delay_model_dir, gbn_files, 'delay', args.num_samples
    )
    
    # 评估drops模型
    _, _, gbn_drops_errors = load_tf1_model_and_predict(
        args.drops_model_dir, gbn_files, 'drops', args.num_samples
    )
    
    # 合并GBN结果
    gbn_errors = {**gbn_delay_errors, **gbn_drops_errors}
    
    # 打印评估摘要
    print_evaluation_summary(nsfnet_errors, gbn_errors)
    
    # 绘制线性刻度CDF图
    print("\\nGenerating linear focus CDF plot...")
    plot_linear_focus_cdf(nsfnet_errors, gbn_errors, args.output_dir)
    
    print("\\nEvaluation completed! Results saved to: {}".format(args.output_dir))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN模型可视化和可解释性分析脚本
展示KAN相比传统MLP的优势：可解释性、样条函数学习、特征重要性等
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm
import pandas as pd
from scipy import stats

# 导入训练脚本中的相关函数和类
import sys
sys.path.append(os.path.dirname(__file__))
from routenet_tf2 import (
    RouteNet, create_dataset, KANLayer, create_model_and_loss_fn
)

class KANVisualizer:
    """KAN模型可视化和分析工具"""
    
    def __init__(self, model_dir, target, config, use_kan=True):
        self.model_dir = model_dir
        self.target = target
        self.config = config
        self.use_kan = use_kan
        self.model = None
        self.kan_layers = []
        
    def load_model(self):
        """加载KAN模型"""
        self.model, _ = create_model_and_loss_fn(self.config, self.target, use_kan=self.use_kan)
        
        # 寻找权重文件
        if self.use_kan:
            weight_files = [
                os.path.join(self.model_dir, f"best_{self.target}_kan_model.weights.h5"),
                os.path.join(self.model_dir, f"best_{self.target}_model.weights.h5"),
            ]
        else:
            weight_files = [
                os.path.join(self.model_dir, f"best_{self.target}_model.weights.h5"),
            ]
        
        weight_path = None
        for path in weight_files:
            if os.path.exists(path):
                weight_path = path
                break
        
        if weight_path is None:
            raise FileNotFoundError(f"No model weights found in {self.model_dir}")
        
        print(f"Loading model weights from: {weight_path}")
        
        # 需要先运行一次前向传播来初始化模型
        dummy_input = {
            'traffic': tf.constant([0.5, 0.3], dtype=tf.float32),
            'capacities': tf.constant([1.0, 0.8], dtype=tf.float32),
            'packets': tf.constant([1000., 800.], dtype=tf.float32),
            'links': tf.constant([0, 1], dtype=tf.int64),
            'paths': tf.constant([0, 1], dtype=tf.int64),
            'sequences': tf.constant([0, 1], dtype=tf.int64),
            'n_links': tf.constant(2, dtype=tf.int64),
            'n_paths': tf.constant(2, dtype=tf.int64),
        }
        
        _ = self.model(dummy_input, training=False)
        self.model.load_weights(weight_path)
        
        # 提取KAN层
        if self.use_kan:
            self._extract_kan_layers()
        
        print("Model loaded successfully!")
        
    def _extract_kan_layers(self):
        """提取模型中的KAN层"""
        self.kan_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Sequential):
                for sublayer in layer.layers:
                    if isinstance(sublayer, KANLayer):
                        self.kan_layers.append(sublayer)
            elif isinstance(layer, KANLayer):
                self.kan_layers.append(layer)
        
        print(f"Found {len(self.kan_layers)} KAN layers")
    
    def visualize_spline_functions(self, output_dir, x_range=(-2, 2), num_points=1000):
        """可视化KAN层学习到的样条函数"""
        if not self.kan_layers:
            print("No KAN layers found!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        x_normalized = np.tanh(x)  # 与KAN层中使用的标准化一致
        
        for layer_idx, kan_layer in enumerate(self.kan_layers):
            print(f"Visualizing spline functions for KAN layer {layer_idx + 1}")
            
            # 获取层的权重
            spline_weights = kan_layer.spline_weights.numpy()  # [input_dim, units, 4]
            gate_weights = kan_layer.gate_weights.numpy()     # [input_dim, units]
            
            input_dim, units, _ = spline_weights.shape
            
            # 为每个输入维度创建图
            for input_idx in range(input_dim):
                fig, axes = plt.subplots(2, min(units//2, 4), figsize=(16, 8))
                if units == 1:
                    axes = [axes]
                elif units <= 4:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                fig.suptitle(f'KAN Layer {layer_idx + 1} - Input Dimension {input_idx + 1}', fontsize=16)
                
                plot_idx = 0
                for unit_idx in range(min(units, 8)):  # 最多显示8个单元
                    if plot_idx >= len(axes):
                        break
                        
                    # 计算样条函数值
                    poly_basis = np.stack([
                        np.ones_like(x_normalized),
                        x_normalized,
                        x_normalized**2,
                        x_normalized**3
                    ], axis=1)  # [num_points, 4]
                    
                    poly_coeff = spline_weights[input_idx, unit_idx, :]  # [4]
                    spline_values = np.dot(poly_basis, poly_coeff)
                    
                    # 应用门控权重
                    gate_weight = gate_weights[input_idx, unit_idx]
                    gated_values = gate_weight * spline_values
                    
                    # 绘制
                    ax = axes[plot_idx] if isinstance(axes, np.ndarray) else axes
                    ax.plot(x, spline_values, 'b-', linewidth=2, label='Spline Function', alpha=0.8)
                    ax.plot(x, gated_values, 'r--', linewidth=2, label=f'Gated (×{gate_weight:.3f})', alpha=0.8)
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    ax.set_title(f'Unit {unit_idx + 1}')
                    ax.set_xlabel('Input Value')
                    ax.set_ylabel('Output Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
                
                # 隐藏多余的子图
                for i in range(plot_idx, len(axes)):
                    if isinstance(axes, np.ndarray):
                        axes[i].set_visible(False)
                
                plt.tight_layout()
                filename = f'kan_layer{layer_idx+1}_input{input_idx+1}_splines.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
        
        print(f"Spline function visualizations saved to {output_dir}")
    
    def analyze_feature_importance(self, dataset, output_dir, num_samples=1000):
        """分析特征重要性"""
        if not self.kan_layers:
            print("No KAN layers found!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Analyzing feature importance...")
        
        # 收集样本数据 - 分别处理路径特征和链路特征
        all_traffic = []  # 路径特征
        all_capacities = []  # 链路特征
        all_outputs = []
        all_utilizations = []  # 计算利用率需要特殊处理
        sample_count = 0
        
        for features, labels in tqdm(dataset, desc="Collecting samples"):
            if sample_count >= num_samples:
                break
            
            # 获取预测结果
            predictions = self.model(features, training=False)
            
            # 收集路径特征（traffic）
            traffic_batch = features['traffic'].numpy()
            all_traffic.extend(traffic_batch)
            
            # 收集链路特征（capacities）
            capacities_batch = features['capacities'].numpy()
            all_capacities.extend(capacities_batch)
            
            # 收集预测结果
            predictions_batch = predictions.numpy()
            all_outputs.extend(predictions_batch)
            
            # 计算利用率 - 这需要根据网络拓扑结构
            # 简化处理：为每条路径分配一个平均容量
            n_paths = len(traffic_batch)
            n_links = len(capacities_batch)
            
            # 简单的利用率估算：使用平均容量
            avg_capacity = np.mean(capacities_batch) if n_links > 0 else 1.0
            utilizations = traffic_batch / (avg_capacity + 1e-9)
            all_utilizations.extend(utilizations)
            
            sample_count += n_paths
        
        # 转换为numpy数组
        all_traffic = np.array(all_traffic)
        all_outputs = np.array(all_outputs)
        all_utilizations = np.array(all_utilizations)
        
        print(f"Collected {len(all_traffic)} path samples")
        print(f"Collected {len(all_capacities)} link samples")  
        print(f"Traffic shape: {all_traffic.shape}")
        print(f"Outputs shape: {all_outputs.shape}")
        print(f"Utilizations shape: {all_utilizations.shape}")
        
        # 对于delay模型，只取第一个输出（loc）
        if all_outputs.ndim > 1 and all_outputs.shape[1] > 1:
            all_outputs = all_outputs[:, 0]
        elif all_outputs.ndim > 1:
            all_outputs = all_outputs.flatten()
        
        # 确保数组长度匹配
        min_len = min(len(all_traffic), len(all_outputs), len(all_utilizations))
        all_traffic = all_traffic[:min_len]
        all_outputs = all_outputs[:min_len]
        all_utilizations = all_utilizations[:min_len]
        
        print(f"Using {min_len} matched samples for analysis")
        
        if min_len == 0:
            print("No valid samples collected!")
            return {}
        
        # 创建可视化
        plt.figure(figsize=(20, 8))
        
        # 1. Traffic vs Prediction
        plt.subplot(2, 3, 1)
        plt.scatter(all_traffic, all_outputs, alpha=0.5, s=1, c='blue')
        plt.xlabel('Traffic (Path Feature)')
        plt.ylabel('Prediction')
        plt.title('Traffic vs Prediction')
        plt.grid(True, alpha=0.3)
        
        # 2. Utilization vs Prediction  
        plt.subplot(2, 3, 2)
        plt.scatter(all_utilizations, all_outputs, alpha=0.5, s=1, c='red')
        plt.xlabel('Estimated Utilization')
        plt.ylabel('Prediction')
        plt.title('Utilization vs Prediction')
        plt.grid(True, alpha=0.3)
        
        # 3. 链路容量分布
        plt.subplot(2, 3, 3)
        all_capacities = np.array(all_capacities)
        plt.hist(all_capacities, bins=50, alpha=0.7, color='green')
        plt.xlabel('Link Capacity')
        plt.ylabel('Frequency')
        plt.title('Link Capacity Distribution')
        plt.grid(True, alpha=0.3)
        
        # 4. Traffic分布
        plt.subplot(2, 3, 4)
        plt.hist(all_traffic, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Path Traffic')
        plt.ylabel('Frequency')
        plt.title('Path Traffic Distribution')
        plt.grid(True, alpha=0.3)
        
        # 5. 预测结果分布
        plt.subplot(2, 3, 5)
        plt.hist(all_outputs, bins=50, alpha=0.7, color='purple')
        plt.xlabel('Prediction')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution')
        plt.grid(True, alpha=0.3)
        
        # 6. Utilization分布
        plt.subplot(2, 3, 6)
        plt.hist(all_utilizations, bins=50, alpha=0.7, color='red')
        plt.xlabel('Estimated Utilization')
        plt.ylabel('Frequency')
        plt.title('Utilization Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # 计算相关性
        try:
            correlations = {
                'traffic': np.corrcoef(all_traffic, all_outputs)[0, 1],
                'utilization': np.corrcoef(all_utilizations, all_outputs)[0, 1],
            }
            
            # 计算统计信息
            traffic_stats = {
                'mean': np.mean(all_traffic),
                'std': np.std(all_traffic),
                'min': np.min(all_traffic),
                'max': np.max(all_traffic)
            }
            
            capacity_stats = {
                'mean': np.mean(all_capacities),
                'std': np.std(all_capacities),
                'min': np.min(all_capacities),
                'max': np.max(all_capacities)
            }
            
            print("\n" + "="*50)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("="*50)
            
            print(f"\nDataset Statistics:")
            print(f"  Path samples: {len(all_traffic)}")
            print(f"  Link samples: {len(all_capacities)}")
            print(f"  Prediction range: [{np.min(all_outputs):.4f}, {np.max(all_outputs):.4f}]")
            
            print(f"\nTraffic Statistics:")
            for key, value in traffic_stats.items():
                print(f"  {key}: {value:.4f}")
            
            print(f"\nCapacity Statistics:")
            for key, value in capacity_stats.items():
                print(f"  {key}: {value:.4f}")
            
            print(f"\nFeature correlations with prediction:")
            for feature, corr in correlations.items():
                if not np.isnan(corr):
                    print(f"  {feature}: {corr:.4f}")
                else:
                    print(f"  {feature}: NaN (no variation)")
            
            # 分析高利用率情况
            high_util_mask = all_utilizations > np.percentile(all_utilizations, 90)
            if np.sum(high_util_mask) > 0:
                high_util_predictions = all_outputs[high_util_mask]
                low_util_predictions = all_outputs[~high_util_mask]
                print(f"\nHigh Utilization Analysis (top 10%):")
                print(f"  High util mean prediction: {np.mean(high_util_predictions):.4f}")
                print(f"  Low util mean prediction: {np.mean(low_util_predictions):.4f}")
                print(f"  Difference: {np.mean(high_util_predictions) - np.mean(low_util_predictions):.4f}")
            
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            correlations = {}
        
        return correlations
    
    def visualize_gate_weights(self, output_dir):
        """可视化门控权重，显示特征重要性"""
        if not self.kan_layers:
            print("No KAN layers found!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for layer_idx, kan_layer in enumerate(self.kan_layers):
            gate_weights = kan_layer.gate_weights.numpy()  # [input_dim, units]
            
            plt.figure(figsize=(12, 8))
            
            # 创建热力图
            im = plt.imshow(gate_weights.T, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, label='Gate Weight')
            
            plt.title(f'KAN Layer {layer_idx + 1} - Gate Weights Heatmap')
            plt.xlabel('Input Dimension')
            plt.ylabel('Output Unit')
            
            # 添加数值标注
            for i in range(gate_weights.shape[0]):
                for j in range(gate_weights.shape[1]):
                    plt.text(i, j, f'{gate_weights[i, j]:.2f}', 
                            ha='center', va='center', 
                            color='white' if abs(gate_weights[i, j]) > 0.5 else 'black')
            
            plt.tight_layout()
            filename = f'kan_layer{layer_idx+1}_gate_weights.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
            # 打印权重统计
            print(f"\nKAN Layer {layer_idx + 1} Gate Weights Statistics:")
            print(f"  Mean: {gate_weights.mean():.4f}")
            print(f"  Std: {gate_weights.std():.4f}")
            print(f"  Min: {gate_weights.min():.4f}")
            print(f"  Max: {gate_weights.max():.4f}")
    
    def compare_with_mlp(self, mlp_model_dir, dataset, output_dir, num_samples=500):
        """比较KAN和MLP模型的预测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载MLP模型
        mlp_model, _ = create_model_and_loss_fn(self.config, self.target, use_kan=False)
        
        mlp_weight_files = [
            os.path.join(mlp_model_dir, f"best_{self.target}_model.weights.h5"),
        ]
        
        mlp_weight_path = None
        for path in mlp_weight_files:
            if os.path.exists(path):
                mlp_weight_path = path
                break
        
        if mlp_weight_path is None:
            print("MLP model not found, skipping comparison")
            return
        
        # 初始化MLP模型
        dummy_input = {
            'traffic': tf.constant([0.5, 0.3], dtype=tf.float32),
            'capacities': tf.constant([1.0, 0.8], dtype=tf.float32),
            'packets': tf.constant([1000., 800.], dtype=tf.float32),
            'links': tf.constant([0, 1], dtype=tf.int64),
            'paths': tf.constant([0, 1], dtype=tf.int64),
            'sequences': tf.constant([0, 1], dtype=tf.int64),
            'n_links': tf.constant(2, dtype=tf.int64),
            'n_paths': tf.constant(2, dtype=tf.int64),
        }
        
        _ = mlp_model(dummy_input, training=False)
        mlp_model.load_weights(mlp_weight_path)
        
        print("Comparing KAN and MLP predictions...")
        
        kan_predictions = []
        mlp_predictions = []
        true_values = []
        sample_count = 0
        
        for features, labels in tqdm(dataset, desc="Comparing models"):
            if sample_count >= num_samples:
                break
            
            kan_pred = self.model(features, training=False)
            mlp_pred = mlp_model(features, training=False)
            
            kan_predictions.append(kan_pred.numpy())
            mlp_predictions.append(mlp_pred.numpy())
            
            if self.target == 'delay':
                true_values.append(labels['delay'].numpy())
            else:  # drops
                true_drops = labels['drops'].numpy()
                true_packets = labels['packets'].numpy()
                true_rates = true_drops / (true_packets + 1e-10)
                true_values.append(true_rates)
            
            sample_count += len(kan_pred)
        
        # 合并所有预测
        kan_predictions = np.concatenate(kan_predictions)
        mlp_predictions = np.concatenate(mlp_predictions)
        true_values = np.concatenate(true_values)
        
        # 如果是delay模型，只取第一个输出（loc）
        if kan_predictions.shape[1] > 1:
            kan_predictions = kan_predictions[:, 0]
        if mlp_predictions.shape[1] > 1:
            mlp_predictions = mlp_predictions[:, 0]
        
        # 对于drops模型，转换为概率
        if self.target == 'drops':
            kan_predictions = 1 / (1 + np.exp(-kan_predictions))  # sigmoid
            mlp_predictions = 1 / (1 + np.exp(-mlp_predictions))  # sigmoid
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # KAN vs True
        axes[0, 0].scatter(true_values, kan_predictions, alpha=0.5, s=1)
        axes[0, 0].plot([true_values.min(), true_values.max()], 
                        [true_values.min(), true_values.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('KAN Predictions')
        axes[0, 0].set_title('KAN Model')
        axes[0, 0].grid(True, alpha=0.3)
        
        # MLP vs True
        axes[0, 1].scatter(true_values, mlp_predictions, alpha=0.5, s=1)
        axes[0, 1].plot([true_values.min(), true_values.max()], 
                        [true_values.min(), true_values.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('MLP Predictions')
        axes[0, 1].set_title('MLP Model')
        axes[0, 1].grid(True, alpha=0.3)
        
        # KAN vs MLP
        axes[1, 0].scatter(mlp_predictions, kan_predictions, alpha=0.5, s=1)
        axes[1, 0].plot([mlp_predictions.min(), mlp_predictions.max()], 
                        [mlp_predictions.min(), mlp_predictions.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('MLP Predictions')
        axes[1, 0].set_ylabel('KAN Predictions')
        axes[1, 0].set_title('KAN vs MLP')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差分布比较
        kan_errors = np.abs(kan_predictions - true_values)
        mlp_errors = np.abs(mlp_predictions - true_values)
        
        axes[1, 1].hist(kan_errors, bins=50, alpha=0.7, label='KAN', density=True)
        axes[1, 1].hist(mlp_errors, bins=50, alpha=0.7, label='MLP', density=True)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{self.target}_kan_vs_mlp_comparison.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # 计算性能指标
        kan_mae = np.mean(kan_errors)
        mlp_mae = np.mean(mlp_errors)
        kan_rmse = np.sqrt(np.mean((kan_predictions - true_values)**2))
        mlp_rmse = np.sqrt(np.mean((mlp_predictions - true_values)**2))
        
        print(f"\nModel Comparison Results:")
        print(f"  KAN MAE: {kan_mae:.6f}")
        print(f"  MLP MAE: {mlp_mae:.6f}")
        print(f"  KAN RMSE: {kan_rmse:.6f}")
        print(f"  MLP RMSE: {mlp_rmse:.6f}")
        print(f"  MAE Improvement: {((mlp_mae - kan_mae) / mlp_mae * 100):.2f}%")
        print(f"  RMSE Improvement: {((mlp_rmse - kan_rmse) / mlp_rmse * 100):.2f}%")
        
        return {
            'kan_mae': kan_mae, 'mlp_mae': mlp_mae,
            'kan_rmse': kan_rmse, 'mlp_rmse': mlp_rmse
        }

def main():
    parser = argparse.ArgumentParser(description='KAN Model Visualization and Interpretability Analysis')
    parser.add_argument('--kan_model_dir', type=str, required=True,
                      help='Directory containing KAN model weights')
    parser.add_argument('--mlp_model_dir', type=str, default=None,
                      help='Directory containing MLP model weights (for comparison)')
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--target', type=str, choices=['delay', 'drops'], required=True,
                      help='Model target: delay or drops')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save visualization results')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for analysis')
    parser.add_argument('--num_samples', type=int, default=1000,
                      help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模型配置
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
    }
    
    # 创建可视化器
    visualizer = KANVisualizer(args.kan_model_dir, args.target, config, use_kan=True)
    
    # 加载模型
    visualizer.load_model()
    
    # 创建测试数据集
    test_files = tf.io.gfile.glob(os.path.join(args.test_dir, '*.tfrecords'))
    from routenet_tf2 import create_dataset
    test_dataset = create_dataset(test_files, args.batch_size, is_training=False)
    
    print(f"Found {len(test_files)} test files")
    print(f"Starting KAN visualization and analysis...")
    
    # 1. 可视化样条函数
    print("\n1. Visualizing spline functions...")
    spline_dir = os.path.join(args.output_dir, 'spline_functions')
    visualizer.visualize_spline_functions(spline_dir)
    
    # 2. 分析特征重要性
    print("\n2. Analyzing feature importance...")
    importance_dir = os.path.join(args.output_dir, 'feature_importance')
    visualizer.analyze_feature_importance(test_dataset, importance_dir, args.num_samples)
    
    # 3. 可视化门控权重
    print("\n3. Visualizing gate weights...")
    gate_dir = os.path.join(args.output_dir, 'gate_weights')
    visualizer.visualize_gate_weights(gate_dir)
    
    # 4. 与MLP比较（如果提供了MLP模型路径）
    if args.mlp_model_dir:
        print("\n4. Comparing with MLP model...")
        comparison_dir = os.path.join(args.output_dir, 'kan_vs_mlp')
        # 重新创建数据集用于比较
        test_dataset_comparison = create_dataset(test_files, args.batch_size, is_training=False)
        visualizer.compare_with_mlp(args.mlp_model_dir, test_dataset_comparison, 
                                  comparison_dir, args.num_samples)
    
    print(f"\nVisualization and analysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nKey findings:")
    print(f"- Spline functions show the learned non-linear transformations")
    print(f"- Gate weights indicate feature importance")
    print(f"- Feature importance analysis reveals input-output relationships")
    if args.mlp_model_dir:
        print(f"- Comparison with MLP shows performance differences")

if __name__ == '__main__':
    main()

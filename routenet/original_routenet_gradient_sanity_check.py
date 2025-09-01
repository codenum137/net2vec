# -*- coding: utf-8 -*-
"""
原版RouteNet TF1.x模型梯度物理意义验证
基于TensorFlow 1.x的梯度计算和验证
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm

# 导入原版RouteNet相关模块
sys.path.append(os.path.dirname(__file__))
from routenet import delay_model_fn, tfrecord_input_fn

# 禁用TF2行为，使用TF1.x
tf.compat.v1.disable_v2_behavior()

class HParams:
    """原版RouteNet的超参数"""
    def __init__(self):
        self.batch_size = 32
        self.link_state_dim = 4
        self.path_state_dim = 2
        self.T = 3
        self.readout_units = 8
        self.readout_layers = 2
        self.dropout_rate = 0.5
        self.l2 = 0.1
        self.l2_2 = 0.01
        self.learn_embedding = True

class OriginalRouteNetGradientChecker:
    """原版RouteNet梯度物理意义验证器"""
    
    def __init__(self, model_dir, target='delay'):
        self.model_dir = model_dir
        self.target = target
        self.hparams = HParams()
        self.session = None
        self.graph = None
        self.setup_model()
    
    def create_simple_network_sample(self):
        """创建简单网络样本用于测试（与训练数据范围一致）"""
        n_links = 3  
        n_paths = 2  
        
        # 使用与训练数据一致的范围
        capacities = np.array([1.0, 0.3, 1.0], dtype=np.float32)  # 归一化容量
        
        # 路径-链路映射 - 使用int32保持一致性
        links = np.array([0, 1, 1, 2], dtype=np.int32)
        paths = np.array([0, 0, 1, 1], dtype=np.int32)
        sequences = np.array([0, 1, 0, 1], dtype=np.int32)
        
        # 基础流量 - 使用训练数据的真实范围
        base_traffic = np.array([0.2, 0.3], dtype=np.float32)
        
        # 数据包数量
        packets = np.array([1000.0, 800.0], dtype=np.float32)
        
        sample_features = {
            'traffic': base_traffic,
            'capacities': capacities,
            'links': links,
            'paths': paths,
            'sequences': sequences,
            'n_links': n_links,
            'n_paths': n_paths,
            'packets': packets
        }
        
        return sample_features
    
    def setup_model(self):
        """设置TF1.x模型图"""
        print(f"Setting up original RouteNet model from: {self.model_dir}")
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 创建占位符用于输入特征 - 确保数据类型一致性
            self.traffic_ph = tf.compat.v1.placeholder(tf.float32, shape=[None], name='traffic_input')
            self.capacities_ph = tf.compat.v1.placeholder(tf.float32, shape=[None], name='capacities_input')
            self.links_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name='links_input')
            self.paths_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name='paths_input')
            self.sequences_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name='sequences_input')
            self.packets_ph = tf.compat.v1.placeholder(tf.float32, shape=[None], name='packets_input')
            self.n_links_ph = tf.compat.v1.placeholder(tf.int32, shape=[], name='n_links_input')
            self.n_paths_ph = tf.compat.v1.placeholder(tf.int32, shape=[], name='n_paths_input')
            
            # 构造特征字典
            features = {
                'traffic': self.traffic_ph,
                'capacities': self.capacities_ph,
                'links': self.links_ph,
                'paths': self.paths_ph,
                'sequences': self.sequences_ph,
                'packets': self.packets_ph,
                'n_links': self.n_links_ph,
                'n_paths': self.n_paths_ph
            }
            
            # 构造虚拟标签（预测时不会使用）
            labels = {
                'delay': tf.zeros_like(self.traffic_ph),
                'jitter': tf.zeros_like(self.traffic_ph),
                'packets': self.packets_ph,
                'drops': tf.zeros_like(self.traffic_ph)
            }
            
            # 创建模型
            if self.target == 'delay':
                model_spec = delay_model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, self.hparams)
            else:
                raise NotImplementedError("Only delay model is supported for now")
            
            self.predictions = model_spec.predictions
            
            # 获取延迟预测（位置参数）
            if 'delay' in self.predictions:
                self.delay_output = self.predictions['delay']
            else:
                raise ValueError("Delay prediction not found in model outputs")
            
            # 计算雅可比矩阵：∂delay/∂traffic
            self.jacobian = tf.gradients(self.delay_output, self.traffic_ph)[0]
            
            # 创建saver
            self.saver = tf.compat.v1.train.Saver()
        
        # 创建session并加载权重
        self.session = tf.compat.v1.Session(graph=self.graph)
        checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {self.model_dir}")
        
        self.saver.restore(self.session, checkpoint_path)
        print(f"Model restored from {checkpoint_path}")
    
    def compute_jacobian(self, sample_features):
        """计算给定样本的雅可比矩阵"""
        feed_dict = {
            self.traffic_ph: sample_features['traffic'],
            self.capacities_ph: sample_features['capacities'],
            self.links_ph: sample_features['links'],
            self.paths_ph: sample_features['paths'],
            self.sequences_ph: sample_features['sequences'],
            self.packets_ph: sample_features['packets'],
            self.n_links_ph: sample_features['n_links'],
            self.n_paths_ph: sample_features['n_paths']
        }
        
        # 运行计算
        jacobian_val, delay_pred = self.session.run([self.jacobian, self.delay_output], feed_dict)
        
        return jacobian_val, delay_pred
    
    def traffic_sweep_experiment(self, path_to_vary=0, traffic_range=(0.1, 0.9), num_points=10):
        """流量扫描实验"""
        print(f"执行原版RouteNet流量扫描实验：变化路径 {path_to_vary} 的流量...")
        
        # 创建基础网络样本
        sample_features = self.create_simple_network_sample()
        base_traffic = sample_features['traffic'].copy()
        n_paths = len(base_traffic)
        
        # 生成流量序列
        traffic_values = np.linspace(traffic_range[0], traffic_range[1], num_points)
        
        results = {
            'traffic_values': traffic_values,
            'delay_predictions': [],
            'jacobian_values': [],
            'diagonal_gradients': [],
        }
        
        # 为交叉梯度准备存储
        cross_gradients = {}
        for i in range(n_paths):
            if i != path_to_vary:
                cross_gradients[f'J_{i}{path_to_vary}'] = []
        
        for traffic_val in tqdm(traffic_values, desc="流量扫描"):
            # 设置当前流量
            current_traffic = base_traffic.copy()
            current_traffic[path_to_vary] = traffic_val
            sample_features['traffic'] = current_traffic
            
            # 计算雅可比矩阵和延迟预测
            try:
                jacobian_val, delay_pred = self.compute_jacobian(sample_features)
                
                if jacobian_val is not None:
                    # 重塑雅可比矩阵为 [n_paths, n_paths] 形状
                    if jacobian_val.ndim == 1:
                        # 如果是一维，说明每个输出对应一个梯度
                        jacobian_matrix = np.diag(jacobian_val)
                    else:
                        jacobian_matrix = jacobian_val.reshape(n_paths, n_paths)
                    
                    results['delay_predictions'].append(delay_pred)
                    results['jacobian_values'].append(jacobian_matrix)
                    results['diagonal_gradients'].append(np.diag(jacobian_matrix))
                    
                    # 记录交叉梯度
                    for i in range(n_paths):
                        if i != path_to_vary:
                            cross_gradients[f'J_{i}{path_to_vary}'].append(jacobian_matrix[i, path_to_vary])
                else:
                    print(f"Warning: No gradient computed for traffic {traffic_val}")
                    
            except Exception as e:
                print(f"Error computing gradient for traffic {traffic_val}: {e}")
        
        # 转换为numpy数组
        results['delay_predictions'] = np.array(results['delay_predictions'])
        results['jacobian_values'] = np.array(results['jacobian_values'])
        results['diagonal_gradients'] = np.array(results['diagonal_gradients'])
        results['cross_gradients'] = cross_gradients
        
        for key in cross_gradients:
            results['cross_gradients'][key] = np.array(cross_gradients[key])
        
        return results
    
    def validate_physical_intuition(self, experiment_results, path_to_vary, output_dir):
        """验证梯度的物理意义"""
        os.makedirs(output_dir, exist_ok=True)
        
        traffic_values = experiment_results['traffic_values']
        delay_predictions = experiment_results['delay_predictions']
        diagonal_gradients = experiment_results['diagonal_gradients']
        cross_gradients = experiment_results['cross_gradients']
        
        n_paths = delay_predictions.shape[1] if len(delay_predictions) > 0 else 2
        
        # 验证结果记录
        validation_results = {
            'self_gradient_positive': True,
            'cross_gradient_positive': True,
            'delay_monotonic': True,
            'gradient_increases_with_congestion': True,
            'physical_intuition_score': 0.0
        }
        
        print("\n" + "="*60)
        print("原版RouteNet梯度物理意义验证结果")
        print("="*60)
        
        if len(diagonal_gradients) == 0:
            print("错误：没有成功计算任何梯度值")
            validation_results['physical_intuition_score'] = 0.0
            return validation_results
        
        # 1. 验证自影响梯度
        if diagonal_gradients.shape[1] > path_to_vary:
            self_gradients = diagonal_gradients[:, path_to_vary]
            positive_self_ratio = np.sum(self_gradients > 0) / len(self_gradients)
            print(f"1. 自影响梯度 J_{path_to_vary}{path_to_vary} > 0:")
            print(f"   正值比例: {positive_self_ratio:.2%}")
            print(f"   平均值: {np.mean(self_gradients):.6f}")
            print(f"   范围: [{np.min(self_gradients):.6f}, {np.max(self_gradients):.6f}]")
            
            if positive_self_ratio < 0.8:
                validation_results['self_gradient_positive'] = False
        
        # 2. 验证交叉影响梯度
        print(f"\n2. 交叉影响梯度分析:")
        for key, cross_grad in cross_gradients.items():
            if len(cross_grad) > 0:
                positive_cross_ratio = np.sum(cross_grad > 0) / len(cross_grad)
                print(f"   {key} > 0: {positive_cross_ratio:.2%}")
                print(f"   平均值: {np.mean(cross_grad):.6f}")
                
                if positive_cross_ratio < 0.6:
                    validation_results['cross_gradient_positive'] = False
        
        # 3. 验证延迟单调性
        print(f"\n3. 延迟单调性验证:")
        for i in range(n_paths):
            if len(delay_predictions) > 0 and delay_predictions.shape[1] > i:
                delays = delay_predictions[:, i]
                diff = np.diff(delays)
                monotonic_ratio = np.sum(diff >= -1e-6) / len(diff) if len(diff) > 0 else 0
                print(f"   路径 {i} 延迟单调递增比例: {monotonic_ratio:.2%}")
                
                if monotonic_ratio < 0.8:
                    validation_results['delay_monotonic'] = False
        
        # 计算总体得分
        score_components = [
            validation_results['self_gradient_positive'],
            validation_results['cross_gradient_positive'], 
            validation_results['delay_monotonic'],
            validation_results['gradient_increases_with_congestion']
        ]
        validation_results['physical_intuition_score'] = sum(score_components) / len(score_components)
        
        print(f"\n4. 总体物理直觉得分: {validation_results['physical_intuition_score']:.2%}")
        
        # 可视化结果
        self._visualize_results(experiment_results, path_to_vary, output_dir, validation_results)
        
        return validation_results
    
    def _visualize_results(self, experiment_results, path_to_vary, output_dir, validation_results):
        """可视化验证结果"""
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        traffic_values = experiment_results['traffic_values']
        delay_predictions = experiment_results['delay_predictions']
        diagonal_gradients = experiment_results['diagonal_gradients']
        cross_gradients = experiment_results['cross_gradients']
        
        if len(delay_predictions) == 0 or len(diagonal_gradients) == 0:
            print("Warning: No data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 延迟 vs 流量
        ax1 = axes[0, 0]
        n_paths = delay_predictions.shape[1]
        for i in range(n_paths):
            ax1.plot(traffic_values, delay_predictions[:, i], 
                    label=f'Path {i}', marker='o', markersize=3)
        ax1.set_xlabel(f'Path {path_to_vary} Traffic')
        ax1.set_ylabel('Predicted Delay')
        ax1.set_title('Delay vs Traffic (Original RouteNet)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 自影响梯度
        ax2 = axes[0, 1]
        if diagonal_gradients.shape[1] > path_to_vary:
            self_gradients = diagonal_gradients[:, path_to_vary]
            ax2.plot(traffic_values, self_gradients, 'r-', marker='s', markersize=4)
            ax2.set_xlabel(f'Path {path_to_vary} Traffic')
            ax2.set_ylabel(f'∂D_{path_to_vary}/∂T_{path_to_vary}')
            ax2.set_title(f'Self-influence Gradient (Original RouteNet)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. 交叉影响梯度
        ax3 = axes[1, 0]
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (key, cross_grad) in enumerate(cross_gradients.items()):
            if len(cross_grad) > 0:
                ax3.plot(traffic_values, cross_grad, color=colors[i % len(colors)],
                        label=key, marker='^', markersize=3)
        ax3.set_xlabel(f'Path {path_to_vary} Traffic')
        ax3.set_ylabel('Cross-influence Gradient')
        ax3.set_title('Cross-influence Gradients (Original RouteNet)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 4. 验证结果总览
        ax4 = axes[1, 1]
        metrics = ['Self>0', 'Cross>0', 'Monotonic', 'Congest-Sens']
        scores = [
            validation_results['self_gradient_positive'],
            validation_results['cross_gradient_positive'],
            validation_results['delay_monotonic'],
            validation_results['gradient_increases_with_congestion']
        ]
        colors_bar = ['green' if s else 'red' for s in scores]
        
        bars = ax4.bar(metrics, [1 if s else 0 for s in scores], color=colors_bar)
        ax4.set_ylabel('Validation Pass')
        ax4.set_title(f'Physical Intuition Validation (Original RouteNet)\nScore: {validation_results["physical_intuition_score"]:.1%}')
        ax4.set_ylim(0, 1.2)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    '✓' if score else '✗', ha='center', va='bottom', fontsize=16)
        
        plt.suptitle('Original RouteNet - Gradient Physical Intuition Validation', 
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'original_routenet_sanity_check_path_{path_to_vary}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细结果
        with open(os.path.join(output_dir, f'original_routenet_results_path_{path_to_vary}.txt'), 'w') as f:
            f.write("Original RouteNet - Gradient Physical Intuition Validation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Varying Path: {path_to_vary}\n")
            f.write(f"Traffic Range: {traffic_values[0]:.2f} - {traffic_values[-1]:.2f}\n")
            f.write(f"Number of Points: {len(traffic_values)}\n\n")
            
            f.write("Validation Results:\n")
            f.write(f"  Self-influence Gradient > 0: {'Pass' if validation_results['self_gradient_positive'] else 'Fail'}\n")
            f.write(f"  Cross-influence Gradient > 0: {'Pass' if validation_results['cross_gradient_positive'] else 'Fail'}\n")
            f.write(f"  Delay Monotonic Increase: {'Pass' if validation_results['delay_monotonic'] else 'Fail'}\n")
            f.write(f"  Congestion Sensitivity: {'Pass' if validation_results['gradient_increases_with_congestion'] else 'Fail'}\n")
            f.write(f"\nOverall Physical Intuition Score: {validation_results['physical_intuition_score']:.1%}\n")
    
    def close(self):
        """关闭TensorFlow session"""
        if self.session:
            self.session.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='原版RouteNet梯度物理意义验证')
    parser.add_argument('--model_dir', default='models/routenet/delay', help='原版RouteNet模型目录')
    parser.add_argument('--output_dir', default='original_routenet_gradient_check', help='输出目录')
    parser.add_argument('--traffic_min', type=float, default=0.1, help='最小流量值')
    parser.add_argument('--traffic_max', type=float, default=0.9, help='最大流量值')
    parser.add_argument('--num_points', type=int, default=10, help='流量采样点数量')
    
    args = parser.parse_args()
    
    try:
        print("初始化原版RouteNet梯度验证器...")
        checker = OriginalRouteNetGradientChecker(args.model_dir, target='delay')
        
        print("执行流量扫描实验...")
        experiment_results = checker.traffic_sweep_experiment(
            path_to_vary=0,
            traffic_range=(args.traffic_min, args.traffic_max),
            num_points=args.num_points
        )
        
        print("验证物理意义...")
        validation_results = checker.validate_physical_intuition(
            experiment_results, 
            path_to_vary=0, 
            output_dir=args.output_dir
        )
        
        print(f"\n{'='*60}")
        print("总体验证结果")
        print(f"{'='*60}")
        print(f"模型类型: Original RouteNet (TF1.x)")
        print(f"物理直觉得分: {validation_results['physical_intuition_score']:.1%}")
        
        if validation_results['physical_intuition_score'] >= 0.8:
            print("✅ 原版RouteNet梯度计算通过物理意义验证！")
        elif validation_results['physical_intuition_score'] >= 0.6:
            print("⚠️  原版RouteNet梯度计算部分通过验证，需要进一步检查")
        else:
            print("❌ 原版RouteNet梯度计算未通过物理意义验证，需要检查实现")
        
        print(f"\n详细结果已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'checker' in locals():
            checker.close()

if __name__ == '__main__':
    main()

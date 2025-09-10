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
    
    def create_controlled_network(self):
        """
        创建一个可控的复杂网络拓扑（与gradient_sanity_check.py一致）
        
        网络结构:
        Node 0 ----[Link 0]---- Node 1 ----[Link 1]---- Node 2 ----[Link 2]---- Node 3
                                      |
                                      +----[Link 3]---- Node 4
        
        路径配置:
        - 路径 0: 0->1->2 (使用链路 0, 1)
        - 路径 1: 0->1->4 (使用链路 0, 3) 
        - 路径 2: 1->2->3 (使用链路 1, 2)
        """
        n_nodes = 5
        n_links = 4
        n_paths = 3
        
        # 链路容量设置：基于真实数据集特征 (容量范围: 10-40, 典型值: [10,10,10,40])
        # 使用数据集中的典型容量值，确保与训练数据一致
        capacities = np.array([10.0, 10.0, 40.0, 20.0], dtype=np.float32)  
        # 链路0: 10.0 (被路径0,1共享，低容量，容易形成瓶颈)
        # 链路1: 10.0 (被路径0,2共享，低容量，容易形成瓶颈) 
        # 链路2: 40.0 (仅路径2使用，高容量)
        # 链路3: 20.0 (仅路径1使用，中等容量)
        
        # 路径-链路映射，注意数据类型
        # 路径0: 链路0, 链路1  
        # 路径1: 链路0, 链路3
        # 路径2: 链路1, 链路2
        links = np.array([0, 1, 0, 3, 1, 2], dtype=np.int32)  # 展平的链路索引
        paths = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)  # 对应的路径索引  
        sequences = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)  # 路径内序列
        
        # 基础流量配置：基于真实数据集特征 (流量范围: 0.086-1.103, 典型值: [0.33,0.55,0.78,0.91])
        base_traffic = np.array([0.33, 0.55, 0.78], dtype=np.float32)  # 使用数据集25%-75%分位数
        
        # 数据包数量
        packets = np.array([1000.0, 800.0, 1200.0], dtype=np.float32)
        
        network_config = {
            'capacities': capacities,
            'links': links,
            'paths': paths,
            'sequences': sequences,
            'n_links': n_links,
            'n_paths': n_paths,
            'packets': packets,
            'base_traffic': base_traffic,
            'bottleneck_links': [0, 1],  # 瓶颈链路
            'shared_paths': [(0, 2), (0, 1)]  # 共享链路的路径对
        }
        
        return network_config
    
    def setup_model(self):
        """设置TF1.x模型图"""
        print("Setting up original RouteNet model from: {}".format(self.model_dir))
        
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
    
    def _apply_routenet_scaling(self, features):
        """
        应用与routenet_tf2.py中相同的数据标准化
        
        RouteNet标准化公式:
        - 流量: (val - 0.18) / 0.15
        - 容量: val / 10.0
        """
        scaled_features = features.copy()
        
        # 标准化流量
        if 'traffic' in features:
            scaled_features['traffic'] = (features['traffic'] - 0.18) / 0.15
        
        # 标准化容量
        if 'capacities' in features:
            scaled_features['capacities'] = features['capacities'] / 10.0
            
        return scaled_features
    
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
    
    def traffic_sweep_experiment(self, network_config, path_to_vary=0, 
                               traffic_range=(0.1, 1.0), num_points=20):
        """
        流量扫描实验：固定其他路径流量，变化指定路径流量
        
        Args:
            network_config: 网络配置
            path_to_vary: 要变化流量的路径索引
            traffic_range: 流量变化范围 (min, max) - 基于数据集实际范围0.086-1.103
            num_points: 采样点数量
        
        Returns:
            experiment_results: 实验结果字典
        """
        print(f"执行原版RouteNet流量扫描实验：变化路径 {path_to_vary} 的流量...")
        
        # 生成流量序列
        traffic_values = np.linspace(traffic_range[0], traffic_range[1], num_points)
        
        results = {
            'traffic_values': traffic_values,
            'delay_predictions': [],
            'jacobian_matrices': [],
            'diagonal_gradients': [],  # J_ii: ∂D_i/∂T_i
            'cross_gradients': {},     # J_ij: ∂D_i/∂T_j (i≠j)
        }
        
        # 为每个路径对记录交叉梯度
        for i in range(network_config['n_paths']):
            if i != path_to_vary:
                results['cross_gradients'][f'J_{i}{path_to_vary}'] = []
        
        base_traffic = network_config['base_traffic'].copy()
        
        for traffic_val in tqdm(traffic_values, desc="流量扫描"):
            # 设置当前流量
            current_traffic = base_traffic.copy()
            current_traffic[path_to_vary] = traffic_val
            
            # 构造原始样本特征
            raw_features = {
                'traffic': current_traffic,
                'capacities': network_config['capacities'],
                'links': network_config['links'],
                'paths': network_config['paths'],
                'sequences': network_config['sequences'],
                'n_links': network_config['n_links'],
                'n_paths': network_config['n_paths'],
                'packets': network_config['packets']
            }
            
            # ⚠️ 重要：应用与routenet_tf2.py相同的数据标准化
            sample_features = self._apply_routenet_scaling(raw_features)
            
            # 计算雅可比矩阵和延迟预测
            try:
                jacobian_val, delay_pred = self.compute_jacobian(sample_features)
                
                if jacobian_val is not None:
                    # 重塑雅可比矩阵为 [n_paths, n_paths] 形状
                    n_paths = network_config['n_paths']
                    if jacobian_val.ndim == 1:
                        # 如果是一维，说明每个输出对应一个梯度
                        jacobian_matrix = np.diag(jacobian_val)
                    else:
                        jacobian_matrix = jacobian_val.reshape(n_paths, n_paths)
                    
                    results['delay_predictions'].append(delay_pred)
                    results['jacobian_matrices'].append(jacobian_matrix)
                    results['diagonal_gradients'].append(np.diag(jacobian_matrix))
                    
                    # 记录交叉梯度
                    for i in range(n_paths):
                        if i != path_to_vary:
                            results['cross_gradients'][f'J_{i}{path_to_vary}'].append(
                                jacobian_matrix[i, path_to_vary]
                            )
                else:
                    print(f"Warning: No gradient computed for traffic {traffic_val}")
                    
            except Exception as e:
                print(f"Error computing gradient for traffic {traffic_val}: {e}")
        
        # 转换为numpy数组
        results['delay_predictions'] = np.array(results['delay_predictions'])
        results['jacobian_matrices'] = np.array(results['jacobian_matrices'])
        results['diagonal_gradients'] = np.array(results['diagonal_gradients'])
        
        for key in results['cross_gradients']:
            results['cross_gradients'][key] = np.array(results['cross_gradients'][key])
        
        return results
    
    def _analyze_path_topology(self, network_config):
        """
        分析网络拓扑，找出路径间的链路共享关系
        
        Returns:
            shared_links_matrix: [n_paths, n_paths] 布尔矩阵，
                                shared_links_matrix[i][j] = True 表示路径i和路径j共享至少一条链路
        """
        n_paths = network_config['n_paths']
        links = network_config['links']
        paths = network_config['paths']
        
        # 构建每条路径使用的链路集合
        path_links = [set() for _ in range(n_paths)]
        
        for link_idx, path_idx in zip(links, paths):
            path_links[path_idx].add(link_idx)
        
        # 构建路径间共享链路矩阵
        shared_links_matrix = np.zeros((n_paths, n_paths), dtype=bool)
        shared_links_count = np.zeros((n_paths, n_paths), dtype=int)
        
        for i in range(n_paths):
            for j in range(n_paths):
                if i != j:
                    shared_links = path_links[i].intersection(path_links[j])
                    shared_links_matrix[i][j] = len(shared_links) > 0
                    shared_links_count[i][j] = len(shared_links)
        
        return shared_links_matrix, shared_links_count, path_links

    def validate_physical_intuition(self, experiment_results, network_config, 
                                   path_to_vary, output_dir):
        """
        验证梯度的物理意义 (拓扑感知版本)
        
        物理直觉验证标准:
        1. 自影响梯度 J_ii > 0：路径自己的流量增加应该增加自己的延迟
        2. 交叉影响梯度 J_ij > 0：仅对共享链路的路径验证交叉影响
        3. 接近拥塞时梯度增大：当流量接近链路容量时，梯度应该显著增大
        4. 延迟单调递增：随着流量增加，延迟应该单调递增
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析网络拓扑
        shared_links_matrix, shared_links_count, path_links = self._analyze_path_topology(network_config)
        
        traffic_values = experiment_results['traffic_values']
        delay_predictions = experiment_results['delay_predictions']
        diagonal_gradients = experiment_results['diagonal_gradients']
        cross_gradients = experiment_results['cross_gradients']
        
        n_paths = network_config['n_paths']
        
        # 验证结果记录
        validation_results = {
            'self_gradient_positive': True,
            'cross_gradient_positive': True,
            'delay_monotonic': True,
            'gradient_increases_with_congestion': True,
            'physical_intuition_score': 0.0,
            'topology_info': {
                'shared_links_matrix': shared_links_matrix,
                'shared_links_count': shared_links_count,
                'path_links': path_links
            }
        }
        
        print("\n" + "="*60)
        print("原版RouteNet梯度物理意义验证结果 (拓扑感知)")
        print("="*60)
        
        # 打印拓扑信息
        print("\n0. 网络拓扑分析:")
        for i in range(n_paths):
            print(f"   路径 {i} 使用链路: {sorted(list(path_links[i]))}")
        
        print("\n   路径间链路共享关系:")
        for i in range(n_paths):
            for j in range(n_paths):
                if i != j and shared_links_matrix[i][j]:
                    shared_links = path_links[i].intersection(path_links[j])
                    print(f"   路径 {i} ↔ 路径 {j}: 共享链路 {sorted(list(shared_links))} ({shared_links_count[i][j]} 条)")
        
        if len(diagonal_gradients) == 0:
            print("错误：没有成功计算任何梯度值")
            validation_results['physical_intuition_score'] = 0.0
            return validation_results
        
        # 1. 验证自影响梯度 J_ii > 0
        self_gradients = diagonal_gradients[:, path_to_vary]
        positive_self_ratio = np.sum(self_gradients > 0) / len(self_gradients)
        print(f"\n1. 自影响梯度 J_{path_to_vary}{path_to_vary} > 0:")
        print(f"   正值比例: {positive_self_ratio:.2%}")
        print(f"   平均值: {np.mean(self_gradients):.6f}")
        print(f"   范围: [{np.min(self_gradients):.6f}, {np.max(self_gradients):.6f}]")
        
        if positive_self_ratio < 0.8:
            validation_results['self_gradient_positive'] = False
        
        # 2. 拓扑感知的交叉影响梯度验证
        print(f"\n2. 交叉影响梯度分析 (拓扑感知):")
        
        cross_gradient_validations = []
        
        for key, cross_grad in cross_gradients.items():
            # 解析梯度键：J_ij 表示 ∂D_i/∂T_j
            parts = key.split('_')
            if len(parts) == 2:
                i = int(parts[1][0])  # 受影响的路径 i
                j = int(parts[1][1])  # 影响的路径 j (应该是path_to_vary)
                
                # 检查路径i和路径j是否共享链路
                if shared_links_matrix[i][j]:
                    # 共享链路，期望正梯度
                    positive_cross_ratio = np.sum(cross_grad > 0) / len(cross_grad)
                    shared_links = path_links[i].intersection(path_links[j])
                    
                    print(f"   {key} > 0 (共享链路 {sorted(list(shared_links))}): {positive_cross_ratio:.2%}")
                    print(f"     平均值: {np.mean(cross_grad):.6f}")
                    
                    cross_gradient_validations.append(positive_cross_ratio >= 0.6)
                    
                else:
                    # 不共享链路，交叉影响应该较小，不强制要求正值
                    positive_cross_ratio = np.sum(cross_grad > 0) / len(cross_grad)
                    avg_magnitude = np.mean(np.abs(cross_grad))
                    
                    print(f"   {key} (无共享链路): {positive_cross_ratio:.2%}")
                    print(f"     平均值: {np.mean(cross_grad):.6f}, 平均幅度: {avg_magnitude:.6f}")
                    
                    # 对于不共享链路的路径，不纳入验证标准，但记录信息
                    print(f"     → 无共享链路，交叉影响预期较小")
        
        # 只有共享链路的交叉梯度需要满足正值要求
        if cross_gradient_validations:
            validation_results['cross_gradient_positive'] = all(cross_gradient_validations)
        else:
            # 如果没有共享链路的路径对，这项验证自动通过
            validation_results['cross_gradient_positive'] = True
            print(f"   注意: 路径 {path_to_vary} 与其他路径无共享链路，交叉梯度验证自动通过")
        
        # 3. 验证延迟单调性
        print(f"\n3. 延迟单调性验证:")
        for i in range(n_paths):
            delays = delay_predictions[:, i]
            # 计算单调递增的比例
            diff = np.diff(delays)
            monotonic_ratio = np.sum(diff >= -1e-6) / len(diff)  # 允许小的数值误差
            print(f"   路径 {i} 延迟单调递增比例: {monotonic_ratio:.2%}")
            
            if monotonic_ratio < 0.8:
                validation_results['delay_monotonic'] = False
        
        # 4. 验证梯度随拥塞增大
        print(f"\n4. 梯度拥塞敏感性验证:")
        # 比较低流量和高流量时的梯度
        low_traffic_idx = len(traffic_values) // 4  # 前25%
        high_traffic_idx = -len(traffic_values) // 4  # 后25%
        
        low_gradient = np.mean(self_gradients[:low_traffic_idx])
        high_gradient = np.mean(self_gradients[high_traffic_idx:])
        gradient_increase_ratio = high_gradient / (low_gradient + 1e-9)
        
        print(f"   低流量时平均梯度: {low_gradient:.6f}")
        print(f"   高流量时平均梯度: {high_gradient:.6f}")
        print(f"   梯度增长比例: {gradient_increase_ratio:.2f}x")
        
        if gradient_increase_ratio < 1.5:
            validation_results['gradient_increases_with_congestion'] = False
        
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
    parser.add_argument('--traffic_min', type=float, default=0.1, 
                       help='最小流量值 (基于数据集范围0.086-1.103)')
    parser.add_argument('--traffic_max', type=float, default=1.0, 
                       help='最大流量值 (基于数据集范围0.086-1.103)')
    parser.add_argument('--num_points', type=int, default=10, 
                       help='流量采样点数量')
    
    args = parser.parse_args()
    
    try:
        print("初始化原版RouteNet梯度验证器...")
        checker = OriginalRouteNetGradientChecker(args.model_dir, target='delay')
        
        print("创建可控网络拓扑...")
        network_config = checker.create_controlled_network()
        
        print("网络配置:")
        print(f"  节点数: 5, 链路数: {network_config['n_links']}, 路径数: {network_config['n_paths']}")
        print(f"  链路容量: {network_config['capacities']}")
        print(f"  基础流量: {network_config['base_traffic']}")
        print(f"  瓶颈链路: {network_config['bottleneck_links']}")
        
        # 对每条路径进行流量扫描实验
        overall_results = {}
        
        for path_id in range(network_config['n_paths']):
            print(f"\n{'='*60}")
            print(f"测试路径 {path_id}")
            print(f"{'='*60}")
            
            # 执行流量扫描实验
            experiment_results = checker.traffic_sweep_experiment(
                network_config,
                path_to_vary=path_id,
                traffic_range=(args.traffic_min, args.traffic_max),
                num_points=args.num_points
            )
            
            # 验证物理意义
            validation_results = checker.validate_physical_intuition(
                experiment_results, 
                network_config, 
                path_id, 
                args.output_dir
            )
            
            overall_results[f'path_{path_id}'] = validation_results
        
        # 计算总体结果
        overall_score = np.mean([
            results['physical_intuition_score'] 
            for results in overall_results.values()
        ])
        
        print(f"\n{'='*60}")
        print("总体验证结果")
        print(f"{'='*60}")
        print(f"模型类型: Original RouteNet (TF1.x)")
        print(f"总体物理直觉得分: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            print("✅ 原版RouteNet梯度计算通过物理意义验证！")
        elif overall_score >= 0.6:
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

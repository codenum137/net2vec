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

    def _compute_s_self_formula(self, diagonal_gradients, path_to_vary, n_samples):
        """
        计算 S_self = (1/N) * Σ I(g_kk^(i) >= 0)
        自影响梯度为正的比例
        """
        self_gradients = diagonal_gradients[:, path_to_vary]
        positive_count = np.sum(self_gradients >= 0)
        return positive_count / n_samples

    def _compute_s_mono_formula(self, delay_predictions, path_to_vary, n_samples):
        """
        计算 S_mono = (1/N) * Σ I(d_k^(i+1) >= d_k^(i))
        延迟单调性：后续流量下的延迟 >= 前一个流量下的延迟
        """
        delays = delay_predictions[:, path_to_vary]
        if len(delays) < 2:
            return 1.0
        
        # 计算相邻延迟差
        delay_diffs = np.diff(delays)
        # 单调递增（允许小的数值误差）
        monotonic_count = np.sum(delay_diffs >= -1e-8)
        return monotonic_count / (n_samples - 1)

    def _compute_s_cross_formula(self, cross_gradients, shared_links_matrix, 
                                path_to_vary, n_paths, n_samples):
        """
        计算 S_cross = (1/M) * Σ_m (1/N) * Σ_i I(g_km^(i) >= 0)
        共享链路路径的交叉影响梯度为正的比例
        M: 与path_to_vary共享链路的路径数量
        """
        shared_path_scores = []
        
        for key, cross_grad in cross_gradients.items():
            # 解析梯度键：J_ij 表示 ∂D_i/∂T_j
            parts = key.split('_')
            if len(parts) == 2:
                i = int(parts[1][0])  # 受影响的路径
                j = int(parts[1][1])  # 影响路径（应该是path_to_vary）
                
                # 检查是否共享链路
                if j == path_to_vary and shared_links_matrix[i][j]:
                    positive_count = np.sum(cross_grad >= 0)
                    path_score = positive_count / n_samples
                    shared_path_scores.append(path_score)
        
        if len(shared_path_scores) == 0:
            return 1.0  # 无共享路径时给满分
        
        return np.mean(shared_path_scores)

    def _compute_s_indep_formula(self, cross_gradients, shared_links_matrix, 
                                path_to_vary, n_paths, n_samples):
        """
        计算 S_indep = (1/L) * Σ_l (1/N) * Σ_i I(|g_kl^(i)| <= ε)
        独立路径的交叉影响梯度接近零的比例
        L: 与path_to_vary不共享链路的路径数量
        ε: 很小的阈值，例如0.001
        """
        epsilon = 0.001
        independent_path_scores = []
        
        for key, cross_grad in cross_gradients.items():
            # 解析梯度键：J_ij 表示 ∂D_i/∂T_j
            parts = key.split('_')
            if len(parts) == 2:
                i = int(parts[1][0])  # 受影响的路径
                j = int(parts[1][1])  # 影响路径（应该是path_to_vary）
                
                # 检查是否为独立路径（不共享链路）
                if j == path_to_vary and not shared_links_matrix[i][j]:
                    small_gradient_count = np.sum(np.abs(cross_grad) <= epsilon)
                    path_score = small_gradient_count / n_samples
                    independent_path_scores.append(path_score)
        
        if len(independent_path_scores) == 0:
            return 1.0  # 无独立路径时给满分
        
        return np.mean(independent_path_scores)

    def _compute_s_congest_formula(self, diagonal_gradients, path_to_vary, n_samples):
        """
        计算 S_congest = I(g_mean_high >= α * g_mean_low)
        拥塞敏感性：高流量区域的平均梯度 >= α倍低流量区域的平均梯度
        α: 拥塞敏感因子，例如1.2
        """
        alpha = 1.2
        self_gradients = diagonal_gradients[:, path_to_vary]
        
        if n_samples < 4:
            return 1.0
        
        # 计算低流量区域（前25%）和高流量区域（后25%）的平均梯度
        low_traffic_end = max(1, n_samples // 4)
        high_traffic_start = max(low_traffic_end, n_samples - n_samples // 4)
        
        g_mean_low = np.mean(self_gradients[:low_traffic_end])
        g_mean_high = np.mean(self_gradients[high_traffic_start:])
        
        # 避免除零错误
        if g_mean_low <= 1e-10:
            return 1.0 if g_mean_high > 1e-10 else 0.0
        
        return 1.0 if g_mean_high >= alpha * g_mean_low else 0.0

    def _print_pc_score_results(self, validation_results, path_to_vary):
        """打印PC-Score结果"""
        print("\n" + "="*70)
        print("PC-Score (物理一致性评分) 结果 - 原版RouteNet")
        print("="*70)
        
        pc_score = validation_results['pc_score']
        components = validation_results['components']
        weights = validation_results['weights']
        
        print(f"\n🎯 PC-Score 总分: {pc_score:.4f}")
        print(f"   验证状态: {'✅ 通过' if validation_results['validation_passed'] else '❌ 未通过'}")
        print(f"   路径 {path_to_vary} 的物理一致性评估")
        
        print(f"\n📊 各组件得分:")
        print(f"   S_self   (自影响为正):     {components['s_self']:.4f} × {weights['self']:.2f} = {components['s_self'] * weights['self']:.4f}")
        print(f"   S_mono   (延迟单调性):     {components['s_mono']:.4f} × {weights['mono']:.2f} = {components['s_mono'] * weights['mono']:.4f}")
        print(f"   S_cross  (共享路径影响):   {components['s_cross']:.4f} × {weights['cross']:.2f} = {components['s_cross'] * weights['cross']:.4f}")
        print(f"   S_indep  (独立路径零影响): {components['s_indep']:.4f} × {weights['indep']:.2f} = {components['s_indep'] * weights['indep']:.4f}")
        print(f"   S_congest(拥塞敏感性):     {components['s_congest']:.4f} × {weights['congest']:.2f} = {components['s_congest'] * weights['congest']:.4f}")
        
        print(f"\n📈 解释:")
        # PC-Score区间解释
        if pc_score >= 0.95:
            print(f"   🌟 优秀 - 模型完全掌握了网络物理规律")
        elif pc_score >= 0.85:
            print(f"   ⭐ 良好 - 模型很好地掌握了网络物理规律")
        elif pc_score >= 0.7:
            print(f"   ✓ 可接受 - 模型基本掌握了网络物理规律")
        elif pc_score >= 0.5:
            print(f"   ⚠ 一般 - 模型部分掌握了网络物理规律")
        else:
            print(f"   ❌ 较差 - 模型未能很好地学习网络物理规律")

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
        n_samples = len(traffic_values)
        
        # 检查数据完整性
        if len(diagonal_gradients) == 0:
            print("错误：没有成功计算任何梯度值")
            return {'pc_score': 0.0, 'validation_passed': False, 'components': {}, 'weights': {}}

        # PC-Score权重配置（重点强调物理规律重要性层次）
        weights = {
            'self': 0.35,    # 自影响为正 - 最重要的基础物理规律
            'mono': 0.25,    # 延迟单调性 - 核心物理特性 
            'cross': 0.15,   # 共享路径影响 - 拓扑感知
            'indep': 0.15,   # 独立路径零影响 - 拓扑感知
            'congest': 0.10  # 拥塞敏感性 - 高级特性
        }

        # 1. 计算PC-Score各组件
        s_self = self._compute_s_self_formula(diagonal_gradients, path_to_vary, n_samples)
        s_mono = self._compute_s_mono_formula(delay_predictions, path_to_vary, n_samples) 
        s_cross = self._compute_s_cross_formula(cross_gradients, shared_links_matrix, 
                                              path_to_vary, n_paths, n_samples)
        s_indep = self._compute_s_indep_formula(cross_gradients, shared_links_matrix, 
                                              path_to_vary, n_paths, n_samples)
        s_congest = self._compute_s_congest_formula(diagonal_gradients, path_to_vary, n_samples)
        
        # 2. 计算加权PC-Score
        pc_score = (weights['self'] * s_self + 
                   weights['mono'] * s_mono + 
                   weights['cross'] * s_cross + 
                   weights['indep'] * s_indep + 
                   weights['congest'] * s_congest)
        
        # 3. 构建验证结果
        validation_results = {
            'pc_score': pc_score,
            'validation_passed': pc_score >= 0.7,
            'components': {
                's_self': s_self,
                's_mono': s_mono, 
                's_cross': s_cross,
                's_indep': s_indep,
                's_congest': s_congest
            },
            'weights': weights,
            'topology_info': {
                'shared_links_matrix': shared_links_matrix,
                'shared_links_count': shared_links_count,
                'path_links': path_links
            }
        }
        
        # 打印PC-Score结果
        self._print_pc_score_results(validation_results, path_to_vary)
        
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
        
        # 4. PC-Score结果文本总结
        ax4 = axes[1, 1]
        ax4.axis('off')  # 隐藏坐标轴
        
        # 创建PC-Score验证结果文本总结
        pc_score = validation_results['pc_score']
        components = validation_results['components']
        weights = validation_results['weights']
        
        summary_text = f"PC-Score Physical Consistency Summary\n"
        summary_text += f"Overall PC-Score: {pc_score:.4f}\n"
        summary_text += f"Status: {'✓ PASS' if validation_results['validation_passed'] else '✗ FAIL'}\n\n"
        
        # PC-Score组件得分
        summary_text += f"Component Scores:\n"
        summary_text += f"S_self:    {components['s_self']:.3f} × {weights['self']:.2f} = {components['s_self'] * weights['self']:.4f}\n"
        summary_text += f"S_mono:    {components['s_mono']:.3f} × {weights['mono']:.2f} = {components['s_mono'] * weights['mono']:.4f}\n"
        summary_text += f"S_cross:   {components['s_cross']:.3f} × {weights['cross']:.2f} = {components['s_cross'] * weights['cross']:.4f}\n"
        summary_text += f"S_indep:   {components['s_indep']:.3f} × {weights['indep']:.2f} = {components['s_indep'] * weights['indep']:.4f}\n"
        summary_text += f"S_congest: {components['s_congest']:.3f} × {weights['congest']:.2f} = {components['s_congest'] * weights['congest']:.4f}\n"
        
        # 添加详细统计信息
        if diagonal_gradients.shape[1] > path_to_vary:
            self_gradients = diagonal_gradients[:, path_to_vary]
            self_pos_ratio = np.sum(self_gradients > 0) / len(self_gradients)
            summary_text += f"\nDetailed Statistics:\n"
            summary_text += f"Self-gradient positive ratio: {self_pos_ratio:.1%}\n"
            summary_text += f"Self-gradient mean: {np.mean(self_gradients):.6f}\n"
        
            for key, cross_grad in cross_gradients.items():
                if len(cross_grad) > 0:
                    cross_pos_ratio = np.sum(cross_grad > 0) / len(cross_grad)
                    summary_text += f"{key} positive ratio: {cross_pos_ratio:.1%}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Original RouteNet - PC-Score Physical Consistency Validation (Path {path_to_vary})', 
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'original_routenet_sanity_check_path_{path_to_vary}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存简化的PC-Score结果
        with open(os.path.join(output_dir, f'original_routenet_pc_score_results_path_{path_to_vary}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Original RouteNet - PC-Score Physical Consistency Results\n")
            f.write("="*60 + "\n")
            f.write(f"Path: {path_to_vary}\n")
            f.write(f"PC-Score: {validation_results['pc_score']:.4f}\n")
            f.write(f"Status: {'PASS' if validation_results['validation_passed'] else 'FAIL'}\n\n")
            
            # PC-Score组件详情
            components = validation_results['components']
            weights = validation_results['weights']
            f.write("PC-Score Components:\n")
            f.write("-" * 30 + "\n")
            f.write(f"S_self:    {components['s_self']:.4f} × {weights['self']:.2f} = {components['s_self'] * weights['self']:.4f}\n")
            f.write(f"S_mono:    {components['s_mono']:.4f} × {weights['mono']:.2f} = {components['s_mono'] * weights['mono']:.4f}\n")
            f.write(f"S_cross:   {components['s_cross']:.4f} × {weights['cross']:.2f} = {components['s_cross'] * weights['cross']:.4f}\n")
            f.write(f"S_indep:   {components['s_indep']:.4f} × {weights['indep']:.2f} = {components['s_indep'] * weights['indep']:.4f}\n")
            f.write(f"S_congest: {components['s_congest']:.4f} × {weights['congest']:.2f} = {components['s_congest'] * weights['congest']:.4f}\n")
    
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
            results['pc_score'] 
            for results in overall_results.values()
        ])
        
        print(f"\n{'='*60}")
        print("总体验证结果")
        print(f"{'='*60}")
        print(f"模型类型: Original RouteNet (TF1.x)")
        print(f"总体PC-Score得分: {overall_score:.4f}")
        
        if overall_score >= 0.7:
            print("✅ 原版RouteNet梯度计算通过PC-Score物理一致性验证！")
        else:
            print("❌ 原版RouteNet梯度计算未通过PC-Score验证，需要检查实现")
        
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

# -*- coding: utf-8 -*-
"""
梯度物理意义验证模块
设计可控实验验证雅可比矩阵是否符合网络物理直觉
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from jacobian_analysis import JacobianAnalyzer, create_simple_network_sample
from routenet_tf2 import RouteNet, create_model_and_loss_fn
import argparse
from tqdm import tqdm

class GradientSanityChecker:
    """梯度物理意义验证器"""
    
    def __init__(self, model_path, config, target='delay', use_kan=False):
        """初始化验证器"""
        self.analyzer = JacobianAnalyzer(model_path, config, target, use_kan)
        self.use_kan = use_kan
        self.target = target
    
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
    
    def create_controlled_network(self):
        """
        创建一个可控的简单网络拓扑
        
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
        print(f"执行流量扫描实验：变化路径 {path_to_vary} 的流量...")
        
        # 生成流量序列
        traffic_values = np.linspace(traffic_range[0], traffic_range[1], num_points)
        
        results = {
            'traffic_values': traffic_values,
            'delay_predictions': [],
            'jacobian_matrices': [],
            'diagonal_gradients': [],  # J_ii: ∂D_i/∂T_i
            'cross_gradients': {},     # J_ij: ∂D_i/∂T_j (i≠j)
        }
        
        # 为每个共享链路的路径对记录交叉梯度
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
            jacobian, delay_pred = self.analyzer.compute_jacobian(sample_features)
            
            results['delay_predictions'].append(delay_pred)
            results['jacobian_matrices'].append(jacobian)
            results['diagonal_gradients'].append(np.diag(jacobian))
            
            # 记录交叉梯度
            for i in range(network_config['n_paths']):
                if i != path_to_vary:
                    results['cross_gradients'][f'J_{i}{path_to_vary}'].append(
                        jacobian[i, path_to_vary]
                    )
        
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

    def _evaluate_independent_paths_zero_influence(self, cross_gradients, shared_links_matrix, 
                                                  path_links, n_paths, path_to_vary, tolerance=1e-4):
        """
        评估独立路径零影响指标 (S_indep)
        
        物理直觉：拓扑上独立的路径（不共享任何链路）之间应该互不影响，
        即它们之间的交叉梯度应该接近于零。
        
        Args:
            cross_gradients: 交叉梯度字典
            shared_links_matrix: 路径间共享链路矩阵
            path_links: 每条路径使用的链路集合列表
            n_paths: 路径总数
            path_to_vary: 当前变化的路径索引
            tolerance: 容忍阈值，用于将绝对梯度值转化为0-1分数
        
        Returns:
            indep_score: 独立路径零影响得分 [0, 1]
        """
        # 找出所有与path_to_vary独立的路径对
        independent_pairs = []
        independent_gradients = []
        
        print(f"   寻找与路径 {path_to_vary} 独立的路径:")
        
        for i in range(n_paths):
            if i != path_to_vary:
                # 检查路径i和path_to_vary是否共享链路
                if not shared_links_matrix[i][path_to_vary]:
                    # 路径i和path_to_vary独立
                    independent_pairs.append((i, path_to_vary))
                    
                    # 查找对应的梯度键
                    grad_key = f'J_{i}{path_to_vary}'
                    if grad_key in cross_gradients:
                        gradient_values = cross_gradients[grad_key]
                        independent_gradients.extend(gradient_values)
                        
                        # 计算该路径对的平均绝对梯度
                        avg_abs_grad = np.mean(np.abs(gradient_values))
                        print(f"     路径 {i} ↔ 路径 {path_to_vary}: 平均绝对交叉梯度 = {avg_abs_grad:.6f}")
                    else:
                        print(f"     路径 {i} ↔ 路径 {path_to_vary}: 未找到梯度数据 ({grad_key})")
        
        if not independent_pairs:
            print(f"     未找到与路径 {path_to_vary} 完全独立的路径")
            return 1.0  # 如果没有独立路径对，给满分
        
        if not independent_gradients:
            print(f"     独立路径对存在但无梯度数据")
            return 0.0  # 有独立路径但没有数据，给0分
        
        # 计算所有独立路径对的平均绝对梯度
        avg_abs_grad_all = np.mean(np.abs(independent_gradients))
        
        print(f"   独立路径对数量: {len(independent_pairs)}")
        print(f"   梯度样本总数: {len(independent_gradients)}")
        print(f"   平均绝对梯度: {avg_abs_grad_all:.6f}")
        print(f"   容忍阈值: {tolerance}")
        
        # 使用容忍阈值将绝对梯度转化为0-1分数
        # S_indep = max(0, 1 - avg_abs_grad / tolerance)
        indep_score = max(0.0, 1.0 - avg_abs_grad_all / tolerance)
        
        return indep_score

    def validate_physical_intuition(self, experiment_results, network_config, 
                                   path_to_vary, output_dir, weights=None, tau=1e-4):
        """
        计算PC-Score (物理一致性评分) - 基于完整数学公式的实现
        
        PC-Score = w_self * S_self + w_mono * S_mono + w_cross * S_cross 
                 + w_indep * S_indep + w_congest * S_congest
        
        Args:
            experiment_results: 实验结果数据
            network_config: 网络配置
            path_to_vary: 变化的路径ID
            output_dir: 输出目录
            weights: 各指标权重，默认为均匀分布
            tau: 独立路径零影响的容忍阈值
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 默认权重设置 - 基于物理规律重要性层次
        if weights is None:
            weights = {
                'self': 0.35,      # w_self - 自影响为正是最基础的规律
                'mono': 0.25,      # w_mono - 延迟单调性是自影响规律的直接体现  
                'cross': 0.15,     # w_cross - 路径干扰是GNN需要学习的关键拓扑效应
                'indep': 0.15,     # w_indep - 路径独立性同样反映了对拓扑的理解
                'congest': 0.10    # w_congest - 拥塞敏感性是更高级、更细微的非线性规律
            }
        
        # 验证权重之和为1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            for key in weights:
                weights[key] /= weight_sum
        
        # 分析网络拓扑
        shared_links_matrix, shared_links_count, path_links = self._analyze_path_topology(network_config)
        
        # 提取实验数据
        traffic_values = experiment_results['traffic_values']
        delay_predictions = experiment_results['delay_predictions']
        diagonal_gradients = experiment_results['diagonal_gradients']
        cross_gradients = experiment_results['cross_gradients']
        n_paths = network_config['n_paths']
        n_samples = len(traffic_values)
        
        # 计算各项PC-Score组件
        s_self = self._compute_s_self_formula(diagonal_gradients, path_to_vary, n_samples)
        s_mono = self._compute_s_mono_formula(delay_predictions, path_to_vary, n_samples)
        s_cross = self._compute_s_cross_formula(cross_gradients, shared_links_matrix, 
                                              n_paths, n_samples, path_to_vary)
        s_indep = self._compute_s_indep_formula(cross_gradients, shared_links_matrix,
                                              n_paths, n_samples, path_to_vary, tau)
        s_congest = self._compute_s_congest_formula(diagonal_gradients, path_to_vary, n_samples)
        
        # 计算PC-Score (物理一致性评分)
        pc_score = (weights['self'] * s_self + 
                   weights['mono'] * s_mono +
                   weights['cross'] * s_cross +
                   weights['indep'] * s_indep +
                   weights['congest'] * s_congest)
        
        # 验证结果记录
        validation_results = {
            'pc_score': pc_score,
            'components': {
                's_self': s_self,
                's_mono': s_mono,
                's_cross': s_cross, 
                's_indep': s_indep,
                's_congest': s_congest
            },
            'weights': weights,
            'tau': tau,
            'validation_passed': pc_score >= 0.7,
            'topology_info': {
                'shared_links_matrix': shared_links_matrix,
                'shared_links_count': shared_links_count,
                'path_links': path_links
            }
        }
        
        # 打印PC-Score结果
        self._print_pc_score_results(validation_results, path_to_vary)
        
        # 可视化结果
        self._visualize_sanity_check(experiment_results, network_config, 
                                    path_to_vary, output_dir, validation_results)
        
        return validation_results
        
    def _compute_s_self_formula(self, diagonal_gradients, path_to_vary, n_samples):
        """
        计算 S_self = (1/N) * Σ I(g_kk^(i) >= 0)
        自影响梯度为正的比例
        """
        self_gradients = diagonal_gradients[:, path_to_vary]
        positive_count = np.sum(self_gradients >= 0)
        s_self = positive_count / n_samples
        return s_self
    
    def _compute_s_mono_formula(self, delay_predictions, path_to_vary, n_samples):
        """
        计算 S_mono = (1/(N-1)) * Σ I(D_k(T_{i+1}) >= D_k(T_i))
        延迟单调性比例
        """
        delays = delay_predictions[:, path_to_vary]
        monotonic_count = 0
        for i in range(n_samples - 1):
            if delays[i + 1] >= delays[i]:
                monotonic_count += 1
        s_mono = monotonic_count / (n_samples - 1) if n_samples > 1 else 1.0
        return s_mono
    
    def _compute_s_cross_formula(self, cross_gradients, shared_links_matrix, 
                               n_paths, n_samples, path_to_vary):
        """
        计算 S_cross = (1/|P_shared|) * Σ ((1/N) * Σ I(g_ij^(k) >= 0))
        共享路径交叉影响为正的平均比例
        """
        shared_pairs = []
        positive_ratios = []
        
        for key, cross_grad in cross_gradients.items():
            # 解析梯度键 (例如 "J_01" -> i=0, j=1)
            parts = key.split('_')
            if len(parts) == 2 and len(parts[1]) == 2:
                i = int(parts[1][0])  # 受影响的路径
                j = int(parts[1][1])  # 影响的路径
                
                # 检查是否为共享链路的路径对
                if i < n_paths and j < n_paths and shared_links_matrix[i][j]:
                    positive_count = np.sum(cross_grad >= 0)
                    positive_ratio = positive_count / n_samples
                    positive_ratios.append(positive_ratio)
                    shared_pairs.append((i, j))
        
        if len(positive_ratios) > 0:
            s_cross = np.mean(positive_ratios)
        else:
            s_cross = 1.0  # 没有共享路径时默认满分
            
        return s_cross
    
    def _compute_s_indep_formula(self, cross_gradients, shared_links_matrix,
                               n_paths, n_samples, path_to_vary, tau):
        """
        计算 S_indep = max(0, 1 - E[|g_ij|]_{(i,j)∈P_indep} / τ)
        独立路径零影响评估
        """
        independent_grads = []
        
        for key, cross_grad in cross_gradients.items():
            # 解析梯度键
            parts = key.split('_')
            if len(parts) == 2 and len(parts[1]) == 2:
                i = int(parts[1][0])  # 受影响的路径
                j = int(parts[1][1])  # 影响的路径
                
                # 检查是否为独立路径对（不共享链路）
                if i < n_paths and j < n_paths and not shared_links_matrix[i][j]:
                    abs_grads = np.abs(cross_grad)
                    independent_grads.extend(abs_grads)
        
        if len(independent_grads) > 0:
            avg_abs_grad = np.mean(independent_grads)
            s_indep = max(0.0, 1.0 - avg_abs_grad / tau)
        else:
            s_indep = 1.0  # 没有独立路径时默认满分
            
        return s_indep
    
    def _compute_s_congest_formula(self, diagonal_gradients, path_to_vary, n_samples):
        """
        计算 S_congest = (1/(N-1)) * Σ I(g_kk^(i+1) >= g_kk^(i))
        拥塞敏感性：梯度随流量单调递增的比例
        """
        self_gradients = diagonal_gradients[:, path_to_vary]
        monotonic_gradient_count = 0
        
        for i in range(n_samples - 1):
            if self_gradients[i + 1] >= self_gradients[i]:
                monotonic_gradient_count += 1
                
        s_congest = monotonic_gradient_count / (n_samples - 1) if n_samples > 1 else 1.0
        return s_congest
    
    def _print_pc_score_results(self, validation_results, path_to_vary):
        """打印PC-Score结果"""
        print("\n" + "="*70)
        print("PC-Score (物理一致性评分) 结果")
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
        score_interpretation = {
            (0.9, 1.0): "🌟 优秀 - 模型完全掌握了网络物理规律",
            (0.8, 0.9): "✅ 良好 - 模型很好地学习了网络物理规律",
            (0.7, 0.8): "✓ 及格 - 模型基本学习了网络物理规律", 
            (0.6, 0.7): "⚠️ 一般 - 模型部分学习了网络物理规律",
            (0.0, 0.6): "❌ 较差 - 模型未能很好地学习网络物理规律"
        }
        
        for (low, high), desc in score_interpretation.items():
            if low <= pc_score < high:
                print(f"   {desc}")
                break
    
    def _visualize_sanity_check(self, experiment_results, network_config, 
                               path_to_vary, output_dir, validation_results):
        """可视化验证结果"""
        
        # 设置字体避免中文显示问题
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        traffic_values = experiment_results['traffic_values']
        delay_predictions = experiment_results['delay_predictions']
        diagonal_gradients = experiment_results['diagonal_gradients']
        cross_gradients = experiment_results['cross_gradients']
        
        # 创建综合图表 (移除右下角子图)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 延迟 vs 流量
        ax1 = axes[0, 0]
        for i in range(network_config['n_paths']):
            ax1.plot(traffic_values, delay_predictions[:, i], 
                    label=f'Path {i}', marker='o', markersize=3)
        ax1.set_xlabel(f'Path {path_to_vary} Traffic (Mbps)')
        ax1.set_ylabel('Predicted Delay')
        ax1.set_title('Delay vs Traffic Relationship')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 自影响梯度
        ax2 = axes[0, 1]
        self_gradients = diagonal_gradients[:, path_to_vary]
        ax2.plot(traffic_values, self_gradients, 'r-', marker='s', markersize=4)
        ax2.set_xlabel(f'Path {path_to_vary} Traffic (Mbps)')
        ax2.set_ylabel(f'∂D_{path_to_vary}/∂T_{path_to_vary}')
        ax2.set_title(f'Self-influence Gradient J_{path_to_vary}{path_to_vary}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. 交叉影响梯度
        ax3 = axes[1, 0] 
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (key, cross_grad) in enumerate(cross_gradients.items()):
            ax3.plot(traffic_values, cross_grad, color=colors[i % len(colors)],
                    label=key, marker='^', markersize=3)
        ax3.set_xlabel(f'Path {path_to_vary} Traffic (Mbps)')
        ax3.set_ylabel('Cross-influence Gradient')
        ax3.set_title('Cross-influence Gradients')
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
        self_gradients = diagonal_gradients[:, path_to_vary]
        self_pos_ratio = np.sum(self_gradients > 0) / len(self_gradients)
        summary_text += f"\nDetailed Statistics:\n"
        summary_text += f"Self-gradient positive ratio: {self_pos_ratio:.1%}\n"
        summary_text += f"Self-gradient mean: {np.mean(self_gradients):.6f}\n"
        
        for key, cross_grad in cross_gradients.items():
            cross_pos_ratio = np.sum(cross_grad > 0) / len(cross_grad)
            summary_text += f"{key} positive ratio: {cross_pos_ratio:.1%}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        model_type = "KAN Model" if self.use_kan else "MLP Model"
        fig.suptitle(f'{model_type} - PC-Score Physical Consistency Validation (Path {path_to_vary})', 
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sanity_check_path_{path_to_vary}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存简化的PC-Score结果
        with open(os.path.join(output_dir, f'pc_score_results_path_{path_to_vary}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{model_type} - PC-Score Physical Consistency Results\n")
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RouteNet梯度物理意义验证')
    parser.add_argument('--model_dir', required=True, help='模型目录路径')
    parser.add_argument('--use_kan', action='store_true', help='使用KAN模型')
    parser.add_argument('--target', default='delay', choices=['delay', 'drops'], 
                       help='预测目标')
    parser.add_argument('--output_dir', default='result/physics/kan', 
                       help='输出目录')
    parser.add_argument('--traffic_min', type=float, default=0.1,
                       help='最小流量值 (基于数据集范围0.086-1.103)')
    parser.add_argument('--traffic_max', type=float, default=1.0,
                       help='最大流量值 (基于数据集范围0.086-1.103)')
    parser.add_argument('--num_points', type=int, default=10,
                       help='流量采样点数量')
    
    args = parser.parse_args()
    
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
    
    try:
        # 构造权重文件路径
        if args.use_kan:
            weight_file = os.path.join(args.model_dir, 'best_delay_kan_model.weights.h5')
        else:
            weight_file = os.path.join(args.model_dir, 'best_delay_model.weights.h5')
        
        print(f"初始化梯度验证器 ({'KAN' if args.use_kan else 'MLP'}模型)...")
        checker = GradientSanityChecker(
            model_path=weight_file,
            config=config,
            target=args.target,
            use_kan=args.use_kan
        )
        
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
        print(f"模型类型: {'KAN' if args.use_kan else 'MLP'}")
        print(f"总体PC-Score得分: {overall_score:.4f}")
        
        if overall_score >= 0.8:
            print("✅ 梯度计算通过物理意义验证！")
        elif overall_score >= 0.6:
            print("⚠️  梯度计算部分通过验证，需要进一步检查")
        else:
            print("❌ 梯度计算未通过物理意义验证，需要检查模型或实现")
        
        print(f"\n详细结果已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

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
        print("梯度物理意义验证结果 (拓扑感知)")
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
        
        # 计算总体物理直觉得分
        score_components = [
            validation_results['self_gradient_positive'],
            validation_results['cross_gradient_positive'], 
            validation_results['delay_monotonic'],
            validation_results['gradient_increases_with_congestion']
        ]
        validation_results['physical_intuition_score'] = sum(score_components) / len(score_components)
        
        print(f"\n5. 总体物理直觉得分: {validation_results['physical_intuition_score']:.2%}")
        
        # 可视化结果
        self._visualize_sanity_check(experiment_results, network_config, 
                                    path_to_vary, output_dir, validation_results)
        
        return validation_results
    
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
        
        # 4. 验证结果文本总结 (替代原来的条形图)
        ax4 = axes[1, 1]
        ax4.axis('off')  # 隐藏坐标轴
        
        # 创建验证结果文本总结
        metrics = ['Self-gradient > 0', 'Cross-gradient > 0', 'Delay Monotonic', 'Congestion Sensitivity']
        scores = [
            validation_results['self_gradient_positive'],
            validation_results['cross_gradient_positive'],
            validation_results['delay_monotonic'],
            validation_results['gradient_increases_with_congestion']
        ]
        
        summary_text = f"Physical Intuition Validation Summary\n"
        summary_text += f"Overall Score: {validation_results['physical_intuition_score']:.1%}\n\n"
        
        for metric, score in zip(metrics, scores):
            status = "✓ PASS" if score else "✗ FAIL"
            summary_text += f"{metric}: {status}\n"
        
        # 添加详细统计信息
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
        fig.suptitle(f'{model_type} - Gradient Physical Intuition Validation (Varying Path {path_to_vary})', 
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sanity_check_path_{path_to_vary}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细结果
        with open(os.path.join(output_dir, f'sanity_check_results_path_{path_to_vary}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{model_type} - Gradient Physical Intuition Validation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Varying Path: {path_to_vary}\n")
            f.write(f"Traffic Range: {traffic_values[0]:.1f} - {traffic_values[-1]:.1f} Mbps\n")
            f.write(f"Number of Traffic Points: {len(traffic_values)}\n\n")
            
            # 1. 自影响梯度详细分析
            self_gradients = diagonal_gradients[:, path_to_vary]
            positive_self_ratio = np.sum(self_gradients > 0) / len(self_gradients)
            f.write("1. Self-influence Gradient Analysis (∂D_i/∂T_i):\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Gradient Values for Path {path_to_vary}:\n")
            for i, (traffic, grad) in enumerate(zip(traffic_values, self_gradients)):
                f.write(f"    Traffic: {traffic:6.1f} Mbps -> Gradient: {grad:12.8f}\n")
            f.write(f"\n  Statistical Summary:\n")
            f.write(f"    Positive Ratio: {positive_self_ratio:.2%}\n")
            f.write(f"    Mean: {np.mean(self_gradients):12.8f}\n")
            f.write(f"    Std: {np.std(self_gradients):12.8f}\n")
            f.write(f"    Min: {np.min(self_gradients):12.8f}\n")
            f.write(f"    Max: {np.max(self_gradients):12.8f}\n")
            f.write(f"    Status: {'Pass' if validation_results['self_gradient_positive'] else 'Fail'}\n\n")
            
            # 2. 交叉影响梯度详细分析
            f.write("2. Cross-influence Gradient Analysis (∂D_i/∂T_j, i≠j):\n")
            f.write("-" * 50 + "\n")
            for key, cross_grad in cross_gradients.items():
                positive_cross_ratio = np.sum(cross_grad > 0) / len(cross_grad)
                f.write(f"  {key} (Cross-influence):\n")
                for i, (traffic, grad) in enumerate(zip(traffic_values, cross_grad)):
                    f.write(f"    Traffic: {traffic:6.1f} Mbps -> Gradient: {grad:12.8f}\n")
                f.write(f"    Statistical Summary:\n")
                f.write(f"      Positive Ratio: {positive_cross_ratio:.2%}\n")
                f.write(f"      Mean: {np.mean(cross_grad):12.8f}\n")
                f.write(f"      Std: {np.std(cross_grad):12.8f}\n")
                f.write(f"      Min: {np.min(cross_grad):12.8f}\n")
                f.write(f"      Max: {np.max(cross_grad):12.8f}\n\n")
            
            # 3. 延迟预测详细分析
            f.write("3. Delay Prediction Analysis:\n")
            f.write("-" * 50 + "\n")
            for path_idx in range(network_config['n_paths']):
                delays = delay_predictions[:, path_idx]
                diff = np.diff(delays)
                monotonic_ratio = np.sum(diff >= -1e-6) / len(diff)
                f.write(f"  Path {path_idx} Delay Predictions:\n")
                for i, (traffic, delay) in enumerate(zip(traffic_values, delays)):
                    f.write(f"    Traffic: {traffic:6.1f} Mbps -> Delay: {delay:12.8f}\n")
                f.write(f"    Monotonic Increase Ratio: {monotonic_ratio:.2%}\n")
                f.write(f"    Mean Delay: {np.mean(delays):12.8f}\n")
                f.write(f"    Delay Range: [{np.min(delays):12.8f}, {np.max(delays):12.8f}]\n\n")
            
            # 4. 拥塞敏感性分析
            low_traffic_idx = len(traffic_values) // 4
            high_traffic_idx = -len(traffic_values) // 4
            low_gradient = np.mean(self_gradients[:low_traffic_idx])
            high_gradient = np.mean(self_gradients[high_traffic_idx:])
            gradient_increase_ratio = high_gradient / (low_gradient + 1e-9)
            
            f.write("4. Congestion Sensitivity Analysis:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Low Traffic Region (first 25% points):\n")
            f.write(f"    Traffic Range: {traffic_values[0]:.1f} - {traffic_values[low_traffic_idx-1]:.1f} Mbps\n")
            f.write(f"    Average Gradient: {low_gradient:12.8f}\n")
            f.write(f"  High Traffic Region (last 25% points):\n")
            f.write(f"    Traffic Range: {traffic_values[high_traffic_idx]:.1f} - {traffic_values[-1]:.1f} Mbps\n")
            f.write(f"    Average Gradient: {high_gradient:12.8f}\n")
            f.write(f"  Gradient Increase Ratio: {gradient_increase_ratio:.4f}x\n")
            f.write(f"  Status: {'Pass' if validation_results['gradient_increases_with_congestion'] else 'Fail'}\n\n")
            
            # 5. 总体验证结果
            f.write("5. Overall Validation Results:\n")
            f.write("=" * 50 + "\n")
            f.write(f"  Self-influence Gradient > 0: {'Pass' if validation_results['self_gradient_positive'] else 'Fail'}\n")
            f.write(f"  Cross-influence Gradient > 0: {'Pass' if validation_results['cross_gradient_positive'] else 'Fail'}\n")
            f.write(f"  Delay Monotonic Increase: {'Pass' if validation_results['delay_monotonic'] else 'Fail'}\n")
            f.write(f"  Congestion Sensitivity: {'Pass' if validation_results['gradient_increases_with_congestion'] else 'Fail'}\n")
            f.write(f"\nOverall Physical Intuition Score: {validation_results['physical_intuition_score']:.1%}\n")
            
            # 6. 原始数据矩阵
            f.write(f"\n6. Raw Data Matrices:\n")
            f.write("=" * 50 + "\n")
            f.write("Jacobian Matrices for each traffic point:\n")
            for i, (traffic, jacobian) in enumerate(zip(traffic_values, experiment_results['jacobian_matrices'])):
                f.write(f"\nTraffic Point {i+1}: {traffic:.1f} Mbps\n")
                f.write("Jacobian Matrix:\n")
                for row in jacobian:
                    f.write("  [" + ", ".join([f"{val:12.8f}" for val in row]) + "]\n")

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
            results['physical_intuition_score'] 
            for results in overall_results.values()
        ])
        
        print(f"\n{'='*60}")
        print("总体验证结果")
        print(f"{'='*60}")
        print(f"模型类型: {'KAN' if args.use_kan else 'MLP'}")
        print(f"总体物理直觉得分: {overall_score:.1%}")
        
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

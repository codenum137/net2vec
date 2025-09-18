# -*- coding: utf-8 -*-
"""
雅可比矩阵分析模块
用于计算RouteNet模型（包括KAN变体）的雅可比矩阵，为流量工程提供梯度信息
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from routenet_tf2 import RouteNet, create_model_and_loss_fn, PhysicsInformedRouteNet
import argparse
from tqdm import tqdm

class JacobianAnalyzer:
    """RouteNet模型雅可比矩阵分析器"""
    
    def __init__(self, model_path, config, target='delay', use_kan=False):
        """
        初始化雅可比分析器
        
        Args:
            model_path: 模型权重文件路径
            config: 模型配置字典
            target: 预测目标 ('delay' 或 'drops')
            use_kan: 是否使用KAN模型
        """
        self.config = config
        self.target = target
        self.use_kan = use_kan
        
        # 创建模型 - 尝试新的 PhysicsInformedRouteNet，如果失败则使用旧方式
        try:
            self.model = PhysicsInformedRouteNet(
                config=config,
                target=target,
                use_kan=use_kan,
                use_physics_loss=False,  # 分析时不需要物理约束
                use_hard_constraint=False,
                lambda_physics=0.0,
                use_curriculum=False
            )
            self.loss_fn = None  # 分析时不需要损失函数
            print(f"Using PhysicsInformedRouteNet for {target} jacobian analysis")
        except Exception as e:
            print(f"Failed to create PhysicsInformedRouteNet: {e}, falling back to original model")
            self.model, self.loss_fn = create_model_and_loss_fn(
                config, target, use_kan=use_kan
            )
        
        # 加载权重
        self._load_model(model_path)
        
    def _load_model(self, model_path):
        """加载模型权重"""
        try:
            if os.path.exists(model_path):
                # 先构建模型：使用简单样本进行一次前向传播
                dummy_sample = create_simple_network_sample()
                dummy_features = {}
                
                # 正确处理数据类型转换
                for key, value in dummy_sample.items():
                    if key in ['links', 'paths', 'sequences', 'n_links', 'n_paths']:
                        # 整数类型
                        dummy_features[key] = tf.convert_to_tensor(value, dtype=tf.int32)
                    else:
                        # 浮点类型
                        dummy_features[key] = tf.convert_to_tensor(value, dtype=tf.float32)
                
                # 构建模型
                _ = self.model(dummy_features)
                print("模型构建完成")
                
                # 加载权重
                try:
                    self.model.load_weights(model_path)
                    print(f"成功加载模型权重: {model_path}")
                except Exception as weight_error:
                    print(f"权重加载失败: {weight_error}")
                    # 尝试重新构建模型
                    self.model.build(input_shape=(None, None))
                    self.model.load_weights(model_path)
                    print(f"重新构建后成功加载模型权重: {model_path}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            raise
    
    @tf.function(reduce_retracing=True)
    def _predict_delay(self, traffic, features):
        """
        预测延迟的内部函数，用于梯度计算
        
        Args:
            traffic: 流量向量 [n_paths]
            features: 其他网络特征（容量、路径结构等）
        
        Returns:
            delay: 预测的延迟向量 [n_paths]
        """
        # 构造完整的输入特征
        inputs = {
            'traffic': traffic,
            'capacities': features['capacities'],
            'links': features['links'],
            'paths': features['paths'],
            'sequences': features['sequences'],
            'n_links': features['n_links'],
            'n_paths': features['n_paths'],
            'packets': features.get('packets', tf.ones_like(traffic) * 1000.0)
        }
        
        # 模型预测
        predictions = self.model(inputs, training=False)
        
        if self.target == 'delay':
            # 延迟预测：取位置参数（均值）
            delay = predictions[:, 0]
        else:
            # 丢包预测：应用sigmoid激活
            delay = tf.nn.sigmoid(predictions[:, 0])
        
        return delay
    
    def compute_jacobian(self, sample_features):
        """
        计算雅可比矩阵 J = ∂D/∂T
        
        Args:
            sample_features: 网络样本特征字典，包含：
                - traffic: 流量向量 [n_paths]
                - capacities: 链路容量 [n_links]
                - links, paths, sequences: 网络拓扑信息
                - n_links, n_paths: 节点和路径数量
        
        Returns:
            jacobian: 雅可比矩阵 [n_paths, n_paths]
            delay_pred: 预测的延迟向量 [n_paths]
        """
        # 提取流量向量
        traffic = tf.convert_to_tensor(sample_features['traffic'], dtype=tf.float32)
        
        # 准备固定特征（除流量外的所有特征）
        fixed_features = {
            key: tf.convert_to_tensor(value) for key, value in sample_features.items()
            if key != 'traffic'
        }
        
        # 使用GradientTape计算雅可比矩阵
        with tf.GradientTape() as tape:
            # 监视流量变量
            tape.watch(traffic)
            
            # 预测延迟
            delay_pred = self._predict_delay(traffic, fixed_features)
        
        # 计算雅可比矩阵：每一行是一个路径延迟对所有流量的梯度
        jacobian = tape.jacobian(delay_pred, traffic)
        
        return jacobian.numpy(), delay_pred.numpy()
    
    def analyze_single_sample(self, sample_features, output_dir=None):
        """
        分析单个样本的雅可比矩阵
        
        Args:
            sample_features: 样本特征
            output_dir: 输出目录（可选）
        
        Returns:
            analysis_results: 分析结果字典
        """
        # 计算雅可比矩阵
        jacobian, delay_pred = self.compute_jacobian(sample_features)
        
        n_paths = len(delay_pred)
        
        # 分析结果
        analysis_results = {
            'jacobian': jacobian,
            'delay_prediction': delay_pred,
            'traffic': sample_features['traffic'],
            'n_paths': n_paths,
            # 雅可比矩阵的统计信息
            'diagonal_elements': np.diag(jacobian),  # 自影响：∂D_i/∂T_i
            'off_diagonal_elements': jacobian[~np.eye(n_paths, dtype=bool)],  # 互影响
            'max_gradient': np.max(np.abs(jacobian)),
            'mean_diagonal': np.mean(np.diag(jacobian)),
            'mean_off_diagonal': np.mean(jacobian[~np.eye(n_paths, dtype=bool)]),
        }
        
        # 可视化（如果指定了输出目录）
        if output_dir:
            self._visualize_jacobian(analysis_results, output_dir)
        
        return analysis_results
    
    def _visualize_jacobian(self, analysis_results, output_dir):
        """可视化雅可比矩阵"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体（如果可用）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        jacobian = analysis_results['jacobian']
        n_paths = analysis_results['n_paths']
        
        # 1. 雅可比矩阵热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(jacobian, annot=True, cmap='RdBu_r', center=0, 
                    fmt='.3f', square=True)
        model_type = "KAN Model" if self.use_kan else "MLP Model"
        plt.title(f'Jacobian Matrix ∂D/∂T ({model_type})')
        plt.xlabel('Traffic Input Path Index')
        plt.ylabel('Delay Output Path Index')
        plt.savefig(os.path.join(output_dir, 'jacobian_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 对角线元素 vs 非对角线元素
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(n_paths), analysis_results['diagonal_elements'])
        plt.title('Diagonal Elements (Self-influence)\n∂D_i/∂T_i')
        plt.xlabel('Path Index')
        plt.ylabel('Gradient Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(analysis_results['off_diagonal_elements'], bins=20, alpha=0.7)
        plt.title('Off-diagonal Elements Distribution (Cross-influence)\n∂D_i/∂T_j (i≠j)')
        plt.xlabel('Gradient Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'jacobian_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 保存数值结果
        with open(os.path.join(output_dir, 'jacobian_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("Jacobian Matrix Analysis Results\n")
            f.write("="*50 + "\n")
            f.write(f"Model Type: {'KAN' if self.use_kan else 'MLP'}\n")
            f.write(f"Number of Paths: {n_paths}\n")
            f.write(f"Max Gradient Value: {analysis_results['max_gradient']:.6f}\n")
            f.write(f"Mean Diagonal: {analysis_results['mean_diagonal']:.6f}\n")
            f.write(f"Mean Off-diagonal: {analysis_results['mean_off_diagonal']:.6f}\n")
            f.write("\nTraffic Input:\n")
            f.write(str(analysis_results['traffic']))
            f.write("\n\nDelay Prediction:\n")
            f.write(str(analysis_results['delay_prediction']))

def create_simple_network_sample():
    """
    创建一个简单的网络样本用于测试
    模拟一个4节点网络，两条路径共享中间链路
    
    Returns:
        sample_features: 网络样本特征字典
    """
    # 简单的4节点网络拓扑
    # 节点: 0-1-2-3
    # 路径1: 0->1->2 (链路: 0->1, 1->2)
    # 路径2: 1->2->3 (链路: 1->2, 2->3)
    # 共享链路: 1->2
    
    n_nodes = 4
    n_links = 3  # 0->1, 1->2, 2->3
    n_paths = 2  # 两条路径
    
    # 链路容量 (Mbps)
    capacities = np.array([100.0, 50.0, 100.0], dtype=np.float32)  # 中间链路容量较小
    
    # 路径的链路序列 - 注意数据类型
    # 路径0: 链路0, 链路1
    # 路径1: 链路1, 链路2
    links = np.array([0, 1, 1, 2], dtype=np.int32)  # 展平的链路索引
    paths = np.array([0, 0, 1, 1], dtype=np.int32)  # 对应的路径索引
    sequences = np.array([0, 1, 0, 1], dtype=np.int32)  # 路径内的序列位置
    
    # 初始流量
    traffic = np.array([20.0, 15.0], dtype=np.float32)  # 路径流量 (Mbps)
    
    # 数据包数量（用于某些损失函数）
    packets = np.array([2000.0, 1500.0], dtype=np.float32)
    
    sample_features = {
        'traffic': traffic,
        'capacities': capacities,
        'links': links,
        'paths': paths,
        'sequences': sequences,
        'n_links': n_links,
        'n_paths': n_paths,
        'packets': packets
    }
    
    return sample_features

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RouteNet雅可比矩阵分析')
    parser.add_argument('--model_dir', required=True, help='模型目录路径')
    parser.add_argument('--use_kan', action='store_true', help='使用KAN模型')
    parser.add_argument('--target', default='delay', choices=['delay', 'drops'], 
                       help='预测目标')
    parser.add_argument('--output_dir', default='jacobian_analysis', 
                       help='输出目录')
    parser.add_argument('--test_simple', action='store_true',
                       help='使用简单网络进行测试')
    
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
    
    # 创建分析器
    try:
        # 构造权重文件路径
        if args.use_kan:
            weight_file = os.path.join(args.model_dir, 'best_delay_kan_model.weights.h5')
        else:
            weight_file = os.path.join(args.model_dir, 'best_delay_model.weights.h5')
        
        analyzer = JacobianAnalyzer(
            model_path=weight_file,
            config=config,
            target=args.target,
            use_kan=args.use_kan
        )
        
        if args.test_simple:
            print("使用简单网络样本进行测试...")
            # 使用简单网络样本
            sample_features = create_simple_network_sample()
        else:
            print("需要提供真实网络样本...")
            # 这里应该从TFRecord文件加载真实样本
            # 暂时使用简单样本
            sample_features = create_simple_network_sample()
        
        print("计算雅可比矩阵...")
        analysis_results = analyzer.analyze_single_sample(
            sample_features, 
            output_dir=args.output_dir
        )
        
        print(f"\n分析完成！结果保存在: {args.output_dir}")
        print(f"路径数量: {analysis_results['n_paths']}")
        print(f"最大梯度值: {analysis_results['max_gradient']:.6f}")
        print(f"对角线平均值 (自影响): {analysis_results['mean_diagonal']:.6f}")
        print(f"非对角线平均值 (互影响): {analysis_results['mean_off_diagonal']:.6f}")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

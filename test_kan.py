#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN (Kolmogorov-Arnold Networks) 功能测试脚本
用于验证 KAN 层的实现是否正确
"""

import tensorflow as tf
import numpy as np
import os
import sys

# 添加路径以导入 routenet_tf2 模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'routenet'))
from routenet_tf2 import KANLayer, RouteNet, create_model_and_loss_fn

def test_kan_layer():
    """测试 KAN 层的基本功能"""
    print("=" * 60)
    print("测试 KAN 层基本功能")
    print("=" * 60)
    
    # 测试参数
    batch_size = 32
    input_dim = 8
    output_dim = 16
    
    # 创建 KAN 层
    kan_layer = KANLayer(units=output_dim, grid_size=5, spline_order=3)
    
    # 创建测试输入
    test_input = tf.random.normal((batch_size, input_dim))
    print(f"输入形状: {test_input.shape}")
    
    # 前向传播测试
    try:
        output = kan_layer(test_input, training=True)
        print(f"输出形状: {output.shape}")
        print(f"预期形状: ({batch_size}, {output_dim})")
        
        if output.shape == (batch_size, output_dim):
            print("✓ KAN 层形状测试通过")
        else:
            print("✗ KAN 层形状测试失败")
            return False
            
        # 检查输出数值是否合理
        if tf.reduce_all(tf.math.is_finite(output)):
            print("✓ KAN 层输出数值有效")
        else:
            print("✗ KAN 层输出包含无效数值 (NaN/Inf)")
            return False
            
    except Exception as e:
        print(f"✗ KAN 层前向传播失败: {e}")
        return False
    
    # 测试梯度计算
    try:
        with tf.GradientTape() as tape:
            output = kan_layer(test_input, training=True)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, kan_layer.trainable_variables)
        
        if all(grad is not None for grad in gradients):
            print("✓ KAN 层梯度计算正常")
        else:
            print("✗ KAN 层梯度计算失败")
            return False
            
    except Exception as e:
        print(f"✗ KAN 层梯度测试失败: {e}")
        return False
    
    print("✓ KAN 层基本功能测试通过\n")
    return True

def test_routenet_with_kan():
    """测试 RouteNet 与 KAN 的集成"""
    print("=" * 60)
    print("测试 RouteNet 与 KAN 集成")
    print("=" * 60)
    
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
    
    # 创建 KAN 版本的 RouteNet
    try:
        model, loss_fn = create_model_and_loss_fn(config, 'drops', use_kan=True)
        print("✓ KAN-RouteNet 模型创建成功")
    except Exception as e:
        print(f"✗ KAN-RouteNet 模型创建失败: {e}")
        return False
    
    # 创建模拟输入数据 - 修复数据类型问题
    n_links = 10
    n_paths = 20
    batch_features = {
        'traffic': tf.random.uniform((n_paths,), 0.1, 1.0),
        'capacities': tf.random.uniform((n_links,), 1.0, 10.0),
        'packets': tf.random.uniform((n_paths,), 100, 1000),
        'links': tf.random.uniform((n_paths * 3,), 0, n_links - 1, dtype=tf.int32),
        'paths': tf.random.uniform((n_paths * 3,), 0, n_paths - 1, dtype=tf.int32),
        'sequences': tf.random.uniform((n_paths * 3,), 0, 3, dtype=tf.int32),
        'n_links': tf.constant(n_links, dtype=tf.int64),  # 转换为张量
        'n_paths': tf.constant(n_paths, dtype=tf.int64),  # 转换为张量
    }
    
    batch_labels = {
        'delay': tf.random.uniform((n_paths,), 0.001, 0.1),
        'jitter': tf.random.uniform((n_paths,), 0.0001, 0.01),
        'drops': tf.cast(tf.random.uniform((n_paths,), 0, 10, dtype=tf.int32), tf.float32),  # 转换为float32
        'packets': batch_features['packets'],
    }
    
    # 测试前向传播
    try:
        predictions = model(batch_features, training=True)
        print(f"✓ KAN-RouteNet 前向传播成功，输出形状: {predictions.shape}")
        
        if predictions.shape[0] == n_paths and predictions.shape[1] == 1:  # drops模型输出1维
            print("✓ KAN-RouteNet 输出形状正确")
        else:
            print(f"✗ KAN-RouteNet 输出形状错误，预期: ({n_paths}, 1)")
            return False
            
    except Exception as e:
        print(f"✗ KAN-RouteNet 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试损失计算
    try:
        loss = loss_fn(batch_labels, predictions)
        print(f"✓ KAN-RouteNet 损失计算成功，损失值: {loss:.4f}")
        
        if tf.math.is_finite(loss):
            print("✓ KAN-RouteNet 损失值有效")
        else:
            print("✗ KAN-RouteNet 损失值无效")
            return False
            
    except Exception as e:
        print(f"✗ KAN-RouteNet 损失计算失败: {e}")
        return False
    
    # 测试梯度计算
    try:
        with tf.GradientTape() as tape:
            predictions = model(batch_features, training=True)
            loss = loss_fn(batch_labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        if all(grad is not None for grad in gradients):
            print("✓ KAN-RouteNet 梯度计算正常")
            
            # 检查梯度是否合理
            gradient_norms = [tf.norm(grad) for grad in gradients if grad is not None]
            max_grad_norm = max(gradient_norms)
            print(f"  最大梯度范数: {max_grad_norm:.6f}")
            
            if max_grad_norm < 100:  # 梯度不应该过大
                print("✓ KAN-RouteNet 梯度范数正常")
            else:
                print("⚠ KAN-RouteNet 梯度范数可能过大")
                
        else:
            print("✗ KAN-RouteNet 梯度计算失败")
            return False
            
    except Exception as e:
        print(f"✗ KAN-RouteNet 梯度测试失败: {e}")
        return False
    
    print("✓ KAN-RouteNet 集成测试通过\n")
    return True

def test_comparison_mlp_vs_kan():
    """比较 MLP 和 KAN 版本的 RouteNet"""
    print("=" * 60)
    print("比较 MLP 和 KAN 版本的 RouteNet")
    print("=" * 60)
    
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
    }
    
    # 创建两种模型
    try:
        mlp_model, mlp_loss_fn = create_model_and_loss_fn(config, 'drops', use_kan=False)
        kan_model, kan_loss_fn = create_model_and_loss_fn(config, 'drops', use_kan=True)
        print("✓ MLP 和 KAN 模型都创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False
    
    # 创建相同的测试数据 - 修复数据类型问题
    n_links = 10
    n_paths = 20
    batch_features = {
        'traffic': tf.random.uniform((n_paths,), 0.1, 1.0),
        'capacities': tf.random.uniform((n_links,), 1.0, 10.0),
        'packets': tf.random.uniform((n_paths,), 100, 1000),
        'links': tf.random.uniform((n_paths * 3,), 0, n_links - 1, dtype=tf.int32),
        'paths': tf.random.uniform((n_paths * 3,), 0, n_paths - 1, dtype=tf.int32),
        'sequences': tf.random.uniform((n_paths * 3,), 0, 3, dtype=tf.int32),
        'n_links': tf.constant(n_links, dtype=tf.int64),  # 转换为张量
        'n_paths': tf.constant(n_paths, dtype=tf.int64),  # 转换为张量
    }
    
    # 测试两个模型
    try:
        mlp_pred = mlp_model(batch_features, training=False)
        kan_pred = kan_model(batch_features, training=False)
        
        print(f"✓ MLP 输出形状: {mlp_pred.shape}")
        print(f"✓ KAN 输出形状: {kan_pred.shape}")
        
        # 统计参数数量
        mlp_params = sum([tf.size(var) for var in mlp_model.trainable_variables])
        kan_params = sum([tf.size(var) for var in kan_model.trainable_variables])
        
        print(f"  MLP 可训练参数数量: {mlp_params}")
        print(f"  KAN 可训练参数数量: {kan_params}")
        print(f"  参数数量比例 (KAN/MLP): {kan_params/mlp_params:.2f}")
        
    except Exception as e:
        print(f"✗ 模型比较失败: {e}")
        return False
    
    print("✓ MLP 和 KAN 模型比较完成\n")
    return True

def main():
    """主测试函数"""
    print("开始 KAN 功能测试...")
    print("TensorFlow 版本:", tf.__version__)
    print()
    
    tests = [
        test_kan_layer,
        test_routenet_with_kan,
        test_comparison_mlp_vs_kan,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试 {test_func.__name__} 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # 总结
    print("=" * 60)
    print("测试结果总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "KAN 层基本功能",
        "RouteNet-KAN 集成",
        "MLP vs KAN 比较"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print(f"\n通过率: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！KAN 功能实现正确。")
        return True
    else:
        print("⚠️  部分测试失败，需要修复 KAN 实现。")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

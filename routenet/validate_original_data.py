# -*- coding: utf-8 -*-
"""
原始数据验证脚本
验证TFRecord训练数据中traffic和delay的关系
这是最基础也是最重要的验证步骤！
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd

def parse_tfrecord_example(example_proto):
    """解析TFRecord中的单个样本"""
    feature_description = {
        'traffic': tf.io.VarLenFeature(tf.float32),
        'delay': tf.io.VarLenFeature(tf.float32),
        'jitter': tf.io.VarLenFeature(tf.float32),
        'drops': tf.io.VarLenFeature(tf.float32),
        'packets': tf.io.VarLenFeature(tf.float32),
        'logdelay': tf.io.VarLenFeature(tf.float32),
        'links': tf.io.VarLenFeature(tf.int64),
        'paths': tf.io.VarLenFeature(tf.int64),
        'sequences': tf.io.VarLenFeature(tf.int64),
        'n_links': tf.io.FixedLenFeature([], tf.int64),
        'n_paths': tf.io.FixedLenFeature([], tf.int64),
        'n_total': tf.io.FixedLenFeature([], tf.int64),
        'capacities': tf.io.VarLenFeature(tf.float32)
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # 转换稀疏张量为稠密张量
    for key in ['traffic', 'delay', 'jitter', 'drops', 'packets', 'logdelay', 'links', 'paths', 'sequences', 'capacities']:
        parsed_features[key] = tf.sparse.to_dense(parsed_features[key])
    
    return parsed_features

def load_random_samples(tfrecord_dir, num_samples=500, max_files=5):
    """从TFRecord文件中随机加载样本"""
    
    # 查找所有TFRecord文件
    tfrecord_files = glob.glob(os.path.join(tfrecord_dir, "*.tfrecords"))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")
    
    # 限制文件数量以避免加载过多数据
    if len(tfrecord_files) > max_files:
        tfrecord_files = tfrecord_files[:max_files]
    
    print(f"Found {len(tfrecord_files)} TFRecord files, using first {len(tfrecord_files)}")
    
    # 创建数据集
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_example)
    
    # 随机采样
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.take(num_samples)
    
    samples = []
    print("Loading samples from TFRecord files...")
    
    for sample in tqdm(dataset, desc="Loading", total=num_samples):
        # 提取numpy数组
        sample_dict = {
            'traffic': sample['traffic'].numpy(),
            'delay': sample['delay'].numpy(),
            'jitter': sample['jitter'].numpy(),
            'drops': sample['drops'].numpy(),
            'packets': sample['packets'].numpy(),
            'capacities': sample['capacities'].numpy(),
            'n_paths': sample['n_paths'].numpy(),
            'n_links': sample['n_links'].numpy()
        }
        samples.append(sample_dict)
    
    print(f"Loaded {len(samples)} samples successfully")
    return samples

def analyze_traffic_delay_relationship(samples, output_dir="data_validation"):
    """分析traffic和delay的关系"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图形样式
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    print("Analyzing traffic-delay relationship...")
    
    # 收集所有traffic和delay数据点
    all_traffic = []
    all_delay = []
    all_capacity = []  # 对应路径的容量信息
    path_info = []  # 路径信息用于分析
    
    for i, sample in enumerate(samples):
        traffic = sample['traffic']
        delay = sample['delay']
        capacities = sample['capacities']
        n_paths = sample['n_paths']
        
        # 每个样本可能有多条路径
        for path_idx in range(n_paths):
            if path_idx < len(traffic) and path_idx < len(delay):
                all_traffic.append(traffic[path_idx])
                all_delay.append(delay[path_idx])
                # 这里简化处理容量信息
                avg_capacity = np.mean(capacities) if len(capacities) > 0 else 100.0
                all_capacity.append(avg_capacity)
                path_info.append({'sample_id': i, 'path_id': path_idx, 'capacity': avg_capacity})
    
    all_traffic = np.array(all_traffic)
    all_delay = np.array(all_delay)
    all_capacity = np.array(all_capacity)
    
    print(f"Total data points: {len(all_traffic)}")
    print(f"Traffic range: [{np.min(all_traffic):.3f}, {np.max(all_traffic):.3f}]")
    print(f"Delay range: [{np.min(all_delay):.6f}, {np.max(all_delay):.6f}]")
    
    # 基础统计分析
    correlation = np.corrcoef(all_traffic, all_delay)[0, 1]
    print(f"Traffic-Delay correlation coefficient: {correlation:.4f}")
    
    # 创建综合分析图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 主散点图：Traffic vs Delay
    ax1 = axes[0, 0]
    scatter = ax1.scatter(all_traffic, all_delay, alpha=0.6, s=20, c=all_capacity, cmap='viridis')
    ax1.set_xlabel('Traffic (Mbps)')
    ax1.set_ylabel('Delay (seconds)')
    ax1.set_title(f'Traffic vs Delay Relationship\nCorrelation: {correlation:.4f}')
    plt.colorbar(scatter, ax=ax1, label='Avg Link Capacity')
    
    # 添加趋势线
    z = np.polyfit(all_traffic, all_delay, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(np.min(all_traffic), np.max(all_traffic), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope={z[0]:.6f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Traffic分布直方图
    ax2 = axes[0, 1]
    ax2.hist(all_traffic, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Traffic (Mbps)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Traffic Distribution')
    ax2.axvline(np.mean(all_traffic), color='red', linestyle='--', label=f'Mean: {np.mean(all_traffic):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Delay分布直方图
    ax3 = axes[0, 2]
    ax3.hist(all_delay, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Delay (seconds)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Delay Distribution')
    ax3.axvline(np.mean(all_delay), color='red', linestyle='--', label=f'Mean: {np.mean(all_delay):.6f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 按流量区间分析延迟
    ax4 = axes[1, 0]
    # 将traffic分成几个区间
    traffic_bins = np.percentile(all_traffic, [0, 20, 40, 60, 80, 100])
    traffic_binned = np.digitize(all_traffic, traffic_bins) - 1
    
    delay_by_traffic_bin = []
    bin_labels = []
    for i in range(len(traffic_bins)-1):
        mask = traffic_binned == i
        if np.sum(mask) > 0:
            delay_by_traffic_bin.append(all_delay[mask])
            bin_labels.append(f'{traffic_bins[i]:.1f}-{traffic_bins[i+1]:.1f}')
    
    ax4.boxplot(delay_by_traffic_bin, labels=bin_labels)
    ax4.set_xlabel('Traffic Bins (Mbps)')
    ax4.set_ylabel('Delay (seconds)')
    ax4.set_title('Delay Distribution by Traffic Bins')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. 对数坐标下的关系
    ax5 = axes[1, 1]
    # 过滤掉非正值
    positive_traffic = all_traffic[all_traffic > 0]
    positive_delay = all_delay[all_traffic > 0]
    positive_delay = positive_delay[positive_delay > 0]
    positive_traffic = positive_traffic[positive_delay > 0]
    
    if len(positive_traffic) > 0:
        ax5.loglog(positive_traffic, positive_delay, 'o', alpha=0.6, markersize=3)
        ax5.set_xlabel('Traffic (Mbps) - Log Scale')
        ax5.set_ylabel('Delay (seconds) - Log Scale')
        ax5.set_title('Log-Log Traffic vs Delay')
        ax5.grid(True, alpha=0.3)
    
    # 6. 样本数据统计表
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats_text = f"""
    Data Statistics:
    
    Total Samples: {len(samples)}
    Total Data Points: {len(all_traffic)}
    
    Traffic Statistics:
      Mean: {np.mean(all_traffic):.3f} Mbps
      Std: {np.std(all_traffic):.3f} Mbps
      Min: {np.min(all_traffic):.3f} Mbps
      Max: {np.max(all_traffic):.3f} Mbps
    
    Delay Statistics:
      Mean: {np.mean(all_delay):.6f} sec
      Std: {np.std(all_delay):.6f} sec
      Min: {np.min(all_delay):.6f} sec
      Max: {np.max(all_delay):.6f} sec
    
    Correlation Analysis:
      Pearson Correlation: {correlation:.4f}
      
    Physical Expectation:
      Expected: Positive correlation
      Actual: {'✓ Positive' if correlation > 0.1 else '✗ Negative' if correlation < -0.1 else '~ Neutral'}
    
    Trend Analysis:
      Slope: {z[0]:.6f}
      Interpretation: {'✓ Delay increases with traffic' if z[0] > 0 else '✗ Delay decreases with traffic' if z[0] < 0 else '~ No clear trend'}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_delay_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存详细的数值分析结果
    with open(os.path.join(output_dir, 'data_validation_report.txt'), 'w') as f:
        f.write("Original Data Validation Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. Dataset Overview:\n")
        f.write(f"   Total TFRecord samples loaded: {len(samples)}\n")
        f.write(f"   Total (traffic, delay) data points: {len(all_traffic)}\n")
        f.write(f"   Average paths per sample: {len(all_traffic)/len(samples):.2f}\n\n")
        
        f.write("2. Traffic Analysis:\n")
        f.write(f"   Mean traffic: {np.mean(all_traffic):.3f} Mbps\n")
        f.write(f"   Standard deviation: {np.std(all_traffic):.3f} Mbps\n")
        f.write(f"   Range: [{np.min(all_traffic):.3f}, {np.max(all_traffic):.3f}] Mbps\n")
        f.write(f"   25th percentile: {np.percentile(all_traffic, 25):.3f} Mbps\n")
        f.write(f"   50th percentile: {np.percentile(all_traffic, 50):.3f} Mbps\n")
        f.write(f"   75th percentile: {np.percentile(all_traffic, 75):.3f} Mbps\n\n")
        
        f.write("3. Delay Analysis:\n")
        f.write(f"   Mean delay: {np.mean(all_delay):.6f} seconds\n")
        f.write(f"   Standard deviation: {np.std(all_delay):.6f} seconds\n")
        f.write(f"   Range: [{np.min(all_delay):.6f}, {np.max(all_delay):.6f}] seconds\n")
        f.write(f"   25th percentile: {np.percentile(all_delay, 25):.6f} seconds\n")
        f.write(f"   50th percentile: {np.percentile(all_delay, 50):.6f} seconds\n")
        f.write(f"   75th percentile: {np.percentile(all_delay, 75):.6f} seconds\n\n")
        
        f.write("4. Traffic-Delay Relationship:\n")
        f.write(f"   Pearson correlation coefficient: {correlation:.6f}\n")
        f.write(f"   Linear trend slope: {z[0]:.8f}\n")
        f.write(f"   Linear trend intercept: {z[1]:.8f}\n\n")
        
        f.write("5. Physical Intuition Check:\n")
        if correlation > 0.1:
            f.write("   ✓ PASS: Positive correlation between traffic and delay\n")
            f.write("   ✓ This matches physical expectation\n")
        elif correlation < -0.1:
            f.write("   ✗ FAIL: Negative correlation between traffic and delay\n") 
            f.write("   ✗ This contradicts physical expectation\n")
        else:
            f.write("   ~ NEUTRAL: Very weak correlation\n")
            f.write("   ~ May indicate data quality issues\n")
        
        f.write(f"\n6. Detailed Correlation Analysis:\n")
        f.write(f"   If correlation > 0.3: Strong positive relationship ✓\n")
        f.write(f"   If correlation > 0.1: Weak positive relationship ✓\n")
        f.write(f"   If -0.1 < correlation < 0.1: No clear relationship ~\n")
        f.write(f"   If correlation < -0.1: Negative relationship (problematic) ✗\n")
        f.write(f"   \n")
        f.write(f"   Actual correlation: {correlation:.6f}\n")
        
        if correlation > 0.3:
            f.write("   Result: ✓ STRONG POSITIVE - Data quality is good for training\n")
        elif correlation > 0.1:
            f.write("   Result: ✓ WEAK POSITIVE - Data is usable but may have noise\n")
        elif correlation > -0.1:
            f.write("   Result: ~ NEUTRAL - Data quality questionable, investigate further\n")
        else:
            f.write("   Result: ✗ NEGATIVE - Data has serious quality issues!\n")
        
        # 样本详细数据（前100个点）
        f.write(f"\n7. Sample Data Points (first 100):\n")
        f.write("-" * 50 + "\n")
        f.write("Traffic(Mbps)  Delay(seconds)  Capacity(Mbps)\n")
        f.write("-" * 50 + "\n")
        for i in range(min(100, len(all_traffic))):
            f.write(f"{all_traffic[i]:11.3f}  {all_delay[i]:13.8f}  {all_capacity[i]:12.3f}\n")
    
    print(f"\nValidation complete! Results saved to {output_dir}")
    print(f"Key findings:")
    print(f"  - Traffic-Delay correlation: {correlation:.4f}")
    print(f"  - Trend slope: {z[0]:.6f}")
    
    if correlation > 0.1:
        print(f"  ✓ Data shows expected positive correlation")
    else:
        print(f"  ✗ WARNING: Data shows unexpected relationship!")
    
    return {
        'correlation': correlation,
        'slope': z[0],
        'traffic_mean': np.mean(all_traffic),
        'delay_mean': np.mean(all_delay),
        'num_points': len(all_traffic)
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Validate original TFRecord training data')
    parser.add_argument('--data_dir', required=True, help='TFRecord data directory')
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples to analyze')
    parser.add_argument('--max_files', type=int, default=3, help='Maximum TFRecord files to load')
    parser.add_argument('--output_dir', default='data_validation', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ORIGINAL DATA VALIDATION")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Max files to process: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # 加载样本数据
        samples = load_random_samples(args.data_dir, args.num_samples, args.max_files)
        
        # 分析traffic-delay关系
        results = analyze_traffic_delay_relationship(samples, args.output_dir)
        
        # 最终判断
        print("\n" + "="*60)
        print("FINAL ASSESSMENT")
        print("="*60)
        
        correlation = results['correlation']
        if correlation > 0.3:
            print("🎉 EXCELLENT: Strong positive correlation found!")
            print("   Your training data is high quality for learning traffic-delay relationships.")
        elif correlation > 0.1:
            print("✅ GOOD: Positive correlation detected.")
            print("   Your training data should work for model training.")
        elif correlation > -0.1:
            print("⚠️  WARNING: Very weak correlation.")
            print("   Model may struggle to learn meaningful relationships.")
        else:
            print("🚨 CRITICAL ISSUE: Negative correlation detected!")
            print("   This contradicts physical expectations. Check data processing pipeline!")
            
        print(f"\nCorrelation coefficient: {correlation:.6f}")
        print(f"Linear trend slope: {results['slope']:.8f}")
        
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果汇总分析器
汇总多组实验的结果并生成对比报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml
import re

class ResultSummarizer:
    """结果汇总器"""
    
    def __init__(self, config_file="experiment_config.yaml", results_dir="experiment_results"):
        self.config_file = config_file
        self.results_dir = results_dir
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self):
        """加载配置文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def collect_results(self):
        """收集所有实验结果"""
        print("📊 正在收集实验结果...")
        
        for model_name in self.config['models'].keys():
            model_dir = os.path.join(self.results_dir, model_name)
            if not os.path.exists(model_dir):
                continue
                
            self.results[model_name] = {}
            
            # 收集各种实验结果
            for exp_type in ['evaluate', 'gradient', 'numerical']:
                exp_dir = os.path.join(model_dir, exp_type)
                if os.path.exists(exp_dir):
                    self.results[model_name][exp_type] = self.parse_experiment_results(exp_dir, exp_type)
        
        print(f"✅ 收集到 {len(self.results)} 个模型的结果")
    
    def parse_experiment_results(self, exp_dir, exp_type):
        """解析单个实验结果"""
        result = {}
        
        try:
            # 读取实验日志
            log_file = os.path.join(exp_dir, "experiment_log.json")
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    result['duration'] = log_data.get('duration', 0)
                    result['success'] = log_data.get('return_code', -1) == 0
            
            # 根据实验类型解析特定结果
            if exp_type == "evaluate":
                result.update(self.parse_evaluate_results(exp_dir))
            elif exp_type == "gradient":
                result.update(self.parse_gradient_results(exp_dir))
            elif exp_type == "numerical":
                result.update(self.parse_numerical_results(exp_dir))
                
        except Exception as e:
            print(f"⚠️  解析结果失败 {exp_dir}: {e}")
            result['error'] = str(e)
        
        return result
    
    def parse_evaluate_results(self, exp_dir):
        """解析评估结果"""
        result = {}
        
        # 查找评估结果文件
        for file in os.listdir(exp_dir):
            if file.endswith('.png'):
                result['plot_file'] = os.path.join(exp_dir, file)
        
        # 尝试从日志中提取数值结果
        log_file = os.path.join(exp_dir, "experiment_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                stdout = log_data.get('stdout', '')
                
                # 提取NSFNet和GBN的性能指标
                nsfnet_stats = self.extract_performance_stats(stdout, "NSFNet")
                gbn_stats = self.extract_performance_stats(stdout, "GBN")
                
                result['nsfnet'] = nsfnet_stats
                result['gbn'] = gbn_stats
        
        return result
    
    def parse_gradient_results(self, exp_dir):
        """解析梯度验证结果"""
        result = {}
        
        # 查找结果文件
        for file in os.listdir(exp_dir):
            if file.startswith('sanity_check_results_'):
                result_file = os.path.join(exp_dir, file)
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # 提取物理直觉得分
                        score_match = re.search(r'Overall Physical Intuition Score: ([\d.]+)%', content)
                        if score_match:
                            result['physics_score'] = float(score_match.group(1)) / 100
                        
                        # 提取各项验证结果
                        result['self_gradient_pass'] = 'Pass' in content and 'Self-influence Gradient > 0: Pass' in content
                        result['cross_gradient_pass'] = 'Cross-influence Gradient > 0: Pass' in content
                        result['monotonic_pass'] = 'Delay Monotonic Increase: Pass' in content
                        result['congestion_pass'] = 'Congestion Sensitivity: Pass' in content
                        
                except Exception as e:
                    print(f"⚠️  解析梯度结果失败: {e}")
        
        return result
    
    def parse_numerical_results(self, exp_dir):
        """解析数值分析结果"""
        result = {}
        
        # 查找CSV结果文件
        csv_file = os.path.join(exp_dir, "detailed_metrics.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                
                # 提取关键指标
                if 'MAE' in df.columns:
                    result['mae_nsfnet_delay'] = df[df['Dataset'] == 'NSFNet'][df['Metric'] == 'Delay']['MAE'].iloc[0]
                    result['mae_gbn_delay'] = df[df['Dataset'] == 'GBN'][df['Metric'] == 'Delay']['MAE'].iloc[0]
                    result['rmse_nsfnet_delay'] = df[df['Dataset'] == 'NSFNet'][df['Metric'] == 'Delay']['RMSE'].iloc[0]
                    result['rmse_gbn_delay'] = df[df['Dataset'] == 'GBN'][df['Metric'] == 'Delay']['RMSE'].iloc[0]
                    result['mape_nsfnet_delay'] = df[df['Dataset'] == 'NSFNet'][df['Metric'] == 'Delay']['MAPE'].iloc[0]
                    result['mape_gbn_delay'] = df[df['Dataset'] == 'GBN'][df['Metric'] == 'Delay']['MAPE'].iloc[0]
                    
            except Exception as e:
                print(f"⚠️  解析数值结果失败: {e}")
        
        return result
    
    def extract_performance_stats(self, text, dataset_name):
        """从文本中提取性能统计"""
        stats = {}
        
        # 查找对应数据集的部分
        pattern = f"{dataset_name}.*?:" + r"(.*?)(?=\n\n|\n[A-Z]|$)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            section = match.group(1)
            
            # 提取延迟指标
            delay_pattern = r"DELAY:.*?Mean Abs Error: ([\d.]+).*?Median Abs Error: ([\d.]+).*?P90 Abs Error: ([\d.]+).*?P95 Abs Error: ([\d.]+)"
            delay_match = re.search(delay_pattern, section, re.DOTALL)
            if delay_match:
                stats['delay_mae'] = float(delay_match.group(1))
                stats['delay_median'] = float(delay_match.group(2))
                stats['delay_p90'] = float(delay_match.group(3))
                stats['delay_p95'] = float(delay_match.group(4))
            
            # 提取抖动指标
            jitter_pattern = r"JITTER:.*?Mean Abs Error: ([\d.]+).*?Median Abs Error: ([\d.]+).*?P90 Abs Error: ([\d.]+).*?P95 Abs Error: ([\d.]+)"
            jitter_match = re.search(jitter_pattern, section, re.DOTALL)
            if jitter_match:
                stats['jitter_mae'] = float(jitter_match.group(1))
                stats['jitter_median'] = float(jitter_match.group(2))
                stats['jitter_p90'] = float(jitter_match.group(3))
                stats['jitter_p95'] = float(jitter_match.group(4))
        
        return stats
    
    def create_comparison_table(self):
        """创建对比表格"""
        print("📋 正在生成对比表格...")
        
        # 准备数据
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            model_config = self.config['models'][model_name]
            
            row = {
                'Model': model_name,
                'Type': model_config['model_type'].upper(),
                'Physics': model_config['physics_type'].title(),
                'Lambda': model_config['lambda_physics'],
            }
            
            # 添加评估结果
            if 'evaluate' in model_results:
                eval_data = model_results['evaluate']
                if 'nsfnet' in eval_data:
                    row.update({
                        'NSFNet_Delay_MAE': eval_data['nsfnet'].get('delay_mae', np.nan),
                        'NSFNet_Delay_P95': eval_data['nsfnet'].get('delay_p95', np.nan),
                        'NSFNet_Jitter_MAE': eval_data['nsfnet'].get('jitter_mae', np.nan),
                    })
                if 'gbn' in eval_data:
                    row.update({
                        'GBN_Delay_MAE': eval_data['gbn'].get('delay_mae', np.nan),
                        'GBN_Delay_P95': eval_data['gbn'].get('delay_p95', np.nan),
                        'GBN_Jitter_MAE': eval_data['gbn'].get('jitter_mae', np.nan),
                    })
            
            # 添加梯度验证结果
            if 'gradient' in model_results:
                grad_data = model_results['gradient']
                row.update({
                    'Physics_Score': grad_data.get('physics_score', np.nan),
                    'Self_Gradient_Pass': grad_data.get('self_gradient_pass', False),
                    'Cross_Gradient_Pass': grad_data.get('cross_gradient_pass', False),
                    'Monotonic_Pass': grad_data.get('monotonic_pass', False),
                })
            
            # 添加数值分析结果
            if 'numerical' in model_results:
                num_data = model_results['numerical']
                row.update({
                    'NSFNet_MAE': num_data.get('mae_nsfnet_delay', np.nan),
                    'GBN_MAE': num_data.get('mae_gbn_delay', np.nan),
                    'NSFNet_RMSE': num_data.get('rmse_nsfnet_delay', np.nan),
                    'GBN_RMSE': num_data.get('rmse_gbn_delay', np.nan),
                })
            
            comparison_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        
        # 保存为CSV
        summary_dir = os.path.join(self.results_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        csv_file = os.path.join(summary_dir, "model_comparison.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"✅ 对比表格保存在: {csv_file}")
        
        # 显示最佳模型
        self.find_best_models(df, summary_dir)
        
        return df
    
    def find_best_models(self, df, summary_dir):
        """找出最佳模型"""
        print("\n🏆 寻找最佳模型...")
        
        best_models = {}
        
        # 按不同指标找最佳模型
        metrics = [
            ('NSFNet_Delay_MAE', '最低NSFNet延迟MAE'),
            ('GBN_Delay_MAE', '最低GBN延迟MAE'),
            ('Physics_Score', '最高物理直觉得分'),
            ('NSFNet_MAE', '最低NSFNet数值MAE'),
            ('GBN_MAE', '最低GBN数值MAE')
        ]
        
        best_summary = []
        
        for metric, description in metrics:
            if metric in df.columns:
                if 'MAE' in metric or 'RMSE' in metric:
                    # 越小越好
                    best_idx = df[metric].idxmin()
                    best_value = df.loc[best_idx, metric]
                    best_model = df.loc[best_idx, 'Model']
                else:
                    # 越大越好
                    best_idx = df[metric].idxmax()
                    best_value = df.loc[best_idx, metric]
                    best_model = df.loc[best_idx, 'Model']
                
                best_models[metric] = (best_model, best_value)
                best_summary.append(f"{description}: {best_model} ({best_value:.4f})")
                print(f"   {description}: {best_model} ({best_value:.4f})")
        
        # 保存最佳模型总结
        best_file = os.path.join(summary_dir, "best_models.txt")
        with open(best_file, 'w', encoding='utf-8') as f:
            f.write("最佳模型总结\n")
            f.write("=" * 50 + "\n\n")
            for summary in best_summary:
                f.write(summary + "\n")
        
        print(f"✅ 最佳模型总结保存在: {best_file}")
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("📊 正在生成可视化图表...")
        
        summary_dir = os.path.join(self.results_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 创建性能对比图
        self.plot_performance_comparison(summary_dir)
        
        # 创建物理约束效果图
        self.plot_physics_effect(summary_dir)
        
        print("✅ 可视化图表生成完成")
    
    def plot_performance_comparison(self, output_dir):
        """绘制性能对比图"""
        # 准备数据
        data_for_plot = []
        
        for model_name, model_results in self.results.items():
            model_config = self.config['models'][model_name]
            
            if 'evaluate' in model_results:
                eval_data = model_results['evaluate']
                
                # NSFNet数据
                if 'nsfnet' in eval_data:
                    data_for_plot.append({
                        'Model': model_name,
                        'Type': model_config['model_type'].upper(),
                        'Physics': model_config['physics_type'],
                        'Lambda': model_config['lambda_physics'],
                        'Dataset': 'NSFNet',
                        'Delay_MAE': eval_data['nsfnet'].get('delay_mae', np.nan),
                        'Jitter_MAE': eval_data['nsfnet'].get('jitter_mae', np.nan)
                    })
                
                # GBN数据
                if 'gbn' in eval_data:
                    data_for_plot.append({
                        'Model': model_name,
                        'Type': model_config['model_type'].upper(),
                        'Physics': model_config['physics_type'],
                        'Lambda': model_config['lambda_physics'],
                        'Dataset': 'GBN',
                        'Delay_MAE': eval_data['gbn'].get('delay_mae', np.nan),
                        'Jitter_MAE': eval_data['gbn'].get('jitter_mae', np.nan)
                    })
        
        if not data_for_plot:
            print("⚠️  没有足够的数据生成性能对比图")
            return
        
        df_plot = pd.DataFrame(data_for_plot)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 延迟MAE对比
        sns.boxplot(data=df_plot, x='Type', y='Delay_MAE', hue='Physics', ax=axes[0,0])
        axes[0,0].set_title('Delay MAE Comparison')
        axes[0,0].set_ylabel('Mean Absolute Error')
        
        # 抖动MAE对比
        sns.boxplot(data=df_plot, x='Type', y='Jitter_MAE', hue='Physics', ax=axes[0,1])
        axes[0,1].set_title('Jitter MAE Comparison')
        axes[0,1].set_ylabel('Mean Absolute Error')
        
        # Lambda参数效果
        sns.scatterplot(data=df_plot, x='Lambda', y='Delay_MAE', hue='Type', style='Physics', ax=axes[1,0])
        axes[1,0].set_title('Lambda Effect on Delay MAE')
        axes[1,0].set_xlabel('Lambda Parameter')
        axes[1,0].set_ylabel('Delay MAE')
        
        # 数据集间性能对比
        sns.boxplot(data=df_plot, x='Dataset', y='Delay_MAE', hue='Type', ax=axes[1,1])
        axes[1,1].set_title('Cross-topology Performance')
        axes[1,1].set_ylabel('Delay MAE')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_physics_effect(self, output_dir):
        """绘制物理约束效果图"""
        # 准备物理约束数据
        physics_data = []
        
        for model_name, model_results in self.results.items():
            model_config = self.config['models'][model_name]
            
            if 'gradient' in model_results:
                grad_data = model_results['gradient']
                
                physics_data.append({
                    'Model': model_name,
                    'Type': model_config['model_type'].upper(),
                    'Physics': model_config['physics_type'],
                    'Lambda': model_config['lambda_physics'],
                    'Physics_Score': grad_data.get('physics_score', np.nan)
                })
        
        if not physics_data:
            print("⚠️  没有足够的数据生成物理约束效果图")
            return
        
        df_physics = pd.DataFrame(physics_data)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 物理直觉得分对比
        sns.boxplot(data=df_physics, x='Type', y='Physics_Score', hue='Physics', ax=axes[0])
        axes[0].set_title('Physics Intuition Score Comparison')
        axes[0].set_ylabel('Physics Score')
        
        # Lambda参数对物理得分的影响
        sns.scatterplot(data=df_physics, x='Lambda', y='Physics_Score', hue='Type', style='Physics', ax=axes[1])
        axes[1].set_title('Lambda Parameter Effect on Physics Score')
        axes[1].set_xlabel('Lambda Parameter')
        axes[1].set_ylabel('Physics Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'physics_effect.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """生成完整报告"""
        print("\n📊 正在生成完整实验报告...")
        
        # 收集结果
        self.collect_results()
        
        if not self.results:
            print("❌ 没有找到任何实验结果")
            return
        
        # 生成对比表格
        df = self.create_comparison_table()
        
        # 创建可视化图表
        self.create_visualizations()
        
        print(f"\n✅ 实验报告生成完成！")
        print(f"📁 结果保存在: {self.results_dir}/summary/")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实验结果汇总分析器')
    parser.add_argument('--config', default='experiment_config.yaml', help='配置文件路径')
    parser.add_argument('--results_dir', default='experiment_results', help='结果目录')
    
    args = parser.parse_args()
    
    try:
        summarizer = ResultSummarizer(args.config, args.results_dir)
        summarizer.generate_report()
        
    except Exception as e:
        print(f"💥 汇总失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

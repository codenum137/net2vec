#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化实验运行器
基于配置文件运行多组RouteNet实验
"""

import yaml
import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config_file="experiment_config.yaml"):
        """初始化实验运行器"""
        self.config_file = config_file
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 已加载配置文件: {self.config_file}")
        print(f"📋 发现 {len(config['models'])} 个模型配置")
        print(f"🧪 发现 {len(config['experiments'])} 种实验类型")
        
        return config
    
    def validate_model_paths(self):
        """验证模型路径是否存在"""
        missing_models = []
        
        for model_name, model_config in self.config['models'].items():
            model_dir = model_config['delay_model_dir']
            if not os.path.exists(model_dir):
                missing_models.append(f"{model_name}: {model_dir}")
        
        if missing_models:
            print("⚠️  以下模型路径不存在:")
            for missing in missing_models:
                print(f"   - {missing}")
            return False
        
        print("✅ 所有模型路径验证通过")
        return True
    
    def build_command(self, experiment_type, model_name, model_config):
        """构建实验命令"""
        exp_config = self.config['experiments'][experiment_type]
        global_settings = self.config['global_settings']
        
        # 基础命令
        script_path = exp_config['script']
        cmd = ["python", script_path]
        
        # 输出目录
        output_dir = os.path.join(
            global_settings['base_output_dir'],
            model_name,
            experiment_type
        )
        
        # 构建参数
        if experiment_type == "evaluate":
            cmd.extend([
                "--delay_model_dir", model_config['delay_model_dir'],
                "--nsfnet_test_dir", global_settings['nsfnet_test_dir'],
                "--gbn_test_dir", global_settings['gbn_test_dir'],
                "--output_dir", output_dir,
                "--batch_size", str(global_settings['batch_size']),
                "--num_samples", str(global_settings['num_samples'])
            ])
            if model_config['use_kan']:
                cmd.append("--kan")
                
        elif experiment_type == "gradient":
            cmd.extend([
                "--model_dir", model_config['delay_model_dir'],
                "--output_dir", output_dir,
                "--target", "delay",
                "--traffic_min", str(global_settings['traffic_min']),
                "--traffic_max", str(global_settings['traffic_max']),
                "--num_points", str(global_settings['num_points'])
            ])
            if model_config['use_kan']:
                cmd.append("--use_kan")
                
        elif experiment_type == "numerical":
            cmd.extend([
                "--model_dir", model_config['delay_model_dir'],
                "--nsfnet_test_dir", global_settings['nsfnet_test_dir'],
                "--gbn_test_dir", global_settings['gbn_test_dir'],
                "--output_dir", output_dir,
                "--batch_size", str(global_settings['batch_size']),
                "--num_samples", str(global_settings['num_samples'])
            ])
            if model_config['use_kan']:
                cmd.append("--kan")
        
        return cmd, output_dir
    
    def run_single_experiment(self, experiment_type, model_name, model_config):
        """运行单个实验"""
        try:
            cmd, output_dir = self.build_command(experiment_type, model_name, model_config)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 记录开始时间
            start_time = time.time()
            
            print(f"🚀 开始实验: {model_name} - {experiment_type}")
            print(f"📁 输出目录: {output_dir}")
            print(f"⚡ 命令: {' '.join(cmd)}")
            
            # 运行实验
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            # 计算运行时间
            duration = time.time() - start_time
            
            # 保存结果
            experiment_result = {
                'model_name': model_name,
                'experiment_type': experiment_type,
                'command': ' '.join(cmd),
                'output_dir': output_dir,
                'duration': duration,
                'return_code': result.returncode,
                'stdout_lines': result.stdout.split('\n') if result.stdout else [],
                'stderr_lines': result.stderr.split('\n') if result.stderr else [],
                'timestamp': datetime.now().isoformat()
            }
            
            if result.returncode == 0:
                print(f"✅ 实验成功: {model_name} - {experiment_type} (耗时: {duration:.1f}s)")
            else:
                print(f"❌ 实验失败: {model_name} - {experiment_type}")
                print(f"💬 错误信息: {result.stderr}")
                
            # 保存实验日志
            log_file = os.path.join(output_dir, "experiment_log.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_result, f, indent=2, ensure_ascii=False)
            
            # 保存一个更易读的文本日志
            readable_log_file = os.path.join(output_dir, "experiment_log.txt")
            with open(readable_log_file, 'w', encoding='utf-8') as f:
                f.write(f"实验: {model_name} - {experiment_type}\n")
                f.write(f"{'='*60}\n")
                f.write(f"开始时间: {experiment_result['timestamp']}\n")
                f.write(f"运行时间: {duration:.2f} 秒\n")
                f.write(f"返回码: {result.returncode}\n")
                f.write(f"命令: {' '.join(cmd)}\n\n")
                
                f.write("标准输出:\n")
                f.write("-" * 40 + "\n")
                f.write(result.stdout if result.stdout else "无输出\n")
                f.write("\n")
                
                if result.stderr:
                    f.write("错误输出:\n")
                    f.write("-" * 40 + "\n")
                    f.write(result.stderr)
                    f.write("\n")
            
            return experiment_result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'experiment_type': experiment_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"💥 实验异常: {model_name} - {experiment_type}: {e}")
            return error_result
    
    def run_experiments(self, selected_models=None, selected_experiments=None, parallel=False, max_workers=4):
        """运行实验"""
        # 筛选模型
        if selected_models:
            models_to_run = {k: v for k, v in self.config['models'].items() if k in selected_models}
        else:
            models_to_run = self.config['models']
        
        # 筛选实验类型
        if selected_experiments:
            experiments_to_run = [exp for exp in selected_experiments if exp in self.config['experiments']]
        else:
            experiments_to_run = list(self.config['experiments'].keys())
        
        print(f"\n🎯 准备运行实验:")
        print(f"📊 模型数量: {len(models_to_run)}")
        print(f"🧪 实验类型: {experiments_to_run}")
        print(f"🔢 总实验数: {len(models_to_run) * len(experiments_to_run)}")
        print(f"⚡ 并行执行: {'是' if parallel else '否'}")
        
        # 确认继续
        response = input("\n确认开始实验? (y/N): ")
        if response.lower() != 'y':
            print("取消实验")
            return
        
        # 记录所有实验任务
        all_tasks = []
        for model_name, model_config in models_to_run.items():
            for experiment_type in experiments_to_run:
                all_tasks.append((experiment_type, model_name, model_config))
        
        # 运行实验
        start_time = time.time()
        
        if parallel and len(all_tasks) > 1:
            # 并行运行
            print(f"\n🔄 使用 {max_workers} 个并行工作进程")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(self.run_single_experiment, exp_type, model_name, model_config): 
                    (exp_type, model_name) 
                    for exp_type, model_name, model_config in all_tasks
                }
                
                for future in as_completed(future_to_task):
                    exp_type, model_name = future_to_task[future]
                    try:
                        result = future.result()
                        self.results[f"{model_name}_{exp_type}"] = result
                    except Exception as e:
                        print(f"💥 任务异常 {model_name}_{exp_type}: {e}")
        else:
            # 串行运行
            print(f"\n🔄 串行运行实验")
            for i, (exp_type, model_name, model_config) in enumerate(all_tasks, 1):
                print(f"\n📊 进度: {i}/{len(all_tasks)}")
                result = self.run_single_experiment(exp_type, model_name, model_config)
                self.results[f"{model_name}_{exp_type}"] = result
        
        # 总结
        total_time = time.time() - start_time
        successful = sum(1 for r in self.results.values() if r.get('return_code') == 0)
        failed = len(self.results) - successful
        
        print(f"\n{'='*60}")
        print(f"📊 实验完成总结")
        print(f"{'='*60}")
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"✅ 成功: {successful} 个")
        print(f"❌ 失败: {failed} 个")
        print(f"📁 结果保存在: {self.config['global_settings']['base_output_dir']}")
        
        # 保存实验总结
        summary_dir = os.path.join(self.config['global_settings']['base_output_dir'], "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_file = os.path.join(summary_dir, f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_experiments': len(all_tasks),
                'successful': successful,
                'failed': failed,
                'total_time': total_time,
                'results': self.results,
                'config': self.config_file
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细总结保存在: {summary_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动化RouteNet实验运行器')
    parser.add_argument('--config', default='experiment_config.yaml', help='配置文件路径')
    parser.add_argument('--models', nargs='+', help='指定要运行的模型 (默认全部)')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['evaluate', 'gradient', 'numerical'],
                       help='指定要运行的实验类型 (默认全部)')
    parser.add_argument('--parallel', action='store_true', help='并行运行实验')
    parser.add_argument('--max_workers', type=int, default=4, help='最大并行工作进程数')
    parser.add_argument('--validate_only', action='store_true', help='仅验证配置，不运行实验')
    
    args = parser.parse_args()
    
    try:
        # 初始化运行器
        runner = ExperimentRunner(args.config)
        
        # 验证模型路径
        if not runner.validate_model_paths():
            print("❌ 模型路径验证失败，请检查配置文件")
            return 1
        
        if args.validate_only:
            print("✅ 配置验证完成")
            return 0
        
        # 运行实验
        runner.run_experiments(
            selected_models=args.models,
            selected_experiments=args.experiments,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
        return 0
        
    except Exception as e:
        print(f"💥 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
RouteNet模型串行训练脚本
支持软硬物理限制、MLP/KAN、不同lambda_physics参数的组合训练
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
import argparse

class ModelTrainer:
    def __init__(self, base_dir="/home/ubantu/net2vec"):
        self.base_dir = Path(base_dir)
        self.train_script = self.base_dir / "routenet" / "routenet_tf2.py"
        self.train_data_dir = self.base_dir / "data" / "routenet" / "nsfnetbw" / "tfrecords" / "train"
        self.eval_data_dir = self.base_dir / "data" / "routenet" / "nsfnetbw" / "tfrecords" / "evaluate"
        self.models_base_dir = self.base_dir / "fixed_model"
        
        # 训练配置
        self.training_configs = self._generate_training_configs()
        
    def _generate_training_configs(self):
        """生成所有训练配置组合"""
        configs = []
        
        # 模型类型和物理限制组合
        model_configs = [
            {"type": "mlp", "use_kan": False, "physics": "soft", "use_physics_loss": True, "use_hard_constraint": False},
            {"type": "mlp", "use_kan": False, "physics": "hard", "use_physics_loss": True, "use_hard_constraint": True},
            {"type": "kan", "use_kan": True, "physics": "soft", "use_physics_loss": True, "use_hard_constraint": False},
            {"type": "kan", "use_kan": True, "physics": "hard", "use_physics_loss": True, "use_hard_constraint": True},
        ]
        
        # lambda_physics参数
        lambda_values = [0.1, 0.5, 1.0]
        
        for model_config in model_configs:
            for lambda_val in lambda_values:
                config = {
                    "name": f"{model_config['type']}_{model_config['physics']}_{lambda_val}",
                    "model_type": model_config["type"],
                    "use_kan": model_config["use_kan"],
                    "physics_type": model_config["physics"],
                    "use_physics_loss": model_config["use_physics_loss"],
                    "use_hard_constraint": model_config["use_hard_constraint"],
                    "lambda_physics": lambda_val,
                    "model_dir": self._get_model_dir(model_config, lambda_val),
                }
                configs.append(config)
        
        return configs
    
    def _get_model_dir(self, model_config, lambda_val):
        """生成模型保存目录 - 优化后的简洁结构"""
        # 使用 fixed_model 作为根目录
        # 目录结构: fixed_model/{model_type}_{physics_type}_{lambda_val}/
        model_dir = self.models_base_dir / f"{model_config['type']}_{model_config['physics']}_{lambda_val}"
        return model_dir
    
    def _build_training_command(self, config):
        """构建训练命令"""
        cmd = [
            "python", str(self.train_script),
            "--train_dir", str(self.train_data_dir),
            "--eval_dir", str(self.eval_data_dir),
            "--model_dir", str(config["model_dir"]),
            "--target", "delay",
            "--epochs", "20",  # 增加训练轮数以获得更好效果
            "--batch_size", "32",
            "--lr_schedule", "plateau",
            "--learning_rate", "0.001",
            # "--plateau_patience", "8",  # 增加耐心值
            "--plateau_factor", "0.5",
            # "--early_stopping_patience", "15",  # 添加早停
        ]
        
        # 添加物理损失相关参数
        cmd.extend(["--physics_loss", "--lambda_physics", str(config["lambda_physics"])])
        
        # 添加约束类型参数
        if config["use_hard_constraint"]:
            cmd.append("--hard_physics")
            
        # 添加KAN相关参数
        if config["use_kan"]:
            cmd.append("--use_kan")
            
        return cmd
    
    def train_model(self, config):
        """训练单个模型"""
        print(f"\n{'='*60}")
        print(f"🚀 开始训练模型: {config['name']}")
        print(f"📁 模型目录: {config['model_dir']}")
        print(f"⚙️  配置: {config['model_type'].upper()}, {config['physics_type']}, λ={config['lambda_physics']}")
        print(f"{'='*60}")
        
        # 创建模型目录
        config["model_dir"].mkdir(parents=True, exist_ok=True)
        
        # 构建训练命令
        cmd = self._build_training_command(config)
        
        print(f"⚡ 执行命令: {' '.join(cmd)}")
        print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # 执行训练
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ 训练成功完成!")
                print(f"⏱️  训练耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
                
                # 保存训练结果
                self._save_training_result(config, True, duration, result.stdout, result.stderr)
                return True
            else:
                print(f"❌ 训练失败!")
                print(f"💬 错误信息: {result.stderr}")
                
                # 保存训练结果
                self._save_training_result(config, False, duration, result.stdout, result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 训练超时 (2小时)")
            self._save_training_result(config, False, 7200, "", "训练超时")
            return False
        except Exception as e:
            print(f"💥 训练异常: {e}")
            self._save_training_result(config, False, 0, "", str(e))
            return False
    
    def _save_training_result(self, config, success, duration, stdout, stderr):
        """保存训练结果"""
        result = {
            "config": config["name"],
            "model_type": config["model_type"],
            "physics_type": config["physics_type"],
            "lambda_physics": config["lambda_physics"],
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "stdout": stdout,
            "stderr": stderr
        }
        
        # 保存到模型目录
        result_file = config["model_dir"] / "training_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        # 保存到汇总日志
        log_dir = self.base_dir / "training_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"training_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def train_all_models(self, start_from=None, models_to_train=None):
        """串行训练所有模型"""
        print(f"🎯 准备训练模型")
        print(f"📊 总计模型数量: {len(self.training_configs)}")
        print(f"🔄 训练方式: 串行 (一个接一个)")
        
        # 显示所有配置
        print(f"\n📋 训练配置列表:")
        for i, config in enumerate(self.training_configs, 1):
            print(f"  {i:2d}. {config['name']} - {config['model_type'].upper()}, {config['physics_type']}, λ={config['lambda_physics']}")
        
        # 过滤要训练的模型
        configs_to_train = self.training_configs.copy()
        
        if models_to_train:
            configs_to_train = [c for c in configs_to_train if c["name"] in models_to_train]
            print(f"\n🎯 仅训练指定模型: {models_to_train}")
            
        if start_from:
            start_idx = next((i for i, c in enumerate(configs_to_train) if c["name"] == start_from), 0)
            configs_to_train = configs_to_train[start_idx:]
            print(f"\n📍 从模型 '{start_from}' 开始训练")
        
        if not configs_to_train:
            print("⚠️  没有找到要训练的模型!")
            return
            
        print(f"\n🚀 开始串行训练 {len(configs_to_train)} 个模型...")
        
        # 开始训练
        successful_trainings = 0
        failed_trainings = 0
        
        for i, config in enumerate(configs_to_train, 1):
            print(f"\n🔢 进度: {i}/{len(configs_to_train)}")
            
            if self.train_model(config):
                successful_trainings += 1
            else:
                failed_trainings += 1
                
            # 训练间隔
            if i < len(configs_to_train):
                print(f"⏸️  等待 10 秒后开始下一个模型训练...")
                time.sleep(10)
        
        # 训练总结
        print(f"\n{'='*60}")
        print(f"📈 训练完成总结")
        print(f"{'='*60}")
        print(f"✅ 成功训练: {successful_trainings}")
        print(f"❌ 失败训练: {failed_trainings}")
        print(f"📊 总计模型: {successful_trainings + failed_trainings}")
        print(f"📈 成功率: {successful_trainings/(successful_trainings + failed_trainings)*100:.1f}%")
        print(f"🕐 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def list_configs(self):
        """列出所有训练配置"""
        print("📋 可用的训练配置:")
        for i, config in enumerate(self.training_configs, 1):
            print(f"  {i:2d}. {config['name']}")
            print(f"      模型类型: {config['model_type'].upper()}")
            print(f"      物理限制: {config['physics_type']}")
            print(f"      Lambda值: {config['lambda_physics']}")
            print(f"      模型目录: {config['model_dir']}")
            print()

def main():
    parser = argparse.ArgumentParser(description="RouteNet模型串行训练脚本")
    parser.add_argument("--list", action="store_true", help="列出所有训练配置")
    parser.add_argument("--start-from", type=str, help="从指定模型开始训练")
    parser.add_argument("--models", nargs="+", help="仅训练指定的模型")
    parser.add_argument("--base-dir", default="/home/ubantu/net2vec", help="项目根目录")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(base_dir=args.base_dir)
    
    if args.list:
        trainer.list_configs()
        return
    
    # 确认开始训练
    if not args.models and not args.start_from:
        response = input(f"\n确认开始训练所有 {len(trainer.training_configs)} 个模型? (y/N): ")
        if response.lower() != 'y':
            print("❌ 取消训练")
            return
    
    trainer.train_all_models(start_from=args.start_from, models_to_train=args.models)

if __name__ == "__main__":
    main()

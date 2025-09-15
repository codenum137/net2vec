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
    def __init__(self, base_dir="/home/ubantu/net2vec", force_retrain=False):
        self.base_dir = Path(base_dir)
        self.train_script = self.base_dir / "routenet" / "routenet_tf2.py"
        self.train_data_dir = self.base_dir / "data" / "routenet" / "nsfnetbw" / "tfrecords" / "train"
        self.eval_data_dir = self.base_dir / "data" / "routenet" / "nsfnetbw" / "tfrecords" / "evaluate"
        self.models_base_dir = self.base_dir / "fixed_model/0915"
        self.force_retrain = force_retrain  # 是否强制重新训练已存在的模型
        
        # 训练配置
        self.training_configs = self._generate_training_configs()
        
    def _generate_training_configs(self):
        """生成所有训练配置组合"""
        configs = []
        
        # 模型类型和物理限制组合
        model_configs = [
            # 不使用物理约束的配置
            {"type": "mlp", "use_kan": False, "physics": "none", "use_physics_loss": False, "use_hard_constraint": False},
            {"type": "kan", "use_kan": True, "physics": "none", "use_physics_loss": False, "use_hard_constraint": False},
            # 使用物理约束的配置
            {"type": "mlp", "use_kan": False, "physics": "soft", "use_physics_loss": True, "use_hard_constraint": False},
            {"type": "mlp", "use_kan": False, "physics": "hard", "use_physics_loss": True, "use_hard_constraint": True},
            {"type": "kan", "use_kan": True, "physics": "soft", "use_physics_loss": True, "use_hard_constraint": False},
            {"type": "kan", "use_kan": True, "physics": "hard", "use_physics_loss": True, "use_hard_constraint": True},
        ]
        
        # lambda_physics参数
        lambda_values = [0.1,0.5, 1.0]
        
        for model_config in model_configs:
            if model_config["use_physics_loss"]:
                # 使用物理约束的配置：生成多个lambda值
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
            else:
                # 不使用物理约束的配置：只生成一个配置
                config = {
                    "name": f"{model_config['type']}_{model_config['physics']}",
                    "model_type": model_config["type"],
                    "use_kan": model_config["use_kan"],
                    "physics_type": model_config["physics"],
                    "use_physics_loss": model_config["use_physics_loss"],
                    "use_hard_constraint": model_config["use_hard_constraint"],
                    "lambda_physics": 0.0,  # 不使用物理约束时lambda值无意义
                    "model_dir": self._get_model_dir(model_config, None),
                }
                configs.append(config)
        
        return configs
    
    def _get_model_dir(self, model_config, lambda_val):
        """生成模型保存目录 - 优化后的简洁结构"""
        # 使用 fixed_model 作为根目录
        if lambda_val is None:
            # 不使用物理约束的情况
            model_dir = self.models_base_dir / f"{model_config['type']}_{model_config['physics']}"
        else:
            # 使用物理约束的情况
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
            "--plateau_patience", "8",  # 增加耐心值
            "--plateau_factor", "0.5",
        ]
        
        # 添加物理损失相关参数（仅在使用物理约束时）
        if config["use_physics_loss"]:
            cmd.extend(["--physics_loss", "--lambda_physics", str(config["lambda_physics"])])
            
            # 添加约束类型参数
            if config["use_hard_constraint"]:
                cmd.append("--hard_physics")
            
        # 添加KAN相关参数
        if config["use_kan"]:
            cmd.append("--kan")  # 修正参数名：使用 --kan 而不是 --use_kan
            
        return cmd
    
    def train_model(self, config):
        """训练单个模型"""
        print(f"\n{'='*60}")
        print(f"🚀 开始训练模型: {config['name']}")
        print(f"📁 模型目录: {config['model_dir']}")
        print(f"⚙️  配置: {config['model_type'].upper()}, {config['physics_type']}, λ={config['lambda_physics']}")
        print(f"{'='*60}")
        
        # 检查模型是否已存在（根据模型类型选择正确的文件名）
        if config["use_kan"]:
            model_file = config["model_dir"] / "best_delay_kan_model.weights.h5"
        else:
            model_file = config["model_dir"] / "best_delay_model.weights.h5"
            
        if model_file.exists() and not self.force_retrain:
            print(f"⏭️  模型已存在，跳过训练: {model_file}")
            print(f"💡 如需重新训练，请使用 --force 参数")
            print(f"{'='*60}")
            return True
        
        # 创建模型目录
        config["model_dir"].mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件路径
        log_file = config["model_dir"] / "training.log"
        
        # 构建训练命令
        cmd = self._build_training_command(config)
        
        print(f"⚡ 执行命令: {' '.join(cmd)}")
        print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 日志文件: {log_file}")
        
        start_time = time.time()
        
        try:
            # 执行训练 - 只显示epoch进度，详细输出写入日志
            print(f"📊 开始训练，监控进度中...")
            print(f"{'='*60}")
            
            process = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 监控输出：控制台显示epoch进度，日志文件保存详细输出
            output_lines = []
            current_epoch = 0
            total_epochs = 20  # 从命令中获取的epoch数
            
            # 打开日志文件
            with open(log_file, 'w', encoding='utf-8') as log_f:
                # 写入日志头部信息
                log_f.write(f"{'='*80}\n")
                log_f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_f.write(f"模型配置: {config['name']}\n")
                log_f.write(f"执行命令: {' '.join(cmd)}\n")
                log_f.write(f"{'='*80}\n\n")
                log_f.flush()
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        output_lines.append(output)
                        
                        # 写入日志文件（实时）
                        log_f.write(output)
                        log_f.flush()
                        
                        # 检查是否包含epoch信息
                        line_lower = output.lower().strip()
                        if 'epoch' in line_lower:
                            # 尝试提取epoch数字 - 支持多种格式
                            import re
                            # 匹配 "Epoch 1/20", "Epoch: 5", "epoch 10", "Training epoch 3" 等格式
                            epoch_patterns = [
                                r'epoch\s*(\d+)(?:/\d+)?',     # Epoch 1/20 或 Epoch 1
                                r'epoch\s*:?\s*(\d+)',         # Epoch: 5 或 Epoch 5
                                r'(\d+)/\d+.*epoch',           # 1/20 epoch 格式
                                r'epoch\s+is\s+(\d+)',         # epoch is 12 格式
                            ]
                            
                            epoch_num = None
                            for pattern in epoch_patterns:
                                epoch_match = re.search(pattern, line_lower)
                                if epoch_match:
                                    epoch_num = int(epoch_match.group(1))
                                    break
                            
                            if epoch_num and epoch_num != current_epoch:
                                current_epoch = epoch_num
                                progress = (current_epoch / total_epochs) * 100
                                progress_msg = f"📈 训练进度: Epoch {current_epoch}/{total_epochs} ({progress:.1f}%)"
                                print(progress_msg)
                                
                                # 同时写入日志文件
                                log_f.write(f"\n[PROGRESS] {progress_msg}\n")
                                log_f.flush()
                
                # 写入日志尾部信息
                log_f.write(f"\n{'='*80}\n")
                log_f.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            process.wait()
            end_time = time.time()
            duration = end_time - start_time
            
            # 显示最终进度
            if current_epoch > 0:
                final_msg = f"📈 训练完成: Epoch {current_epoch}/{total_epochs} (100%)"
                print(final_msg)
                
                # 写入最终状态到日志
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n[FINAL] {final_msg}\n")
                    log_f.write(f"训练耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)\n")
                    log_f.write(f"{'='*80}\n")
            
            full_output = ''.join(output_lines)
            
            if process.returncode == 0:
                print(f"{'='*60}")
                print(f"✅ 训练成功完成!")
                print(f"⏱️  训练耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
                print(f"📝 详细日志: {log_file}")
                
                # 写入成功状态到日志
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n[SUCCESS] 训练成功完成!\n")
                    log_f.write(f"返回码: {process.returncode}\n")
                
                # 保存训练结果
                self._save_training_result(config, True, duration, full_output, "")
                return True
            else:
                print(f"{'='*60}")
                print(f"❌ 训练失败!")
                print(f"💬 返回码: {process.returncode}")
                print(f"📝 详细日志: {log_file}")
                
                # 写入失败状态到日志
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n[ERROR] 训练失败!\n")
                    log_f.write(f"返回码: {process.returncode}\n")
                
                # 保存训练结果
                self._save_training_result(config, False, duration, full_output, f"Process returned {process.returncode}")
                return False
                
        except KeyboardInterrupt:
            print(f"{'='*60}")
            print(f"🛑 训练被用户中断")
            print(f"📝 详细日志: {log_file}")
            if 'process' in locals():
                process.terminate()
                process.wait()
            duration = time.time() - start_time
            
            # 写入中断状态到日志
            try:
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n[INTERRUPTED] 训练被用户中断\n")
                    log_f.write(f"中断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_f.write(f"已运行时间: {duration:.1f}秒\n")
            except:
                pass
            
            self._save_training_result(config, False, duration, "", "用户中断")
            return False
        except Exception as e:
            print(f"{'='*60}")
            print(f"💥 训练异常: {e}")
            print(f"📝 详细日志: {log_file}")
            duration = time.time() - start_time
            
            # 写入异常状态到日志
            try:
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n[EXCEPTION] 训练异常: {e}\n")
                    log_f.write(f"异常时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_f.write(f"已运行时间: {duration:.1f}秒\n")
            except:
                pass
            
            self._save_training_result(config, False, duration, "", str(e))
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
    parser.add_argument("--force", action="store_true", help="强制重新训练已存在的模型")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(base_dir=args.base_dir, force_retrain=args.force)
    
    if args.list:
        trainer.list_configs()
        return
    
    # 显示训练模式
    if args.force:
        print("🔄 强制重新训练模式：将覆盖已存在的模型")
    else:
        print("⏭️  跳过模式：已存在的模型将被跳过")
    
    # 确认开始训练
    if not args.models and not args.start_from:
        response = input(f"\n确认开始训练所有 {len(trainer.training_configs)} 个模型? (y/N): ")
        if response.lower() != 'y':
            print("❌ 取消训练")
            return
    
    trainer.train_all_models(start_from=args.start_from, models_to_train=args.models)

if __name__ == "__main__":
    main()

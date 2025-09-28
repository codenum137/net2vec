#!/usr/bin/env python3
"""精简版 RouteNet 串行训练脚本 (仅控制台输出)"""

import sys
import time
import subprocess
import argparse
import re
from pathlib import Path
from datetime import datetime

class ModelTrainer:
    def __init__(self, base_dir: str = './', force_retrain: bool = False,
                 enable_early_stopping: bool = True, early_stopping_patience: int = 5):
        self.base_dir = Path(base_dir)
        self.train_script = self.base_dir / 'routenet' / 'routenet_tf2.py'
        self.train_data_dir = self.base_dir / 'data' / 'routenet' / 'nsfnetbw' / 'tfrecords' / 'train'
        self.eval_data_dir = self.base_dir / 'data' / 'routenet' / 'nsfnetbw' / 'tfrecords' / 'evaluate'
        self.models_base_dir = self.base_dir / 'fixed_model/0928'
        self.force_retrain = force_retrain
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.training_configs = self._generate_training_configs()

    def _generate_training_configs(self):
        configs = []
        model_configs = [
            # # 不使用物理约束的配置
            {"type": "mlp", "use_kan": False, "physics": "none", "use_physics_loss": False, "use_hard_constraint": False},
            {"type": "kan", "use_kan": True, "physics": "none", "use_physics_loss": False, "use_hard_constraint": False},
            # # 使用物理约束的配置 - 传统固定lambda
            {"type": "mlp", "use_kan": False, "physics": "soft", "use_physics_loss": True, "use_hard_constraint": False},
            {"type": "mlp", "use_kan": False, "physics": "hard", "use_physics_loss": True, "use_hard_constraint": True},
            {"type": "kan", "use_kan": True, "physics": "soft", "use_physics_loss": True, "use_hard_constraint": False},
            {"type": "kan", "use_kan": True, "physics": "hard", "use_physics_loss": True, "use_hard_constraint": True},
            # 使用物理约束的配置 - 课程学习
            # {"type": "mlp", "use_kan": False, "physics": "soft_cl", "use_physics_loss": True, "use_hard_constraint": False, "use_curriculum": True},
            # {"type": "mlp", "use_kan": False, "physics": "hard_cl", "use_physics_loss": True, "use_hard_constraint": True, "use_curriculum": True},
            # {"type": "kan", "use_kan": True, "physics": "soft_cl", "use_physics_loss": True, "use_hard_constraint": False, "use_curriculum": True},
            # {"type": "kan", "use_kan": True, "physics": "hard_cl", "use_physics_loss": True, "use_hard_constraint": True, "use_curriculum": True},
        ]
        for mc in model_configs:
            cfg = {
                'name': f"{mc['type']}_{mc['physics']}",
                'model_type': mc['type'],
                'use_kan': mc['use_kan'],
                'physics_type': mc['physics'],
                'use_physics_loss': mc['use_physics_loss'],
                'use_hard_constraint': mc['use_hard_constraint'],
                'lambda_physics': 0.05,
                'use_curriculum': False,
                'model_dir': self._get_model_dir(mc, None)
            }
            configs.append(cfg)
        return configs

    def _get_model_dir(self, mc, lam):
        return self.models_base_dir / f"{mc['type']}_{mc['physics']}"

    def _build_training_command(self, cfg):
        cmd = [
            'python', str(self.train_script),
            '--train_dir', str(self.train_data_dir),
            '--eval_dir', str(self.eval_data_dir),
            '--model_dir', str(cfg['model_dir']),
            '--target', 'delay',
            '--epochs', '20',
            '--batch_size', '32',
            '--lr_schedule', 'plateau',
            '--learning_rate', '0.001',
            '--plateau_patience', '8',
            '--plateau_factor', '0.5',
            '--seed', '137'
        ]
        if cfg['use_kan']:
            cmd.append('--kan')
        if self.enable_early_stopping:
            cmd.extend([
                '--early_stopping',
                '--early_stopping_patience', str(self.early_stopping_patience),
                '--early_stopping_min_delta', '1e-6',
                '--early_stopping_restore_best'
            ])
        return cmd

    def train_model(self, cfg, raw_output: bool = False):
        print(f"\n{'='*70}\n🚀 模型: {cfg['name']}  (类型: {cfg['model_type'].upper()}  物理: {cfg['physics_type']})")
        model_file = cfg['model_dir'] / ('best_delay_kan_model.weights.h5' if cfg['use_kan'] else 'best_delay_model.weights.h5')
        if model_file.exists() and not self.force_retrain:
            print(f"⏭️  已存在，跳过: {model_file}")
            return True
        cfg['model_dir'].mkdir(parents=True, exist_ok=True)
        cmd = self._build_training_command(cfg)
        print(f"⚡ 命令: {' '.join(cmd)}")
        print(f"🕐 开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start = time.time()
        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            epoch_patterns = [
                r'epoch\s*(\d+)(?:/\d+)?',
                r'epoch\s*:?\s*(\d+)',
                r'(\d+)/\d+.*epoch',
                r'epoch\s+is\s+(\d+)',
            ]
            current_epoch = 0
            total_epochs = 20
            buffer = []
            last_progress_print = 0.0
            last_step_shown = -1
            progress_regex = re.compile(r'Training Epoch (\d+): .*loss=([0-9.]+), step=(\d+)\]')
            eval_progress_regex = re.compile(r'Evaluating Epoch (\d+): .*loss=([0-9.]+)')

            def print_progress(compact_line: str, force=False):
                nonlocal last_progress_print
                now = time.time()
                if force or now - last_progress_print >= 1.0:  # 最多每秒刷新一次
                    sys.stdout.write('\r' + compact_line + ' '*20)
                    sys.stdout.flush()
                    last_progress_print = now

            while True:
                line = p.stdout.readline()
                if line == '' and p.poll() is not None:
                    break
                if not line:
                    continue
                buffer.append(line)

                if raw_output:
                    # 原始模式直接透传
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue

                stripped = line.strip()
                lower = stripped.lower()

                # 捕获训练进度行 (tqdm) 并压缩展示
                if stripped.startswith('Training Epoch'):
                    m = progress_regex.search(stripped)
                    if m:
                        ep = int(m.group(1))
                        loss_val = m.group(2)
                        step = int(m.group(3))
                        if ep != current_epoch:
                            current_epoch = ep
                            last_step_shown = -1
                            print()  # 换行，结束上一epoch进度
                            print(f"🟢 开始 Epoch {ep}")
                        # 仅在步数前进且步数为10的倍数或最终刷新节流时间超出时更新
                        if step != last_step_shown and (step % 10 == 0):
                            print_progress(f"Epoch {ep} Step {step} loss={loss_val}")
                            last_step_shown = step
                    continue  # 不直接输出原始行

                # 捕获评估进度
                if stripped.startswith('Evaluating Epoch'):
                    m2 = eval_progress_regex.search(stripped)
                    if m2:
                        ep_eval = m2.group(1)
                        loss_eval = m2.group(2)
                        print_progress(f"[Eval] Epoch {ep_eval} loss={loss_eval}")
                    continue

                # Epoch 结束/结果行 - 正常打印换行
                if 'finished' in lower and 'epoch' in lower:
                    print()  # 确保进度行换行
                    print(stripped)
                    continue
                if 'evaluation loss improved' in lower or 'evaluation loss did not improve' in lower:
                    print(stripped)
                    continue
                if 'saving model' in lower or 'best validation loss' in lower:
                    print(stripped)
                    continue
                if stripped.startswith('📚 Curriculum Learning') or 'Curriculum Learning' in stripped:
                    print(stripped)
                    continue
                # 其他非刷屏输出（保持）
                if stripped:
                    print(stripped)
            p.wait()
            dur = time.time() - start
            if not raw_output:
                print()  # 结束最后一行的\r覆盖
            if p.returncode == 0:
                print(f"✅ 完成: {cfg['name']}  用时 {dur/60:.1f} 分钟")
                return True
            else:
                print(f"❌ 失败: {cfg['name']}  code={p.returncode}")
                print(''.join(buffer[-40:]))
                return False
        except KeyboardInterrupt:
            print(f"🛑 中断: {cfg['name']}")
            if 'p' in locals():
                p.terminate(); p.wait()
            return False
        except Exception as e:
            print(f"💥 异常: {cfg['name']} -> {e}")
            return False

    def train_all_models(self, start_from=None, models_to_train=None, raw_output: bool = False):
        print(f"🎯 总模型数: {len(self.training_configs)}")
        for i, c in enumerate(self.training_configs, 1):
            print(f"  {i:2d}. {c['name']}")
        cfgs = self.training_configs
        if models_to_train:
            cfgs = [c for c in cfgs if c['name'] in models_to_train]
            print(f"🎯 过滤后: {len(cfgs)} -> {models_to_train}")
        if start_from:
            try:
                idx = next(i for i, c in enumerate(cfgs) if c['name'] == start_from)
                cfgs = cfgs[idx:]
                print(f"📍 从 {start_from} 开始")
            except StopIteration:
                print(f"⚠️ 未找到 {start_from}, 从头开始")
        if not cfgs:
            print("⚠️ 无可训练模型")
            return
        success = fail = 0
        for i, c in enumerate(cfgs, 1):
            print(f"\n🔢 进度 {i}/{len(cfgs)} -> {c['name']}")
            ok = self.train_model(c, raw_output=raw_output)
            success += int(ok)
            fail += int(not ok)
            if i < len(cfgs):
                time.sleep(2)
        print(f"\n{'='*60}\n📊 训练完成  成功:{success}  失败:{fail}  总计:{success+fail}\n{'='*60}")

    def list_configs(self):
        print("📋 配置列表:")
        for i, c in enumerate(self.training_configs, 1):
            print(f"  {i:2d}. {c['name']}  type={c['model_type']}  physics={c['physics_type']}")

def main():
    parser = argparse.ArgumentParser(description='RouteNet 串行训练脚本 (精简版)')
    parser.add_argument('--list', action='store_true', help='列出所有配置')
    parser.add_argument('--start-from', type=str, help='从指定模型名称开始训练')
    parser.add_argument('--models', nargs='+', help='只训练这些模型')
    parser.add_argument('--base-dir', default='./', help='项目根目录')
    parser.add_argument('--force', action='store_true', help='已存在时重新训练')
    parser.add_argument('--yes', '-y', action='store_true', help='无需确认直接开始')
    parser.add_argument('--no-early-stopping', action='store_true', help='禁用早停')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='早停耐心')
    parser.add_argument('--raw-output', action='store_true', help='显示原始子进程输出(可能刷屏)')
    args = parser.parse_args()

    trainer = ModelTrainer(
        base_dir=args.base_dir,
        force_retrain=args.force,
        enable_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience
    )

    if args.list:
        trainer.list_configs(); return
    if not args.models and not args.start_from and not args.yes:
        ans = input(f"确认训练全部 {len(trainer.training_configs)} 个模型? (y/N): ")
        if ans.lower() != 'y':
            print('取消'); return
    elif args.yes:
        print(f"✅ 自动确认，开始训练 {len(trainer.training_configs)} 个模型")
    trainer.train_all_models(start_from=args.start_from, models_to_train=args.models, raw_output=args.raw_output)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
RouteNet 串行训练脚本 (精简 + 单行刷新进度)
"""
import argparse
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import re

# Python版本保护：提示未激活正确环境时的错误（需要 Python 3.7+）
if sys.version_info < (3, 7):
    raise RuntimeError("当前使用的解释器版本为 Python {}.{}. 需要使用 Python 3.7+ 并先激活 conda 环境: 'conda activate routenet-tf2-env'".format(sys.version_info.major, sys.version_info.minor))

class ModelTrainer:
    def __init__(self, base_dir="./", force_retrain=False, enable_early_stopping=True, early_stopping_patience=5):
        self.base_dir = Path(base_dir)
        self.train_script = self.base_dir / "routenet" / "routenet_tf2.py"
        self.train_data_dir = self.base_dir / "data" / "routenet" / "nsfnetbw" / "tfrecords" / "train"
        self.eval_data_dir = self.base_dir / "data" / "routenet" / "nsfnetbw" / "tfrecords" / "evaluate"
        self.models_base_dir = self.base_dir / "fixed_model/0927"
        self.force_retrain = force_retrain
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.training_configs = self._generate_training_configs()

    def _generate_training_configs(self):
        cfgs = []
        base = [
            {"type": "mlp", "use_kan": False, "physics": "none"},
            {"type": "kan", "use_kan": True,  "physics": "none"},
        ]
        for mc in base:
            name = "{}_{}".format(mc['type'], mc['physics'])
            cfgs.append({
                "name": name,
                "model_type": mc['type'],
                "use_kan": mc['use_kan'],
                "physics_type": mc['physics'],
                "use_physics_loss": False,
                "use_hard_constraint": False,
                "lambda_physics": 0.0,
                "use_curriculum": False,
                "model_dir": self._get_model_dir(mc, None)
            })
        return cfgs

    def _get_model_dir(self, mc, lam):
        return self.models_base_dir / (f"{mc['type']}_{mc['physics']}" if lam is None else f"{mc['type']}_{mc['physics']}_{lam}")

    def _build_training_command(self, cfg):
        # 使用当前解释器而不是硬编码 'python'，避免落到系统 Python2.7
        cmd = [
            sys.executable, str(self.train_script),
            "--train_dir", str(self.train_data_dir),
            "--eval_dir", str(self.eval_data_dir),
            "--model_dir", str(cfg['model_dir']),
            "--target", "delay",
            "--epochs", "40",
            "--batch_size", "32",
            "--lr_schedule", "plateau",
            "--learning_rate", "0.001",
            "--plateau_patience", "8",
            "--plateau_factor", "0.5",
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

    def train_model(self, cfg):
        print(f"\n{'='*60}")
        print(f"🚀 开始训练: {cfg['name']}")
        print(f"📁 目录: {cfg['model_dir']}")
        print(f"⚙️  配置: {cfg['model_type'].upper()}, λ={cfg['lambda_physics']}")
        print(f"{'='*60}")
        model_file = cfg['model_dir'] / ('best_delay_kan_model.weights.h5' if cfg['use_kan'] else 'best_delay_model.weights.h5')
        if model_file.exists() and not self.force_retrain:
            print(f"⏭️  已存在, 跳过: {model_file}")
            return True
        cfg['model_dir'].mkdir(parents=True, exist_ok=True)
        cmd = self._build_training_command(cfg)
        print('⚡ CMD:', ' '.join(cmd))
        print('📝 输出: 单行刷新 Epoch 进度 (失败仅输出尾部)')

        start = time.time()
        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            lines = []
            cur = -1
            total = None
            last_len = 0
            patterns = [
                re.compile(r'epoch\s*(\d+)\s*/\s*(\d+)', re.IGNORECASE),
                re.compile(r'(\d+)\s*/\s*(\d+).*epoch', re.IGNORECASE),
                re.compile(r'epoch\s*[:：]?\s*(\d+)', re.IGNORECASE),
                re.compile(r'epoch\s+is\s+(\d+)', re.IGNORECASE),
            ]
            # 从命令行推断总轮数
            try:
                i = cmd.index('--epochs'); total = int(cmd[i+1])
            except Exception:
                pass
            while True:
                line = p.stdout.readline()
                if line == '' and p.poll() is not None:
                    break
                if not line:
                    continue
                lines.append(line)
                low = line.lower()
                if 'epoch' not in low:
                    continue
                for rg in patterns:
                    m = rg.search(low)
                    if not m:
                        continue
                    try:
                        ep = int(m.group(1))
                    except Exception:
                        continue
                    if m.lastindex and m.lastindex >= 2:
                        try:
                            total = int(m.group(2))
                        except Exception:
                            pass
                    cur = ep
                    pct = (cur / total * 100.0) if total else 0.0
                    status = f"[{cfg['name']}] Epoch {cur}/{total or '?'}  {pct:5.1f}%"
                    pad = ' ' * max(0, last_len - len(status))
                    sys.stdout.write('\r' + status + pad)
                    sys.stdout.flush()
                    last_len = len(status)
                    break
            p.wait()
            if last_len:
                sys.stdout.write('\n'); sys.stdout.flush()
            dur = time.time() - start
            if p.returncode == 0:
                print(f"✅ 完成: {cfg['name']} ⏱️ {dur:.1f}s ({dur/60:.1f}m)")
                return True
            print(f"❌ 失败: {cfg['name']} code={p.returncode}")
            print('--- 最后 50 行输出 ---')
            for l in lines[-50:]:
                print(l.rstrip())
            return False
        except KeyboardInterrupt:
            print('\n🛑 用户中断')
            if 'p' in locals():
                p.terminate(); p.wait()
            return False
        except Exception as e:
            print('\n💥 异常:', e)
            return False

    def train_all_models(self, start_from=None, models_to_train=None):
        print('🎯 准备训练')
        print(f'📊 总计: {len(self.training_configs)}')
        print('🔄 串行模式')
        for i, c in enumerate(self.training_configs, 1):
            print(f"  {i:2d}. {c['name']} -> {c['model_dir']}")
        cfgs = self.training_configs
        if models_to_train:
            sel = set(models_to_train)
            cfgs = [c for c in cfgs if c['name'] in sel]
            print('🎯 仅训练:', sorted(sel))
        if start_from:
            idx = next((i for i, c in enumerate(cfgs) if c['name'] == start_from), 0)
            cfgs = cfgs[idx:]
            print('📍 从', start_from, '开始')
        if not cfgs:
            print('⚠️ 无匹配配置'); return
        ok = fail = 0
        for i, c in enumerate(cfgs, 1):
            print(f"\n🔢 模型 {i}/{len(cfgs)} -> {c['name']}")
            if self.train_model(c): ok += 1
            else: fail += 1
            if i < len(cfgs): time.sleep(2)
        print('\n' + '='*60)
        print('📈 训练总结')
        print('='*60)
        print('✅ 成功:', ok)
        print('❌ 失败:', fail)
        tot = ok + fail
        if tot: print(f'� 成功率: {ok/tot*100:.1f}%')
        print('🕐 完成时间:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def list_configs(self):
        for i, c in enumerate(self.training_configs, 1):
            print(f"  {i:2d}. {c['name']} ({c['model_dir']})")

def main():
    ap = argparse.ArgumentParser(description='RouteNet 串行训练脚本')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--start-from')
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--base-dir', default='./')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--yes', '-y', action='store_true')
    ap.add_argument('--no-early-stopping', action='store_true')
    ap.add_argument('--early-stopping-patience', type=int, default=5)
    args = ap.parse_args()
    trainer = ModelTrainer(
        base_dir=args.base_dir,
        force_retrain=args.force,
        enable_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.list:
        trainer.list_configs(); return
    print('🔄 强制训练模式' if args.force else '⏭️ 已存在模型跳过')
    print('🛑 早停: ' + ('启用' if trainer.enable_early_stopping else '禁用'))
    if not args.models and not args.start_from and not args.yes:
        resp = input(f"确认训练全部 {len(trainer.training_configs)} 个模型? (y/N): ")
        if resp.lower() != 'y':
            print('❌ 取消'); return
    elif args.yes:
        print(f'✅ 自动确认 {len(trainer.training_configs)} 个模型')
    trainer.train_all_models(start_from=args.start_from, models_to_train=args.models)

if __name__ == '__main__':
    main()

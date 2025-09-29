#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合多个模型 (mlp_none / kan_none / kan_bspline 等) 的 numerical_performance_analysis.csv 结果，
仅生成一个统一汇总 CSV：
    aggregate_numerical_summary.csv  (行=模型, 列=指标)

使用方式：
 1. 修改下方 ROOT_DIR 常量为你的实验根目录 (其下包含各模型子目录: mlp_none / kan_none / kan_bspline ...)
 2. 运行：python aggregate_numerical_results.py
 3. 如需临时覆盖，可加 --root /some/other/path

说明：按你的需求，已移除 JSON 与 pretty_print 文本输出，仅保留 CSV。
"""
import csv
import argparse
import sys
from pathlib import Path

# 强制使用 Python3，避免在 Python2 下触发语法错误 (类型注解等)
if sys.version_info[0] < 3:
    sys.stderr.write("请使用 python3 运行该脚本，当前版本: %d.%d\n" % (sys.version_info[0], sys.version_info[1]))
    sys.exit(1)

EXPECTED_FILE = 'numerical_performance_analysis.csv'

# ====== 配置区：在此设置默认根目录 ======
# 示例: ROOT_DIR = Path('experiment_results/kan_model/137')
ROOT_DIR = Path('experiment_results/kan_model/137')  # 按需修改
# =======================================

# 我们关心的指标 (从单模型 CSV 中提取)
# 表格中有多行：Target, Metric, NSFNet..., GBN..., Degradation
# 我们希望汇总时展开为列：
# <TARGET>_<METRIC>_NSF, <TARGET>_<METRIC>_GBN, <TARGET>_<METRIC>_DEGRADATION


def parse_single_csv(csv_path: Path):
    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_model_name(path: Path):
    return path.name  # 目录名直接作为模型名


def extract_metrics(rows):
    """将单个模型 CSV 行结构展开成字典。"""
    data = {}
    for row in rows:
        target = row['Target'].strip().upper()
        metric = row['Metric'].strip().upper()
        # 列名
        nsf_col = f"{target}_{metric}_NSF"
        gbn_col = f"{target}_{metric}_GBN"
        deg_col = f"{target}_{metric}_DEG"
        # 原始值字符串
        nsf_raw = row['NSFNet (Training Topology)']
        gbn_raw = row['GBN (Test Topology)']
        deg_raw = row['Degradation']
        # 尝试把带百分号的去掉转成 float
        def to_float(val: str):
            v = val.strip().replace('%','')
            try:
                return float(v)
            except Exception:
                return None
        data[nsf_col] = to_float(nsf_raw)
        data[gbn_col] = to_float(gbn_raw)
        data[deg_col] = to_float(deg_raw)
    return data


def aggregate(root: Path, output_dir: Path, include_models=None):
    if include_models:
        include_set = set(include_models)
    else:
        include_set = None

    model_dirs = [d for d in root.iterdir() if d.is_dir() and d.name not in ('summary',)]
    results = []
    missing = []

    for mdir in model_dirs:
        model_name = normalize_model_name(mdir)
        if include_set and model_name not in include_set:
            continue
        csv_path = mdir / 'numerical' / EXPECTED_FILE
        if not csv_path.exists():
            missing.append(model_name)
            continue
        rows = parse_single_csv(csv_path)
        metric_dict = extract_metrics(rows)
        metric_dict['model'] = model_name
        results.append(metric_dict)

    # 汇总所有列
    all_cols = set()
    for r in results:
        all_cols.update(r.keys())
    # 确保 model 列在首位
    all_cols.discard('model')
    ordered_cols = ['model'] + sorted(all_cols)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 写 CSV
    csv_out = output_dir / 'aggregate_numerical_summary.csv'
    with csv_out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[OK] Aggregated CSV written: {csv_out}")
    if missing:
        print(f"Warning: {len(missing)} models missing results: {missing}")


def main():
    parser = argparse.ArgumentParser(description='聚合 numerical 分析结果 (使用文件内 ROOT_DIR 常量)')
    parser.add_argument('--root', type=str, help='可选：覆盖文件内 ROOT_DIR 常量')
    parser.add_argument('--output', type=str, default='summary', help='输出子目录名称 (默认: summary)')
    parser.add_argument('--models', nargs='+', help='只聚合指定模型 (可选)')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve() if args.root else ROOT_DIR.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f'Root path not found: {root}')
    print(f'[Config] 使用根目录: {root}')
    output_dir = root / args.output
    aggregate(root, output_dir, include_models=args.models)

if __name__ == '__main__':
    main()

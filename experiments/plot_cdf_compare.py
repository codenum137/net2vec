#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 mlp_none 与 kan_none 的相对误差 CDF 放入同一张 1x2 图进行对比。

数据来源：加载 evaluation 缓存的 npz 文件（来自 evaluate_routenet_tf2.py 脚本生成的缓存），
假设 npz 文件中包含以下键（delay/jitter）：
  nsfnet_rel_errors_delay, nsfnet_rel_errors_jitter, gbn_rel_errors_delay, gbn_rel_errors_jitter

使用方式示例：
python experiments/plot_cdf_compare.py \
  --mlp_cache experiment_results/0926-sample-100/mlp_none/evaluate/eval_cache_mlp.npz \
  --kan_cache experiment_results/0926-sample-100/kan_none/evaluate/eval_cache_kan.npz \
  --output experiment_results/0926-sample-100/cdf_compare_mlp_vs_kan.png

图像规格：
  - 字体: Times New Roman
  - fig 大小: (7, 3.19)
  - 子图: 1x2, 左: NSFNet, 右: GBN
    - 使用带符号的 relative error（展示正负偏差分布）
  - 指标: delay, jitter （如数据缺失则跳过）
  - CDF 线型: delay 实线, jitter 虚线；MLP 蓝色(KAN橙色)；或再区分可加透明度
  - 图例放置避免遮挡 (右上 / 内部自动)
  - X 轴统一范围：根据两模型 + 两拓扑所有绝对误差的 P99 或最大值裁剪 (避免尾部极端值拉伸)
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, List

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8
})

METRICS = ['delay', 'jitter']
TOPO_MAP = {
    'nsfnet': 'NSFNet',
    'gbn': 'GBN'
}

COLORS = {
    'MLP': {'delay': '#1f77b4', 'jitter': '#1f77b4'},  # 同色不同线型
    'KAN': {'delay': '#ff7f0e', 'jitter': '#ff7f0e'}
}
LINESTYLES = {
    'delay': '-',
    'jitter': '--'
}

REQUIRED_KEYS = [
    'nsfnet_rel_errors_delay', 'nsfnet_rel_errors_jitter',
    'gbn_rel_errors_delay', 'gbn_rel_errors_jitter'
]

def load_cache(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def prepare_signed_errors(cache: Dict[str, np.ndarray], topo_prefix: str) -> Dict[str, np.ndarray]:
    """从缓存中抽取某个拓扑的【带符号】相对误差数组。
    支持两种命名：
      1) {topo}_rel_errors_delay / {topo}_rel_errors_jitter (原设想)
      2) {topo}_delay / {topo}_jitter (evaluate_routenet_tf2.py 实际保存的键)
    返回字典: metric -> signed relative error ndarray
    """
    out: Dict[str, np.ndarray] = {}
    for m in METRICS:
        candidates = [f"{topo_prefix}_rel_errors_{m}", f"{topo_prefix}_{m}"]
        chosen_key = None
        for ck in candidates:
            if ck in cache:
                arr = cache[ck]
                if arr is not None and len(arr) > 0:
                    out[m] = np.asarray(arr)
                    chosen_key = ck
                    break
        if chosen_key is None:
            print(f"[Info] No data for {topo_prefix} {m} (tried keys: {candidates})")
        else:
            med_abs = np.median(np.abs(out[m]))
            mean_signed = np.mean(out[m])
            print(f"[Load] {topo_prefix} {m}: key='{chosen_key}', n={out[m].size}, med_abs={med_abs:.4f}, mean_signed={mean_signed:.4f}")
    return out

def compute_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.sort(values)
    cdf = np.arange(1, len(s) + 1) / len(s)
    return s, cdf

def determine_xmax_signed(all_arrays: List[np.ndarray], quantile: float = 0.995) -> float:
    """决定对称 X 轴最大绝对值（用于带符号误差）。
    基于所有数组的绝对值分布取给定分位数，同时限制极端值影响。
    """
    valid = [a for a in all_arrays if a is not None and len(a) > 0]
    if not valid:
        print("[Warn] No error arrays found; using default xmax=1.0")
        return 1.0
    merged = np.concatenate(valid)
    abs_vals = np.abs(merged)
    q = np.quantile(abs_vals, quantile)
    maxv = abs_vals.max()
    med = np.median(abs_vals)
    xmax = min(maxv, max(q, med * 5))
    if xmax < 1e-3:
        xmax = 1e-3
    return float(xmax)

def plot_cdf_panel(ax, topo: str, mlp_errors: Dict[str, np.ndarray], kan_errors: Dict[str, np.ndarray], xmax: float, panel_tag: str = None):
    """绘制单个拓扑的带符号相对误差 CDF 面板。

    panel_tag: 诸如 '(a)', '(b)' 的标签，会通过换行放在 xlabel 第二行，便于复用 (c)(d) 等。
    """
    ax.set_title(TOPO_MAP.get(topo, topo))
    has_any = False
    for model_label, errors_dict in [('MLP', mlp_errors), ('KAN', kan_errors)]:
        for metric in METRICS:
            if metric in errors_dict and len(errors_dict[metric]) > 0:
                s, cdf = compute_cdf(errors_dict[metric])
                ax.plot(s, cdf,
                        color=COLORS[model_label][metric],
                        linestyle=LINESTYLES[metric],
                        linewidth=1.5,
                        label=f"{model_label} {metric.capitalize()} ")
                has_any = True
    ax.axvline(0.0, color='red', linestyle=':', linewidth=1.2, alpha=0.9)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    if has_any:
        ax.legend(loc='lower right', framealpha=0.8)
    xlabel = 'Relative Error'
    if panel_tag:
        xlabel += f"\n{panel_tag}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')


def main():
    parser = argparse.ArgumentParser(description='Plot MLP vs KAN Relative Error CDF (1x2)')
    parser.add_argument('--mlp_cache', type=str, required=True, help='MLP evaluation cache npz path')
    parser.add_argument('--kan_cache', type=str, required=True, help='KAN evaluation cache npz path')
    parser.add_argument('--output', type=str, required=True, help='Output image path (.png)')
    parser.add_argument('--quantile', type=float, default=0.99, help='Quantile to determine x-axis max (default 0.99)')
    args = parser.parse_args()

    mlp_cache = load_cache(args.mlp_cache)
    kan_cache = load_cache(args.kan_cache)

    # 准备带符号误差
    mlp_nsf = prepare_signed_errors(mlp_cache, 'nsfnet')
    mlp_gbn = prepare_signed_errors(mlp_cache, 'gbn')
    kan_nsf = prepare_signed_errors(kan_cache, 'nsfnet')
    kan_gbn = prepare_signed_errors(kan_cache, 'gbn')

    # 计算全局对称 xmax
    all_arrays: List[np.ndarray] = []
    for d in [mlp_nsf, mlp_gbn, kan_nsf, kan_gbn]:
        for v in d.values():
            if v is not None and len(v) > 0:
                all_arrays.append(v)
    xmax = determine_xmax_signed(all_arrays, quantile=args.quantile)
    # 强制限制到 ±1 范围
    if xmax > 1.0:
        print(f"[Info] Clamping symmetric x-axis max from ±{xmax:.5f} to ±1.0 as requested")
        xmax = 1.0
    else:
        print(f"[Info] Symmetric x-axis max (<=1): ±{xmax:.5f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.19))  # 保持固定物理尺寸

    plot_cdf_panel(ax1, 'nsfnet', mlp_nsf, kan_nsf, xmax, panel_tag='(a)')
    plot_cdf_panel(ax2, 'gbn', mlp_gbn, kan_gbn, xmax, panel_tag='(b)')

    # 调整边距；之前 right=0.995 导致右侧最大刻度 "1.0" 被裁切，这里留出更多空间
    fig.subplots_adjust(left=0.08, right=0.982, top=0.90, bottom=0.20, wspace=0.30)

    # 打印尺寸以供确认
    w, h = fig.get_size_inches()
    print(f"[Info] Figure size (inches): {w:.2f} x {h:.2f}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 不使用 bbox_inches='tight' 以保持声明的物理尺寸；需要裁剪可改为 pad_inches 调整
    fig.savefig(args.output, dpi=300)
    print(f"Saved figure to {args.output}")

if __name__ == '__main__':
    main()

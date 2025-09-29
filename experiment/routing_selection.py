# -*- coding: utf-8 -*-
"""Port-Matrix Routing Selection (单一模式版)
====================================================
本版本仅保留一种路由解析模式：
    - graph_attr.txt 描述拓扑与每个 (source, port) 对应的 target 节点
    - Routing.txt（位于每个 tar.gz 内）是一个 N×N 端口号矩阵：entry[i][j] = 从 i 发往 j 应使用的出口端口号；-1 表示 i==j 或无路由

流程：
    1. 解析 graph_attr.txt -> 构建 port_map: source -> {port: target}
    2. 解压各 tar.gz, 读取其中 Routing.txt，解析为端口矩阵
    3. 对所有 (src != dst) 根据端口矩阵 + port_map 重建路径（检测循环与最大跳数）
    4. 基于真实或合成 traffic / capacities 使用已训练的 RouteNet 预测延迟
    5. 根据目标（加权总延迟 or 平均延迟）排序候选路由方案

已移除：行式路径格式、节点ID型下一跳矩阵、自动模式等所有其它解析路径，只保留端口矩阵逻辑以简化使用。
"""
from __future__ import annotations
import os
import tarfile
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
import json
import math
import random
import tensorflow as tf

# 导入模型 (假设脚本位置在仓库根路径下的 experiment/ )
from routenet.routenet_tf2 import RouteNet

#############################################
#  已移除 TFRecord 相关代码：仅使用合成流量
#############################################

#############################################
#  数据结构
#############################################
@dataclass
class RoutingCandidate:
    name: str
    path_node_lists: List[List[int]]  # 每条路径的节点序列（节点ID）
    path_links: List[List[int]]       # 每条路径对应的 link index 序列（填充后）
    n_paths: int
    n_links: int
    link_map: Dict[Tuple[int, int], int]  # 映射 (u,v) -> link_idx
    # 可选：记录 (src,dst) 对便于与 TFRecord traffic 对齐
    sd_pairs: Optional[List[Tuple[int,int]]] = None

#############################################
#  工具函数 & 解析函数
#############################################

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover
        pass

#############################################
#  端口矩阵解析辅助
#############################################

def _read_matrix(lines: List[str]) -> Optional[List[List[int]]]:
    data: List[List[int]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # 支持空格或逗号分隔
        if ',' in ln:
            parts = [p for p in ln.split(',') if p.strip() != '']
        else:
            parts = [p for p in ln.split() if p.strip() != '']
        row: List[int] = []
        for p in parts:
            try:
                row.append(int(p))
            except ValueError:
                return None
        data.append(row)
    # 检查是否为方阵
    if not data:
        return None
    n = len(data)
    for r in data:
        if len(r) != n:
            return None
    return data


def parse_routing_file_port_matrix(mat: List[List[int]], port_map: Dict[int, Dict[int,int]], max_hops: int = 64) -> Tuple[List[List[int]], List[Tuple[int,int]]]:
    """根据端口矩阵与 graph_attr 的 (src,port)->next hop 映射重建所有 (src,dst) 路径。

    规则：
      - mat[i][j] = 出口端口号，-1 表示 i==j 或无路由
      - 若端口号在 port_map[i] 中不存在 => 路由无效
      - 检测环：若下一跳已在当前路径，判为无效
      - 超过 max_hops 也判无效
    返回：有效的节点序列列表、对应 (src,dst) 对列表。
    """
    N = len(mat)
    paths: List[List[int]] = []
    sd_pairs: List[Tuple[int,int]] = []
    for src in range(N):
        for dst in range(N):
            if src == dst:
                continue
            port = mat[src][dst]
            if port < 0:
                continue  # 明确无路由
            if src not in port_map or port not in port_map[src]:
                continue
            cur = src
            seq = [src]
            visited = {src}
            valid = True
            hops = 0
            while cur != dst and hops < max_hops:
                p = mat[cur][dst]
                if p < 0:
                    valid = False
                    break
                if cur not in port_map or p not in port_map[cur]:
                    valid = False
                    break
                nxt = port_map[cur][p]
                if nxt in visited:
                    valid = False
                    break
                seq.append(nxt)
                visited.add(nxt)
                cur = nxt
                hops += 1
            if cur != dst:
                valid = False
            if valid and len(seq) >= 2:
                # 确保终点是 dst
                if seq[-1] != dst:
                    seq.append(dst)
                paths.append(seq)
                sd_pairs.append((src, dst))
    return paths, sd_pairs


def parse_graph_attr(graph_attr_path: str) -> Dict[int, Dict[int,int]]:
    """解析 graph_attr.txt，构建 (source -> {port: target}) 映射。
    只关心 edge 块中的 source / target / port。
    """
    port_map: Dict[int, Dict[int,int]] = {}
    if not os.path.exists(graph_attr_path):
        raise FileNotFoundError(f"graph_attr 不存在: {graph_attr_path}")
    with open(graph_attr_path, 'r', encoding='utf-8', errors='ignore') as f:
        in_edge = False
        cur = {}
        for line in f:
            line = line.strip()
            if line.startswith('edge'):  # 进入 edge [ 或单行 edge
                in_edge = True
                cur = {}
            if in_edge:
                if line.startswith('source'):
                    cur['source'] = int(line.split()[1])
                elif line.startswith('target'):
                    cur['target'] = int(line.split()[1])
                elif line.startswith('port'):
                    # 行格式: port 0
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            cur['port'] = int(parts[1])
                        except ValueError:
                            pass
                elif line.startswith(']') and in_edge:
                    # 结束一个 edge block
                    if {'source','target','port'} <= cur.keys():
                        src = cur['source']; tgt = cur['target']; p = cur['port']
                        port_map.setdefault(src, {})[p] = tgt
                    in_edge = False
    return port_map


def parse_port_routing_matrix(lines: List[str], port_map: Dict[int, Dict[int,int]], max_hops: int) -> Tuple[List[List[int]], List[Tuple[int,int]]]:
    """专用：读取端口矩阵，直接按端口模式解析。"""
    mat = _read_matrix(lines)
    if mat is None:
        return [], []
    return parse_routing_file_port_matrix(mat, port_map, max_hops=max_hops)


def build_link_indices(path_node_lists: List[List[int]]) -> Tuple[Dict[Tuple[int,int], int], List[List[int]]]:
    """为所有 (u,v) 有向边分配 link index，并返回每条路径的 link index 序列列表。"""
    link_map: Dict[Tuple[int,int], int] = {}
    next_idx = 0
    path_links: List[List[int]] = []
    for nodes in path_node_lists:
        links_seq = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            key = (u, v)
            if key not in link_map:
                link_map[key] = next_idx
                next_idx += 1
            links_seq.append(link_map[key])
        path_links.append(links_seq)
    return link_map, path_links


def flatten_paths(path_links: List[List[int]]):
    links = []
    paths = []
    sequences = []
    for p_idx, link_seq in enumerate(path_links):
        for s_idx, l in enumerate(link_seq):
            links.append(l)
            paths.append(p_idx)
            sequences.append(s_idx)
    return np.array(links, dtype=np.int64), np.array(paths, dtype=np.int64), np.array(sequences, dtype=np.int64)


def discover_routing_candidates(dataset_dir: str, limit: int | None = None, max_hops: int = 64, port_map: Optional[Dict[int, Dict[int,int]]] = None) -> List[RoutingCandidate]:
    """扫描 dataset_dir 下的 *.tar.gz，尝试解压并读取 Routing.txt。

    返回每个候选的节点路径与分配好的 link 索引。
    如果一个压缩包没有 Routing.txt 或解析失败，则跳过。
    """
    candidates: List[RoutingCandidate] = []
    tgzs = [f for f in os.listdir(dataset_dir) if f.endswith('.tar.gz')]
    if limit is not None:
        tgzs = tgzs[:limit]
    for fname in tgzs:
        full_path = os.path.join(dataset_dir, fname)
        try:
            with tarfile.open(full_path, 'r:gz') as tar:
                # 找 Routing.txt
                routing_member = None
                for m in tar.getmembers():
                    nm_low = m.name.lower()
                    if nm_low.endswith('routing.txt'):
                        routing_member = m
                        break
                if routing_member is None:
                    print(f"[WARN] {fname} 没找到 Routing.txt，跳过")
                    continue
                f = tar.extractfile(routing_member)
                if f is None:
                    print(f"[WARN] {fname} Routing.txt 无法读取，跳过")
                    continue
                raw = f.read().decode('utf-8', errors='ignore').splitlines()
                tmp_path_lists, tmp_sd_pairs = parse_port_routing_matrix(raw, port_map, max_hops=max_hops)
                if not tmp_path_lists:
                    print(f"[WARN] {fname} 端口矩阵解析失败或为空，跳过 (请检查格式 & graph_attr 对应性)")
                    continue
                link_map, path_links = build_link_indices(tmp_path_lists)
                cand = RoutingCandidate(
                    name=fname.replace('.tar.gz',''),
                    path_node_lists=tmp_path_lists,
                    path_links=path_links,
                    n_paths=len(tmp_path_lists),
                    n_links=len(link_map),
                    link_map=link_map,
                    sd_pairs=tmp_sd_pairs
                )
                candidates.append(cand)
                print(f"[INFO] 解析 {fname} (port-matrix): 有效路径 {cand.n_paths}, 唯一链路 {cand.n_links}")
        except Exception as e:  # pragma: no cover
            print(f"[ERROR] 读取 {fname} 出错: {e}")
            continue
    return candidates


def generate_traffic_for_candidate(cand: RoutingCandidate, mode: str, seed: int) -> np.ndarray:
    """根据模式生成 traffic 向量 (长度 = n_paths)。

    modes:
      - uniform: 全 0.18 (接近原 scale 反标准化前中值) -> 经 scale: (0.18-0.18)/0.15=0
      - random: U[0.15,0.21) 稳定随机
      - onehot: 仅第一条路径 0.25 其余 0.18
    最终会做同训练 parse_fn 中的 scale: (x-0.18)/0.15
    """
    rng = np.random.default_rng(seed)
    if mode == 'uniform':
        raw = np.full(cand.n_paths, 0.18, dtype=np.float32)
    elif mode == 'random':
        raw = rng.uniform(0.15, 0.21, size=cand.n_paths).astype(np.float32)
    elif mode == 'onehot':
        raw = np.full(cand.n_paths, 0.18, dtype=np.float32)
        if cand.n_paths > 0:
            raw[0] = 0.25
    else:
        raise ValueError(f"Unsupported traffic_mode: {mode}")
    scaled = (raw - 0.18) / 0.15
    return scaled


def generate_capacities(cand: RoutingCandidate, default_capacity: float = 10.0) -> np.ndarray:
    """默认所有链路容量相同。模型内部期望已经 /10 归一化，所以此处返回 (capacity/10)。"""
    caps = np.full(cand.n_links, default_capacity, dtype=np.float32)
    return caps / 10.0


def build_model(config_overrides: Dict, model_dir: str) -> RouteNet:
    # 基础配置与训练脚本一致 (仅推理使用)；可通过 overrides 修改。
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
        'use_dropout': False,  # 推理关闭 dropout
        'sae_enabled': False,
        'sae_hidden_dim': 64,
        'sae_latent_dim': 16,
        'sae_activation': 'relu',
        'loss_variant': 'standard'
    }
    config.update(config_overrides)
    model = RouteNet(config, output_units=2, final_activation=None)
    weights_path = os.path.join(model_dir, 'best_delay_model.weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")
    # 构造一次假输入以建立变量
    # 这里使用最小 dummy 图
    dummy_features = {
        'traffic': tf.zeros([1], tf.float32),
        'capacities': tf.zeros([1], tf.float32),
        'links': tf.constant([0], tf.int64),
        'paths': tf.constant([0], tf.int64),
        'sequences': tf.constant([0], tf.int64),
        'n_links': tf.constant(1, tf.int64),
        'n_paths': tf.constant(1, tf.int64)
    }
    _ = model(dummy_features, training=False)
    model.load_weights(weights_path)
    print(f"[MODEL] Loaded weights from {weights_path}")
    return model


def predict_candidate(model: RouteNet, cand: RoutingCandidate, traffic: np.ndarray, capacities: np.ndarray, packets: Optional[np.ndarray] = None, objective_weight: str = 'traffic', subset_indices: Optional[Sequence[int]] = None) -> Dict:
    """对单个候选路由方案进行推理，并返回指标。

    objective_weight:
      - 'traffic': Σ traffic_i * delay_i
      - 'packets': 若提供 packets，则 Σ packets_i * delay_i，否则退化为 traffic
      - 'uniform': 直接平均 (与 mode='mean' 类似，但仍记录 sum_weighted_delay)
    """
    # 可能对路径做子集过滤
    if subset_indices is not None:
        subset_indices = list(subset_indices)
        path_links = [cand.path_links[i] for i in subset_indices]
        links_arr, paths_arr, seqs_arr = flatten_paths(path_links)
        traffic = traffic[subset_indices]
        if packets is not None:
            packets = packets[subset_indices]
        n_paths = len(subset_indices)
    else:
        links_arr, paths_arr, seqs_arr = flatten_paths(cand.path_links)
        n_paths = cand.n_paths
    features = {
        'traffic': tf.convert_to_tensor(traffic, tf.float32),
        'capacities': tf.convert_to_tensor(capacities, tf.float32),
        'links': tf.convert_to_tensor(links_arr, tf.int64),
        'paths': tf.convert_to_tensor(paths_arr, tf.int64),
        'sequences': tf.convert_to_tensor(seqs_arr, tf.int64),
        'n_links': tf.constant(cand.n_links, tf.int64),
        'n_paths': tf.constant(n_paths, tf.int64)
    }
    preds = model(features, training=False)
    # preds: [n_paths, 2] -> 第一列 loc (预测 delay 均值)，第二列 raw scale
    delay_loc = preds[:, 0].numpy()
    raw_scale = preds[:, 1].numpy()
    sigma = np.log1p(np.exp(raw_scale)) + 1e-6  # softplus
    if objective_weight == 'packets' and packets is not None:
        weight_vec = packets
    elif objective_weight == 'uniform':
        weight_vec = np.ones_like(traffic, dtype=np.float32)
    else:
        weight_vec = traffic
    sum_weighted = float(np.sum(weight_vec * delay_loc))
    return {
        'delay_mean': delay_loc,
        'delay_sigma': sigma,
        'traffic': traffic,
        'packets': packets,
        'n_paths': n_paths,
        'objective_sum': sum_weighted,
        'objective_mean': float(np.mean(delay_loc)),
    }


def select_od_subset(cand: RoutingCandidate, od_limit: Optional[int], od_fraction: Optional[float], seed: int) -> Optional[List[int]]:
    """选择 (src,dst) 对子集，返回所选路径索引列表。

    逻辑：
      - 若 od_limit is None 且 od_fraction is None -> 返回 None (不裁剪)
      - 基于 cand.sd_pairs 顺序（已与 path_node_lists 对齐）抽样
      - od_fraction 优先于 od_limit? 此处：若同时给出，先用 fraction 计算数量，再与 limit 取最小
    """
    if cand.sd_pairs is None:
        return None
    total = len(cand.sd_pairs)
    if total == 0:
        return None
    if od_limit is None and od_fraction is None:
        return None
    if od_fraction is not None:
        k = max(1, int(math.ceil(total * min(max(od_fraction, 0.0), 1.0))))
    else:
        k = total
    if od_limit is not None:
        k = min(k, max(1, od_limit))
    if k >= total:
        return None  # 不需要裁剪
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    chosen = sorted(indices[:k])
    return chosen


def main(args):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 发现所有候选 routing 方案
    port_map = parse_graph_attr(args.graph_attr)
    print(f"[GRAPH] 解析 graph_attr: {len(port_map)} 个源节点含端口映射")
    candidates = discover_routing_candidates(
        args.dataset_dir,
        limit=args.limit_candidates,
        max_hops=args.max_hops,
        port_map=port_map
    )
    if not candidates:
        print("[FATAL] 未找到任何候选 routing 方案。")
        return

    # 构建模型
    model = build_model({}, args.model_dir)

    results = []
    for cand in candidates:
        # 始终使用合成流量（与 TFRecord 解耦）
        traffic_vec = generate_traffic_for_candidate(cand, args.traffic_mode, seed=args.seed + cand.n_paths)
        capacities_vec = generate_capacities(cand, default_capacity=args.default_capacity)
        subset_indices = select_od_subset(cand, args.od_limit, args.od_fraction, seed=args.seed)
        if args.objective_weight == 'packets':
            packets_vec = traffic_vec * args.packet_scale  # 简单放大作为伪 packets
        else:
            packets_vec = None
        source_mode = 'synthetic'
        pred = predict_candidate(
            model, cand, traffic_vec, capacities_vec,
            packets=packets_vec,
            objective_weight=args.objective_weight,
            subset_indices=subset_indices
        )
        objective = pred['objective_sum'] if args.mode == 'sum' else pred['objective_mean']
        results.append({
            'name': cand.name,
            'n_paths': cand.n_paths,
            'n_links': cand.n_links,
            'objective': objective,
            'subset_used': subset_indices is not None,
            'subset_size': pred['n_paths'],
            'objective_type': args.mode,
            'sum_weighted_delay': pred['objective_sum'],
            'mean_delay': pred['objective_mean'],
            'data_source': source_mode,
            'objective_weight': args.objective_weight
        })
        print(f"[CAND] {cand.name}: objective={objective:.6f} (sum={pred['objective_sum']:.6f}, mean={pred['objective_mean']:.6f}) src={source_mode}" + (f" SUBSET {pred['n_paths']} paths" if subset_indices is not None else ""))

    # 排序
    results_sorted = sorted(results, key=lambda x: x['objective'])

    # 取前 K
    top_k = results_sorted[:args.top_k]
    print("\n=== Top {} Routing Schemes (mode: {}) ===".format(len(top_k), args.mode))
    for i, r in enumerate(top_k, 1):
        print(f"{i}. {r['name']} objective={r['objective']:.6f} (sum={r['sum_weighted_delay']:.6f}, mean={r['mean_delay']:.6f}) paths={r['n_paths']} links={r['n_links']}")

    # 保存 JSON/CSV
    json_path = os.path.join(args.output_dir, 'routing_ranking.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'mode': args.mode, 'results': results_sorted, 'top_k': top_k}, f, ensure_ascii=False, indent=2)
    print(f"[OUT] 排序结果写入 {json_path}")

    csv_path = os.path.join(args.output_dir, 'routing_ranking.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('rank,name,objective,objective_type,sum_weighted_delay,mean_delay,n_paths,n_links,data_source,objective_weight\n')
        for idx, r in enumerate(results_sorted, 1):
            f.write(f"{idx},{r['name']},{r['objective']},{r['objective_type']},{r['sum_weighted_delay']},{r['mean_delay']},{r['n_paths']},{r['n_links']},{r.get('data_source','')},{r.get('objective_weight','')}\n")
    print(f"[OUT] 排序结果写入 {csv_path}")

    # 保存最优 routing 的 meta
    if top_k:
        best_meta_path = os.path.join(args.output_dir, 'best_routing_meta.json')
        with open(best_meta_path, 'w', encoding='utf-8') as f:
            json.dump(top_k[0], f, ensure_ascii=False, indent=2)
        print(f"[OUT] 最优方案信息写入 {best_meta_path}")

        # 若开启 SAE 且请求导出 latent，则对最优方案再跑一次并输出每条链路 latent
        if args.enable_sae and args.dump_sae_latent:
            if not model.sae_enabled:
                print("[WARN] enable_sae=True 但模型未启用 SAE (权重不匹配?)，跳过 latent 导出")
            else:
                # 重新推理，使用完整路径（不裁剪），请求返回 aux
                traffic_vec = generate_traffic_for_candidate(candidates[0], args.traffic_mode, seed=args.seed + candidates[0].n_paths)
                capacities_vec = generate_capacities(candidates[0], default_capacity=args.default_capacity)
                if args.objective_weight == 'packets':
                    packets_vec = traffic_vec * args.packet_scale
                else:
                    packets_vec = None
                links_arr, paths_arr, seqs_arr = flatten_paths(candidates[0].path_links)
                full_features = {
                    'traffic': tf.convert_to_tensor(traffic_vec, tf.float32),
                    'capacities': tf.convert_to_tensor(capacities_vec, tf.float32),
                    'links': tf.convert_to_tensor(links_arr, tf.int64),
                    'paths': tf.convert_to_tensor(paths_arr, tf.int64),
                    'sequences': tf.convert_to_tensor(seqs_arr, tf.int64),
                    'n_links': tf.constant(candidates[0].n_links, tf.int64),
                    'n_paths': tf.constant(candidates[0].n_paths, tf.int64)
                }
                preds, aux = model(full_features, training=False, return_aux=True)
                if aux is None:
                    print("[WARN] SAE 未返回辅助输出，跳过 latent 导出")
                else:
                    latent = aux['latent'].numpy()  # [n_links, latent_dim]
                    # 构造 link_idx -> (u,v) 反向映射
                    inv_map = [None] * len(candidates[0].link_map)
                    for (u,v), idx in candidates[0].link_map.items():
                        inv_map[idx] = (u,v)
                    latent_csv = os.path.join(args.output_dir, 'best_sae_latent.csv')
                    with open(latent_csv, 'w', encoding='utf-8') as f:
                        header = ['link_index','src','dst'] + [f'z{i}' for i in range(latent.shape[1])]
                        f.write(','.join(header) + '\n')
                        for idx,(u,v) in enumerate(inv_map):
                            row = [str(idx), str(u), str(v)] + [f"{x:.6f}" for x in latent[idx]]
                            f.write(','.join(row) + '\n')
                    print(f"[OUT] SAE latent 写入 {latent_csv} (shape {latent.shape})")

    print("完成 Routing 方案评估。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Routing Scheme Selection via Trained RouteNet Delay Model')
    parser.add_argument('--model_dir', type=str, required=True, help='包含 best_delay_model.weights.h5 的目录')
    parser.add_argument('--dataset_dir', type=str, required=True, help='包含若干 *.tar.gz (含 Routing.txt) 的目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出排名结果目录')
    parser.add_argument('--mode', type=str, choices=['sum','mean'], default='sum', help='目标：sum 加权总延迟 或 mean 平均延迟')
    parser.add_argument('--traffic_mode', type=str, choices=['uniform','random','onehot'], default='uniform', help='生成 traffic 的方式 (后续可扩展为从 TFRecord 读取)')
    parser.add_argument('--default_capacity', type=float, default=10.0, help='默认链路容量 (未提供真实数据时使用)')
    parser.add_argument('--top_k', type=int, default=5, help='输出前 K 个方案')
    parser.add_argument('--limit_candidates', type=int, default=None, help='限制候选数量 (调试用)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # 使用真实 TFRecord 样本
    parser.add_argument('--objective-weight', type=str, choices=['traffic','packets','uniform'], default='traffic', help='加权方式：traffic=Σ(traffic*delay), packets=Σ(packets*delay), uniform=平均')
    parser.add_argument('--packet-scale', type=float, default=1000.0, help='objective-weight=packets 时：packets=traffic*packet_scale')
    parser.add_argument('--ignore-real-capacities', action='store_true', help='(已无效) 为兼容保留，不再使用真实容量')
    parser.add_argument('--max-hops', type=int, default=64, help='端口矩阵解析路径允许的最大跳数')
    parser.add_argument('--graph-attr', type=str, required=True, help='graph_attr.txt 路径 (端口->下一跳 映射数据)')
    # 新增: 限制 (src,dst) 对子集
    parser.add_argument('--od-limit', type=int, default=None, help='只评估前/随机子集的若干 (src,dst) 对 (路径集合)')
    parser.add_argument('--od-fraction', type=float, default=None, help='随机抽样比例 (0,1]，与 --od-limit 并存时取较小结果')
    # SAE latent 导出
    parser.add_argument('--enable-sae', action='store_true', help='启用并加载带 SAE 的模型 (需权重兼容)')
    parser.add_argument('--dump-sae-latent', action='store_true', help='输出最优路由方案的所有链路 SAE latent')

    args = parser.parse_args()
    # 将配置覆盖项临时存入，供 main 中使用；为简化直接在 args 上挂一个字段
    args.model_overrides = {}
    if args.enable_sae:
        args.model_overrides.update({'sae_enabled': True})
    # 为了使用 overrides，需要修改 main 内部的 build_model 调用
    # 这里简单地 monkey patch 原 build_model 调用方式：重新定义一个封装 main 的逻辑
    original_main = main
    def wrapped_main(a):
        # 复制原 main 但在创建模型前注入 overrides
        # 为避免重写大量代码，这里在 global 作用域临时替换 build_model 行为
        global build_model
        _orig_build = build_model
        def _patched_build(overrides, model_dir):
            merged = dict(overrides)
            merged.update(a.model_overrides)
            return _orig_build(merged, model_dir)
        build_model = _patched_build
        try:
            original_main(a)
        finally:
            build_model = _orig_build
    wrapped_main(args)

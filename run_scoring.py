# -*- coding: utf-8 -*-
import os
import argparse
from collections import defaultdict
from typing import List, Set, Dict
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import util
from tqdm import tqdm
from run_recall import get_llm_data, get_query, get_active_i_set, get_or_compute_entity_embeddings, get_semantically_similar_nodes, get_structural_nodes,get_candidate_set


# =====================================================================================
# 1. 从 TPNet.py 移植的核心模块
# =====================================================================================

class RandomProjectionModule(nn.Module):
    """
    该模块通过随机特征传播维护一系列时序游走矩阵，并从中提取成对特征。
    移植自 lxd99/tpnet/TPNet.py
    """
    def __init__(self, node_num: int, edge_num: int, dim_factor: int, num_layer: int, time_decay_weight: float,
                 device: str, beginning_time: np.float64, enforce_dim: int):
        super(RandomProjectionModule, self).__init__()
        self.node_num = node_num
        self.edge_num = edge_num
        if enforce_dim != -1:
            self.dim = enforce_dim
        else:
            # 维度设置为 min(dim_factor * log(2*edge_num), node_num)
            self.dim = min(int(math.log(self.edge_num * 2)) * dim_factor, node_num)
        
        print(f"--- RandomProjectionModule 初始化 ---")
        print(f"   结构向量维度 (dim): {self.dim}")
        
        self.num_layer = num_layer
        self.time_decay_weight = time_decay_weight
        self.beginning_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.now_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.device = device
        self.random_projections = nn.ParameterList()

        for i in range(self.num_layer + 1):
            if i == 0:
                # 第0层是随机初始化的基础投影
                self.random_projections.append(
                    nn.Parameter(torch.normal(0, 1 / math.sqrt(self.dim), (self.node_num, self.dim)),
                                 requires_grad=False))
            else:
                # 更高阶的游走投影初始化为0
                self.random_projections.append(
                    nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """按批次更新时序游走矩阵。"""
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)
        time_weight = torch.exp(-self.time_decay_weight * (next_time - node_interact_times))[:, None]

        # 时间衰减：更新从上次更新时间到当前批次时间的衰减
        for i in range(1, self.num_layer + 1):
            decay_factor = np.power(np.exp(-self.time_decay_weight * (next_time - self.now_time.cpu().numpy())), i)
            self.random_projections[i].data *= decay_factor

        # 增量更新：加入新交互的影响
        for i in range(self.num_layer, 0, -1):
            src_update_messages = self.random_projections[i - 1][dst_node_ids] * time_weight
            dst_update_messages = self.random_projections[i - 1][src_node_ids] * time_weight
            self.random_projections[i].scatter_add_(dim=0, index=src_node_ids[:, None].expand(-1, self.dim),
                                                    src=src_update_messages)
            self.random_projections[i].scatter_add_(dim=0, index=dst_node_ids[:, None].expand(-1, self.dim),
                                                    src=dst_update_messages)
        
        self.now_time.data = torch.tensor(next_time, device=self.device)
    
    def get_all_projections(self) -> torch.Tensor:
        """获取所有节点的最终结构向量 (所有阶的投影拼接)。"""
        # 我们将不同阶数的投影拼接起来，形成一个更丰富的结构表示
        return torch.cat(list(self.random_projections), dim=1)



# =====================================================================================
# 计算历史伙伴权重的函数
# =====================================================================================

'''
def calculate_historical_partner_weights(
    q_head_id: int, 
    q_time: int, 
    all_history_events: list, 
    time_decay_rate: float
) -> dict:
    """
    为查询节点的每一个历史伙伴计算一个综合权重。
    该权重综合了交互频率和时间衰减。

    参数:
        q_head_id (int): 查询节点的ID。
        q_time (int): 当前的查询时间戳, 用于计算时间差。
        all_history_events (list): 包含所有历史交互事件的列表 (训练集+验证集)。
        time_decay_rate (float): 时间衰减公式中的lambda参数。

    返回:
        dict: 一个字典，键是历史伙伴的ID，值是其对应的计算权重。
    """
    # 步骤 1: 遍历历史，统计每个伙伴的交互频率和最近交互时间
    # partner_stats 的结构: {partner_id: {'count': int, 'last_time': int}}
    partner_stats = defaultdict(lambda: {'count': 0, 'last_time': 0})
    
    # 我们只关心在查询时间点之前发生的交互
    for u, r, i, ts, _, _ in all_history_events:
        if ts >= q_time:
            continue
        
        partner_id = -1
        if u == q_head_id:
            partner_id = i
        elif i == q_head_id:
            partner_id = u
        
        if partner_id != -1:
            # 更新交互次数
            partner_stats[partner_id]['count'] += 1
            # 更新最近交互时间
            if ts > partner_stats[partner_id]['last_time']:
                partner_stats[partner_id]['last_time'] = ts

    # 步骤 2: 计算每个伙伴的最终权重
    partner_weights = {}
    for partner_id, stats in partner_stats.items():
        frequency = stats['count']
        last_interaction_time = stats['last_time']
        
        # 计算时间衰减因子
        time_diff = q_time - last_interaction_time
        time_decay_factor = np.exp(-time_decay_rate * time_diff)
        
        # 最终权重 = 频率 * 时间衰减因子
        final_weight = frequency * time_decay_factor
        partner_weights[partner_id] = final_weight
        
    return partner_weights
'''
def calculate_historical_partner_weights(
    q_head_id: int, 
    q_time: int, 
    node_history: defaultdict[int, list], 
    time_decay_rate: float
) -> dict:
    """
    【优化版】
    使用预处理好的节点历史记录，只遍历当前查询节点的相关历史，避免全量扫描。
    """
    partner_stats = defaultdict(lambda: {'count': 0, 'last_time': 0})
    
    # 只需遍历当前查询节点的历史，效率极高
    if q_head_id not in node_history:
        return {}

    for partner_id, _, timestamp in node_history.get(q_head_id, []):
        if timestamp >= q_time:
            break
        partner_stats[partner_id]['count'] += 1
        partner_stats[partner_id]['last_time'] = timestamp

    partner_weights = {}
    for partner_id, stats in partner_stats.items():
        frequency = stats['count']
        last_interaction_time = stats['last_time']
        
        time_diff = q_time - last_interaction_time
        time_decay_factor = np.exp(-time_decay_rate * time_diff)
        final_weight = frequency * time_decay_factor
        partner_weights[partner_id] = final_weight
        
    return partner_weights


def compute_similarity_matrix(
    candidate_list: List[int],
    historical_partner_ids: List[int],
    all_embeddings: torch.Tensor,
    sim_type: str = 'semantic'
) -> torch.Tensor:
    """
    计算候选节点列表和历史伙伴列表之间的相似度矩阵。

    参数:
        candidate_list (List[int]): 候选节点的ID列表。
        historical_partner_ids (List[int]): 历史伙伴的ID列表。
        all_embeddings (torch.Tensor): 包含所有节点特征向量的张量。
                                      对于语义是 'all_entity_embeddings'，
                                      对于结构可以是 'all_structural_embeddings'。
        sim_type (str): 'semantic' 或 'structural'，用于打印信息。

    返回:
        torch.Tensor: 一个形状为 [候选节点数, 历史伙伴数] 的相似度矩阵。
    """
    # print(f"--- 正在计算 {sim_type} 相似度矩阵 ---")
    
    # 步骤 1: 使用高级索引从总向量张量中高效地抽取出所需的向量
    # .to(device) 确保所有张量在同一设备上
    device = all_embeddings.device
    candidate_embeddings = all_embeddings[torch.tensor(candidate_list, device=device)]
    historical_embeddings = all_embeddings[torch.tensor(historical_partner_ids, device=device)]

    # 步骤 2: 计算两组向量之间的余弦相似度矩阵
    # util.cos_sim 会返回一个 [len(candidate_list), len(historical_partner_ids)] 的矩阵
    similarity_matrix = util.cos_sim(candidate_embeddings, historical_embeddings)
    
    # print(f"   {sim_type} 相似度矩阵计算完成，矩阵形状为: {similarity_matrix.shape}")
    return similarity_matrix

def calculate_candidate_scores(
    q_head_id: int,
    q_time: int,
    candidate_set: Set[int],
    node_history: Dict,
    all_semantic_embeddings: torch.Tensor,
    all_structural_embeddings: torch.Tensor,
    lambda_time: float,
    alpha: float,
    beta: float
) -> Dict[int, float]:
    """
    计算所有候选节点的最终价值评分。

    参数:
        q_head_id, q_time: 查询信息。
        candidate_set: 候选节点ID集合。
        node_history: 历史事件字典。
        all_semantic_embeddings, all_structural_embeddings: 节点的语义和结构向量。
        lambda_time, alpha, beta: 超参数。

    返回:
        一个字典，键是候选节点ID，值是其最终分数。
    """
    # 步骤 1: 计算历史伙伴权重
    partner_weights_dict = calculate_historical_partner_weights(q_head_id, q_time, node_history, lambda_time)
    if not partner_weights_dict:
        print("警告: 查询节点没有历史伙伴，无法计算评分。")
        return {node_id: 0.0 for node_id in candidate_set}

    # 步骤 2: 准备有序列表，用于矩阵运算
    candidate_list = sorted(list(candidate_set))
    historical_partner_ids = sorted(list(partner_weights_dict.keys()))
    # 创建一个与 historical_partner_ids 顺序一致的权重向量
    device = all_semantic_embeddings.device
    W_hist_partners = torch.tensor([partner_weights_dict[pid] for pid in historical_partner_ids], device=device, dtype=torch.float32)

    # 步骤 3: 计算语义和结构相似度矩阵
    semantic_sim_matrix = compute_similarity_matrix(candidate_list, historical_partner_ids, all_semantic_embeddings, 'semantic')
    structural_sim_matrix = compute_similarity_matrix(candidate_list, historical_partner_ids, all_structural_embeddings, 'structural')

    # 步骤 4: 计算总相似度矩阵
    total_sim_matrix = alpha * structural_sim_matrix + (1 - alpha) * semantic_sim_matrix

    # 步骤 5: 计算历史相似度得分 (Score_hist)
    # 使用矩阵乘法高效完成加权聚合
    # [num_candidates, num_partners] @ [num_partners] -> [num_candidates]
    score_hist_vec = total_sim_matrix @ W_hist_partners

    # 步骤 6: 计算自身相似度得分 (Score_self)
    q_head_semantic_emb = all_semantic_embeddings[q_head_id].unsqueeze(0)
    q_head_structural_emb = all_structural_embeddings[q_head_id].unsqueeze(0)
    candidate_semantic_embs = all_semantic_embeddings[torch.tensor(candidate_list, device=device)]
    candidate_structural_embs = all_structural_embeddings[torch.tensor(candidate_list, device=device)]
    
    self_sim_semantic = util.cos_sim(q_head_semantic_emb, candidate_semantic_embs).squeeze(0)
    self_sim_structural = util.cos_sim(q_head_structural_emb, candidate_structural_embs).squeeze(0)
    score_self_vec = alpha * self_sim_structural + (1 - alpha) * self_sim_semantic

    # 步骤 7: 计算最终混合得分 (Score_final)
    final_score_vec = beta * score_hist_vec + (1 - beta) * score_self_vec
    
    # 将最终得分打包成字典返回
    final_scores_dict = {node_id: score.item() for node_id, score in zip(candidate_list, final_score_vec)}
    
    return final_scores_dict


# =====================================================================================
# 主程序入口
# =====================================================================================

def main():
    # --- 设置命令行参数 ---
    parser = argparse.ArgumentParser(description="动态图链接预测候选集价值评分脚本")
    # 数据和路径参数
    parser.add_argument('--dataset_name', type=str, default='ICEWS1819', help='数据集名称')
    parser.add_argument('--base_data_dir', type=str, default='/home/gtang/DTGB-main/DyLink_Datasets', help='数据集存放的基础目录')
    parser.add_argument('--local_model_path', type=str, default='/home/gtang/pretrain_model/all-mpnet-base-v2', help='本地语义模型路径')
    # 召回参数
    parser.add_argument('--k_time', type=int, default=20, help='时间维度召回数量')
    parser.add_argument('--k_struc', type=int, default=20, help='结构维度召回数量')
    parser.add_argument('--k_sem', type=int, default=20, help='语义维度召回数量')
    # 结构向量模型 (RP) 参数
    parser.add_argument('--rp_dim_factor', type=int, default=10, help='结构向量维度因子')
    parser.add_argument('--rp_num_layer', type=int, default=3, help='结构向量的游走最大跳数')
    parser.add_argument('--rp_time_decay_weight', type=float, default=1e-7, help='结构向量的时间衰减权重')
    # 评分超参数
    parser.add_argument('--lambda_time', type=float, default=0.01, help='历史伙伴权重的时间衰减率')
    parser.add_argument('--alpha', type=float, default=0.5, help='结构/语义相似度的加权系数')
    parser.add_argument('--beta', type=float, default=0.5, help='历史/自身相似度得分的加权系数')
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备 (cuda or cpu)')
    args = parser.parse_args()

    # --- 1. 加载数据 ---
    print("--- 1. 正在加载和处理数据... ---")
    entities, _, train_list, val_list, test_list, node_num, rel_num = get_llm_data(
        args.dataset_name, args.base_data_dir, val_ratio=0.15, test_ratio=0.15
    )
    all_events = train_list + val_list + test_list
    all_history_events = train_list + val_list
    
    # --- 2. 准备向量 ---
    print("\n--- 2. 正在准备节点特征向量... ---")
    # a. 获取语义向量
    embedding_file_path = os.path.join(args.base_data_dir, args.dataset_name, f"{args.dataset_name}_entity_embeddings.pt")
    all_semantic_embeddings = get_or_compute_entity_embeddings(
        entities=entities, model_path=args.local_model_path, file_path=embedding_file_path, device=args.device
    )
    
    # b. 【新】计算结构向量
    print("\n--- 正在计算结构向量 (Temporal Walk Matrix Projection)... ---")
    rp_module = RandomProjectionModule(
        node_num=node_num, edge_num=len(all_history_events), dim_factor=args.rp_dim_factor,
        num_layer=args.rp_num_layer, time_decay_weight=args.rp_time_decay_weight,
        device=args.device, beginning_time=all_history_events[0][3], enforce_dim=-1
    ).to(args.device)

    # 分批次“训练”RP模块以构建结构向量
    batch_size = 200
    for i in tqdm(range(0, len(all_history_events), batch_size), desc="  构建结构向量中"):
        batch_events = all_history_events[i: i + batch_size]
        src_nodes = np.array([event[0] for event in batch_events])
        dst_nodes = np.array([event[2] for event in batch_events])
        timestamps = np.array([event[3] for event in batch_events])
        rp_module.update(src_nodes, dst_nodes, timestamps)

    # 获取最终的结构向量
    all_structural_embeddings = rp_module.get_all_projections().detach()
    print(f"--- 结构向量构建完成，向量形状为: {all_structural_embeddings.shape} ---")

    # --- 3. 获取候选集 ---
    query_event = test_list[0]
    candidate_set_C = get_candidate_set(
        query_event=query_event, all_events=all_events, all_entity_embeddings=all_semantic_embeddings,
        k_time=args.k_time, k_struc=args.k_struc, k_sem=args.k_sem
    )

    # --- 4. 执行价值评分 ---
    if not candidate_set_C:
        print("候选集为空，无法进行评分。")
        return
        
    print("\n--- 4. 正在对候选集节点进行价值评分... ---")
    q_head, _, _, q_time = get_query(query_event)
    final_scores = calculate_candidate_scores(
        q_head_id=q_head, q_time=q_time, candidate_set=candidate_set_C,
        all_history_events=all_history_events,
        all_semantic_embeddings=all_semantic_embeddings,
        all_structural_embeddings=all_structural_embeddings,
        lambda_time=args.lambda_time, alpha=args.alpha, beta=args.beta
    )

    # --- 5. 打印最终结果 ---
    print("\n--- 评分完成 ---")
    print(f"得分最高的 Top-20 候选节点:")
    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    for node_id, score in sorted_scores[:20]:
        print(f"  节点ID: {node_id:<6}  最终得分: {score:.4f}")


if __name__ == '__main__':
    main()
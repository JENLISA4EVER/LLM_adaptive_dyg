# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict
from typing import List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# =====================================================================================
# 1. 数据加载与预处理函数
# =====================================================================================

def get_llm_data(dataset_name: str, base_data_dir: str, val_ratio: float, test_ratio: float):
    """
    加载并处理动态图数据集。
    """
    base_path = os.path.join(base_data_dir, dataset_name)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"数据集路径不存在: {base_path}")
        
    graph_df = pd.read_csv(os.path.join(base_path, 'edge_list.csv'), index_col=0)
    entity_df = pd.read_csv(os.path.join(base_path, 'entity_text.csv'), index_col=0)
    relation_df = pd.read_csv(os.path.join(base_path, 'relation_text.csv'), index_col=0)
    
    entities = entity_df.sort_index()[['i', 'text']].values.tolist()
    relations = relation_df.sort_index()[['i', 'text']].values.tolist()
    
    node_num = max(graph_df['u'].max(), graph_df['i'].max()) + 1
    rel_num = graph_df['r'].max() + 1
    
    graph_df = graph_df.sort_values('ts').reset_index(drop=True)
    
    n_total = len(graph_df)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    
    train_df = graph_df.iloc[:n_train]
    val_df = graph_df.iloc[n_train : n_train + n_val]
    test_df = graph_df.iloc[n_train + n_val:]
    
    train_list = train_df[['u', 'r', 'i', 'ts', 'label', 'idx']].values.tolist()
    val_list = val_df[['u', 'r', 'i', 'ts', 'label', 'idx']].values.tolist()
    test_list = test_df[['u', 'r', 'i', 'ts', 'label', 'idx']].values.tolist()
    
    return entities, relations, train_list, val_list, test_list, node_num, rel_num

def get_query(qualruple: list):
    q_head = qualruple[0]
    q_rel = qualruple[1]
    q_tail = qualruple[2]
    q_time = qualruple[3]
    return q_head, q_rel, q_tail, q_time

# =====================================================================================
# 2. 三路召回的实现函数 
# =====================================================================================

def get_active_i_set(q_time: int, top_k: int, all_events: list) -> list:
    """
    获取在 q_time 这个时间戳之前，最近 top_k 个发生过交互的节点列表。
    *注：此函数已修改为直接接收所需参数，而不是依赖全局变量。*
    """
    active_nodes = {}
    for event in all_events:
        e_time = event[3]
        if e_time >= q_time:
            break
        active_nodes[event[0]] = e_time
        active_nodes[event[2]] = e_time
        
    sorted_active_nodes = sorted(active_nodes.items(), key=lambda item: item[1], reverse=True)
    return [(node, time) for node, time in sorted_active_nodes[:top_k]]

def get_or_compute_entity_embeddings(entities: list, model_path: str, file_path: str, device: str) -> torch.Tensor:
    """
    加载或计算所有实体的语义向量。
    """
    if os.path.exists(file_path):
        print(f"发现已缓存的向量文件，正在从 '{file_path}' 加载...")
        embeddings = torch.load(file_path, map_location=device)
        print("向量加载完成。")
        return embeddings
    
    print(f"未找到缓存文件。正在从本地路径 '{model_path}' 加载模型...")
    model = SentenceTransformer(model_path, device=device)
    
    print(f"正在为 {len(entities)} 个实体计算语义向量...")
    texts_to_encode = [item[1] for item in entities]
    embeddings = model.encode(
        texts_to_encode, 
        convert_to_tensor=True, 
        show_progress_bar=True,
        device=device
    )
    
    print(f"计算完成，正在将向量保存到 '{file_path}'...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(embeddings, file_path)
    print("保存成功。")
    return embeddings

def get_semantically_similar_nodes(q_head_id: int, top_k: int, all_embeddings: torch.Tensor) -> list:
    """
    获取与查询节点 q_head 在语义上最相似的 top_k 个节点。
    """
    if q_head_id >= len(all_embeddings):
        raise ValueError(f"查询ID {q_head_id} 超出实体列表范围。")
    query_embedding = all_embeddings[q_head_id]
    cos_scores = util.cos_sim(query_embedding, all_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k + 1)
    
    similar_nodes = []
    for score, idx in zip(top_results.values, top_results.indices):
        if idx.item() == q_head_id:
            continue
        similar_nodes.append((idx.item(), score.item()))
    return similar_nodes


def get_structural_nodes_old(q_head_id: int, q_time: int, top_k: int, all_events: list) -> list:
    """
    获取与查询节点 q_head_id 具有最多共同邻居的 top_k 个节点。
    *注：此函数已修改为直接接收所需参数，而不是依赖全局变量。*
    """
    adj = defaultdict(set)
    # 仅使用查询时间点之前的事件构建邻接表
    filtered_events = [event for event in all_events if event[3] < q_time]
    for u, r, i, ts, label, idx in filtered_events:
        adj[u].add(i)
        adj[i].add(u)

    if q_head_id not in adj:
        return []
    q_head_neighbors = adj[q_head_id]

    common_neighbors_count = []
    for node_id in adj.keys():
        if node_id == q_head_id or node_id in q_head_neighbors:
            continue
        count = len(q_head_neighbors.intersection(adj[node_id]))
        if count > 0:
            common_neighbors_count.append((node_id, count))

    common_neighbors_count.sort(key=lambda item: item[1], reverse=True)
    return common_neighbors_count[:top_k]

def get_structural_nodes(
    q_head_id: int, 
    top_k: int, 
    adj: defaultdict[int, set]
) -> list:
    """
    【优化版】
    接收一个已经构建好的邻接表，避免对全部历史事件的重复扫描和构建。
    """
    if q_head_id not in adj:
        return []
    q_head_neighbors = adj[q_head_id]

    common_neighbors_count = []
    # 遍历图中所有节点，寻找共同邻居
    for node_id in adj.keys():
        if node_id == q_head_id or node_id in q_head_neighbors:
            continue
        count = len(q_head_neighbors.intersection(adj[node_id]))
        if count > 0:
            common_neighbors_count.append((node_id, count))

    common_neighbors_count.sort(key=lambda item: item[1], reverse=True)
    return common_neighbors_count[:top_k]

# =====================================================================================
# 3. 整合后的召回函数 
# =====================================================================================

def get_candidate_set(
    query_event: List,
    all_events: List,
    all_entity_embeddings: torch.Tensor,
    k_time: int,
    k_struc: int,
    k_sem: int
) -> Set[int]:
    """
    通过时间、结构、语义三路召回，整合并返回一个最终的候选节点集合。
    *注：此函数已修改为直接接收所需参数，而不是依赖全局变量。*
    """
    q_head, _, _, q_time = get_query(query_event)
    print(f"\n--- 开始为查询节点 u={q_head} 在时间戳 t={q_time} 进行候选集召回 ---")

    time_recalled_nodes_with_info = get_active_i_set(q_time, k_time, all_events)
    time_recalled_ids = {node_id for node_id, time in time_recalled_nodes_with_info}
    print(f"   时间维度召回 {len(time_recalled_ids)} 个节点。")

    struc_recalled_nodes_with_info = get_structural_nodes_old(q_head, q_time, k_struc, all_events)
    struc_recalled_ids = {node_id for node_id, count in struc_recalled_nodes_with_info}
    print(f"   结构维度召回 {len(struc_recalled_ids)} 个节点。")

    sem_recalled_nodes_with_info = get_semantically_similar_nodes(q_head, k_sem, all_entity_embeddings)
    sem_recalled_ids = {node_id for node_id, score in sem_recalled_nodes_with_info}
    print(f"   语义维度召回 {len(sem_recalled_ids)} 个节点。")

    final_candidate_set = time_recalled_ids.union(struc_recalled_ids, sem_recalled_ids)
    
    print(f"--- 召回完成，最终候选节点集合 C 的大小为: {len(final_candidate_set)} ---")
    return final_candidate_set

# =====================================================================================
# 4. 主程序入口
# =====================================================================================

def main():
    # --- 设置命令行参数 ---
    parser = argparse.ArgumentParser(description="动态图链接预测候选集召回脚本")
    parser.add_argument('--dataset_name', type=str, default='ICEWS1819', help='数据集名称')
    parser.add_argument('--base_data_dir', type=str, default='/home/gtang/DTGB-main/DyLink_Datasets', help='数据集存放的基础目录')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--local_model_path', type=str, default='/home/gtang/pretrain_model/all-mpnet-base-v2', help='本地预训练模型路径')
    parser.add_argument('--k_time', type=int, default=50, help='时间维度召回数量')
    parser.add_argument('--k_struc', type=int, default=50, help='结构维度召回数量')
    parser.add_argument('--k_sem', type=int, default=50, help='语义维度召回数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备 (cuda or cpu)')
    args = parser.parse_args()

    # --- 1. 加载数据 ---
    print("--- 1. 正在加载和处理数据... ---")
    entities, relations, train_list, val_list, test_list, node_num, rel_num = get_llm_data(
        args.dataset_name, args.base_data_dir, args.val_ratio, args.test_ratio
    )

    # --- 2. 获取实体语义向量 ---
    print("\n--- 2. 正在获取实体语义向量... ---")
    embedding_file_path = os.path.join(args.base_data_dir, args.dataset_name, f"{args.dataset_name}_entity_embeddings.pt")
    all_entity_embeddings = get_or_compute_entity_embeddings(
        entities=entities,
        model_path=args.local_model_path,
        file_path=embedding_file_path,
        device=args.device
    )

    # --- 3. 执行候选集召回 ---
    # 计算召回率
    total_test_list = len(test_list)
    recall_time = 0
    for idx, query_event in enumerate(tqdm(test_list, desc=' 评估召回率')):
        q_tail = query_event[2]
        all_events_for_recall = train_list + val_list + test_list

        candidate_set_C = get_candidate_set(
            query_event=query_event,
            all_events=all_events_for_recall,
            all_entity_embeddings=all_entity_embeddings,
            k_time=args.k_time,
            k_struc=args.k_struc,
            k_sem=args.k_sem
        )

        if q_tail in candidate_set_C: 
            recall_time += 1
        
        print(f"当前候选召回率：{recall_time / (idx + 1)}")

    
    print(f"候选召回率：{recall_time / total_test_list}")

    # --- 4. 打印最终结果 ---
    # print("\n最终候选集内容 (部分):")
    # print(list(candidate_set_C)[:50]) # 打印最多50个看看


if __name__ == '__main__':
    main()
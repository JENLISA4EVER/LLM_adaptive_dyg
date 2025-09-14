# -*- coding: utf-8 -*-
import os
import argparse
from typing import List, Set, Dict, Tuple
import random
import itertools
from collections import defaultdict
import math  # 【新增】引入math库

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import util # 【新增】为计算语义相似度标准差

# 从我们之前的脚本中导入必要的函数
from utils.realtime_plotter import RealtimePlotter
from utils.dynamic_few_shot import FewShotManager
from run_recall import get_query, load_data_for_evaluation, get_active_i_set, get_candidate_set, get_structural_nodes, get_semantically_similar_nodes, get_or_compute_entity_embeddings
from run_scoring import calculate_candidate_scores, RandomProjectionModule

# =====================================================================================
# 3. 第三部分：样本定义与划分 & LLM 预测 (整合了所有自适应修改)
# =====================================================================================

def get_adaptive_hyperparameters(
    args: argparse.Namespace,
    q_head: int,
    q_time: int,
    node_history: defaultdict,
    initial_history: List,
    all_semantic_embeddings: torch.Tensor,
    global_stats: Dict
) -> Dict:
    """
    【升级版】根据当前查询的特征，动态计算三路召回k值和评分权重。
    """
    adaptive_params = {}
    q_head_degree = len(node_history.get(q_head, []))
    log_degree = math.log(q_head_degree + 1)
    
    # --- 1. 自适应调整 k_struc ---
    if args.adaptive_k_struc:
        log_avg_degree = global_stats.get('log_avg_degree', math.log(1))
        scaling_factor = 1.0 + args.k_struc_deg_factor * (log_degree - log_avg_degree)
        k_struc_adaptive = int(args.k_struc * scaling_factor)
        adaptive_params['k_struc'] = max(10, min(args.k_struc * 2, k_struc_adaptive))
    else:
        adaptive_params['k_struc'] = args.k_struc

    # --- 2. 自适应调整 k_time ---
    if args.adaptive_k_time:
        base_k = args.k_time
        # 注意：这里为了效率，只在initial_history的末尾部分查找，因为它是时间有序的
        # 对于非常大的图，可以进一步优化，但对于当前实现是足够的
        recent_events = [e for e in initial_history if e[3] < q_time]
        if len(recent_events) > base_k:
            k_th_timestamp = sorted([e[3] for e in recent_events], reverse=True)[base_k-1]
            delta_t_k = q_time - k_th_timestamp
            
            avg_delta_t = global_stats.get('avg_delta_t_k', 1)
            # 使用tanh进行平滑和有界调整
            scaling_factor = 1.0 - args.k_time_factor * math.tanh((delta_t_k - avg_delta_t) / (avg_delta_t + 1e-6))
            k_time_adaptive = int(base_k * scaling_factor)
            adaptive_params['k_time'] = max(10, min(args.k_time * 2, k_time_adaptive))
        else:
            adaptive_params['k_time'] = args.k_time
    else:
        adaptive_params['k_time'] = args.k_time

    # --- 3. 自适应调整 k_sem ---
    if args.adaptive_k_sem:
        pool_size = args.k_sem * 2
        query_embedding = all_semantic_embeddings[q_head]
        cos_scores = util.cos_sim(query_embedding, all_semantic_embeddings)[0]
        top_results = torch.topk(cos_scores, k=pool_size)
        top_scores_std = torch.std(top_results.values).item()

        avg_std_sem = global_stats.get('avg_std_sem', 0.1)
        # 如果标准差大，说明分数有断崖，应减少k；反之增加k
        scaling_factor = 1.0 - args.k_sem_factor * math.tanh((top_scores_std - avg_std_sem) / (avg_std_sem + 1e-6))
        k_sem_adaptive = int(args.k_sem * scaling_factor)
        adaptive_params['k_sem'] = max(10, min(args.k_sem * 2, k_sem_adaptive))
    else:
        adaptive_params['k_sem'] = args.k_sem

    # --- 4. 自适应调整 alpha 和 beta ---
    if args.adaptive_alpha_beta:
        q_head_hist_count = len(node_history.get(q_head, []))
        beta_offset = args.beta_range * math.tanh((q_head_hist_count - global_stats.get('avg_hist_count', 10)) / args.beta_maturity_factor)
        beta_adaptive = args.beta + beta_offset
        adaptive_params['beta'] = max(0.05, min(0.95, beta_adaptive))

        log_avg_degree = global_stats.get('log_avg_degree', math.log(1))
        alpha_offset = args.alpha_range * math.tanh((log_degree - log_avg_degree) / args.alpha_deg_factor)
        alpha_adaptive = args.alpha + alpha_offset
        adaptive_params['alpha'] = max(0.05, min(0.95, alpha_adaptive))
    else:
        adaptive_params['alpha'] = args.alpha
        adaptive_params['beta'] = args.beta
        
    return adaptive_params

def partition_samples_adaptive(
    scores: Dict[int, float],
    p: float,
    q: float,
    p_std_factor: float,
    p_min_count: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    【新增】自适应样本划分: 先保证基数，再用统计数据提纯，并有底线保障。
    """
    if not scores:
        return [], [], []

    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    num_nodes = len(sorted_nodes)

    p_base_idx = int(num_nodes * p)
    q_idx = int(num_nodes * (1 - q))
    negative_nodes = [node_id for node_id, score in sorted_nodes[q_idx:]]
    
    initial_positive_pool = sorted_nodes[:p_base_idx]
    
    if num_nodes > 1 and p_std_factor > 0:
        score_values = np.array(list(scores.values()))
        mean_score = np.mean(score_values)
        std_score = np.std(score_values)
        quality_threshold = mean_score + p_std_factor * std_score
        purified_positive_nodes = [node_id for node_id, score in initial_positive_pool if score > quality_threshold]
    else:
        purified_positive_nodes = [node_id for node_id, score in initial_positive_pool]

    if len(purified_positive_nodes) < p_min_count and initial_positive_pool:
        final_future_positive_nodes = [node_id for node_id, score in initial_positive_pool[:p_min_count]]
    else:
        final_future_positive_nodes = purified_positive_nodes
        
    final_positive_set = set(final_future_positive_nodes)
    negative_set = set(negative_nodes)
    ambiguous_nodes = [node_id for node_id, score in sorted_nodes if node_id not in final_positive_set and node_id not in negative_set]

    return final_future_positive_nodes, negative_nodes, ambiguous_nodes

# (partition_samples, get_last_interaction_info_optimized, verify_and_create_sample_quadruples_optimized, build_few_shot_prompt, predict_with_llm 这些原有的、未修改的函数保持不变)
def partition_samples(
    scores: Dict[int, float], 
    p: float, 
    q: float
) -> Tuple[List[int], List[int], List[int]]:
    if not scores:
        return [], [], []
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    num_nodes = len(sorted_nodes)
    p_idx = int(num_nodes * p)
    q_idx = int(num_nodes * (1 - q))
    future_positive_nodes = [node_id for node_id, score in sorted_nodes[:p_idx]]
    negative_nodes = [node_id for node_id, score in sorted_nodes[q_idx:]]
    ambiguous_nodes = [node_id for node_id, score in sorted_nodes[p_idx:q_idx]]
    return future_positive_nodes, negative_nodes, ambiguous_nodes

def get_last_interaction_info_optimized(
    node_id: int, 
    q_time: int,
    node_history: defaultdict[int, list],
    relations_map: Dict[int, str],
    cache: Dict
) -> Tuple[str, int, int]:
    if node_id in cache:
        return cache[node_id]
    last_rel_id, last_ts = -1, -1
    if node_id in node_history:
        for partner_id, rel_id, timestamp in reversed(node_history[node_id]):
            if timestamp < q_time:
                last_rel_id = rel_id
                last_ts = timestamp
                break
    if last_ts == -1:
        cache[node_id] = ("<NO_HISTORY>", -1, -1)
        return "<NO_HISTORY>", -1, -1
    rel_text = relations_map.get(last_rel_id, f"Relation_{last_rel_id}")
    cache[node_id] = (rel_text, last_ts, last_rel_id)
    return rel_text, last_ts, last_rel_id

def verify_and_create_sample_quadruples_optimized(
    u_id: int,
    q_time: int,
    future_positive_nodes: List[int],
    ambiguous_nodes: List[int],
    negative_nodes: List[int],
    u_true_history_set: set,
    node_history: defaultdict[int, list],
    relations_map: Dict[int, str]
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    final_future_pos_samples, final_neg_samples, final_ambiguous_samples = [], [], []
    interaction_cache = {}
    for v_node in future_positive_nodes:
        r_text, t, r_id = get_last_interaction_info_optimized(v_node, q_time, node_history, relations_map, interaction_cache)
        if t != -1 and (r_id, v_node, t) not in u_true_history_set:
            final_future_pos_samples.append((u_id, r_text, v_node, t))
    for v_node in negative_nodes:
        r_text, t, r_id = get_last_interaction_info_optimized(v_node, q_time, node_history, relations_map, interaction_cache)
        if t != -1:
            final_neg_samples.append((u_id, r_text, v_node, t))
    for v_node in ambiguous_nodes:
        r_text, t, r_id = get_last_interaction_info_optimized(v_node, q_time, node_history, relations_map, interaction_cache)
        if t != -1:
            final_ambiguous_samples.append((u_id, r_text, v_node, t))
    return final_future_pos_samples, final_neg_samples, final_ambiguous_samples

def build_few_shot_prompt(
    u_id: int,
    u_time: int, 
    golden_pos_quads: List,
    future_pos_quads: List,
    neg_quads: List,
    ambiguous_quads: List
) -> str:
    """构建用于LLM few-shot学习的完整提示词。"""
    prompt = ""
    # prompt += "You are a link prediction expert in a dynamic graph. Based on the examples of past interactions, determine which of the two new nodes is more likely to connect with the query node.\n\n"
    # prompt += "In the following quadruple `(u, r, v, t)` examples, `u` is the source node ID, `r` is the text describing the link type, `v` is the destination node ID, and `t` is the timestamp of the interaction.\n\n"
    
    prompt += "### Golden Positive Examples (Confirmed Past Interactions)\n"
    prompt += "These links have definitely happened:\n"
    for u, r, v, t in golden_pos_quads:
        prompt += f"({u}, {r}, {v}, {t})\n"
        
    prompt += "\n### Potential Future Links (High-Confidence Candidates)\n"
    prompt += "Based on analysis, these links are highly likely to happen in the future:\n"
    for u, r, v, t in future_pos_quads:
        prompt += f"({u}, might interact with, {v}, {u_time})\n"

    # <--- 新增: 模糊样本部分 --- >
    prompt += "\n### Ambiguous or Neutral Links (Uncertain Candidates)\n"
    prompt += "Based on analysis, the likelihood of these links happening is uncertain:\n"
    for u, r, v, t in ambiguous_quads:
        prompt += f"({u}, might interact with, {v}, {u_time})\n"

    prompt += "\n### Unlikely Links (Low-Confidence Candidates)\n"
    prompt += "Based on analysis, these links are very unlikely to happen:\n"
    for u, r, v, t in neg_quads:
        prompt += f"({u}, interact with, {v}, {u_time})\n"
        
    prompt += "\n---\n"
    prompt += "Now, based on all the examples above, analyze the examples and answer the following question. The correct answer could be either A or B. You must only output the letter of the correct option (A or B).\n"
    
    return prompt

def predict_with_llm(
    prompt_template: str,
    u_id: int,
    v_true: int,
    v_neg: int,
    tokenizer,
    model
) -> Tuple[str, Dict[str, float]]:
    """使用LLM进行选择题形式的链接预测。"""
    question = f"Question: Which node is more likely to link with node {u_id}?\n  A: {v_true}\n  B: {v_neg}\nAnswer: "

    full_prompt = prompt_template + question
    # print(full_prompt)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs['attention_mask'].sum(dim=1)
    # print(f'token长度：{prompt_length}')
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # 获取最后一个token的logits
        logits = outputs.logits[0, -1, :]

    # 获取'A'和'B'的token ID
    token_id_A = tokenizer.convert_tokens_to_ids('A')
    token_id_B = tokenizer.convert_tokens_to_ids('B')
    
    # 获取对应ID的logits
    logit_A = logits[token_id_A]
    logit_B = logits[token_id_B]
    
    # 使用softmax计算概率
    probs = torch.nn.functional.softmax(torch.tensor([logit_A, logit_B]), dim=0)
    prob_A = probs[0].item()
    prob_B = probs[1].item()
    
    prediction = 'A' if prob_A > prob_B else 'B'
    # print(prediction)
    probabilities = {'A': prob_A, 'B': prob_B}
    
    return prediction, probabilities

# =====================================================================================
# 主评估流程
# =====================================================================================

def evaluate_link_prediction_optimized(args, train_list, val_list, full_test_flow, eval_event_set, all_semantic_embeddings, relations_map, node_num):
    """
    【最终版的主评估函数】
    集成了三路召回、评分权重、样本划分的完整自适应逻辑。
    """
    PROMPT_PREAMBLE = ""
    PROMPT_PREAMBLE += "You are a link prediction expert in a dynamic graph. Based on the examples of past interactions, determine which of the two new nodes is more likely to connect with the query node.\n\n"
    PROMPT_PREAMBLE += "In the following quadruple `(u, r, v, t)` examples, `u` is the source node ID, `r` is the text describing the link type, `v` is the destination node ID, and `t` is the timestamp of the interaction.\n\n"


    print("\n--- 5. 开始在测试集上进行链接预测评估 ---")

    # =======================================================================
    # Phase 0: 全局信息预计算
    # =======================================================================
    print("   - Phase 0: 预计算全局统计信息...")
    initial_history = train_list + val_list
    initial_history.sort(key=lambda x: x[3])
    
    node_history_for_stats = defaultdict(list)
    for u, r, i, ts, _, _ in initial_history:
        node_history_for_stats[u].append(i)
        node_history_for_stats[i].append(u)
    
    degrees = [len(neighbors) for neighbors in node_history_for_stats.values()]
    log_degrees = [math.log(d + 1) for d in degrees if d > 0]
    hist_counts = [len(neighbors) for neighbors in node_history_for_stats.values()]

    sample_size = 1000
    delta_t_samples = []
    timestamps_history = sorted([e[3] for e in initial_history])
    if args.adaptive_k_time:
        for _ in tqdm(range(sample_size), desc="   - 采样计算 avg_delta_t"):
            rand_idx = random.randint(len(timestamps_history)//2, len(timestamps_history)-1)
            q_time_sample = timestamps_history[rand_idx]
            events_before = [ts for ts in timestamps_history if ts < q_time_sample]
            if len(events_before) > args.k_time:
                k_th_ts = sorted(events_before, reverse=True)[args.k_time-1]
                delta_t_samples.append(q_time_sample - k_th_ts)

    std_sem_samples = []
    if args.adaptive_k_sem:
        node_indices = list(range(node_num))
        for _ in tqdm(range(sample_size), desc="   - 采样计算 avg_std_sem"):
            rand_node_idx = random.choice(node_indices)
            pool_size = args.k_sem * 2
            query_emb = all_semantic_embeddings[rand_node_idx]
            cos_scores = util.cos_sim(query_emb, all_semantic_embeddings)[0]
            top_results = torch.topk(cos_scores, k=pool_size)
            std_sem_samples.append(torch.std(top_results.values).item())

    global_stats = {
        'log_avg_degree': np.mean(log_degrees) if log_degrees else math.log(1),
        'avg_hist_count': np.mean(hist_counts) if hist_counts else 1,
        'avg_delta_t_k': np.mean(delta_t_samples) if delta_t_samples else 1,
        'avg_std_sem': np.mean(std_sem_samples) if std_sem_samples else 0.1,
    }
    print(f"   - 全局信息: avg_log_degree={global_stats['log_avg_degree']:.2f}, avg_hist={global_stats['avg_hist_count']:.2f}, avg_delta_t={global_stats['avg_delta_t_k']:.2f}, avg_std_sem={global_stats['avg_std_sem']:.3f}")

    # =======================================================================
    # Phase 1: 初始化动态数据结构和模型 (代码不变)
    # =======================================================================
    print("   - Phase 1: 初始化动态模块 (RPModule, Adjacency List, Node History)...")
    initial_history = train_list + val_list
    initial_history.sort(key=lambda x: x[3])
    
    # 1a. 初始化 RPModule (只创建一次)
    rp_module = RandomProjectionModule(
        node_num=node_num, edge_num=len(initial_history), dim_factor=args.rp_dim_factor,
        num_layer=args.rp_num_layer, time_decay_weight=args.rp_time_decay_weight,
        device=args.device, beginning_time=initial_history[0][3], enforce_dim=-1
    ).to(args.device)
    batch_size = 200
    for i in tqdm(range(0, len(initial_history), batch_size), desc="   - 初始化 RPModule"):
        batch = initial_history[i:i+batch_size]
        rp_module.update(np.array([e[0] for e in batch]), np.array([e[2] for e in batch]), np.array([e[3] for e in batch]))

    # 1b. 初始化邻接表 (只创建一次)
    adj = defaultdict(set)
    for u, r, i, ts, _, _ in tqdm(initial_history, desc="   - 初始化邻接表"):
        adj[u].add(i); adj[i].add(u)
        
    # 1c. 初始化节点交互历史 (只创建一次)
    node_history = defaultdict(list)
    for u, r, i, ts, _, _ in tqdm(initial_history, desc="   - 初始化节点历史"):
        node_history[u].append((i, r, ts)) # 存储伙伴、关系ID、时间
        node_history[i].append((u, r, ts)) # 存储伙伴、关系ID、时间

    # 1d. 初始化LLM (只加载一次)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    plotter = None
    if args.enable_plotting:
        plotter = RealtimePlotter(
            task_name='Link Prediction',
            save_path='link_prediction_metrics.pdf',
            metrics=['ROC AUC', 'AP']
        )

    few_shot_manager = None
    if args.enable_dynamic_few_shot:
        few_shot_manager = FewShotManager(max_examples=args.num_fs)

    print("   - 所有模块初始化完成.")

    # =======================================================================
    # Phase 2: 按时间顺序处理测试事件并增量更新
    # =======================================================================
    print("   - Phase 2: 按时间戳进行评估和增量更新...")
    all_calibrated_scores, all_labels = [], []
    all_node_ids = list(range(node_num))
    total_test_events = len(full_test_flow)
    processed_events_count = 0

    test_events_grouped_by_time = itertools.groupby(full_test_flow, key=lambda x: x[3])

    for timestamp, events_at_this_time_iter in tqdm(test_events_grouped_by_time, desc="  评估测试集 (按时间戳)"):
        events_at_this_time = list(events_at_this_time_iter)
        current_structural_embeddings = rp_module.get_all_projections().detach()
        
        for test_event in events_at_this_time:
            if tuple(test_event) in eval_event_set:
                processed_events_count += 1
                q_head, q_rel, q_tail_true, q_time = get_query(test_event)

                # --- 【修改】获取所有自适应超参数 ---
                adaptive_params = get_adaptive_hyperparameters(
                    args, q_head, q_time, node_history, initial_history, all_semantic_embeddings, global_stats
                )
                if processed_events_count % args.print_interval == 0:
                    print(f"\n   [Debug] 事件 {processed_events_count}: q_head={q_head}, "
                        f"k_time={adaptive_params['k_time']}, k_struc={adaptive_params['k_struc']}, k_sem={adaptive_params['k_sem']}, "
                        f"alpha={adaptive_params['alpha']:.3f}, beta={adaptive_params['beta']:.3f}")
                
                # --- 【修改】三路召回 (使用自适应k) ---
                time_recalled_ids = {node_id for node_id, _ in get_active_i_set(q_time, adaptive_params['k_time'], initial_history)}
                struc_recalled_ids = {node_id for node_id, _ in get_structural_nodes(q_head, adaptive_params['k_struc'], adj)}
                sem_recalled_ids = {node_id for node_id, _ in get_semantically_similar_nodes(q_head, adaptive_params['k_sem'], all_semantic_embeddings)}
                candidate_set_C = time_recalled_ids.union(struc_recalled_ids, sem_recalled_ids)
                
                # 【修改】确保召回的节点总数不会太少
                if len(candidate_set_C) < args.min_recall_num:
                    time_recalled_ids_fb = {node_id for node_id, _ in get_active_i_set(q_time, args.k_time, initial_history)}
                    struc_recalled_ids_fb = {node_id for node_id, _ in get_structural_nodes(q_head, args.k_struc, adj)}
                    sem_recalled_ids_fb = {node_id for node_id, _ in get_semantically_similar_nodes(q_head, args.k_sem, all_semantic_embeddings)}
                    candidate_set_C.update(time_recalled_ids_fb)
                    candidate_set_C.update(struc_recalled_ids_fb)
                    candidate_set_C.update(sem_recalled_ids_fb)

                if not candidate_set_C: continue
                
                # --- 【修改】评分 (使用自适应alpha, beta) ---
                final_scores = calculate_candidate_scores(
                    q_head, q_time, candidate_set_C, node_history,
                    all_semantic_embeddings, current_structural_embeddings,
                    args.lambda_time, adaptive_params['alpha'], adaptive_params['beta'] # 使用自适应alpha, beta
                )
                
                # --- 【修改】构建 Prompt (使用自适应样本划分) ---
                if args.adaptive_pq:
                    future_pos_nodes, negative_nodes, ambiguous_nodes = partition_samples_adaptive(
                        final_scores, args.p, args.q, args.p_std_factor, args.p_min_count
                    )
                else:
                    future_pos_nodes, negative_nodes, ambiguous_nodes = partition_samples(final_scores, args.p, args.q)

                
                # --- [修改] 负采样策略 ---
                q_tail_neg = None
                fallback_sampling = False

                if args.neg_sampling_strategy == 'recall_pool':
                    # 策略5: 为负采样，使用默认k值重新构建一个独立的召回池
                    
                    # 1. 使用默认k值 (args.k_*) 进行三路召回
                    time_recalled_ids_neg = {node_id for node_id, _ in get_active_i_set(q_time, 55, initial_history)}
                    struc_recalled_ids_neg = {node_id for node_id, _ in get_structural_nodes(q_head, 55, adj)}
                    sem_recalled_ids_neg = {node_id for node_id, _ in get_semantically_similar_nodes(q_head, 55, all_semantic_embeddings)}
                    
                    # 2. 合并成一个专用于负采样的召回池
                    neg_sampling_recall_pool = time_recalled_ids_neg.union(struc_recalled_ids_neg, sem_recalled_ids_neg)

                    # 3. 从这个新生成的、非自适应的池子中进行负采样
                    potential_negatives = list(neg_sampling_recall_pool - {q_head, q_tail_true})
                    
                    if potential_negatives:
                        q_tail_neg = random.choice(potential_negatives)
                    else:
                        # 如果新池子为空或只有无效节点，则退回全局采样
                        fallback_sampling = True
                
                if args.neg_sampling_strategy == 'hard_positive':
                    # 策略2: 从未来正样本中采样，且不能是历史伙伴或未来真实伙伴
                    # 1. 获取所有在历史中交互过的伙伴
                    historical_partners = {partner for partner, r_id, t in node_history.get(q_head, []) if t < q_time}
                    # 2. [新增] 获取所有在未来 (t' > q_time) 将会交互的真实伙伴
                    #    这需要遍历整个测试集来预知未来
                    future_partners = {event[2] for event in full_test_flow if event[0] == q_head and event[3] > q_time}
                    # 3. 合并历史与未来两种需要排除的节点
                    nodes_to_exclude = historical_partners.union(future_partners)
                    # 4. 从“未来正样本”池中过滤掉所有应排除的节点以及本次查询的正确答案
                    potential_negatives = [
                        node for node in future_pos_nodes 
                        if node != q_tail_true and node not in nodes_to_exclude
                    ]
                    if potential_negatives:
                        q_tail_neg = random.choice(potential_negatives)
                    else:
                        # 如果过滤后没有候选者，则启用备用采样
                        fallback_sampling = True

                elif args.neg_sampling_strategy == 'ambiguous':
                    # 策略3: 从模糊样本中采样
                    potential_negatives = [node for node in ambiguous_nodes if node != q_tail_true]
                    if potential_negatives:
                        q_tail_neg = random.choice(potential_negatives)
                    else:
                        fallback_sampling = True

                elif args.neg_sampling_strategy == 'negative':
                    # 策略4: 从负样本中采样
                    potential_negatives = [node for node in negative_nodes if node != q_tail_true]
                    if potential_negatives:
                        q_tail_neg = random.choice(potential_negatives)
                    else:
                        fallback_sampling = True
                
                
                
                # 策略1 (global) 或任何策略的备用方案
                if args.neg_sampling_strategy == 'global' or fallback_sampling:
                    q_tail_neg = random.choice(all_node_ids)
                    while q_tail_neg == q_tail_true or q_tail_neg == q_head:
                        q_tail_neg = random.choice(all_node_ids)
                # --- [修改] 负采样结束 ---


                # --- [修改] 确保负样本不出现在Prompt的正面证据中 ---
                # 这一步是为了防止给LLM矛盾的信息
                prompt_future_pos_nodes = [node for node in future_pos_nodes if node != q_tail_neg]
                prompt_ambiguous_nodes = [node for node in ambiguous_nodes if node != q_tail_neg]

                

                # (后续的样本构建、Prompt、预测、更新逻辑完全不变)
                # 1. 高效地为验证准备u的历史集合
                u_history_for_verification = set()
                if q_head in node_history:
                    for partner, r_id, t in node_history[q_head]:
                        if t < q_time:
                            u_history_for_verification.add((r_id, partner, t))
                
                # 2. 调用新的高效函数填充 quad 列表
                future_pos_quads, neg_quads, ambiguous_quads = verify_and_create_sample_quadruples_optimized(
                    q_head, q_time, prompt_future_pos_nodes, prompt_ambiguous_nodes, negative_nodes, 
                    u_history_for_verification, node_history, relations_map
                )

                # 3. 获取黄金正样本
                recent_u_history = node_history.get(q_head, [])[-args.num_golden_samples:]
                # 注意: 此处的关系获取需要适配新node_history格式
                golden_pos_quads = []
                for partner, r_id, t in recent_u_history:
                    if t < q_time:
                        r_text = relations_map.get(r_id, f"Rel_{r_id}")
                        golden_pos_quads.append((q_head, r_text, partner, t))

                # 4. 构建最终Prompt
                # 【修改】在构建Prompt前，先获取few-shot上下文
                few_shot_context = ""
                if few_shot_manager:
                    few_shot_context = few_shot_manager.get_few_shot_context()

                base_prompt_template = build_few_shot_prompt(
                    q_head, q_time, golden_pos_quads, future_pos_quads, neg_quads[:20], ambiguous_quads
                )
                # 【修改】将few-shot上下文添加到主prompt之前
                current_prompt_template = few_shot_context + PROMPT_PREAMBLE + base_prompt_template
                # if len(few_shot_context) > 0: print(current_prompt_template)

                # --- 预测与位置校准 ---
                prediction1, probabilities1 = predict_with_llm(current_prompt_template, q_head, q_tail_true, q_tail_neg, tokenizer, model)
                prediction2, probabilities2 = predict_with_llm(current_prompt_template, q_head, q_tail_neg, q_tail_true, tokenizer, model)
                
                calibrated_score_for_true_node = (probabilities1['A'] + probabilities2['B']) / 2.0
                calibrated_score_for_neg_node = (probabilities1['B'] + probabilities2['A']) / 2.0

                # 【新增】判断预测是否错误，并生成新的few-shot样本
                if few_shot_manager and few_shot_manager.can_add_more() and probabilities1['A'] >= probabilities1['B']:
                    few_shot_manager.generate_and_add_example(
                        prompt_preamble=PROMPT_PREAMBLE,
                        failed_prompt_template=base_prompt_template, # 使用不包含旧示例的模板
                        q_head=q_head,
                        v_true=q_tail_true,
                        v_neg=q_tail_neg,
                        model=model,
                        tokenizer=tokenizer,
                        device=args.device
                    )


                all_calibrated_scores.extend([calibrated_score_for_true_node, calibrated_score_for_neg_node])
                all_labels.extend([1, 0])

                # --- 打印中间结果 ---
                if processed_events_count % args.print_interval == 0 and len(all_labels) > 1:
                    current_auc = roc_auc_score(all_labels, all_calibrated_scores)
                    current_ap = average_precision_score(all_labels, all_calibrated_scores)
                    print(f"\n--- 中间指标 (事件 {processed_events_count}/{total_test_events}) --- ROC AUC: {current_auc:.4f}, AP: {current_ap:.4f}")

                    # 【新增】调用绘图模块
                    if plotter:
                        plotter.update_and_plot(
                            event_count=processed_events_count,
                            current_metrics={'ROC AUC': current_auc, 'AP': current_ap}
                        )


        # 2b. 更新阶段：用刚刚处理完的事件来更新状态
        rp_module.update(np.array([e[0] for e in events_at_this_time]), np.array([e[2] for e in events_at_this_time]), np.array([e[3] for e in events_at_this_time]))
        for u, r, i, ts, _, _ in events_at_this_time:
            adj[u].add(i); adj[i].add(u)
            node_history[u].append((i, r, ts)) # 更新时也使用新格式
            node_history[i].append((u, r, ts))
        initial_history.extend(events_at_this_time) # 更新 get_active_i_set 所需的历史

    # 4. 计算并打印最终指标
    final_auc = roc_auc_score(all_labels, all_calibrated_scores)
    final_ap = average_precision_score(all_labels, all_calibrated_scores)
    if plotter and len(all_labels) > 1:
        plotter.update_and_plot(
                event_count=processed_events_count,
                current_metrics={'ROC AUC': final_auc, 'AP': final_ap}
        )
    print(f"\n--- 最终测试指标 ---\nROC AUC: {final_auc:.4f}, AP: {final_ap:.4f}")

    return final_auc, final_ap


def main():
    parser = argparse.ArgumentParser(description="动态图链接预测-完全自适应LLM预测脚本")
    
    # --- 【新增】实验设置参数 ---
    parser.add_argument(
        '--setting',
        type=str,
        default='transductive',
        choices=['transductive', 'inductive'],
        help="实验设置: 'transductive' (默认) 或 'inductive'."
    )

    # 数据和路径参数
    parser.add_argument('--dataset_name', type=str, default='ICEWS1819', help='数据集名称')
    parser.add_argument('--base_data_dir', type=str, default='/home/gtang/DTGB-main/DyLink_Datasets', help='数据集存放的基础目录')
    parser.add_argument('--local_model_path', type=str, default='/home/gtang/pretrain_model/all-mpnet-base-v2', help='本地语义模型路径')
    parser.add_argument('--llm_model_path', type=str, default='/home/gtang/pretrain_model/Qwen3-8B', help='本地LLM模型路径')
    
    # 召回基准参数
    parser.add_argument('--k_time', type=int, default=55, help='【基准】时间维度召回数量')
    parser.add_argument('--k_struc', type=int, default=55, help='【基准】结构维度召回数量')
    parser.add_argument('--k_sem', type=int, default=55, help='【基准】语义维度召回数量')
    
    # 结构向量模型 (RP) 参数
    parser.add_argument('--rp_dim_factor', type=int, default=10, help='结构向量维度因子')
    parser.add_argument('--rp_num_layer', type=int, default=3, help='结构向量的游走最大跳数')
    parser.add_argument('--rp_time_decay_weight', type=float, default=1e-7, help='结构向量的时间衰减权重')
    
    # 评分基准超参数
    parser.add_argument('--lambda_time', type=float, default=0.01, help='历史伙伴权重的时间衰减率')
    parser.add_argument('--alpha', type=float, default=0.5, help='【基准】结构/语义相似度的加权系数')
    parser.add_argument('--beta', type=float, default=0.5, help='【基准】历史/自身相似度得分的加权系数')
    
    # 样本划分与Prompt参数
    parser.add_argument('--p', type=float, default=0.15, help='【基准】未来正样本比例')
    parser.add_argument('--q', type=float, default=0.50, help='负样本比例')
    parser.add_argument('--num_golden_samples', type=int, default=100, help='用于Prompt的黄金正样本数量')
    
    # 其他参数
    parser.add_argument('--print_interval', type=int, default=50, help='每隔多少个测试事件打印一次中间指标')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备 (cuda or cpu)')
    parser.add_argument('--enable_plotting', action='store_true', help='Enable real-time plotting of performance metrics.')
    parser.add_argument('--seed_value', type=int, default=42, help='随机种子数值')
    parser.add_argument('--num_fs', type=int, default=3, help='少样本示例个数')

    # --- 负采样策略参数 ---
    parser.add_argument(
        '--neg_sampling_strategy',
        type=str,
        default='global',
        choices=['global', 'hard_positive', 'ambiguous', 'negative', 'recall_pool'],
        help="""负采样策略:
        'global': 全局随机负采样 (原始策略);
        'hard_positive': 从未来正样本中采样 (且非历史伙伴);
        'ambiguous': 从模糊样本中采样;
        'negative': 从划分出的负样本中采样.
        'recall_pool': 从完整召回池C中随机采样.
        """
    )

    # --- 【新增】自适应超参数的控制开关和元参数 ---
    # 开关
    parser.add_argument('--adaptive_k_struc', action='store_true', help='开启k_struc的自适应调整')
    parser.add_argument('--adaptive_k_time', action='store_true', help='开启k_time的自适应调整')
    parser.add_argument('--adaptive_k_sem', action='store_true', help='开启k_sem的自适应调整')
    parser.add_argument('--adaptive_alpha_beta', action='store_true', help='开启alpha和beta的自适应调整')
    parser.add_argument('--adaptive_pq', action='store_true', help='开启p/q划分的自适应调整')
    parser.add_argument('--enable_dynamic_few_shot', action='store_true', help='Enable dynamic few-shot example generation for link prediction.')

    # k 自适应的元参数
    parser.add_argument('--k_struc_deg_factor', type=float, default=0.2, help='k_struc根据log度进行缩放的因子')
    parser.add_argument('--k_time_factor', type=float, default=0.5, help='k_time根据时间间隔进行缩放的因子')
    parser.add_argument('--k_sem_factor', type=float, default=0.5, help='k_sem根据相似度标准差进行缩放的因子')
    parser.add_argument('--min_recall_num', type=int, default=120, help='确保最低总召回数的底线')

    # alpha/beta 自适应的元参数
    parser.add_argument('--alpha_range', type=float, default=0.15, help='alpha浮动的最大范围 (基准值±此范围)')
    parser.add_argument('--alpha_deg_factor', type=float, default=1.0, help='alpha根据log度调整的平滑因子')
    parser.add_argument('--beta_range', type=float, default=0.2, help='beta浮动的最大范围 (基准值±此范围)')
    parser.add_argument('--beta_maturity_factor', type=float, default=20.0, help='beta根据历史交互数调整的平滑因子 (分母)')
    
    # p/q 划分自适应的元参数
    parser.add_argument('--p_std_factor', type=float, default=1.0, help='划分高质量正样本的标准差倍数 (0表示不使用统计提纯)')
    parser.add_argument('--p_min_count', type=int, default=8, help='保证最少的未来正样本数量')

    args = parser.parse_args()

    # --- [新增] 在所有操作开始前设置随机种子 ---
    seed_value = args.seed_value  # 选择任何固定的整数
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # --- 主流程 ---
    print(f"--- 1. 正在加载和处理数据 (模式: {args.setting}) ---")
    entities, relations, train_list, val_list, test_list_trans, test_list_ind, node_num = load_data_for_evaluation(
        args.dataset_name, args.base_data_dir, val_ratio=0.15, test_ratio=0.15
    )
    
    relations_map = {rel_id: rel_text for rel_id, rel_text in relations}
    
    print("\n--- 2. 正在准备节点语义向量... ---")
    embedding_file_path = os.path.join(args.base_data_dir, args.dataset_name, f"{args.dataset_name}_entity_embeddings.pt")
    all_semantic_embeddings = get_or_compute_entity_embeddings(
        entities=entities, model_path=args.local_model_path, file_path=embedding_file_path, device=args.device
    )

    if args.setting == 'transductive':
        test_list_for_eval = test_list_trans
        full_test_flow = test_list_trans  # Transductive模式下，评估流就是测试集本身
    else: # inductive
        test_list_for_eval = test_list_ind
        # Inductive模式下，评估流是完整的transductive测试集，以保证时序连续性
        full_test_flow = test_list_trans 
    
    # 将 test_list_for_eval 转换为一个set，方便快速查找
    eval_event_set = {tuple(event) for event in test_list_for_eval}

    # 调用评估函数，传入 full_test_flow 和 eval_event_set
    evaluate_link_prediction_optimized(
        args, train_list, val_list, full_test_flow, eval_event_set,
        all_semantic_embeddings, relations_map, node_num
    )


if __name__ == '__main__':
    main()
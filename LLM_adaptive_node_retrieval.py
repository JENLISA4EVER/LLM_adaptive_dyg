# -*- coding: utf-8 -*-
"""
最终版动态图节点检索任务评估脚本。
该脚本整合了以下核心功能：
1.  支持多种可配置的负采样策略来构建包含100个节点的检索池。
2.  利用高效的批量推理（Batch Inference）为100个候选节点进行打分。
3.  使用标准的检索指标 MRR 和 Hits@K 进行评估。
"""
import os
import argparse
import random
import itertools
from collections import defaultdict
import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import util
from vllm import LLM, SamplingParams

# 直接从我们现有的脚本中导入所有需要的函数
from run_recall import get_query, get_llm_data, get_active_i_set, get_candidate_set, get_structural_nodes, get_semantically_similar_nodes, get_or_compute_entity_embeddings
from run_scoring import calculate_candidate_scores, RandomProjectionModule
from LLM_adaptive_link_prediction import get_adaptive_hyperparameters, partition_samples, build_few_shot_prompt, verify_and_create_sample_quadruples_optimized
from utils.realtime_plotter import RealtimePlotter

# =====================================================================================
# 1. 高效的批量打分函数 (来自之前的修改)
# =====================================================================================
def batch_score_candidates_with_vllm(
    prompt_template: str,
    q_head: int,
    retrieval_pool: list,
    anchor_node: int,
    llm: LLM, # 注意：这里传入的是vLLM实例
    tokenizer # vLLM内部也需要tokenizer来获取token ID
) -> list:
    """
    [vLLM版] 使用vLLM引擎通过批量推理，高效地为检索池中所有候选节点打分。
    """
    batch_prompts = []
    for candidate_node in retrieval_pool:
        question1 = f"Question: Which node is more likely to link with node {q_head}?\n  A: {candidate_node}\n  B: {anchor_node}\nOnly A or B can be output, and it is strictly prohibited to output any other content.\nAnswer: "
        batch_prompts.append(prompt_template + question1)
        question2 = f"Question: Which node is more likely to link with node {q_head}?\n  A: {anchor_node}\n  B: {candidate_node}\nOnly A or B can be output, and it is strictly prohibited to output any other content.\nAnswer: "
        batch_prompts.append(prompt_template + question2)

    # 步骤1: 定义采样参数
    # 我们需要获取'A'和'B'的logprobs，所以我们让模型“生成”一个token
    # 并请求返回词汇表中所有token的logprobs

    sampling_params = SamplingParams(
        max_tokens=1,
        logprobs=20
    )

    # 步骤2: 执行一次vLLM的批量生成/推理 (不变)
    outputs = llm.generate(batch_prompts, sampling_params)
    # print(outputs)
    # 步骤3: [修改] 从输出中解析logprobs并计算分数
    token_id_A = tokenizer.convert_tokens_to_ids('A')
    token_id_B = tokenizer.convert_tokens_to_ids('B')

    candidate_scores = []
    for i, candidate_node in enumerate(retrieval_pool):
        # 第一次推理的结果 (A=candidate, B=anchor)
        output1 = outputs[2 * i]
        # [修改] 从生成结果(output)的logprobs中获取概率字典
        # .outputs[0] -> 第一个返回序列 (我们只要求返回1个)
        # .logprobs[0] -> 第一个生成的token的logprobs
        logprobs1 = output1.outputs[0].logprobs[0]

        # 第二次推理的结果 (A=anchor, B=candidate)
        output2 = outputs[2 * i + 1]
        logprobs2 = output2.outputs[0].logprobs[0]

        # 从logprobs字典中获取A和B的对数概率 (这部分逻辑不变)
        logprob_A1 = logprobs1.get(token_id_A, -100.0) 
        logprob_B1 = logprobs1.get(token_id_B, -100.0)
        
        logprob_A2 = logprobs2.get(token_id_A, -100.0)
        logprob_B2 = logprobs2.get(token_id_B, -100.0)

        # 应用Softmax将logprobs转换为概率 (这部分逻辑不变，但需确保是浮点数)
        probs1 = torch.nn.functional.softmax(torch.tensor([logprob_A1, logprob_B1], dtype=torch.float), dim=0)
        probs2 = torch.nn.functional.softmax(torch.tensor([logprob_A2, logprob_B2], dtype=torch.float), dim=0)
        
        # 校准分数 (不变)
        calibrated_score = (probs1[0] + probs2[1]).item() / 2.0
        candidate_scores.append({'node_id': candidate_node, 'score': calibrated_score})
        
    return candidate_scores

def batch_score_candidates_with_llm(
    prompt_template: str,
    q_head: int,
    retrieval_pool: list,
    anchor_node: int,
    tokenizer,
    model
) -> list:
    """
    [修改版] 使用LLM通过批量推理，高效地为检索池中所有候选节点打分。
    内置了mini-batch处理机制，以防止显存溢出。
    """
    # 步骤1: 准备一个包含所有待推理Prompt的大列表 (逻辑不变)
    batch_prompts = []
    for candidate_node in retrieval_pool:
        question1 = f"Question: Which node is more likely to link with node {q_head}?\n  A: {candidate_node}\n  B: {anchor_node}\nAnswer: "
        batch_prompts.append(prompt_template + question1)
        question2 = f"Question: Which node is more likely to link with node {q_head}?\n  A: {anchor_node}\n  B: {candidate_node}\nAnswer: "
        batch_prompts.append(prompt_template + question2)

    # --- [新增] Mini-batch 处理逻辑 ---
    mini_batch_size = 8  # 设置您希望的小批次大小
    all_probs_list = []  # 用于收集每个小批次的处理结果

    # 设置一次tokenizer，供循环使用
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用tqdm为mini-batch处理添加进度条
    for i in tqdm(range(0, len(batch_prompts), mini_batch_size), desc="    - Scoring mini-batches", leave=False):
        # 步骤2: 取出一个小批次的prompts
        mini_batch = batch_prompts[i : i + mini_batch_size]
        
        # 步骤3: 对这个小批次进行Tokenize和推理
        inputs = tokenizer(
            mini_batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]

        # 步骤4: 从Logits计算概率 (仅针对小批次)
        token_id_A = tokenizer.convert_tokens_to_ids('A')
        token_id_B = tokenizer.convert_tokens_to_ids('B')
        
        logits_A = last_token_logits[:, token_id_A]
        logits_B = last_token_logits[:, token_id_B]
        
        stacked_logits = torch.stack([logits_A, logits_B], dim=-1)
        # 将计算出的概率张量移至CPU，以释放GPU显存给下一个批次
        mini_batch_probs = torch.nn.functional.softmax(stacked_logits, dim=-1).cpu()
        
        all_probs_list.append(mini_batch_probs)

    # 步骤5: 将所有小批次的结果合并成一个大张量
    all_probs = torch.cat(all_probs_list, dim=0)
    # --- [新增] Mini-batch 处理结束 ---

    # 步骤6: 校准分数并整理结果 (逻辑不变)
    candidate_scores = []
    for i, candidate_node in enumerate(retrieval_pool):
        prob1 = all_probs[2 * i]
        prob2 = all_probs[2 * i + 1]
        calibrated_score = (prob1[0] + prob2[1]).item() / 2.0
        candidate_scores.append({'node_id': candidate_node, 'score': calibrated_score})
        
    return candidate_scores

# =====================================================================================
# 2. 核心评估函数：整合了所有策略
# =====================================================================================

def evaluate_retrieval_task(args, train_list, val_list, test_list, all_semantic_embeddings, relations_map, node_num):
    print(f"\n--- 5. 开始节点检索评估 [采样策略: {args.neg_sampling_strategy}] ---")
    
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

    # 初始化模块 (与之前脚本相同)
    print("   - Phase 1: 初始化动态模块...")
    initial_history = train_list + val_list
    initial_history.sort(key=lambda x: x[3])
    rp_module = RandomProjectionModule(
        node_num=node_num, edge_num=len(initial_history), dim_factor=args.rp_dim_factor,
        num_layer=args.rp_num_layer, time_decay_weight=args.rp_time_decay_weight,
        device=args.device, beginning_time=initial_history[0][3], enforce_dim=-1
    ).to(args.device)
    batch_size = 200
    for i in tqdm(range(0, len(initial_history), batch_size), desc="   - 初始化 RPModule"):
        batch = initial_history[i:i+batch_size]
        rp_module.update(np.array([e[0] for e in batch]), np.array([e[2] for e in batch]), np.array([e[3] for e in batch]))

    adj = defaultdict(set); node_history = defaultdict(list)
    for u, r, i, ts, _, _ in tqdm(initial_history, desc="   - 初始化邻接表和节点历史"):
        adj[u].add(i); adj[i].add(u)
        node_history[u].append((i, r, ts)); node_history[i].append((u, r, ts))
        
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    '''
    llm = LLM(
        model=args.llm_model_path, 
        trust_remote_code=True, 
        enable_prefix_caching=True  # [新增] 开启前缀缓存功能的关键参数
    )
    tokenizer = llm.get_tokenizer() # 从vLLM实例获取tokenizer
    '''

    plotter = None
    if args.enable_plotting:
        metrics_to_plot = ['MRR'] + [f'Hits@{k}' for k in args.k_values]
        plotter = RealtimePlotter(
            task_name='Node Retrieval',
            save_path='node_retrieval_metrics.pdf',
            metrics=metrics_to_plot
        )

    print("   - 所有模块初始化完成.")

    # 初始化评估指标
    mrr_sum = 0.0
    hits_at_k = {k: 0 for k in args.k_values}
    processed_events_count = 0
    all_node_ids = list(range(node_num))
    
    # 按时间戳处理测试事件
    test_events_grouped_by_time = itertools.groupby(test_list, key=lambda x: x[3])
    for timestamp, events_at_this_time_iter in tqdm(test_events_grouped_by_time, desc="  评估检索任务 (按时间戳)"):
        events_at_this_time = list(events_at_this_time_iter)
        current_structural_embeddings = rp_module.get_all_projections().detach()
        
        for test_event in events_at_this_time:
            processed_events_count += 1
            q_head, q_rel, q_tail_true, q_time = get_query(test_event)

            # 步骤1: 获取自适应超参数
            # 注意：这里需要从您的 LLM_adaptive_link_prediction.py 导入 get_adaptive_hyperparameters
            # 并且在 main 函数中需要定义相关的 argparse 参数
            adaptive_params = get_adaptive_hyperparameters(
                 args, q_head, q_time, node_history, initial_history, all_semantic_embeddings, global_stats
            )

            # 使用自适应参数进行召回、打分、划分
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

            final_scores = calculate_candidate_scores(
                q_head, q_time, candidate_set_C, node_history,
                all_semantic_embeddings, current_structural_embeddings,
                args.lambda_time, 
                adaptive_params['alpha'], 
                adaptive_params['beta']
            )
            future_pos_nodes, negative_nodes, ambiguous_nodes = partition_samples(final_scores, args.p, args.q)

            
            # 步骤2: 根据采样策略，构建包含num_neg个负样本的池子
            negative_samples = set()
            nodes_to_exclude_base = {q_head, q_tail_true} # 基础排除列表
            
            # 定义采样源
            sampling_pool_source = []
            if args.neg_sampling_strategy == 'hard_positive':
                historical_partners = {p for p, _, _ in node_history.get(q_head, []) if _ < q_time}
                future_partners = {e[2] for e in test_list if e[0] == q_head and e[3] > q_time}
                nodes_to_exclude = nodes_to_exclude_base.union(historical_partners, future_partners)
                sampling_pool_source = [n for n in future_pos_nodes if n not in nodes_to_exclude]
            elif args.neg_sampling_strategy == 'ambiguous':
                sampling_pool_source = [n for n in ambiguous_nodes if n not in nodes_to_exclude_base]
            elif args.neg_sampling_strategy == 'negative':
                sampling_pool_source = [n for n in negative_nodes if n not in nodes_to_exclude_base]
            elif args.neg_sampling_strategy == 'recall_pool':
                sampling_pool_source = [n for n in candidate_set_C if n not in nodes_to_exclude_base]
            
            # 从采样源中采样
            if sampling_pool_source:
                num_to_sample = min(args.num_neg, len(sampling_pool_source))
                negative_samples.update(random.sample(sampling_pool_source, num_to_sample))

            # 如果采样不足num_neg个 (源太小或全局采样)，用全局随机采样补足
            while len(negative_samples) < args.num_neg:
                neg_sample = random.choice(all_node_ids)
                if neg_sample not in nodes_to_exclude_base and neg_sample not in negative_samples:
                    negative_samples.add(neg_sample)
            
            # 步骤3: 构建最终的、打乱顺序的检索池
            retrieval_pool = [q_tail_true] + list(negative_samples)
            random.shuffle(retrieval_pool)

            # 步骤4: 构建Prompt模板 (仅提供上下文，不包含具体候选者)
            # 复用划分结果来创建丰富的上下文
            u_history_for_verification = {(r_id, partner, t) for partner, r_id, t in node_history.get(q_head, []) if t < q_time}
            future_pos_quads, neg_quads, ambiguous_quads = verify_and_create_sample_quadruples_optimized(
                q_head, q_time, future_pos_nodes, ambiguous_nodes, negative_nodes, 
                u_history_for_verification, node_history, relations_map
            )
            recent_u_history = node_history.get(q_head, [])[-args.num_golden_samples:]
            golden_pos_quads = [(q_head, relations_map.get(r_id, f"Rel_{r_id}"), partner, t) for partner, r_id, t in recent_u_history if t < q_time]
            prompt_template = build_few_shot_prompt(
                q_head, q_time, golden_pos_quads, future_pos_quads, neg_quads[:20], ambiguous_quads
            )
            
            # 步骤5: 调用批量打分函数
            anchor_node = random.choice(all_node_ids)
            while anchor_node in negative_samples or anchor_node in nodes_to_exclude_base:
                anchor_node = random.choice(all_node_ids)
            
            '''
            candidate_scores = batch_score_candidates_with_vllm(
                prompt_template, q_head, retrieval_pool, anchor_node, llm=llm, tokenizer=tokenizer
            )

            '''
            candidate_scores = batch_score_candidates_with_llm(
                prompt_template, q_head, retrieval_pool, anchor_node, tokenizer, model
            )
            
            
            
            # 步骤6: 排序并计算指标
            sorted_candidates = sorted(candidate_scores, key=lambda x: x['score'], reverse=True)
            try:
                rank = [c['node_id'] for c in sorted_candidates].index(q_tail_true) + 1
            except ValueError:
                rank = 101
            
            mrr_sum += 1.0 / rank
            for k in args.k_values:
                if rank <= k:
                    hits_at_k[k] += 1
            
            # 步骤7: 打印中间结果
            if processed_events_count % args.print_interval == 0:
                current_mrr = mrr_sum / processed_events_count
                current_hits_str = ", ".join([f"H@{k}: {hits_at_k[k] / processed_events_count:.4f}" for k in args.k_values])
                print(f"\n--- 中间指标 (事件 {processed_events_count}/{len(test_list)}) ---")
                print(f"  MRR: {current_mrr:.4f}, {current_hits_str}")
                
                if plotter:
                    metrics_for_plot = {f'Hits@{k}': hits_at_k[k] / processed_events_count for k in args.k_values}
                    metrics_for_plot['MRR'] = current_mrr
                    plotter.update_and_plot(
                        event_count=processed_events_count,
                        current_metrics=metrics_for_plot
                    )

        # 增量更新
        rp_module.update(np.array([e[0] for e in events_at_this_time]), np.array([e[2] for e in events_at_this_time]), np.array([e[3] for e in events_at_this_time]))
        for u, r, i, ts, _, _ in events_at_this_time:
            adj[u].add(i); adj[i].add(u)
            node_history[u].append((i, r, ts)); node_history[i].append((u, r, ts))
        initial_history.extend(events_at_this_time)

    # 打印最终指标
    final_mrr = mrr_sum / processed_events_count
    final_hits_str = "\n".join([f"  Hits@{k}: {hits_at_k[k] / processed_events_count:.4f}" for k in args.k_values])
    print(f"\n--- 最终检索任务指标 ---\n  MRR: {final_mrr:.4f}\n{final_hits_str}")

# =====================================================================================
# 3. 主程序入口
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="动态图节点检索任务评估脚本")
    
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
    parser.add_argument('--print_interval', type=int, default=5, help='每隔多少个测试事件打印一次中间指标')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备 (cuda or cpu)')
    parser.add_argument('--num_neg', type=int, default=99, help='每条数据的负采样数量')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 10], help='用于评估Hits@K的K值列表')
    parser.add_argument('--enable_plotting', action='store_true', help='Enable real-time plotting of performance metrics.')
    
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

    # 设置随机种子
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    # --- 主流程 ---
    print("--- 1. 正在加载和处理数据... ---")
    entities, relations, train_list, val_list, test_list, node_num, rel_num = get_llm_data(
        args.dataset_name, args.base_data_dir, val_ratio=0.15, test_ratio=0.15
    )
    relations_map = {rel_id: rel_text for rel_id, rel_text in relations}

    print("\n--- 2. 正在准备节点语义向量... ---")
    embedding_file_path = os.path.join(args.base_data_dir, args.dataset_name, f"{args.dataset_name}_entity_embeddings.pt")
    all_semantic_embeddings = get_or_compute_entity_embeddings(
        entities=entities, model_path=args.local_model_path, file_path=embedding_file_path, device=args.device
    )

    evaluate_retrieval_task(
        args, train_list, val_list, test_list, 
        all_semantic_embeddings, relations_map, node_num
    )

if __name__ == '__main__':
    main()
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

# 引入我们的模块
from src.config import load_config
from src.data.dataset import UnifiedICLDataset, collate_fn
from src.model.components import RandomProjector
from src.model.llama_wrapper import LlamaWrapper
from src.utils.metrics import extract_answer, calculate_metrics

def compute_alignment_stats(model, projector, train_loader, device, max_samples=2000):
    """
    [核心逻辑] 计算对齐参数
    无需训练，只需计算统计量 (Mean, Std)
    """
    print("📊 Computing Alignment Statistics (Training-Free)...")
    
    # 1. 计算目标分布 (Target Stats) - 来自 LLM 自身的 Embedding
    # 获取 LLM 的 Embedding 权重矩阵
    llm_embeddings = model.llm.get_input_embeddings().weight.detach() # (Vocab, 4096)
    
    # 过滤掉 padding 等零向量 (参考原论文逻辑)
    non_zero_mask = torch.abs(llm_embeddings).sum(dim=1) > 1e-9
    valid_llm_embeds = llm_embeddings[non_zero_mask]
    
    target_mean = valid_llm_embeds.mean().item()
    target_std = valid_llm_embeds.std().item()
    
    print(f"   Target (LLM) Mean: {target_mean:.6f}, Std: {target_std:.6f}")
    
    # 2. 计算源分布 (Source Stats) - 来自我们的 Projector 输出
    # 我们需要跑一部分训练数据，经过 Projector，看看分布长啥样
    source_embeds_bucket = []
    sample_count = 0
    
    for batch in tqdm(train_loader, desc="Calibrating"):
        # 取出 embedding: (Batch, 640)
        embs = batch['net_input']['query_emb'].to(device).to(model.llm.dtype)
        
        # 经过随机投影: (Batch, 640) -> (Batch, 4096)
        with torch.no_grad():
            proj_embs = projector(embs)
            
        source_embeds_bucket.append(proj_embs)
        sample_count += len(embs)
        if sample_count >= max_samples:
            break
            
    all_source_embs = torch.cat(source_embeds_bucket, dim=0)
    
    # 对齐统计量结构体
    align_stats = {
        'target_mean': torch.tensor(target_mean, device=device),
        'target_std': torch.tensor(target_std, device=device),
        'source_mean': all_source_embs.mean(), # 这里可以更精细地按维度算，原论文通常是全局算
        'source_std': all_source_embs.std()
    }
    
    print(f"   Source (Proj) Mean: {align_stats['source_mean']:.6f}, Std: {align_stats['source_std']:.6f}")
    
    return align_stats

def main():
    # 1. 加载配置
    cfg = load_config()
    print(f"🚀 Starting ICRL Inference: {cfg.experiment.name}")
    print(f"   Task: {cfg.data.task_name} | Model: {cfg.llm.model_path}")
    
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据
    data_path = os.path.join(cfg.data.get("data_root", "./data"), "processed_data.pt")
    # 如果 preprocess 保存路径不一样，请在这里修改，或者从 cfg.experiment.save_dir 读取
    # 假设 preprocess.py 保存到了 logs/... 下，为了方便我们先试探性读取
    if not os.path.exists(data_path):
        # 尝试从 logs 目录找
        data_path = os.path.join(cfg.experiment.save_dir, "processed_data.pt")
        
    print(f"📂 Reading data from: {data_path}")
    
    # 训练集用于提供 Shots 和 校准分布
    train_dataset = UnifiedICLDataset(data_path, n_shots=cfg.icl.n_total_shots, mode='train')
    # 测试集用于评估
    test_dataset = UnifiedICLDataset(data_path, n_shots=cfg.icl.n_total_shots, mode='test')
    
    # DataLoader
    # 校准不需要 shuffle，也不需要 shots，只取 query_emb 即可，但复用 collate_fn 方便
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.icl.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 3. 初始化组件
    # 自动检测输入维度 (从数据集里拿一个看看)
    sample_emb = train_dataset[0]['query']['emb']
    input_dim = sample_emb.shape[0] # 通常是 640 (ESM-2 150M) 或 1280
    output_dim = 4096 # Llama-3 8B hidden size
    
    print(f"🔧 Initializing Projector: {input_dim} -> {output_dim}")
    projector = RandomProjector(input_dim, output_dim).to(device)
    
    # 加载 Llama
    model = LlamaWrapper(cfg, projector).to(device)
    
    # 4. [Phase 1] 统计对齐 (Calibration)
    align_stats = compute_alignment_stats(model, projector, train_loader, device)
    
    # 5. [Phase 2] 推理评估 (Inference)
    print("\n🔮 Starting Inference on Test Set...")
    all_preds = []
    all_targets = []
    
    # 创建结果保存目录
    res_dir = os.path.join(cfg.experiment.save_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    f_log = open(os.path.join(res_dir, "predictions.txt"), "w")
    
    for batch in tqdm(test_loader, desc="Testing"):
        net_input = batch['net_input']
        targets = batch['target'].numpy()
        
        # 生成文本
        # 注意：generate 内部会自动调用 projector 和 apply_alignment
        decoded_outputs = model.generate(net_input, align_stats=align_stats)
        
        for i, text in enumerate(decoded_outputs):
            # 提取数值
            pred_val = extract_answer(text)
            
            all_preds.append(pred_val)
            all_targets.append(targets[i])
            
            # 实时打印/保存日志
            # Llama3 的输出可能包含 prompt，我们需要截取 assistant 的部分
            # 由于 decode_outputs 是纯生成的文本（如果使用 model.generate 的话）
            # 或者包含 prompt（如果配置不同）。
            # 这里的 LlamaWrapper.generate 返回的是纯生成的文本部分吗？
            # 检查 LlamaWrapper 代码: tokenizer.batch_decode(outputs) 会包含所有 input_ids
            # 所以我们需要截取。
            
            # 简单处理：提取 prompt 之后的文本
            # 更好的做法是在 LlamaWrapper 里只 decode 新生成的 token
            # 假设 metrics.extract_answer 足够鲁棒能处理全文
            
            log_str = f"GT: {targets[i]:.4f} | Pred: {pred_val} | Raw: {text[-50:].replace(chr(10), ' ')}"
            f_log.write(log_str + "\n")
            
            # 简单 debug
            if i == 0:
                print(f"\n[Sample Output]\n{text[-100:]}\n--> Parsed: {pred_val}")

    f_log.close()
    
    # 6. 计算最终指标
    print("\n📈 Calculating Metrics...")
    metrics = calculate_metrics(all_preds, all_targets)
    
    print("="*40)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"Valid Outputs: {metrics['valid_count']}/{metrics['total_count']}")
    print("="*40)
    
    # 保存指标
    pd.DataFrame([metrics]).to_csv(os.path.join(res_dir, "metrics.csv"), index=False)

if __name__ == "__main__":
    main()
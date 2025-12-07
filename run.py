import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入我们重构的模块
from src.config import load_config
from src.data.loader import get_data_loader, collate_fn
from src.data.processors.factory import ProcessorFactory
from src.model.llama_wrapper import LlamaWrapper
from src.data.utils.metrics import calculate_metrics, extract_answer

def main():
    # ==========================================
    # 1. 初始化与配置加载
    # ==========================================
    cfg = load_config()
    print(f"🚀 Starting Experiment: {cfg.experiment.name}")
    print(f"   Task: {cfg.data.task_name} | Method: {cfg.method.name}")
    
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.experiment.save_dir, exist_ok=True)

    # ==========================================
    # 2. 数据加载 (Data Loading)
    # ==========================================
    # get_data_loader 内部使用了 TaskFactory 和 Cache 自动对齐
    print("\n📂 Loading Data & Aligning Caches...")
    datasets = get_data_loader(cfg)
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    print(f"   Train Size: {len(train_dataset)} | Test Size: {len(test_dataset)}")

    # ==========================================
    # 3. 特征处理 (Processing / Alignment)
    # ==========================================
    # 这一步是核心：无论是 Spectral 还是 Original ICRL，都在这里把
    # 原始的 [N, 512] 变为对齐后的 [N, 4096]
    print(f"\n⚙️ Running Processor: {cfg.method.name}...")
    processor = ProcessorFactory.get_processor(cfg)
    
    # Fit: 在训练集上计算统计量 (如 Mean, Std, Fiedler Vector, PCA Matrix)
    processor.fit(train_dataset)
    
    # Transform: 将变换应用到训练集和测试集
    # 注意：这会直接修改 dataset 内部的 input1_emb 等字段，或者返回新的 Tensor
    # 建议 processor 内部实现 update_dataset 方法
    processor.transform_dataset(train_dataset)
    processor.transform_dataset(test_dataset)
    
    print("   ✅ Processing Complete. Embeddings are now aligned to LLM space.")

    # ==========================================
    # 4. 准备 DataLoader
    # ==========================================
    # Process 之后再建立 Loader，因为 Tensor 已经变了
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.icl.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn # 需确保 collate_fn 能处理 input1_emb
    )

    # ==========================================
    # 5. 模型初始化 (Model Initialization)
    # ==========================================
    print(f"\n🤖 Initializing Llama Wrapper...")
    model = LlamaWrapper(cfg).to(device)
    model.eval()

    # ==========================================
    # 6. 推理循环 (Inference Loop)
    # ==========================================
    print("\n🔮 Starting Inference...")
    all_preds = []
    all_targets = []
    results_log = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # batch 已经在 collate_fn 里被处理成 Tensor (如果是 embedding)
            
            # 生成文本
            outputs = model.generate(batch)
            
            # 后处理与评估
            targets = batch['label'] # 假设 dataset 返回 label
            
            for i, text in enumerate(outputs):
                # 提取数值答案
                pred_val = extract_answer(text)
                target_val = targets[i]
                
                all_preds.append(pred_val)
                all_targets.append(target_val)
                
                results_log.append({
                    "target": target_val,
                    "prediction": pred_val,
                    "output_text": text
                })

    # ==========================================
    # 7. 结果保存与计算 (Metrics)
    # ==========================================
    print("\n📈 Calculating Metrics...")
    # 过滤无效预测 (None)
    valid_preds = [p if p is not None else 0.0 for p in all_preds]
    
    metrics = calculate_metrics(valid_preds, all_targets)
    metrics['valid_rate'] = sum(1 for p in all_preds if p is not None) / len(all_preds)
    
    print("="*40)
    print(f"RMSE: {metrics.get('rmse', 'N/A')}")
    print(f"Pearson: {metrics.get('pearson', 'N/A')}")
    print(f"Valid Rate: {metrics['valid_rate']:.2%}")
    print("="*40)
    
    # 保存结果
    res_path = os.path.join(cfg.experiment.save_dir, "results.csv")
    pd.DataFrame(results_log).to_csv(res_path, index=False)
    print(f"💾 Results saved to {res_path}")

if __name__ == "__main__":
    main()